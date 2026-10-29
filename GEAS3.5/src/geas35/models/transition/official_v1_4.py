"""Official six-combination transition rerun and MLP restart orchestration v1.4."""
from __future__ import annotations
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence
import numpy as np

from geas35.experiments.transition.cli import run_from_config
from geas35.experiments.transition.config import load_transition_experiment_config, read_configured_transition_frames
from geas35.io_utils import write_json
from geas35.models.transition.action_response import evaluate_action_response
from geas35.models.transition.causal_state import load_frozen_observation_scaler
from geas35.models.transition.datasets import build_transition_dataset_from_rl_frame
from geas35.models.transition.evaluator import evaluate_transition_model_one_step
from geas35.models.transition.feature_variants import observation_columns_for_variant, prepare_frame_for_feature_variant, project_frozen_observation_scaler
from geas35.models.transition.hyperparameters import transition_target_normalization_scales, validation_rollout_weighted_score
from geas35.models.transition.sklearn_models import MLPTransitionModel
from geas35.models.transition.trainer import _evaluate_rollouts
from geas35.models.transition.base import TransitionTargetFormulation

SEEDS=(42,43,44)


def run_official_combination(config_path: str | Path, *, seeds: Sequence[int]=SEEDS) -> dict[str,Any]:
    loaded=load_transition_experiment_config(config_path)
    output=loaded.experiment_config.output_root
    try:
        experiment=run_from_config(config_path)
    except Exception as exc:
        payload={"status":"execution_error","stage":"candidate_experiment","error_type":type(exc).__name__,"message":str(exc),"test_opened":False}
        _atomic_json(output/"official_status.json",payload)
        return payload
    failures=list(experiment.experiment_result.training_result.candidate_failures)
    mlp=next((r for r in experiment.experiment_result.training_result.candidate_results if r.model_name=="mlp"),None)
    if mlp is None or not mlp.hpo_result.has_eligible_trial:
        payload={"status":"no_eligible_model","candidate_failures":failures,"test_opened":False}
        _atomic_json(output/"official_status.json",payload)
        return payload
    restart=run_mlp_restarts(config_path,mlp.hpo_result.best_params,seeds=seeds)
    payload={"status":"completed" if restart["status"]=="completed" else restart["status"],"candidate_failures":failures,"mlp_restarts":restart,"test_opened":False}
    _atomic_json(output/"official_status.json",payload)
    return payload


def run_mlp_restarts(config_path: str|Path, best_params: Mapping[str,Any], *, seeds:Sequence[int]=SEEDS)->dict[str,Any]:
    loaded=load_transition_experiment_config(config_path)
    variant=loaded.experiment_config.feature_variant
    if variant is None: raise ValueError("v1.4 official restart requires an explicit feature variant")
    train,val,test,rollout=read_configured_transition_frames(loaded,include_test=False)
    if test is not None: raise AssertionError("Test opened during official restart")
    train=prepare_frame_for_feature_variant(train,variant); val=prepare_frame_for_feature_variant(val,variant)
    rollout=val if rollout is None else prepare_frame_for_feature_variant(rollout,variant)
    columns=observation_columns_for_variant(variant)
    original,_=load_frozen_observation_scaler(loaded.scaler_path)
    scaler=project_frozen_observation_scaler(original,columns)
    train_ds=build_transition_dataset_from_rl_frame(train,observation_columns=columns).dataset
    val_ds=build_transition_dataset_from_rl_frame(val,observation_columns=columns).dataset
    scales=transition_target_normalization_scales(val_ds.y,val_ds.target_columns)
    root=loaded.experiment_config.output_root/loaded.experiment_config.crop/"mlp"/"restarts"
    rows=[]
    for seed in seeds:
        destination=root/f"seed_{seed}"; destination.mkdir(parents=True,exist_ok=True)
        try:
            params=json.loads(json.dumps(best_params)); params["random_state"]=int(seed)
            model=MLPTransitionModel(**params).configure_target_formulation(TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA,observation_scaler=scaler)
            model.fit(train_ds,validation_dataset=val_ds)
            one=evaluate_transition_model_one_step(model,val_ds,normalization_scales=scales)
            multi=_evaluate_rollouts(model,rollout_frame=rollout,rollout_horizon_steps=(3,6,12),rollout_step_minutes=5,action_provider_factory=None,normalization_scales=scales,observation_scaler=scaler,hpo_mode=False)
            score,components=validation_rollout_weighted_score(one,multi,objective_weights=loaded.experiment_config.hpo_config.objective_weights)
            action=evaluate_action_response(model,val_ds)
            finite=np.isfinite(score) and all(np.isfinite(v) for v in one.aggregate_metrics.values() if isinstance(v,(int,float)))
            safety=all(float(m.get("safety_gate_passed",0))==1 and float(m.get("physical_violation_rate",0))==0 and float(m.get("nan_inf_count",0))==0 for m in multi.values())
            eligible=bool(finite and safety and action.eligible)
            with (destination/"model.pkl.tmp").open("wb") as handle: pickle.dump(model,handle)
            (destination/"model.pkl.tmp").replace(destination/"model.pkl")
            row={"seed":int(seed),"status":"completed","eligible":eligible,"validation_rollout_weighted_score":float(score),"objective_components":components,"one_step":one.to_artifact(),"rollout":multi,"action_response":action.to_artifact(),"params":params,"test_opened":False}
            _atomic_json(destination/"completed.json",row); rows.append(row)
        except Exception as exc:
            row={"seed":int(seed),"status":"execution_error","eligible":False,"error_type":type(exc).__name__,"message":str(exc),"test_opened":False}
            _atomic_json(destination/"failure.json",row); rows.append(row)
    eligible=sorted((r for r in rows if r.get("eligible")),key=lambda r:(r["validation_rollout_weighted_score"],r["seed"]))
    if not eligible:
        result={"status":"no_eligible_seed","rows":rows,"selected":None,"test_opened":False}
    else:
        selected=eligible[len(eligible)//2]
        result={"status":"completed","selection_rule":"median_validation_score","rows":rows,"selected":{"seed":selected["seed"],"validation_rollout_weighted_score":selected["validation_rollout_weighted_score"],"checkpoint":str(root/f"seed_{selected['seed']}"/"model.pkl")},"score_summary":{"mean":float(np.mean([r['validation_rollout_weighted_score'] for r in eligible])),"std":float(np.std([r['validation_rollout_weighted_score'] for r in eligible])),"min":float(np.min([r['validation_rollout_weighted_score'] for r in eligible])),"max":float(np.max([r['validation_rollout_weighted_score'] for r in eligible]))},"test_opened":False}
    _atomic_json(root.parent/"restart_summary.json",result)
    return result


def _atomic_json(path:Path,payload:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp")
    write_json(temporary,payload,convert=True); temporary.replace(path)

__all__=["run_mlp_restarts","run_official_combination"]
