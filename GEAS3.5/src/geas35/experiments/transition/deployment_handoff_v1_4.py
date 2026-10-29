"""Step 14 v1.4: freeze researcher choices, evaluate Test once, and hand off."""
from __future__ import annotations
import hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from geas35.experiments.transition.config import load_transition_experiment_config, read_configured_transition_frames
from geas35.experiments.transition.reproducibility_review_v1_4 import SELECTED_CANDIDATES, STEP13_V1_4_VERSION
from geas35.io_utils import write_json
from geas35.models.transition.causal_state import load_frozen_observation_scaler
from geas35.models.transition.datasets import build_transition_dataset_from_rl_frame
from geas35.models.transition.evaluator import evaluate_transition_model_one_step
from geas35.models.transition.feature_variants import observation_columns_for_variant, prepare_frame_for_feature_variant, project_frozen_observation_scaler
from geas35.models.transition.hyperparameters import transition_target_normalization_scales
from geas35.models.transition.inference import load_transition_model_from_artifact_dir, predict_next_observation_reward
from geas35.models.transition.trainer import _evaluate_rollouts
from geas35.rl import MDP_V1_ACTION_COLUMNS

STEP14_V1_4_VERSION="geas35.transition.step14.v1_4"
DECISION_VERSION="geas35.transition.researcher_decision.v1_4"
CROPS=("strawberry","melon","cucumber")

class Step14V14Error(ValueError): pass

def run_step14_v1_4(*, project_root:str|Path, decision_author:str="researcher",
                    step13_manifest_path:str|Path|None=None, output_root:str|Path|None=None)->Path:
    root=Path(project_root).resolve()
    step13_path=_resolved(step13_manifest_path,root/"experiments/transition_model_selection/artifacts/step13_v1_4/step13_v1_4_reproducibility_manifest.json")
    out=_resolved(output_root,root/"experiments/transition_model_selection/artifacts/step14_v1_4")
    official=root/"experiments/transition_model_selection/artifacts/step12_v1_4_root_cause/official_rerun"
    s13=_read(step13_path)
    _require(s13.get("schema_version")==STEP13_V1_4_VERSION and s13.get("status")=="passed","Step 13 v1.4 is not passed")
    _require(s13.get("test_opened") is False,"Step 13 Test lock is invalid")
    _require(not (out/"transition_test_access_log.json").exists(),"Step 14 Test was already accessed; refusing a second evaluation")
    out.mkdir(parents=True,exist_ok=True)

    decisions={}
    limitations={
      "strawberry":[],
      "melon":["Negative 60-minute CO2 Persistence skill.","Only one action has sufficient action-change support."],
      "cucumber":[],
    }
    for crop,(variant,model) in SELECTED_CANDIDATES.items():
        ad=official/variant/crop/crop/model
        tr=_read(ad/"training_summary.json"); ro=_read(ad/"rollout_metrics.json"); ma=_read(ad/"manifest.json")
        decisions[crop]={"candidate_name":model,"feature_variant":variant,"model_sha256":_sha(ad/"model.pkl"),
          "schema_hash":ma["schema_hash"],"selection_basis":"Validation only",
          "validation_rollout_weighted_score":tr["validation_rollout_weighted_score"],
          "validation_60min_nrmse":ro["60min"]["trajectory_nrmse"],
          "validation_60min_physical_violation_rate":ro["60min"]["physical_violation_rate"],
          "limitations":limitations[crop],"test_observed_at_decision_time":False}
    decision={"schema_version":DECISION_VERSION,"decision_author":decision_author,"decided_at_utc":_now(),
      "decision_source":"researcher_explicit_approval","framework_generated_recommendation":False,
      "reselection_after_test_prohibited":True,"decisions":decisions}
    decision_path=out/"researcher_decision.json"; write_json(decision_path,decision,convert=True)
    auth={"schema_version":STEP14_V1_4_VERSION,"stage":"transition_test_authorization","authorized_at_utc":_now(),
      "authorized":True,"scope":"one Test evaluation for each of the three frozen candidates",
      "researcher_decision":_record(decision_path,root),"step13_manifest":_record(step13_path,root),
      "authorized_candidates":decisions,"test_results_must_not_change_selection":True}
    auth_path=out/"transition_test_authorization_manifest.json"; write_json(auth_path,auth,convert=True)
    decision_hash=_sha(decision_path); auth_hash=_sha(auth_path)

    events=[]; results={}; handoff={}
    for crop in CROPS:
        variant,model=SELECTED_CANDIDATES[crop]
        suffix="with_flags" if variant.endswith("with_quality_flags") else "without_flags"
        cfg=root/"experiments/transition_model_selection/configs"/f"{crop}_v1_4_{suffix}.yaml"
        loaded=load_transition_experiment_config(cfg)
        ad=official/variant/crop/crop/model
        _require(_sha(ad/"model.pkl")==decisions[crop]["model_sha256"],f"{crop}: model changed after decision")
        _require(_sha(decision_path)==decision_hash and _sha(auth_path)==auth_hash,"Decision or authorization changed")
        opened=_now(); train,val,test,rollout=read_configured_transition_frames(loaded,include_test=True)
        del train,val,rollout
        _require(test is not None,f"{crop}: Test not loaded")
        events.append({"crop":crop,"split":"test","opened_at_utc":opened,"path":str(loaded.test_path),
          "dataset_sha256":_sha(loaded.test_path),"candidate":model,"feature_variant":variant,"authorization_sha256":auth_hash})
        frame=prepare_frame_for_feature_variant(test,variant); obs=observation_columns_for_variant(variant)
        ds=build_transition_dataset_from_rl_frame(frame,observation_columns=obs).dataset
        source,_=load_frozen_observation_scaler(loaded.scaler_path); scaler=project_frozen_observation_scaler(source,obs)
        bundle=load_transition_model_from_artifact_dir(ad,expected_target_formulation="residual_physical_delta_v1")
        scales=transition_target_normalization_scales(ds.y,ds.target_columns)
        one=evaluate_transition_model_one_step(bundle.model,ds,normalization_scales=scales).to_artifact()
        multi=_evaluate_rollouts(bundle.model,rollout_frame=frame,rollout_horizon_steps=(3,6,12),rollout_step_minutes=5,
          action_provider_factory=None,normalization_scales=scales,observation_scaler=scaler,hpo_mode=False)
        row=frame.loc[frame["rl_valid_transition"].astype(bool)].iloc[0]
        action={c:float(row[c]) for c in MDP_V1_ACTION_COLUMNS}; reward=predict_next_observation_reward(bundle,row,action)
        safety=all(float(multi[h]["physical_violation_rate"])==0 and float(multi[h]["nan_inf_count"])==0 for h in ("15min","30min","60min"))
        status="passed" if safety and math.isfinite(reward.reward) else "deployment_hold"
        payload={"schema_version":STEP14_V1_4_VERSION,"crop":crop,"candidate":model,"feature_variant":variant,
          "status":status,"test_access_count":1,"test_dataset_sha256":_sha(loaded.test_path),"one_step":one,"rollout":multi,
          "reward_handoff":{"finite":math.isfinite(reward.reward),"reward":reward.reward,"term_count":len(reward.reward_terms)},
          "test_used_for_selection":False,"reselection_after_test_prohibited":True}
        rp=out/crop/"selected_model_test_metrics.json"; rp.parent.mkdir(parents=True,exist_ok=True); write_json(rp,payload,convert=True)
        results[crop]={"status":status,"metrics":_record(rp,root)}
        handoff[crop]={"candidate_name":model,"feature_variant":variant,"candidate_artifact_dir":_rel(ad,root),
          "candidate_model_sha256":_sha(ad/"model.pkl"),"schema_hash":decisions[crop]["schema_hash"],
          "observation_scaler":_record(official/variant/crop/"observation_scaler.json",root),"test_status":status,
          "test_metrics":_record(rp,root),"artifact_copied":False,"explicit_load_verified":True,"reward_handoff_verified":math.isfinite(reward.reward)}

    log_path=out/"transition_test_access_log.json"; write_json(log_path,{"schema_version":STEP14_V1_4_VERSION,"total_access_count":3,"events":events},convert=True)
    overall="passed" if all(x["status"]=="passed" for x in results.values()) else "deployment_hold"
    hp=out/"explicit_candidate_deployment_handoff.json"; write_json(hp,{"schema_version":STEP14_V1_4_VERSION,
      "stage":"explicit_transition_deployment_handoff","status":overall,"created_at_utc":_now(),"automatic_model_selection":False,
      "candidate_artifact_copy_count":0,"researcher_decision":_record(decision_path,root),"step13_manifest":_record(step13_path,root),"candidates":handoff},convert=True)
    audit_path=out/"transition_v1_4_completion_audit.json"; write_json(audit_path,{"schema_version":STEP14_V1_4_VERSION,
      "status":overall,"checks":{"step13_passed":True,"decision_frozen_before_test":True,"authorization_frozen_before_test":True,
      "only_selected_candidates_tested":True,"test_access_count":3,"test_used_for_reselection":False,
      "all_test_safety_gates_passed":overall=="passed","candidate_artifact_copy_count":0},"test_results":results,
      "test_access_log":_record(log_path,root),"deployment_handoff":_record(hp,root)},convert=True)
    manifest={"schema_version":STEP14_V1_4_VERSION,"step":14,"status":overall,"researcher_decision":_record(decision_path,root),
      "test_authorization":_record(auth_path,root),"test_access_log":_record(log_path,root),"deployment_handoff":_record(hp,root),
      "completion_audit":_record(audit_path,root),"selected_candidate_count":3,"test_access_count":3,
      "automatic_model_selection":False,"candidate_artifact_copy_count":0}
    mp=out/"step14_handoff_manifest.json"; write_json(mp,manifest,convert=True)
    lines=["# Step 14 Transition Model Handoff","",f"- Status: {overall}","- Selection basis: Validation only","- Test access: once per selected crop model","- Artifact copies: 0",""]
    lines += [f"- {c}: `{handoff[c]['candidate_name']}` / `{handoff[c]['feature_variant']}` — Test {handoff[c]['test_status']}" for c in CROPS]
    lines += ["","Test results were not used for reselection.",""]
    (out/"step14_handoff_summary.md").write_text("\n".join(lines),encoding="utf-8")
    return mp

def _now(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def _read(p:Path):
    d=json.loads(p.read_text(encoding="utf-8")); _require(isinstance(d,dict),f"JSON object required: {p}"); return d
def _sha(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def _rel(p:Path,r:Path):
    try:return str(p.resolve().relative_to(r.resolve()))
    except ValueError:return str(p.resolve())
def _record(p:Path,r:Path): return {"path":_rel(p,r),"sha256":_sha(p),"size_bytes":p.stat().st_size}
def _resolved(v,default): return (default if v is None else Path(v)).expanduser().resolve()
def _require(ok,msg):
    if not ok: raise Step14V14Error(msg)

__all__=["DECISION_VERSION","STEP14_V1_4_VERSION","Step14V14Error","run_step14_v1_4"]
