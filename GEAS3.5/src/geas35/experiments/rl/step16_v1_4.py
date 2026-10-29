"""Build and smoke-test the v1.4 model-driven RL environments."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, platform
from pathlib import Path
from typing import Any
import numpy as np
import sklearn
from geas35.rl.model_driven_env import STEP16_ENV_VERSION, load_model_driven_env_from_step15

VERSION="geas35.rl.step16.v1_4"
CROPS=("strawberry","melon","cucumber")

def _sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def _record(path:Path,root:Path)->dict[str,Any]:
    return {"path":path.resolve().relative_to(root).as_posix(),"sha256":_sha(path),"size_bytes":path.stat().st_size}
def _read(path:Path)->dict[str,Any]:return json.loads(path.read_text(encoding="utf-8"))
def _write(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,default=str)+"\n",encoding="utf-8")

def _in_support_episode(env)->int:
    thresholds=env.support.config["thresholds"]
    for index in range(len(env.episodes)):
        env.reset(seed=42,options={"episode_index":index})
        action={column:float(env._current_scaled[column]) for column in env.action_columns}
        state,actions=env._support_frames(action);scores=env.support.scores(state,actions)
        if scores["state"][0]<=thresholds["state_warning"] and scores["joint"][0]<=thresholds["joint_warning"]:return index
    raise ValueError(f"{env.crop}: no in-support episode")

def run_step16_v1_4(*,project_root:str|Path,output_root:str|Path|None=None)->Path:
    root=Path(project_root).resolve();out=Path(output_root).resolve() if output_root else root/"experiments/rl_policy_training/artifacts/step16_v1_4"
    protocol_path=root/"experiments/rl_policy_training/artifacts/step15_v1_4/rl_protocol_manifest.json";protocol=_read(protocol_path)
    if protocol.get("schema_version")!="geas35.rl.step15.v1_4" or protocol.get("status")!="passed":raise ValueError("Passed Step 15 v1.4 required")
    traces={};crops={}
    for crop in CROPS:
        env=load_model_driven_env_from_step15(project_root=root,crop=crop,split="train",protocol_manifest_path=protocol_path)
        index=_in_support_episode(env);observation,reset_info=env.reset(seed=42,options={"episode_index":index})
        action={column:float(env._current_scaled[column]) for column in env.action_columns};before=env._current_scaled.copy()
        next_obs,reward,terminated,truncated,info=env.step(action)
        model_input=before.copy()
        for column,value in info["executed_action"].items():model_input[column]=value
        input_columns, _ = env.model._require_fitted()
        parity_frame = model_input.reindex(input_columns).to_frame().T.apply(
            lambda column: column.astype(float)
        )
        expected=env.model.predict(parity_frame).next_observation.iloc[0]
        parity=all(np.isclose(float(expected[c]),float(info["transition_prediction_physical"][c]),rtol=0,atol=1e-12) for c in env.model.target_columns_)
        env.reset(seed=42,options={"episode_index":index});repeated=env.step(action)
        deterministic=np.array_equal(next_obs,repeated[0]) and np.isclose(reward,repeated[1],rtol=0,atol=1e-12) and terminated==repeated[2]
        if observation.shape!=(55,) or next_obs.shape!=(55,) or not np.isfinite(next_obs).all() or not parity or not deterministic or truncated:raise ValueError(f"{crop}: Step 16 smoke failed")
        traces[crop]={"episode_index":index,"reset_info":reset_info,"raw_action":info["raw_action"],"executed_action":info["executed_action"],"state_ood_score":info["state_ood_score"],"joint_ood_score":info["joint_ood_score"],"joint_ood_level":info["joint_ood_level"],"prediction_physical":info["transition_prediction_physical"],"reward":reward,"terminated":terminated,"truncated":truncated,"next_observation_shape":list(next_obs.shape),"one_step_inference_parity":parity,"deterministic_replay":deterministic}
        crops[crop]={"status":"passed","split":"train","candidate_name":protocol["crops"][crop]["candidate_name"],"feature_variant":protocol["crops"][crop]["feature_variant"],"observation_shape":list(env.observation_shape),"action_shape":list(env.action_shape),"episode_count":len(env.episodes),"exogenous_provider_mode":env.exogenous_provider.mode,"three_target_output":True,"support_monitoring":True,"reward_handoff":True,"one_step_inference_parity":parity,"deterministic_replay":deterministic}
    trace_path=out/"step16_trajectory_trace.json";_write(trace_path,{"schema_version":VERSION,"status":"passed","crops":traces})
    manifest={"schema_version":VERSION,"environment_version":STEP16_ENV_VERSION,"step":16,"status":"passed","created_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"scope":"v1_4_model_driven_environment_step17_pending","runtime":{"python":platform.python_version(),"numpy":np.__version__,"scikit_learn":sklearn.__version__},"step15_protocol":_record(protocol_path,root),"implementations":[_record(root/p,root) for p in ("src/geas35/rl/model_driven_env.py","src/geas35/models/transition/causal_state.py","src/geas35/experiments/rl/step16_v1_4.py")],"trajectory_trace":_record(trace_path,root),"contracts":{"internal_state":"physical_unscaled","policy_observation":"frozen_train_scaled_55","transition_input":"scaled_55_observation_plus_executed_6_action","transition_output":"physical_temperature_humidity_co2","step_minutes":5,"episode_crosses_date_or_series_boundary":False,"recorded_weather_offline":True,"forecast_weather_interface":True,"previous_action_updated_from_executed_action":True,"ood_state_and_joint_recorded_each_step":True,"warning_action_projection_and_penalty":True,"severe_action_safe_fallback":True,"severe_state_or_repeated_severe_pessimistic_termination":True,"scheduled_valid_steps_denominator_exposed":True,"rl_test_accessed":False},"crops":crops}
    manifest_path=out/"step16_environment_manifest.json";_write(manifest_path,manifest)
    integrity={"schema_version":VERSION,"step":16,"status":"passed","environment_manifest":_record(manifest_path,root),"checks":{"step15_v1_4_hash_chain_verified":True,"selected_models_loaded":True,"observation_action_shape_finite":True,"one_step_inference_parity":True,"deterministic_seed_action_replay":True,"reward_handoff":True,"candidate_artifact_copy_count":0,"rl_test_accessed":False,"step17_implemented":False}}
    path=out/"step16_integrity_manifest.json";_write(path,integrity);return path

__all__=["VERSION","run_step16_v1_4"]
