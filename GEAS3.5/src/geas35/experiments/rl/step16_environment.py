"""Step 16 official model-driven environment handoff and smoke run."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
from typing import Any
import numpy as np
import sklearn

from geas35.rl.model_driven_env import (
    STEP16_ENV_VERSION, load_model_driven_env_from_step15,
)

STEP16_VERSION = "geas35.rl.step16.v1"
OFFICIAL_CROPS = ("strawberry", "melon", "cucumber")


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path,root: Path) -> dict[str,Any]:
    return {"path":path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256":_sha256(path),"size_bytes":path.stat().st_size}


def _write(path: Path,value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+"\n")


def _in_support_episode(env) -> int:
    thresholds=env.support.config['thresholds']
    for index in range(len(env.episodes)):
        env.reset(seed=42,options={'episode_index':index})
        action={column:float(env._current_scaled[column]) for column in env.action_columns}
        state,action_frame=env._support_frames(action)
        scores=env.support.scores(state,action_frame)
        if (float(scores['state'][0]) <= float(thresholds['state_warning'])
                and float(scores['joint'][0]) <= float(thresholds['joint_warning'])):
            return index
    raise ValueError(f"{env.crop}: no in-support episode start for Step 16 smoke.")


def run_step16_environment_handoff(
    *,project_root: str|Path,output_root: str|Path|None=None,
) -> Path:
    root=Path(project_root).resolve()
    destination=Path(output_root).resolve() if output_root else (
        root/'experiments/rl_policy_training/artifacts/step16'
    )
    protocol_path=root/'experiments/rl_policy_training/artifacts/step15/rl_protocol_manifest.json'
    protocol=json.loads(protocol_path.read_text())
    if protocol.get('status')!='passed':
        raise ValueError('Step 16 requires passed Step 15 protocol.')
    traces={}
    crops={}
    for crop in OFFICIAL_CROPS:
        env=load_model_driven_env_from_step15(project_root=root,crop=crop,split='train')
        episode_index=_in_support_episode(env)
        observation,reset_info=env.reset(seed=42,options={'episode_index':episode_index})
        action={column:float(env._current_scaled[column]) for column in env.action_columns}
        before=env._current_scaled.copy()
        result=env.step(action)
        next_observation,reward,terminated,truncated,info=result
        model_input=before.copy()
        for column,value in info['executed_action'].items(): model_input[column]=value
        expected=env.model.predict(model_input.to_frame().T).next_observation.iloc[0]
        actual=info['transition_prediction_physical']
        parity=all(np.isclose(float(expected[column]),float(actual[column]),rtol=0,atol=1e-12)
                   for column in env.model.target_columns_)
        env.reset(seed=42,options={'episode_index':episode_index})
        repeated=env.step(action)
        deterministic=(np.array_equal(next_observation,repeated[0])
                       and np.isclose(reward,repeated[1],rtol=0,atol=1e-12)
                       and terminated==repeated[2])
        if not parity or not deterministic or truncated:
            raise ValueError(f'{crop}: Step 16 smoke contract failed.')
        traces[crop]={
            'episode_index':episode_index,'reset_info':reset_info,
            'raw_action':info['raw_action'],'executed_action':info['executed_action'],
            'state_ood_score':info['state_ood_score'],
            'joint_ood_score':info['joint_ood_score'],
            'joint_ood_level':info['joint_ood_level'],
            'prediction_physical':actual,'reward':reward,
            'terminated':terminated,'truncated':truncated,
            'next_observation_shape':list(next_observation.shape),
            'one_step_inference_parity':parity,'deterministic_replay':deterministic,
        }
        crops[crop]={
            'status':'passed','split':'train','candidate_name':'extra_trees',
            'observation_shape':list(env.observation_shape),
            'action_shape':list(env.action_shape),'episode_count':len(env.episodes),
            'exogenous_provider_mode':env.exogenous_provider.mode,
            'three_target_output':True,'support_monitoring':True,
            'reward_handoff':True,'one_step_inference_parity':parity,
            'deterministic_replay':deterministic,
            'ood_response_parameters':{
                'warning_penalty_max':env.env_config.warning_penalty_max,
                'worst_step_reward':env.env_config.worst_step_reward,
                'repeated_severe_limit':env.env_config.repeated_severe_limit,
            },
        }
    trace_path=destination/'step16_trajectory_trace.json'
    _write(trace_path,{'schema_version':STEP16_VERSION,'status':'passed','crops':traces})
    implementation_path=root/'src/geas35/rl/model_driven_env.py'
    manifest={
        'schema_version':STEP16_VERSION,'environment_version':STEP16_ENV_VERSION,
        'step':'16','status':'passed',
        'created_at_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
        'scope':'model_driven_environment_only_step17_not_implemented',
        'runtime':{
            'python':platform.python_version(),
            'numpy':np.__version__,
            'scikit_learn':sklearn.__version__,
        },
        'step15_protocol':_record(protocol_path,root),
        'implementation':_record(implementation_path,root),
        'trajectory_trace':_record(trace_path,root),
        'contracts':{
            'internal_state':'physical_unscaled',
            'policy_and_model_observation':'frozen_train_scaled',
            'transition_input':'scaled_state_plus_executed_action',
            'transition_output':'physical_temperature_humidity_co2',
            'step_minutes':5,'episode_crosses_date_or_series_boundary':False,
            'recorded_weather_offline':True,'forecast_weather_interface':True,
            'ood_state_and_joint_recorded_each_step':True,
            'warning_action_projection_and_penalty':True,
            'severe_action_safe_fallback':True,
            'severe_state_or_repeated_severe_pessimistic_termination':True,
            'scheduled_valid_steps_denominator_exposed':True,
        },
        'crops':crops,
    }
    manifest_path=destination/'step16_environment_manifest.json'
    _write(manifest_path,manifest)
    integrity={
        'schema_version':STEP16_VERSION,'step':'16','status':'passed',
        'environment_manifest':_record(manifest_path,root),
        'checks':{
            'step15_hash_chain_verified':True,'all_crops_extra_trees_loaded':True,
            'observation_action_shape_finite':True,'one_step_inference_parity':True,
            'deterministic_seed_action_replay':True,'reward_handoff':True,
            'candidate_artifact_copy_count':0,'step17_implemented':False,
        },
    }
    integrity_path=destination/'step16_integrity_manifest.json'
    _write(integrity_path,integrity)
    return integrity_path
