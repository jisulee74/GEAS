"""Read-only Step 19.5 diagnostics. Never trains policies or changes official gates."""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import csv
import fcntl
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch

from geas35.experiments.rl.checkpoint_v2 import load_ppo_checkpoint_v2
from geas35.experiments.rl import step19_5_runner_v2_2 as audit
from geas35.experiments.rl.step19_5_calibration_v2_2 import read_json, write_json, sha256
from geas35.experiments.rl.step20_hpo import evaluate_paired_validation_episodes, aggregate_validation_metrics
from geas35.experiments.rl.step20_runner_v1_4 import _artifact_hashes, _limit_transition_threads
from geas35.rl.model_driven_env import load_model_driven_env_from_step15, _episode_registry

CROPS = ('strawberry', 'melon', 'cucumber')
ART = Path('experiments/rl_policy_training/artifacts')
OUT = ART / 'step19_5_diagnostics_v1'
STAGES = ('inventory', 'evaluator-parity', 'rollout-diagnosis', 'learning-analysis', 'report')
RTOL, ATOL = 1e-6, 1e-8
_ENV = {}


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, (pd.Timestamp, Path)): return str(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True).encode()).hexdigest()


def check_split(split):
    if split not in ('train', 'validation'):
        raise ValueError('Diagnostics allow Train/Validation only; Test is locked')


def indices(count, split):
    check_split(split)
    return list(range(count)) if split == 'validation' else sorted(set(np.linspace(0, count - 1, min(24, count), dtype=int).tolist()))


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError(f'Already running: {path}') from exc
        yield


def checkpoint(root, crop, config_id, seed, steps):
    base = 'step19_5_v2_2/extended_training' if steps == 1048576 else 'step19_5_v1_4'
    if steps not in (524288, 1048576): raise ValueError('No substitute for an unavailable checkpoint')
    return root / ART / base / crop / config_id / f'seed_{seed}.pt'


def get_env(root, crop, split):
    check_split(split)
    key = (str(root), crop, split)
    if key not in _ENV:
        env = load_model_driven_env_from_step15(project_root=root, crop=crop, split=split,
            protocol_manifest_path=root / ART / 'step15_v1_4/rl_protocol_manifest.json')
        _limit_transition_threads(env, 1)
        _ENV[key] = env
    return _ENV[key]


def verify_inputs(root, manifest):
    errors = [record['path'] for record in manifest['files'] if not (root / record['path']).is_file()
              or sha256(root / record['path']) != record['sha256']]
    if errors: raise ValueError(f'Frozen diagnostic inputs changed: {errors}')
    return len(manifest['files'])


def inventory(root):
    root = Path(root).resolve(); output = root / OUT
    with lock(output / 'inventory.lock'):
        files = {};code_drift=[]
        def add(path, expected=None):
            path = Path(path); path = path if path.is_absolute() else root / path
            if not path.is_file(): raise FileNotFoundError(path)
            key = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            actual = sha256(path)
            if expected is not None and actual != expected:
                if path.suffix=='.py':code_drift.append({'path':key,'frozen_sha256':expected,'current_sha256':actual})
                else:raise ValueError(f'Hash mismatch: {path}')
            files[key] = {'path': key, 'sha256': actual, 'size_bytes': path.stat().st_size}
        def records(obj):
            if isinstance(obj, dict):
                if 'path' in obj and 'sha256' in obj:
                    # Never open Test datasets/reports while traversing provenance metadata.
                    if 'test' not in str(obj['path']).lower(): add(obj['path'], obj['sha256'])
                else:
                    for k,v in obj.items():
                        if 'test' not in k.lower(): records(v)
            elif isinstance(obj, list):
                for v in obj: records(v)
        for name in ('step15_v1_4/rl_protocol_manifest.json', 'step16_v1_4/step16_environment_manifest.json',
                     'step17_v1_4/step17_integrity_manifest.json', 'step18_v1_4/step18_integrity_manifest.json',
                     'step19_v1_4/step19_integrity_manifest.json', 'step19_5_v2_2/source_v1_4_freeze_manifest.json',
                     'step19_5_v2_2/step19_5_integrity_manifest.json'):
            p = root / ART / name
            if p.exists(): add(p); records(read_json(p))
        protocol = read_json(root / ART / 'step15_v1_4/rl_protocol_manifest.json')
        split_policy_path = root / protocol['split_access_policy']['path']
        split_policy = read_json(split_policy_path)
        registries = {}
        for crop in CROPS:
            registries[crop] = {}
            for split in ('train','validation'):
                rec = split_policy['datasets'][crop][split]; add(rec['path'], rec['sha256'])
                frame = pd.read_parquet(rec['path'], columns=['reg_date','rl_valid_transition','crop','series_id','segment_id','episode_id','mdp_v1_rollout_id'])
                eps = _episode_registry(frame); chosen = indices(len(eps), split)
                registries[crop][split] = {'indices': chosen, 'episodes': [dict(episode_index=i, **eps[i]) for i in chosen]}
        for base in ('step19_5_v1_4', 'step19_5_v2_2'):
            for p in (root / ART / base).rglob('*'):
                if p.is_file() and p.suffix in ('.json','.pt') and 'diagnostic' not in str(p): add(p)
        for p in (root/'experiments/rl_policy_training/configs').glob('*.json'): add(p)
        for directory in ('src/geas35/rl','src/geas35/experiments/rl','src/geas35/models/transition'):
            for p in (root/directory).glob('*.py'): add(p)
        add(root/'experiments/rl_policy_training/scripts/run_step19_5_diagnostics.py')
        checkpoints = []
        for crop in CROPS:
            for steps in ((524288,1048576) if crop=='strawberry' else (524288,)):
                for n in range(5):
                    for seed in (42,43,44):
                        p=checkpoint(root,crop,f'representative_{n}',seed,steps)
                        payload=torch.load(p,map_location='cpu',weights_only=False)
                        assert payload['extra_state']['training_steps']==steps,(p,'wrong step')
                        assert payload['artifact_hashes']==_artifact_hashes(root,crop),(p,'dependency mismatch')
                        checkpoints.append({'crop':crop,'config_id':f'representative_{n}','seed':seed,'steps':steps,
                            'path':str(p.relative_to(root)),'sha256':sha256(p),
                            'optimizer_present':bool(payload.get('optimizer_state')),'rng_present':bool(payload.get('rng_state'))})
        body={'source_code_drift':code_drift,'historical_environment_equivalence_proven':not code_drift,'version':'step19_5_diagnostics_v1','files':sorted(files.values(),key=lambda x:x['path']),
            'trace_state_columns':{crop:read_json(root/protocol['crops'][crop]['state_action_support']['path'])['state_columns'] for crop in CROPS},'registries':registries,'checkpoints':checkpoints,'test_accessed':False,
            'missing_checkpoint_steps':[262144], 'historical_262144_policy_replay_available':False,
            'tolerance':{'rtol':RTOL,'atol':ATOL}}
        target=output/'input_manifest.json'
        if target.exists() and read_json(target)!=clean(body):
            raise ValueError('Inventory already frozen with different inputs; use a new diagnostic version')
        write_json(target,clean(body));return target


def state_contributions(env, state):
    s=env.support; q=np.asarray(state,dtype=float)*s.state_weights
    distance,neighbors=s._state_nn.kneighbors(q.reshape(1,-1),return_distance=True)
    kth=int(neighbors[0,-1]); terms=(q-s._state_reference[kth])**2
    return {'state_score':float(distance[0,-1]),'neighbor_index':kth,
        'squared_distance_components':dict(zip(env.observation_columns,map(float,terms))),
        'component_sum_matches_distance':bool(np.isclose(terms.sum(),distance[0,-1]**2,rtol=RTOL,atol=ATOL)),
        'interpretation':'weighted kth-neighbor squared-distance components, not causal importance'}


def reward_components(reward, info):
    base=float(info.get('base_reward',0.0)); warning=-float(info.get('warning_ood_penalty',0.0))
    terminal=float(info.get('pessimistic_terminal_cost',0.0))
    return {'base':base,'warning':warning,'terminal':terminal,
            'clamp_or_override':float(reward)-base-warning-terminal}


class TraceEnv:
    """Observation-only proxy. The wrapped environment owns every transition."""
    def __init__(self, env, episode_index, stream=None, recorded_actions=False):
        self.env=env;self.index=episode_index;self.stream=stream;self.recorded_actions=recorded_actions
        self.episodes=[env.episodes[episode_index]];self.records=[];self.reset_info=None
    def __getattr__(self,name): return getattr(self.env,name)
    def reset(self,*,seed=None,options=None):
        obs,info=self.env.reset(seed=seed,options={'episode_index':self.index})
        self.reset_info=clean(info);self.records=[]
        return obs,info
    def step(self, action):
        env=self.env;before=env._current_scaled.loc[list(env.observation_columns)].to_numpy(dtype=float).copy()
        physical=env._current_physical.reindex(env.observation_columns).to_numpy(dtype=float).copy()
        position=int(env._position); history_len=len(env.state_updater._history)
        result=env.step(action);obs,reward,terminated,truncated,info=result
        record={'episode_index':self.index,'step':len(self.records),'source_position':position,
            'scaled_state':before.tolist(),'physical_state':physical.tolist(),
            'next_scaled_state':obs.tolist(),'history_length':history_len,
            'history_reference':'previous trace states; initial history is the recorded reset state',
            'action':np.asarray(action,dtype=float).tolist(),'reward':float(reward),
            'terminated':bool(terminated),'truncated':bool(truncated),'info':clean(info),
            'reward_components':reward_components(reward,info)}
        if info.get('state_ood_level') in ('warning','severe') and (len(self.records)==0 or info.get('state_ood_level')=='severe'):
            record['state_distance']=state_contributions(env,before)
        if self.recorded_actions and info.get('transition_prediction_physical'):
            truth=env.scaler.inverse_transform_frame(env.frame.iloc[[position+1]]).iloc[0]
            record['recorded_path_error_physical']={k:float(v-truth[k]) for k,v in info['transition_prediction_physical'].items()}
            record['recorded_action_modified']=not np.allclose(np.asarray(action),np.array([info['executed_action'][k] for k in env.action_columns]),rtol=RTOL,atol=ATOL)
        # Keep only a compact comparison signature in memory; full states remain in gzip trace.
        self.records.append({'reward':record['reward'],'terminated':record['terminated'],
            'truncated':record['truncated'],'info':record['info'], 'reward_components':record['reward_components'],
            'trajectory_hash':digest({k:record[k] for k in ('scaled_state','next_scaled_state','action')})})
        if self.stream is not None:self.stream.write(json.dumps(clean(record),allow_nan=False)+'\n')
        return result


def summarize_trace(records, reset_info):
    executed=len(records);scheduled=int(reset_info['scheduled_valid_steps'])
    severe=sum(r['info'].get('joint_ood_level')=='severe' or r['info'].get('state_ood_level')=='severe' for r in records)
    fallback=sum(bool(r['info'].get('safety_fallback_applied')) for r in records)
    warning=sum(r['info'].get('joint_ood_level')=='warning' for r in records)
    hard=0
    for r in records:
        info=r['info'];action=info.get('executed_action')
        if action is not None:
            values=np.array(list(action.values()),dtype=float)
            hard+=int(not np.isfinite(values).all() or (values<0).any() or (values>1).any()
                or any(action[n] not in (0.,1.) for n in ('heat_run','cool_run','fan_run'))
                or action['vent_pct']>info.get('weather_constraint',{}).get('vent_open_cap',1.)+1e-7)
    reward=float(sum(r['reward'] for r in records));components={k:sum(r['reward_components'][k] for r in records) for k in ('base','warning','terminal','clamp_or_override')}
    if not np.isclose(sum(components.values()),reward,rtol=RTOL,atol=ATOL):raise ValueError('Reward decomposition mismatch')
    terms=Counter()
    for row in records:
        for k,v in row['info'].get('reward_terms',{}).items():
            if isinstance(v,(float,int)):terms[k]+=v
    return {'scheduled_valid_steps':scheduled,'executed_steps':executed,'reward_sum':reward,
        'mean_reward_per_scheduled_valid_step':reward/scheduled,'severe_steps':severe,
        'severe_ood_rate':severe/max(executed,1),'severe_ood_per_scheduled_step':severe/scheduled,
        'fallback_steps':fallback,'fallback_rate':fallback/max(executed,1),'warning_steps':warning,
        'warning_ood_rate':warning/max(executed,1),'hard_constraint_violations':hard,
        'reward_components':components,'reward_terms':dict(terms),
        'termination_reason':records[-1]['info'].get('termination_reason') if records else None,
        'first_severe_step':next((i for i,r in enumerate(records) if r['info'].get('state_ood_level')=='severe'),None),
        'trajectory_hash':digest([r['trajectory_hash'] for r in records])}


def compare_episodes(left, right):
    differences=[]
    for key in ('scheduled_valid_steps','executed_steps','termination_reason','first_severe_step'):
        if left[key]!=right[key]:differences.append(key)
    for key in ('reward_sum','severe_ood_rate','fallback_rate','warning_ood_rate','hard_constraint_violations'):
        if not np.isclose(left[key],right[key],rtol=RTOL,atol=ATOL):differences.append(key)
    return differences


def evaluate_episode(env,trainer,index,ordinal,gates,evaluator,stream):
    wrapped=TraceEnv(env,index,stream)
    if evaluator=='legacy':
        rows,_=evaluate_paired_validation_episodes(wrapped,trainer,episode_indices=[0],seed=42+ordinal)
        metrics=aggregate_validation_metrics(rows)
    else:
        # The unmodified audit evaluator requests seed 42 for its single visible episode.
        original_reset=wrapped.reset
        wrapped.reset=lambda **kw:original_reset(seed=42+ordinal,options=kw.get('options'))
        original_loader=audit.load_model_driven_env_from_step15
        audit.load_model_driven_env_from_step15=lambda **kwargs:wrapped
        try:metrics,*_=audit._evaluate_audit(root=Path('.'),crop=env.crop,trainer=trainer,gates=gates,transition_n_jobs=1)
        finally:audit.load_model_driven_env_from_step15=original_loader
    summary=summarize_trace(wrapped.records,wrapped.reset_info)
    for key in ('mean_reward_per_scheduled_valid_step','scheduled_valid_steps','executed_steps',
                'hard_constraint_violations','severe_ood_rate','fallback_rate','warning_ood_rate'):
        if not np.isclose(metrics[key],summary[key],rtol=RTOL,atol=ATOL):raise ValueError(f'Trace aggregation mismatch: {key}')
    return {'episode_index':index,'episode_id':wrapped.reset_info['episode_id'],**summary}


def atomic_episode(path, fingerprint, operation):
    """Commit trace before its completion record; validate both before reuse."""
    trace=path.with_suffix('.jsonl.gz')
    if path.exists():
        value=read_json(path)
        if value['fingerprint']!=fingerprint:raise ValueError(f'Stale diagnostic result: {path}')
        if not trace.exists() or sha256(trace)!=value['trace_sha256']:raise ValueError(f'Trace hash mismatch: {trace}')
        return value
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=trace.with_name(trace.name+'.tmp')
    with gzip.open(temporary,'wt',encoding='utf-8') as stream: result=operation(stream)
    temporary.replace(trace)
    value={'fingerprint':fingerprint,'trace_sha256':sha256(trace),'test_accessed':False,**clean(result)}
    write_json(path,value);return value


def job_id(job):return '__'.join(str(job[k]) for k in ('crop','split','mode','config_id','seed','steps','repeat'))


def aggregate_episodes(episodes):
    scheduled=sum(e['scheduled_valid_steps'] for e in episodes);executed=sum(e['executed_steps'] for e in episodes)
    return {'scheduled_valid_steps':scheduled,'executed_steps':executed,
        'mean_reward_per_scheduled_valid_step':sum(e['reward_sum'] for e in episodes)/scheduled,
        'severe_ood_rate':sum(e['severe_steps'] for e in episodes)/max(executed,1),
        'severe_ood_per_scheduled_step':sum(e['severe_steps'] for e in episodes)/scheduled,
        'fallback_rate':sum(e['fallback_steps'] for e in episodes)/max(executed,1),
        'hard_constraint_violations':sum(e['hard_constraint_violations'] for e in episodes),
        'termination_reasons':dict(Counter(e['termination_reason'] for e in episodes)),
        'reward_components':{k:sum(e['reward_components'][k] for e in episodes)/scheduled for k in ('base','warning','terminal','clamp_or_override')}}


def run_job(root_text, manifest_hash, job):
    root=Path(root_text);output=root/OUT;ident=job_id(job);directory=output/'jobs'/ident
    torch.set_num_threads(1);check_split(job['split'])
    manifest=read_json(output/'input_manifest.json')
    if digest(manifest)!=manifest_hash:raise ValueError('Inventory changed during execution')
    fingerprint=digest({'manifest':manifest_hash,'job':job})
    with lock(directory/'job.lock'):
        final=directory/'result.json'
        if final.exists():
            result=read_json(final)
            if result['fingerprint']!=fingerprint:raise ValueError('Stale job result')
            for rec in result['episode_files']:
                if sha256(directory/rec['path'])!=rec['sha256']:raise ValueError('Episode summary changed')
                ep=read_json(directory/rec['path'])
                if sha256((directory/rec['path']).with_suffix('.jsonl.gz'))!=ep['trace_sha256']:raise ValueError('Episode trace changed')
            return str(final)
        write_json(directory/'status.json',{'status':'running','pid':os.getpid(),'job':job,'started':time.time()})
        print('START',ident,flush=True)
        try:
            env=get_env(root,job['crop'],job['split'])
            selected=manifest['registries'][job['crop']][job['split']]['indices']
            if env.episodes and [dict(episode_index=i,**env.episodes[i]) for i in selected]!=manifest['registries'][job['crop']][job['split']]['episodes']:
                raise ValueError('Episode registry changed')
            trainer=None
            if job['mode'] in ('legacy','audit'):
                p=checkpoint(root,job['crop'],job['config_id'],job['seed'],job['steps'])
                trainer,payload=load_ppo_checkpoint_v2(p,expected_artifact_hashes=_artifact_hashes(root,job['crop']),device=job['device'],restore_rng=True)
                if payload['extra_state']['training_steps']!=job['steps']:raise ValueError('Wrong checkpoint step')
            gates=read_json(root/ART/'step19_5_v1_4/train_derived_safety_gates.json')['crops'][job['crop']]
            episodes=[];episode_files=[]
            for ordinal,index in enumerate(selected):
                path=directory/'episodes'/f'{ordinal:04d}.json'
                if trainer is not None:
                    operation=lambda stream:evaluate_episode(env,trainer,index,ordinal,gates,job['mode'],stream)
                else:operation=lambda stream:recorded_episode(env,index,job['mode'],stream)
                result=atomic_episode(path,digest({'job':fingerprint,'index':index}),operation)
                episodes.append(result);episode_files.append({'path':str(path.relative_to(directory)),'sha256':sha256(path)})
                write_json(directory/'status.json',{'status':'running','pid':os.getpid(),'job':job,'episodes_done':len(episodes),'episodes_total':len(selected),'updated':time.time()})
            final_result={'job':job,'fingerprint':fingerprint,'episodes':episodes,'episode_files':episode_files,
                'metrics':aggregate_episodes(episodes),'test_accessed':False}
            write_json(final,final_result)
            write_json(directory/'status.json',{'status':'complete','job':job,'episodes_done':len(episodes),'updated':time.time()})
            print('DONE',ident,flush=True);return str(final)
        except BaseException as exc:
            write_json(directory/'status.json',{'status':'failed','job':job,'error':repr(exc),'updated':time.time()});raise


def recorded_episode(env,index,mode,stream):
    """Recorded-state scoring + teacher-forced prediction, or guarded logged-action rollout."""
    if mode=='recorded-rollout':
        wrapped=TraceEnv(env,index,stream,recorded_actions=True);_,reset=wrapped.reset(seed=42+index)
        terminated=truncated=False
        while not (terminated or truncated):
            action=env.frame.iloc[env._position].loc[list(env.action_columns)].to_numpy(dtype=float)
            _,_,terminated,truncated,_=wrapped.step(action)
        return {'episode_index':index,'episode_id':reset['episode_id'],**summarize_trace(wrapped.records,reset),
                'logged_actions_subject_to_existing_safety_projection':True}
    if mode!='recorded-one-step':raise ValueError(f'Unknown mode: {mode}')
    episode=env.episodes[index];positions=list(range(episode['start'],episode['final_observation']))
    frame=env.frame.iloc[positions];state=frame.loc[:,list(env.observation_columns)];actions=frame.loc[:,list(env.action_columns)]
    scores=env.support.scores(state,actions);inputs,targets=env.model._require_fitted()
    pred=env.model.predict(frame.loc[:,list(inputs)].apply(pd.to_numeric,errors='coerce')).next_observation
    truth=env.scaler.inverse_transform_frame(env.frame.iloc[np.array(positions)+1].copy())
    physical=env.scaler.inverse_transform_frame(frame.copy());threshold=env.support.config['thresholds']
    errors={t:[] for t in targets};state_severe=[];joint_severe=[];first=None
    for ordinal,pos in enumerate(positions):
        severe=bool(scores['state'][ordinal]>threshold['state_severe']);state_severe.append(severe)
        joint_severe.append(bool(scores['joint'][ordinal]>threshold['joint_severe']))
        if severe and first is None:first=ordinal
        error={t:float(pred.iloc[ordinal][t]-truth.iloc[ordinal][t]) for t in targets}
        for t,v in error.items():errors[t].append(v)
        row={'episode_index':index,'step':ordinal,'source_position':pos,'scaled_state':state.iloc[ordinal].tolist(),
            'physical_state':physical.iloc[ordinal].reindex(env.observation_columns).tolist(),'action':actions.iloc[ordinal].tolist(),
            'state_ood_score':float(scores['state'][ordinal]),'joint_ood_score':float(scores['joint'][ordinal]),
            'thresholds':threshold,'prediction_physical':pred.iloc[ordinal].to_dict(),'one_step_error_physical':error,
            'history_reference':{'kind':'recorded_dataset','source_position':pos},'reward':None}
        if severe:row['state_distance']=state_contributions(env,state.iloc[ordinal].to_numpy())
        stream.write(json.dumps(clean(row),allow_nan=False)+'\n')
    count=len(positions);severe=sum(a or b for a,b in zip(state_severe,joint_severe))
    return {'episode_index':index,'episode_id':episode['episode_id'],'scheduled_valid_steps':count,'executed_steps':count,
        'reward_sum':0.,'reward_not_evaluated':True,'severe_steps':severe,'severe_ood_rate':severe/count,
        'state_severe_steps':sum(state_severe),'fallback_steps':0,'fallback_rate':0.,'hard_constraint_violations':0,
        'termination_reason':'recorded_episode_boundary','first_severe_step':first,
        'reward_components':dict.fromkeys(('base','warning','terminal','clamp_or_override'),0.),
        'prediction_errors':{t:{'count':len(v),'sum':float(np.sum(v)),'sum_abs':float(np.abs(v).sum()),'sum_square':float(np.square(v).sum())} for t,v in errors.items()}}


def make_job(crop,split='validation',mode='audit',config_id='representative_0',seed=42,steps=524288,repeat=0,device='cpu'):
    check_split(split)
    if crop not in CROPS:raise ValueError('Unsupported crop')
    return dict(crop=crop,split=split,mode=mode,config_id=config_id,seed=seed,steps=steps,repeat=repeat,device=device)


def stage_jobs(stage,crops,device):
    jobs=[]
    for crop in crops:
        if stage=='evaluator-parity':
            for mode in ('legacy','audit'):
                for repeat in (0,1):jobs.append(make_job(crop,mode=mode,repeat=repeat,device=device))
        elif stage=='rollout-diagnosis' and crop in ('melon','cucumber'):
            for split in ('validation','train'):
                for mode in ('recorded-one-step','recorded-rollout'):jobs.append(make_job(crop,split=split,mode=mode,device=device))
            jobs.append(make_job(crop,split='train',device=device))
            for n in range(5):
                for seed in (42,43,44):jobs.append(make_job(crop,config_id=f'representative_{n}',seed=seed,device=device))
        elif stage=='learning-analysis' and crop=='strawberry':
            for steps in (524288,1048576):
                for n in range(5):
                    for seed in (42,43,44):jobs.append(make_job(crop,config_id=f'representative_{n}',seed=seed,steps=steps,device=device))
    return jobs


def run_stage(root,stage,crops=CROPS,device='cpu',workers=3):
    root=Path(root).resolve()
    if stage not in STAGES or not 1<=workers<=3:raise ValueError('Invalid stage or workers (1..3)')
    if stage=='inventory':return inventory(root)
    manifest=read_json(root/OUT/'input_manifest.json');verify_inputs(root,manifest)
    if stage=='report':
        from geas35.experiments.rl.step19_5_diagnostics_report import generate_report
        return generate_report(root,manifest,device)
    if stage!='evaluator-parity':
        for crop in crops:
            parity=root/OUT/'parity'/f'{crop}.json'
            if not parity.exists() or not read_json(parity)['reproducible']:
                raise ValueError(f'Complete reproducible evaluator parity first: {crop}')
    jobs=stage_jobs(stage,crops,device); fingerprint=digest(manifest)
    with lock(root/OUT/f'{stage}.lock'):
        status_path=root/OUT/f'{stage}_status.json'
        write_json(status_path,{'status':'running','jobs':len(jobs),'completed':0,'workers':workers})
        done=0
        try:
            if workers==1:
                for job in jobs:
                    run_job(str(root),fingerprint,job);done+=1
                    write_json(status_path,{'status':'running','jobs':len(jobs),'completed':done,'workers':workers})
            else:
                with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
                    futures=[pool.submit(run_job,str(root),fingerprint,job) for job in jobs]
                    for future in as_completed(futures):
                        future.result();done+=1
                        write_json(status_path,{'status':'running','jobs':len(jobs),'completed':done,'workers':workers})
            if stage=='evaluator-parity':
                for crop in crops:build_parity(root,crop,device)
            write_json(status_path,{'status':'complete','jobs':len(jobs),'completed':done,'workers':workers})
        except BaseException as exc:
            write_json(status_path,{'status':'failed','jobs':len(jobs),'completed':done,'error':repr(exc)});raise
    return status_path


def result_for(root,job):return read_json(root/OUT/'jobs'/job_id(job)/'result.json')


def first_trace_difference(root,left_job,right_job,ordinal):
    from itertools import zip_longest
    base=root/OUT/'jobs'
    a=base/job_id(left_job)/'episodes'/f'{ordinal:04d}.jsonl.gz';b=base/job_id(right_job)/'episodes'/f'{ordinal:04d}.jsonl.gz'
    with gzip.open(a,'rt') as left,gzip.open(b,'rt') as right:
        for i,(x,y) in enumerate(zip_longest(left,right)):
            if x is None or y is None:return {'step':i,'field':'trace_length'}
            x,y=json.loads(x),json.loads(y)
            for key in ('scaled_state','action','next_scaled_state','reward'):
                if not np.allclose(x[key],y[key],rtol=RTOL,atol=ATOL):return {'step':i,'field':key}
            for key in ('terminated','truncated'):
                if x[key]!=y[key]:return {'step':i,'field':key}
            if x['info'].get('termination_reason')!=y['info'].get('termination_reason'):return {'step':i,'field':'termination_reason'}
    return None


def build_parity(root,crop,device):
    pairs=[('legacy',0,'legacy',1),('audit',0,'audit',1),('legacy',0,'audit',0)]
    comparisons=[]
    for lm,lr,rm,rr in pairs:
        lj=make_job(crop,mode=lm,repeat=lr,device=device);rj=make_job(crop,mode=rm,repeat=rr,device=device)
        left=result_for(root,lj);right=result_for(root,rj);differences=[]
        if len(left['episodes'])!=len(right['episodes']):raise ValueError('Different episode registries')
        for ordinal,(a,b) in enumerate(zip(left['episodes'],right['episodes'])):
            fields=compare_episodes(a,b);trace_difference=None
            if a['trajectory_hash']!=b['trajectory_hash']:trace_difference=first_trace_difference(root,lj,rj,ordinal)
            if fields or trace_difference:differences.append({'episode_index':a['episode_index'],'fields':fields,'first_trace_difference':trace_difference})
        comparisons.append({'left':[lm,lr],'right':[rm,rr],'passed':not differences,'differences':differences})
    value={'crop':crop,'reproducible':all(x['passed'] for x in comparisons[:2]),
        'evaluators_equivalent':comparisons[2]['passed'],'comparisons':comparisons,'rtol':RTOL,'atol':ATOL,'test_accessed':False}
    write_json(root/OUT/'parity'/f'{crop}.json',value);return value
