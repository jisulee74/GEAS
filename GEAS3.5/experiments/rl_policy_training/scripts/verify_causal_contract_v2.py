#!/usr/bin/env python3
"""No fitting: verify opt-in solar contract on frozen Train/Validation episodes."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,json,gzip,hashlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.recorded_light_env_v1 import SOLAR_FEATURES
from geas35.rl.causal_contract_env_v2 import load_causal_contract_env as load_recorded_light_env,VERSION,CONTRACT
from geas35.experiments.rl.step19_5_diagnostics import (ART,OUT,read_json,write_json,sha256,lock,indices,
    _limit_transition_threads,TraceEnv,summarize_trace,atomic_episode,aggregate_episodes,digest)
DEST=ART/'step19_5_causal_contract_v2'


def run(crop,split):
    root=ROOT;out=root/DEST;directory=out/f'{crop}_{split}';original=root/OUT
    with lock(directory/'run.lock'):
        env=load_recorded_light_env(project_root=root,crop=crop,split=split,protocol_manifest_path=root/ART/'step15_v1_4/rl_protocol_manifest.json')
        _limit_transition_threads(env,1)
        manifest=read_json(original/'input_manifest.json');selected=manifest['registries'][crop][split]['indices']
        legacy_dir=root/ART/'step19_5_light_contract_v1'/f'{crop}_{split}'
        legacy=read_json(legacy_dir/'result.json')
        frozen=read_json(out/'verification_manifest.json');fingerprint=digest(frozen)
        physical=env.scaler.inverse_transform_frame(env.frame.copy());columns=list(env.observation_columns)
        all_results=[];comparison=[];first=[]
        for ordinal,index in enumerate(selected):
            old_ep=read_json(legacy_dir/'episodes'/f'{ordinal:04d}.json');old_trace=legacy_dir/'episodes'/f'{ordinal:04d}.jsonl.gz'
            if sha256(old_trace)!=old_ep['trace_sha256']:raise ValueError('Legacy trace changed')
            def operation(stream):
                wrapped=TraceEnv(env,index,stream,recorded_actions=True);_,reset=wrapped.reset(seed=42+index)
                finished=False;errors={c:[] for c in SOLAR_FEATURES}
                while not finished:
                    pos=env._position
                    for c in SOLAR_FEATURES:errors[c].append(float(env._current_physical[c]-physical.iloc[pos][c]))
                    action=env.frame.iloc[pos].loc[list(env.action_columns)].to_numpy(dtype=float)
                    _,_,terminated,truncated,info=wrapped.step(action);finished=terminated or truncated
                    for c in SOLAR_FEATURES:errors[c].append(float(env._current_physical[c]-physical.iloc[env._position][c]))
                return {'episode_index':index,'episode_id':reset['episode_id'],**summarize_trace(wrapped.records,reset),
                    'feature_errors':{c:{'max_abs':float(np.max(np.abs(v))),'mae':float(np.mean(np.abs(v)))} for c,v in errors.items()}}
            p=directory/'episodes'/f'{ordinal:04d}.json';new=atomic_episode(p,digest({'manifest':fingerprint,'index':index,'split':split,'crop':crop}),operation)
            all_results.append(new)
            old_rows=[]
            with gzip.open(old_trace,'rt') as f:
                for line in f:old_rows.append(json.loads(line))
            new_trace=p.with_suffix('.jsonl.gz');new_rows=[]
            with gzip.open(new_trace,'rt') as f:
                for line in f:new_rows.append(json.loads(line))
            for label,trace_rows in [('legacy',old_rows),('corrected',new_rows)]:
                divergences=set()
                for row in trace_rows:
                    pos=row['source_position'];logged=physical.iloc[pos];step=row['step']
                    entry={'crop':crop,'split':split,'episode_index':index,'episode_id':new['episode_id'],'version':label,
                        'step':step,'timestamp':str(logged['reg_date']),'outdoor_light':float(logged.obs_outdoor_light),
                        'day_boundary':bool(pos>0 and pd.Timestamp(logged.reg_date).date()!=pd.Timestamp(physical.iloc[pos-1].reg_date).date()),
                        'terminal':row['info'].get('termination_reason')}
                    for c in SOLAR_FEATURES:
                        actual=row['physical_state'][columns.index(c)];target=float(logged[c]);entry[c+'_recorded']=target;entry[c+'_rollout']=actual
                        if c not in divergences and not np.isclose(actual,target,rtol=1e-6,atol=1e-8):
                            prev=trace_rows[step-1] if step else None
                            first.append({**entry,'feature':c,'recorded':target,'rollout':actual,'recorded_zero':target==0.,
                                'recorded_missing':not np.isfinite(target),'previous_rollout':prev['physical_state'][columns.index(c)] if prev else None,
                                'previous_recorded':float(physical.iloc[pos-1][c]) if pos else None,
                                'dt_seconds':float((pd.Timestamp(logged.reg_date)-pd.Timestamp(physical.iloc[pos-1].reg_date)).total_seconds()) if pos else None})
                            divergences.add(c)
                    comparison.append(entry)
            print(crop,split,ordinal+1,'/',len(selected),old_ep['executed_steps'],'->',new['executed_steps'],new['termination_reason'],flush=True)
            write_json(directory/'status.json',{'status':'running','episodes_done':ordinal+1,'episodes_total':len(selected)})
        pd.DataFrame(comparison).to_csv(directory/'feature_comparison.csv.gz',index=False,compression='gzip')
        pd.DataFrame(first).to_csv(directory/'first_divergence.csv',index=False)
        changes=[{'episode_index':b['episode_index'],'old_steps':a['executed_steps'],'new_steps':b['executed_steps'],
            'old_reason':a['termination_reason'],'new_reason':b['termination_reason'],
            'old_first_ood':a['first_severe_step'],'new_first_ood':b['first_severe_step']} for a,b in zip([read_json(legacy_dir/'episodes'/f'{i:04d}.json') for i in range(len(selected))],all_results)]
        pd.DataFrame(changes).to_csv(directory/'episode_changes.csv',index=False)
        result={'crop':crop,'split':split,'version':VERSION,'old':legacy['new'],'new':aggregate_episodes(all_results),
            'max_feature_errors':{c:max(e['feature_errors'][c]['max_abs'] for e in all_results) for c in SOLAR_FEATURES},
            'episodes':changes,'old_ood_resolved_to_boundary':sum(e['old_reason']=='severe_state_ood' and e['new_reason']=='episode_boundary' for e in changes),
            'old_ood_delayed':sum(e['old_reason']=='severe_state_ood' and e['new_steps']>e['old_steps'] for e in changes),
            'test_accessed':False,'official_thresholds_changed':False}
        write_json(directory/'result.json',result);write_json(directory/'status.json',{'status':'complete','episodes_done':len(selected)})
        return result


def main():
    out=ROOT/DEST;out.mkdir(parents=True,exist_ok=True)
    with lock(out/'coordinator.lock'):
        paths=[ROOT/'src/geas35/rl/causal_contract_env_v2.py',ROOT/'src/geas35/rl/recorded_light_env_v1.py',Path(__file__),ROOT/'src/geas35/models/transition/causal_state.py',ROOT/'src/geas35/features/derived.py']
        paths += [ROOT/f'offline_dataset_preparation/datasets/03_quality_controlled/{c}/{s}.parquet' for c in ('melon','cucumber') for s in ('train','validation')]
        original=read_json(ROOT/OUT/'input_manifest.json')
        from geas35.experiments.rl.step19_5_diagnostics import verify_inputs
        verify_inputs(ROOT,original)
        body={'version':VERSION,'contract':CONTRACT,'original_input_manifest_sha256':sha256(ROOT/OUT/'input_manifest.json'),
            'files':[{'path':str(p.relative_to(ROOT)),'sha256':sha256(p)} for p in paths],
            'test_accessed':False,'official_decision':'hold_step20_hpo','new_training':False}
        p=out/'verification_manifest.json'
        if p.exists() and read_json(p)!=body:raise ValueError('Changed verification inputs')
        write_json(p,body)
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(run,crop,split) for crop in ('melon','cucumber') for split in ('validation','train')]
            results=[future.result() for future in futures]
        verify_inputs(ROOT,original)
        write_json(out/'verification_summary.json',{'status':'complete','results':results,'original_inputs_unchanged':True,'test_accessed':False})
        print('VERIFICATION COMPLETE',flush=True)


if __name__=='__main__':main()
