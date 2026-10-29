"""Build two independent Phase0 copies, publish only byte-identical results.

Reads explicit parent Train/Validation files only. No model or reward calls.
"""
from pathlib import Path
import sys, json, tempfile, shutil
import numpy as np
import pandas as pd
R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R/'src'))
from geas35.models.transition.four_channel_phase0 import (
    TARGETS, RAW, FLAGS, KEYS, ALIASES, MODEL_NAMESPACE, VERSION,
    sha, canonical, valid_raw, physical_gate, fit_scaler, scaler_from, FourChannelRuntime)
BASE = R/'experiments/rl_policy_training/artifacts'
PREREG = BASE/'transition_4ch_migration_prereg_v1'
P = json.loads((PREREG/'migration_preregistration.json').read_text())
PARENT = BASE/P['source_dataset']
DEST = BASE/P['next_dataset_namespace']
REPORT = BASE/'transition_4ch_phase0_v1'
READS = {}

def read(crop, split, kind):
    if crop not in ('melon','cucumber','strawberry') or split not in ('train','validation') or kind not in ('physical','registry','prepared_qc','windows','provenance'):
        raise ValueError('Explicit Historical-H Train/Validation allowlist only')
    path = PARENT/crop/f'{split}_{kind}.parquet'
    h = sha(path)
    expected = P['admission']['parent_hashes'].get(f'{crop}/{path.name}')
    if expected and h != expected:
        raise ValueError(f'Parent hash changed: {path}')
    READS[str(path.relative_to(R))] = h
    return pd.read_parquet(path)

def save(obj, path):
    path.write_bytes(canonical(obj))

def subset_windows(old, reg):
    by_id = {r.row_id:i for i,r in enumerate(reg.itertuples())}
    rows = []
    for w in old.itertuples(index=False):
        i = by_id.get(w.start_row_id)
        if i is None: continue
        block = reg.iloc[i:i+int(w.horizon)]
        if len(block) != w.horizon: continue
        if block.mdp_v1_rollout_id.nunique() != 1 or int(block.mdp_v1_rollout_id.iloc[0]) != int(w.rollout_id): continue
        if not all(block[k].nunique() == 1 for k in KEYS[:-1]+['source_confidence']): continue
        if not block.reg_date.diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all(): continue
        if block.target_timestamp.iloc[-1] != w.end_target_timestamp: continue
        rows.append(dict(zip(old.columns, w)))
    return pd.DataFrame(rows, columns=old.columns)

def build(out):
    out.mkdir(parents=True)
    schema = {**P['schema'], 'model_namespace':MODEL_NAMESPACE, 'dataset_namespace':P['next_dataset_namespace']}
    save(schema, out/'schema.json')
    sh = sha(out/'schema.json')
    result = {}
    obs = schema['observation_columns']; actions = schema['action_columns']
    for crop in ('melon','cucumber','strawberry'):
        dst = out/crop; dst.mkdir()
        datasets = {}
        for split in ('train','validation'):
            qc = read(crop,split,'prepared_qc'); registry = read(crop,split,'registry')
            physical = read(crop,split,'physical'); oldwin = read(crop,split,'windows')
            pro = read(crop,split,'provenance')
            assert qc.crop.eq(crop).all() and qc.split.eq(split).all()
            assert not qc.duplicated(KEYS).any() and not registry.row_id.duplicated().any()
            assert physical.row_id.equals(registry.row_id)
            assert (registry.target_timestamp-registry.reg_date).eq(pd.Timedelta(minutes=5)).all()
            q = qc.set_index(KEYS)
            cur = q.loc[pd.MultiIndex.from_frame(registry[KEYS])].reset_index()
            nxt = q.loc[pd.MultiIndex.from_frame(registry.drop(columns='reg_date').rename(columns={'target_timestamp':'reg_date'})[KEYS])].reset_index()
            for k in ('sensor_source','source_table','source_confidence'):
                if k in cur: assert cur[k].equals(nxt[k]), ('source mismatch',crop,split,k)
            good = valid_raw(cur) & valid_raw(nxt)
            assert int(good.sum()) == P['admission']['expected_pair_counts'][f'{crop}/{split}'], 'PAIR_COUNT_MISMATCH'
            for pre, raw in [('',cur),('next_',nxt)]:
                for alias,s in [('obs_indoor_temp_c','in_temp'),('obs_indoor_humidity_pct','in_hum')]:
                    np.testing.assert_allclose(physical.loc[good,pre+alias],raw.loc[good,s+'_raw_value'],rtol=1e-6,atol=1e-8)
            reg = registry.loc[good].reset_index(drop=True)
            meta = list(registry.columns)+['split']
            data = physical.loc[good,meta].reset_index(drop=True)
            for pre, raw in [('',cur),('next_',nxt)]:
                for c,s in zip(TARGETS,RAW): data[pre+c] = raw.loc[good,s+'_raw_value'].to_numpy(float)
                for c in obs[4:]: data[pre+c] = physical.loc[good,pre+c].to_numpy(float)
                physical_gate(data[[pre+c for c in TARGETS]].to_numpy(float))
            for a in actions: data[a] = physical.loc[good,a].to_numpy(float)
            # Explicit order, no legacy reward, targets, aliases, quality flags, or CO2 in model frame.
            data = data[meta+obs+actions+['next_'+c for c in obs]]
            assert np.isfinite(data[obs+actions+['next_'+c for c in obs]].to_numpy(float)).all()
            assert not any('co2' in c.lower() for c in data)
            provenance = pro.set_index('row_id').loc[reg.row_id].reset_index()
            for a in actions:
                np.testing.assert_allclose(data['obs_prev_'+a],provenance['previous_'+a],rtol=1e-6,atol=1e-8)
                np.testing.assert_allclose(data[a],provenance['current_'+a],rtol=1e-6,atol=1e-8)
                np.testing.assert_allclose(data['next_obs_prev_'+a],data[a],rtol=1e-6,atol=1e-8)
                assert np.isfinite(provenance['previous_previous_'+a]).all()
            wins = subset_windows(oldwin,reg)
            one = pd.DataFrame({'rollout_id':reg.mdp_v1_rollout_id,'horizon':1,'start_row_id':reg.row_id,'end_target_timestamp':reg.target_timestamp})
            wins = pd.concat([one,wins],ignore_index=True)
            assert set(wins.horizon) == {1,3,6,12}
            histcols = [c for c in qc if c in obs or c in ALIASES or FourChannelRuntime.legal_exogenous(c)]
            # Historical physical aliases retain frozen HLV15 prehistory. They are
            # not model columns and never read after reset timestamp.
            qc[histcols].to_parquet(dst/f'{split}_sensor_prehistory.parquet', index=False)
            data.to_parquet(dst/f'{split}_physical.parquet',index=False)
            reg.to_parquet(dst/f'{split}_registry.parquet',index=False)
            wins.to_parquet(dst/f'{split}_windows.parquet',index=False)
            provenance.to_parquet(dst/f'{split}_provenance.parquet',index=False)
            admission = registry.copy();admission['current_raw_valid']=valid_raw(cur);admission['target_raw_valid']=valid_raw(nxt);admission['eligible']=good
            admission.to_parquet(dst/f'{split}_admission.parquet',index=False)
            for suffix, raw in [('current',cur),('target',nxt)]:
                cols=[s+'_raw_value' for s in RAW]+[s+'_'+f for s in RAW for f in FLAGS]+['is_resampled_row']
                evidence=raw.loc[good,cols].reset_index(drop=True);evidence.insert(0,'row_id',reg.row_id)
                evidence.to_parquet(dst/f'{split}_{suffix}_raw_evidence.parquet',index=False)
            result[f'{crop}/{split}'] = {'old_pairs':len(registry),'pairs':len(reg),'excluded':int((~good).sum()),'windows':{str(h):int((wins.horizon==h).sum()) for h in [1,3,6,12]},'sources':reg.source_confidence.value_counts().to_dict()}
            datasets[split]=data
        trainreg=pd.read_parquet(dst/'train_registry.parquet')
        trainreg.to_parquet(dst/'scaler_fit_registry.parquet',index=False)
        payload=fit_scaler(datasets['train'],obs)
        payload.update(schema_sha256=sh,fit_registry_sha256=sha(dst/'scaler_fit_registry.parquet'),dataset_namespace=P['next_dataset_namespace'],crop=crop)
        save(payload,dst/'observation_scaler.json')
        scaler=scaler_from(payload)
        for split,data in datasets.items():
            scaled=scaler.transform_frame(data)
            # Target physical values remain physical, clearly prefixed next_.
            np.testing.assert_allclose(scaler.inverse_transform_frame(scaled)[obs].to_numpy(float),data[obs].to_numpy(float),rtol=1e-6,atol=1e-8)
            assert np.isfinite(scaled[obs].to_numpy(float)).all()
            scaled.to_parquet(dst/f'{split}_scaled.parquet',index=False)
        normalizers={c:float(datasets['validation']['next_'+c].std(ddof=0)) for c in TARGETS}
        assert all(v>0 and np.isfinite(v) for v in normalizers.values())
        save({'role':'fixed Validation evaluation denominator only; not input scaler fitting','std_ddof0':normalizers,'fallbacks':[]},dst/'evaluation_normalization.json')
    return result

def main():
    lock=json.loads((PREREG/'preregistration_lock.json').read_text())
    assert sha(PREREG/lock['file'])==lock['sha256']
    REPORT.mkdir(exist_ok=True)
    if DEST.exists(): raise RuntimeError('New namespace already exists; never overwrite')
    with tempfile.TemporaryDirectory(prefix='geas4ch_') as tmp:
        a,b=Path(tmp)/'a',Path(tmp)/'b'
        stats=build(a);assert stats==build(b)
        ha={str(p.relative_to(a)):sha(p) for p in a.rglob('*') if p.is_file()}
        hb={str(p.relative_to(b)):sha(p) for p in b.rglob('*') if p.is_file()}
        assert ha==hb,'Independent rebuild differs'
        shutil.copytree(a,DEST)
    manifest={'status':'DATASET_BUILT — RUNTIME_GATES_PENDING','preregistration_sha256':lock['sha256'],'dataset_namespace':DEST.name,'model_fit':False,'input_hashes':READS,'artifact_hashes':ha,'counts':stats,'deterministic_independent_rebuild':True,'runtime_adapter_version':VERSION,'runtime_adapter_sha256':sha(R/'src/geas35/models/transition/four_channel_phase0.py'),'builder_sha256':sha(Path(__file__)),'forbidden_reads':{'Test':0,'P0':0},'gates':{'counts':True,'raw_identity':True,'exact_join':True,'Train_only_scaler':True,'command_carry':True,'schema_no_CO2':True,'deterministic_rebuild':True}}
    save(manifest,REPORT/'phase0_manifest.json')
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':main()
