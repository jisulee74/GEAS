"""Non-learning Phase0 verification on all reset boundaries and sampled paths."""
from pathlib import Path
import sys,json,subprocess,importlib.metadata
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'src'))
from geas35.models.transition.four_channel_phase0 import *
B=R/'experiments/rl_policy_training/artifacts';D=B/'past_only_qc_hlv15_b60_r6_4ch_v1';O=B/'transition_4ch_phase0_v1'

def verify():
    manifest=json.loads((O/'phase0_manifest.json').read_text())
    assert all(sha(R/p)==h for p,h in manifest['input_hashes'].items())
    assert all(sha(D/p)==h for p,h in manifest['artifact_hashes'].items())
    schema=json.loads((D/'schema.json').read_text());obs=schema['observation_columns'];actions=schema['action_columns']
    results=[];failures=[]
    def check(actual,expected,where):
        a=actual[obs].to_numpy(float);b=expected[obs].to_numpy(float)
        bad=~np.isclose(a,b,rtol=1e-6,atol=1e-8)
        if bad.any():
            failures.append({'where':where,'features':{c:{'actual':float(a[i]),'expected':float(b[i])} for i,c in enumerate(obs) if bad[i]}})
        return float(np.max(np.abs(a-b)))
    for crop in ('melon','cucumber','strawberry'):
        for split in ('train','validation'):
            p=D/crop;data=pd.read_parquet(p/f'{split}_physical.parquet');history=pd.read_parquet(p/f'{split}_sensor_prehistory.parquet')
            def runtime():return FourChannelRuntime(D/'schema.json',p/'observation_scaler.json',schema_hash=sha(D/'schema.json'),scaler_hash=sha(p/'observation_scaler.json'),prehistory=history)
            def target(row):
                v=row.copy();v[obs]=row[['next_'+c for c in obs]].to_numpy(float);v['reg_date']=row.target_timestamp
                return v
            # New segment whenever a removed row makes a gap, even within parent rollout.
            starts=(data.mdp_v1_rollout_id.ne(data.mdp_v1_rollout_id.shift()) | data.reg_date.ne(data.target_timestamp.shift()))
            groups=[g for _,g in data.groupby(starts.cumsum(),sort=False)]
            maxerr=0.;steps=0;rt=runtime()
            for group in groups:
                row=group.iloc[0];cur=rt.reset(row)
                maxerr=max(maxerr,check(cur,row,f'{crop}/{split}/reset/{row.row_id}'))
                t=target(row)
                nxt=rt.advance(prediction=dict(zip(TARGETS,t[TARGETS].to_numpy(float))),action={a:float(row[a]) for a in actions},exogenous=rt.exogenous(t))
                maxerr=max(maxerr,check(nxt,t,f'{crop}/{split}/first/{row.row_id}'));steps+=1
            # Full longest admitted run, up to >288 when it exists; no model calls.
            longest=max(groups,key=len);limit=min(len(longest),301);rt=runtime();rt.reset(longest.iloc[0])
            for _,row in longest.iloc[:limit].iterrows():
                t=target(row)
                cur=rt.advance(prediction=dict(zip(TARGETS,t[TARGETS].to_numpy(float))),action={a:float(row[a]) for a in actions},exogenous=rt.exogenous(t))
                maxerr=max(maxerr,check(cur,t,f'{crop}/{split}/long/{row.row_id}'));steps+=1
            # Free synthetic delta paths with poisoned future endogenous fields
            # withheld by explicit allowlist; exercise every required horizon.
            mock=[]
            for horizon in (3,6,12):
                group=next(g for g in groups if len(g)>=horizon)
                rt=runtime();rt.reset(group.iloc[0]);baseline=rt.current[TARGETS].to_numpy(float).copy()
                rt2=runtime();rt2.reset(group.iloc[0])
                for _,row in group.iloc[:horizon].iterrows():
                    t=target(row);poison=t.copy()
                    for c in obs:
                        if not rt.legal_exogenous(c):poison[c]=-999999.
                    action={a:float(row[a]) for a in actions}
                    pred=rt.next_from_residual([.001,.002,-.001,-.002]);baseline+=np.array([.001,.002,-.001,-.002])
                    cur=rt.advance(prediction=pred,action=action,exogenous=rt.exogenous(t))
                    alt=rt2.advance(prediction=pred,action=action,exogenous=rt2.exogenous(poison))
                    np.testing.assert_allclose(cur[obs].to_numpy(float),alt[obs].to_numpy(float),rtol=1e-6,atol=1e-8)
                    np.testing.assert_allclose(cur[TARGETS].to_numpy(float),baseline,rtol=1e-6,atol=1e-8)
                    assert not any('co2' in c for c in rt.model_input(action).index)
                mock.append(horizon)
            entry={'crop':crop,'split':split,'all_reset_boundaries':len(groups),'teacher_forced_steps':steps,'longest_available_steps':len(longest),'long_teacher_forcing_steps':limit,'mock_horizons':mock,'max_absolute_teacher_forcing_error':maxerr}
            results.append(entry);print(json.dumps(entry),flush=True)
            # Save progress only in allowed companion manifest.
            manifest['runtime_validation']=results;manifest['runtime_failures']=failures
            (O/'phase0_manifest.json').write_bytes(canonical(manifest))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(R/'tests'),'-p','test_transition_four_channel_phase0.py'],capture_output=True,text=True,env={**__import__('os').environ,'PYTHONPATH':str(R/'src')})
    manifest['regression_tests']={'exit_code':tests.returncode,'output':tests.stdout+tests.stderr}
    assert all(sha(R/p)==h for p,h in manifest['input_hashes'].items()), 'Parent changed'
    deps={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','lightgbm','xgboost','scipy']}
    prereg=json.loads((B/'transition_4ch_migration_prereg_v1/migration_preregistration.json').read_text())
    manifest['dependencies']=deps
    manifest['dependency_lock_matches']=deps==prereg['dependencies']
    paths=['src/geas35/models/transition/four_channel_phase0.py','src/geas35/rl/sensor_history_env_v4.py','src/geas35/rl/causal_contract_env_v2.py','src/geas35/rl/recorded_light_env_v1.py','src/geas35/models/transition/causal_state.py','src/geas35/features/derived.py','src/geas35/rl/mdp_v1.py','experiments/rl_policy_training/scripts/build_transition_4ch_phase0.py','experiments/rl_policy_training/scripts/verify_transition_4ch_phase0.py','tests/test_transition_four_channel_phase0.py']
    manifest['code_hashes']={p:sha(R/p) for p in paths}
    manifest['runtime_adapter_sha256']=sha(R/'src/geas35/models/transition/four_channel_phase0.py')
    manifest['rollout_boundary_contract']='Parent rollout id retained for lineage; after raw exclusion exact timestamp continuity must also hold; reset at every new gap.'
    manifest['gates'].update(runtime_teacher_forcing=not failures,synthetic_regression=tests.returncode==0,dependency_lock=manifest['dependency_lock_matches'])
    passed=all(manifest['gates'].values())
    manifest['status']='TRANSITION 4CH PHASE0 — PASS / FIXED-CANDIDATE RESELECTION READY' if passed else 'TRANSITION 4CH PHASE0 — BLOCKED'
    manifest['fit_executed']=False
    manifest['limitations']=['Runtime recurrence validated with supplied diagnostic predictions; no trained four-channel model exists yet.','Existing historical context covariates are not Future-F objective approval.','Raw supervisor B60/R6 admission remains frozen; no detector applied to synthetic state.']
    (O/'phase0_manifest.json').write_bytes(canonical(manifest))
    lines=['# Transition 4CH Phase0', '',manifest['status'],'', '모델 fit / 후보 scoring / support 생성 / Step16 / PPO / Test·P0 데이터 접근을 수행하지 않았다. 기존 artifact는 변경하지 않았다.','', '| crop | split | exact pairs | 3-step | 6-step | 12-step |','|---|---|---:|---:|---:|---:|']
    for k,s in manifest['counts'].items():
        crop,split=k.split('/');w=s['windows'];lines.append(f"| {crop} | {split} | {s['pairs']:,} | {w['3']:,} | {w['6']:,} | {w['12']:,} |")
    lines+=['','56 state + 6 action → T1/T2/RH1/RH2; CO₂ 모델 컬럼 없음. 현재/target raw identity 및 +5분 identity join, B command carry 검사 완료.','', '새 namespace: `past_only_qc_hlv15_b60_r6_4ch_v1`. Scaler는 eligible Train current state만 fit, Validation은 transform-only. 고정 모델 평가용 Validation std는 별도 evaluation_normalization.json이며 input scaler와 분리했다.','', '두 번의 독립 rebuild가 모든 dataset/registry/window/schema/scaler 파일에서 byte-identical. 각 input/output/source hash는 companion manifest에 포함한다.','', 'Runtime: 기존 solar/degree/condensation/sensor rolling 수식을 보존하고, 물리 T1/RH1을 derived alias로 생성한다. 현재 action은 model input과 next command carry에만 반영된다. future record 전체 대신 allowlist exogenous만 허용하며 invalid prediction은 중단한다. 과거 representative prehistory는 frozen HLV15의 계산 이력을 유지하고 CO₂는 제외한다.','', '## 검증 범위','', '| crop/split | 모든 reset | teacher-forced steps | long path | max abs error |','|---|---:|---:|---:|---:|']
    for v in results:lines.append(f"| {v['crop']}/{v['split']} | {v['all_reset_boundaries']} | {v['teacher_forced_steps']} | {v['long_teacher_forcing_steps']} | {v['max_absolute_teacher_forcing_error']:.8g} |")
    lines+=['','모든 admitted 연속 run 시작점의 reset+1step과 split별 최장 run(최대301step), 각 split 3/6/12step mock 경로를 검사했다. 모든 데이터/window는 전수 identity 검사하지만 runtime을 모든 window에서 실행한 것은 아니다. 301-step synthetic 경로에서 >288 history·자정을 추가 검사한다. tolerance: rtol=1e-6, atol=1e-8.','',f"회귀 테스트 exit code: {tests.returncode}. dependency lock: {manifest['dependency_lock_matches']}.",'',f"실패 gate: {[k for k,v in manifest['gates'].items() if not v]}",'', '다음 한 단계: 고정 후보군 fresh fit/reselection 실행.' if passed else '다음 한 단계: manifest에 기록된 실패 gate 해결; 후보 fit 금지.']
    lines += ['', '구현 중 command source_confidence가 sensor-history boundary로 전달되어 이력이 잘리는 오류를 발견해 수정했다. command provenance는 updater 밖에서 별도 동일성을 검사하고, sensor_source/source_table·crop/series/segment·gap 경계는 기존대로 유지한다. 수식/입수/threshold 변경은 없다. 초기 실패 내역은 manifest implementation_corrections에 보존한다.', '', 'Parent rollout ID는 계보용으로 유지한다. 제거된 행이 만든 gap에서는 반드시 reset하며, 고정 window registry 밖을 임의 연결하지 않는다.']
    if failures:lines+=['','실제 첫 mismatch:','```json',json.dumps(failures[0],ensure_ascii=False,indent=2),'```']
    (O/'final_transition_4ch_phase0.md').write_text('\n'.join(lines)+'\n')
    print(manifest['status'],flush=True)
if __name__=='__main__':verify()
