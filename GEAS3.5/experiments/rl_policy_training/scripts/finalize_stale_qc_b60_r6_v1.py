from pathlib import Path
import json,hashlib,sys,tempfile,dataclasses,shutil
import pandas as pd,numpy as np
from build_stale_qc_b60_r6_v1 import D,OLD,REC,ROOT,assemble,sha,write,fit_b_scaler,OBS
summaries=[];checks=[]
for crop in ('melon','cucumber','strawberry'):
 for split in ('train','validation'):
  dest=D/crop;summary=json.loads((dest/'verification'/f'{split}_summary.json').read_text());assert summary['pass'];summaries.append(summary)
  b=pd.read_parquet(OLD/crop/f'{split}_physical.parquet');f=pd.read_parquet(REC/f'{crop}_{split}_recovery_sidecar.parquet');out,w,raw,block=assemble(b,f)
  # Independently saved rebuilding must reproduce parquet bytes, not just numeric equality.
  with tempfile.TemporaryDirectory(prefix='geas_b60_verify_') as tmp:
   for name,frame in [('physical',out),('windows',w)]:
    p=Path(tmp)/(name+'.parquet');frame.to_parquet(p,index=False);assert sha(p)==sha(dest/f'{split}_{name}.parquet')
  expected=pd.read_csv(REC/'admission_impact.csv');row=expected.loc[expected.crop.eq(crop)&expected.split.eq(split)&expected['mask'].eq('latched_6')].iloc[0];assert len(out)==row.expected_transitions
  for h in (3,6,12):assert w.horizon.eq(h).sum()==row[f'expected_windows_{h}'] and w.horizon.eq(h).sum()>0
  assert list(out.filter(regex='^obs_').columns)==list(b.filter(regex='^obs_').columns)
  checks.append({'crop':crop,'split':split,'parquet_hash_deterministic':True,'shadow_admission_expected_exact':True,'schema_unchanged':True})
 train=pd.read_parquet(D/crop/'train_physical.parquet');sc=fit_b_scaler(train);saved=json.loads((D/crop/'observation_scaler.json').read_text());assert json.loads(json.dumps(dataclasses.asdict(sc)))==saved
 registry=pd.read_parquet(D/crop/'scaler_fit_registry.parquet');assert registry.row_id.tolist()==train.row_id.tolist() and registry.split.eq('train').all()
for p,h in json.loads((D/'input_manifest.json').read_text()).items():assert sha(ROOT/p)==h
log=Path('/tmp/geas_stale_phase0_tests.log');assert '78 passed' in log.read_text();shutil.copyfile(log,D/'regression_tests.log')
write(D/'gate_results.json',{'status':'PASS','scope':'offline research Phase0 only','online_deployment_safety':'BLOCKED','teacher_feature_reward_errors':sum(x['teacher_feature_reward_errors'] for x in summaries),'teacher_steps':sum(x['teacher_steps'] for x in summaries),'reset_checks':sum(x['reset_checks'] for x in summaries),'regression_tests_passed':78,'rtol':1e-6,'atol':1e-8,'independent_checks':checks,'official_model_training_not_yet_done':True})
stats=json.loads((D/'build_summary.json').read_text());lines=['# B60/R6 offline Phase0 재구축: PASS\n','Offline research와 online deployment gate를 분리했다. **offline Phase0 PASS**, online actuator-safe handling은 **BLOCKED 유지**. 60분과6개 변화 관측 recovery는 프로젝트의 보수적 QC 선택이며 물리 고장/최적 복구 시간의 진실값이 아니다.\n','|crop|split|기존 eligible|stale 제외|recovery-only 제외|최종 eligible|3/6/12 windows|source|\n|---|---|---|---|---|---|---|---|']
for r in stats:lines.append(f"|{r['crop']}|{r['split']}|{r['old_eligible']}|{r['stale_excluded']}|{r['recovery_pending_only_excluded']}|{r['final_eligible']}|{'/'.join(str(r['windows'][str(h)]) for h in (3,6,12))}|{r['source_counts']}|")
lines += ['\nB60 causal-forward current/target、trajectory mask만 admission에 사용했다. retrospective는 감사 전용. episode 전체를 삭제하거나 끊긴 구간을 연결하지 않았다. 같은 row_id를 보존하고 새 admission gap에서 rollout_id/done을 다시 정의했다.','\n기존 B command carry, solar/current alias/degree carry/condensation prehistory/sensor-history/reward의 physical 값은 보존했다. raw sensor history 자체를 임의 삭제·대체·보간하지 않았다. stale/recovery metadata는 모델 feature가 아니며 기존55 observation schema를 유지했다. 기존 v4 및 그 판정을 덮어쓰지 않았다.','\n새 Train eligible만으로 scaler를 fit했고 Validation은 transform만 했다. 기존 scaler 계약처럼 binary observation은 그대로 유지한다. scaler row registry, hash 및 역변환을 확인했다. 동일 입력 재구축 parquet hash 일치와 shadow 예상 모집단 일치를 검증했다.','\n기존78개 회귀 테스트 및 실제 eligible episode reset, 새 admission run reset, split별 최장 teacher-forcing 및 중간 reset, command 변형 invariant를 통과했다. 허용오차 rtol=1e-6/atol=1e-8 유지. [gate_results.json](gate_results.json).','\n딸기 high-CO₂/high-vent coverage 부족 및 legacy snapshot assumption은 유지된다. 향후 개선은 stale/recovery-pending 적용 domain 분리 효과이지 고농도 dynamics를 새로 학습한 증거가 아니다.','\n이 PASS는 고정 후보군 Transition Phase1/2 재학습 진입을 허용한다. support/PPO/HPO/Step16–19/Step19.5 또는 실제 배포 승인은 아니다.','\n첫 builder의 scaler binary-column 검사 오류 실행은 `_incomplete_attempt1`로 보존하고 기존 binary 비표준화 계약대로 수정했다. 공식 과거 입력은 불변이다.']
(D/'final_decision_ko.md').write_text('\n'.join(lines))
write(D/'artifact_integrity.json',[{'path':str(p.relative_to(D)),'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file() and p.name!='artifact_integrity.json'])
print('PHASE0 PASS',json.loads((D/'gate_results.json').read_text()),flush=True)
