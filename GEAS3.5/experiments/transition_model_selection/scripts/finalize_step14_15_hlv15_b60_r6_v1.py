"""Finalize HLV15 handoff/support, independent geometry and input binding audit."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'src'))
from geas35.models.transition.handoff_hlv15_b60_r6_v1 import SELECTED,VERSION,sha
from geas35.rl.bound_support_hlv15_b60_r6_v1 import load_bound_research_artifacts
H=R/'experiments/transition_model_selection/artifacts/step14_hlv15_b60_r6_v1';S=R/'experiments/rl_policy_training/artifacts/step15_hlv15_b60_r6_v1'
P=R/'experiments/rl_policy_training/artifacts/past_only_qc_hlv15_b60_r6_v1';D=R/'experiments/transition_model_selection/artifacts/command_carry_hlv15_b60_r6_v1';OLD=S.parent/'step15_b60_r6_v1'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def table(f):
 def v(x):return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)
 return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(v(x) for x in row)+' |' for row in f.itertuples(index=False,name=None)])+'\n'
def main():
 pro=read(H/'protocol.json');assert sha(H/'protocol.json')==read(H/'protocol_lock.json')['sha256']
 for rec in pro['inputs']+pro['code']:assert sha(R/rec['path'])==rec['sha256'],rec['path']
 assert read(S/'bound_loader_validation.json')['status']=='PASS'
 hands=[];supports=[];cal=[];action=[];geometry=[];local=[];selected=[]
 for crop,name in SELECTED.items():
  h=read(H/crop/'gate.json');g=read(S/crop/'gate.json');assert h['status']==g['status']=='PASS';hands.append(h);supports.append(g)
  path=S/crop/'binding.json';rt,sup=load_bound_research_artifacts(path,binding_sha256=sha(path),crop=crop,root=R)
  manifest=read(H/crop/'manifest.json');sel=read(D/crop/'selection.json');selected.append(dict(crop=crop,algorithm=name,weighted_score=manifest['selection_score'],persistence_score=manifest['persistence_score'],model_hash=sha(H/crop/'model.pkl'),manifest_hash=sha(H/crop/'manifest.json')))
  # Independent Euclidean k50 calculation against the newly generated reference.
  frame=pd.read_parquet(P/crop/'validation_scaled.parquet');ix=np.unique(np.linspace(0,len(frame)-1,24).astype(int));x=frame.iloc[ix][rt.obs].to_numpy(float);a=frame.iloc[ix][rt.actions].to_numpy(float)
  sw=np.array(sup.config['state_weights']);aw=np.array(sup.config['action_weights'])
  for group in ('continuous_state_columns','binary_state_columns'):
   cols=sup.config[group]
   for col in cols:assert sw[rt.obs.index(col)]==1/np.sqrt(len(cols))
  for group in ('continuous_action_columns','binary_action_columns'):
   cols=sup.config[group]
   for col in cols:assert aw[rt.actions.index(col)]==1/np.sqrt(len(cols))
  scores=sup.scores(frame.iloc[ix][rt.obs],frame.iloc[ix][rt.actions]);maxdiff=0.
  for j in range(len(x)):
   dstate=((sup.reference_state_raw-x[j])*sw)**2
   daction=((sup.reference_action_raw-a[j])*aw)**2
   for kind,dist in [('state',np.sqrt(dstate.sum(axis=1))),('joint',np.sqrt((dstate.sum(axis=1)+daction.sum(axis=1))/2))]:
    expected=np.partition(dist,49)[49];np.testing.assert_allclose(expected,scores[kind][j],rtol=1e-6,atol=1e-8);maxdiff=max(maxdiff,float(abs(expected-scores[kind][j])))
  geometry.append(dict(crop=crop,rows=len(ix),state_joint_checks=len(ix)*2,k=50,max_abs_diff=maxdiff,mismatches=0,weights_verified=True,joint_sqrt2_verified=True))
  m=sup.config
  for kind in ('state','joint'):
   cal.append(dict(crop=crop,kind=kind,reference_rows=m['reference_rows'],calibration_rows=m['calibration_rows'],reference_episodes=m['reference_episode_count'],calibration_episodes=m['calibration_episode_count'],warning_threshold=m['thresholds'][kind+'_warning'],severe_threshold=m['thresholds'][kind+'_severe'],warning_coverage=m['empirical_calibration_coverage'][kind+'_warning'],severe_coverage=m['empirical_calibration_coverage'][kind+'_severe']))
  report=read(H/crop/'action_response.json')['report'];oldreport=read(D/crop/name/'fit/completed.json')['action_response'];assert report['eligible']==oldreport['eligible']
  for key,value in report['actions'].items():
   expected=oldreport['actions'][key];assert value['support_count']==expected['support_count'] and value['status']==expected['status']
   np.testing.assert_allclose(value['counterfactual_mean_absolute_sensitivity'],expected['counterfactual_mean_absolute_sensitivity'],rtol=1e-6,atol=1e-8,equal_nan=True)
  for act,v in report['actions'].items():
   mag=v['counterfactual_mean_absolute_sensitivity'];action.append(dict(crop=crop,action=act,support_count=v['support_count'],global_action_support_eligible=v['status']=='eligible',response_magnitude=mag,nonzero_finite_response=bool(np.isfinite(mag) and mag>1e-12),candidate_gate=report['eligible'],population='new Validation adjacent changes, fixed action-response protocol'))
  # Local support geometry is distinct from global response. Preserve exact metadata.
  local.append(dict(crop=crop,**m['local_action_support']))
 summaries=pd.concat([pd.read_csv(S/c/'support_summary.csv') for c in SELECTED],ignore_index=True)
 summaries.to_csv(S/'support_summary.csv',index=False);pd.DataFrame(cal).to_csv(S/'calibration_summary.csv',index=False);pd.DataFrame(action).to_csv(S/'action_response_summary.csv',index=False)
 write(S/'geometry_validation.json',dict(status='PASS',crops=geometry));write(S/'local_action_geometry.json',local)
 historical=pd.read_csv(OLD/'support_summary.csv')
 ref=summaries.merge(historical,on=['crop','split','kind'],suffixes=('_new','_historical'))
 ref['interpretation']='historical metric preserved / causal availability gate superseded by past-only QC rebuild'
 ref.to_csv(S/'historical_comparison.csv',index=False)
 rows=[]
 for c in SELECTED:
  summary=summaries.loc[summaries.crop.eq(c)&summaries.split.eq('validation')].set_index('kind');h=next(x for x in hands if x['crop']==c)
  rows.append(dict(crop=c,model=SELECTED[c],Step14='PASS',one_step_rows=h['one_step_rows'],sequential_checks=h['rollout_steps'],one_step_max_abs_diff=h['one_step_max_abs_diff'],rollout_max_abs_diff=h['rollout_max_abs_diff'],state_severe_rate=summary.loc['state','severe_rate'],joint_severe_rate=summary.loc['joint','severe_rate']))
 write(H/'step14_handoff_manifest.json',dict(version=VERSION,status='PASS',dataset=P.name,protocol=dict(path=str((H/'protocol.json').relative_to(R)),sha256=sha(H/'protocol.json')),crops={h['crop']:h['manifest'] for h in hands}))
 gate=dict(status='PASS',Step14='PASS',Step15='PASS',Step16_entry='eligible for separate research validation only',Step16_executed=False,strawberry_Step16='BLOCKED maintained',overall_Step17='BLOCKED maintained',online_deployment='BLOCKED unverified actuator-safe handling',handoffs=hands,supports=supports,inputs_unchanged=True,model_fit=False,model_reselection=False,Test_accessed=False,threshold_tuned=False)
 write(S/'final_gate.json',gate);write(H/'final_gate.json',dict(status='PASS',crops=hands,Step15_started_only_after_all_crops_PASS=True))
 tests=Path('/tmp/hlv15_step14_history_regression.log').read_text();assert '57 passed' in tests;(S/'history_regression.log').write_text(tests)
 text=['# HLV15 causal Step14 handoff + Step15 support — PASS\n','작성일: 2026-10-12 KST\n','**Step14 PASS / Step15 PASS.** Step16은 실행하지 않았다. strawberry Step16 / 전체 Step17 / online deployment BLOCKED는 유지한다.\n',table(pd.DataFrame(rows)),
 '## 공식 binding\n',table(pd.DataFrame(selected).drop(columns=['model_hash','manifest_hash'])),
 '새 Phase0 `past_only_qc_hlv15_b60_r6_v1`과 Transition `command_carry_hlv15_b60_r6_v1`만 공식 입력이다. melon=XGBoost, cucumber=ExtraTrees, strawberry=LightGBM을 그대로 사용했다. 재선정/fit 없음. 이전 strawberry KNN은 로드하지 않았다. 모델/scaler/schema/runtime contract는 byte-identical 복사했고 physical/scaled dataset, Train/Validation row/window registry, provenance/quality sidecar, scaler-fit registry를 hash binding했다.\n',
 '전용 loader는 `src/geas35/models/transition/handoff_hlv15_b60_r6_v1.py`, support loader는 `src/geas35/rl/bound_support_hlv15_b60_r6_v1.py`다. caller-pinned manifest hash와 HLV15 dataset version, crop/algorithm, 55 observation+6 action order, T/RH/CO₂ target order 및 embedded scaler를 검사한다. old loader/env factory는 공식 경로가 아니다.\n',
 '## Step14 parity / causal wiring\n','전체 Validation one-step(5985/4663/17399행), 작물당 8개 고정 60분 window×12step=96 sequential checks를 수행했다. offline 저장 prediction 대 runtime, batch 대 scalar 55-state, next command carry의 mismatch는 모두0이다. rtol=1e-6, atol=1e-8. 모든 one-step output은 finite이며 action perturbation probe physical/nonfinite violation도0이다. 잘못된 crop 및 model/scaler/schema/contract hash 변조 거부를 검증했다.\n',
 'observation의 e_(t-1)와 current e_t input, next carry=e_t, 분리된 sensor/command history를 유지한다. HLV15/QC를 loader에서 재구성하지 않는다. admitted registry만 평가하고 raw B60 stale detector를 simulation prediction에 실행하지 않는다. history/HLV/command 회귀57개 PASS. 작물별 runtime_parity_trace.parquet에 current/action/predicted/next 전체 trace가 있다.\n',
 '## Step15 고정 Train-only calibration\n','reference≤10000, calibration≤5000, Train episode20%, seed42, k50, 기존 continuous/binary 그룹별 1/sqrt(n) weights, joint sqrt(2), conformal95/99를 유지했다. Train 모집단28713/14192/43909만 fit/calibration, Validation5985/4663/17399는 score-only다. reference/calibration episode disjoint, deterministic rebuild array equality, exact Train row registry와 독립 Euclidean k50 계산을 검증했다. Validation 결과를 보고 threshold 또는 sampling을 변경하지 않았다.\n',table(pd.DataFrame(cal)),
 table(summaries.loc[summaries.split.eq('validation'),['crop','kind','rows','warning_threshold','severe_threshold','warning_including_severe_rate','severe_rate']]),
 'warning rate는 severe를 포함하며 distance>threshold 기준이다. 높은 Validation severe는 downstream 분포 문제이며 이번 implementation/support construction gate에 새 automatic fail로 추가하지 않았다. 전체 거리 p0/p50/p90/p95/p99/p100은 support_summary.csv, source/episode별 분해와 row score는 각 crop 디렉터리에 있다.\n',
 '## Action-response와 local geometry 분리\n',table(pd.DataFrame(action).drop(columns=['population'])),
 '위 support count는 기존 action-response protocol의 새 Validation 연속 action 변화 수다. candidate gate는 지원≥100인 action 중 반응이 존재하는 기존 규칙이다. 여섯 action 전부≥100 또는 전부nonzero라는 새 gate가 아니다. Step15 local action geometry는 Train k50 reference의 continuous interval/conformal adjustment와 binary neighbor count이며 local_action_geometry.json에 별도 보존했다. 두 support 의미를 혼합하지 않는다.\n',
 'strawberry LightGBM shade_curtain_pct response=0은 Transition provenance와 이번 재검사 모두 유지됐다. 이를 support threshold나 action eligibility 변경 근거로 사용하지 않았다. 일부 action의 지원 부족/무반응을 실제 제어 안전성 보증으로 해석하지 않는다.\n',
 '## Historical 비교 및 limitation\n','`historical metric preserved / causal availability gate superseded by past-only QC rebuild`\n',
 'historical_comparison.csv는 old B60/R6의 threshold/거리분포/비율과 새 결과를 나란히 보여준다. 과거 자료는 비교만 했고 reference/calibration/threshold/model/scaler를 공식 입력으로 재사용하지 않았다.\n',
 'strawberry Train CO₂≥1500=22행, +previous vent≥0.75=3행, +current vent≥0.75=2행의 희소성은 남는다. geas2_5 unseen series, melon/strawberry legacy_command_snapshot_assumption, clean5 historical autoregressive accumulation, reward imbalance, online actuator-safe fallback 미검증도 유지한다. 지원 충분성이나 고농도 dynamics 일반화 확인이라고 주장하지 않는다.\n',
 '## 다음 작업\n','**HLV15 causal Step16 research environment validation** 하나를 권장한다. 이번에는 Step16/domain-v2/C/E/reward 변경/PPO/Step17–19/Step19.5/HPO/Test 및 online deployment approval을 수행하지 않았다.\n']
 (S/'final_decision_ko.md').write_text('\n'.join(text));(H/'final_decision_ko.md').write_text('\n'.join(text))
 for folder in (H,S):
  write(folder/'artifact_integrity.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact_integrity.json'})
 print('FINAL Step14 PASS Step15 PASS; no Step16',flush=True)
if __name__=='__main__':main()
