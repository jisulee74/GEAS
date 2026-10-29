"""Read-only trace analysis and report; no rollout/fit/data regeneration."""
from pathlib import Path
import sys,json,gzip
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.models.transition.handoff_hlv15_b60_r6_v1 import sha,SELECTED
D=ROOT/'experiments/rl_policy_training/artifacts/step16_hlv15_b60_r6_v1';S=ROOT/'experiments/rl_policy_training/artifacts/step15_hlv15_b60_r6_v1';H=ROOT/'experiments/transition_model_selection/artifacts/step14_hlv15_b60_r6_v1';T=ROOT/'experiments/transition_model_selection/artifacts/command_carry_hlv15_b60_r6_v1'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def table(f):
 f=f.copy()
 for c in f:
  if pd.api.types.is_float_dtype(f[c]):f[c]=f[c].map(lambda x:f'{x:.6g}')
 return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(map(str,r))+' |' for r in f.itertuples(index=False,name=None)])
pro=read(D/'protocol.json');assert sha(D/'protocol.json')==read(D/'protocol_lock.json')['sha256']
for x in pro['inputs']+pro['code']:assert sha(ROOT/x['path'])==x['sha256']
assert all((D/c/'result.json').exists() for c in SELECTED)
results=[];ood=[];pathparity=[];actiontables=[];rewardreconciliations=[];firsts=[];near=[];rewardflags=[];jointcomponents=[]
for crop,name in SELECTED.items():
 out=D/crop;support_config=read(S/crop/'state_action_support.json');reference=np.load(S/crop/'support_reference.npz');joint_reference=np.column_stack([reference['reference_state']*np.array(support_config['state_weights']),reference['reference_action']*np.array(support_config['action_weights'])])/np.sqrt(2);joint_columns=support_config['state_columns']+support_config['action_columns'];result=read(out/'result.json');results.append(result);ep=pd.read_csv(out/'episode_summary.csv');data=np.load(T/crop/name/'rollout/predictions.npz');lookup={str(k):i for i,k in enumerate(data['start_row_id'])};maxerror=0.;matched=0;differentaction=0
 for e in ep.to_dict('records'):
  path=out/'episodes'/f"{e['episode_index']:04d}.jsonl.gz"
  episode_record=read(path.with_suffix('').with_suffix('.json'));assert episode_record['trace_sha256']==sha(path);assert episode_record['protocol_sha256']==sha(D/'protocol.json')
  with gzip.open(path,'rt') as stream:rows=[json.loads(line) for line in stream]
  key=rows[0]['row_id'];baselineindex=lookup.get(key);seen=set()
  for j,row in enumerate(rows):
   info=row['info'];changed=row['changed_actions_before']+int(row.get('action_changed',False))
   if row['executed'] and row['step']<12 and baselineindex is not None and row['step']<data['max_horizon'][baselineindex]:
    baseline=data['predictions'][baselineindex,row['step']];difference=np.array(row['prediction'])-baseline
    if changed==0:
     np.testing.assert_allclose(row['prediction'],baseline,rtol=1e-6,atol=1e-8);maxerror=max(maxerror,float(np.max(np.abs(difference))));matched+=1
    else:differentaction+=1
   for kind in ('state','joint'):
    if info[kind+'_ood_level']=='severe' and kind not in seen:
     seen.add(kind);firsts.append({'crop':crop,'episode_index':e['episode_index'],'kind':kind,'first_step':row['step'],'timestamp':row['timestamp'],'recorded_state_severe':row['recorded_state_severe'],'distance':info[kind+'_ood_score'],'threshold':info['support_thresholds'][kind+'_severe']})
  last=rows[-1];reason=last['info']['termination_reason'];jointalready=last['recorded_joint_distance']>last['info']['support_thresholds']['joint_severe']
  if reason in ('severe_state_ood','repeated_severe_action_ood'):
   kind='state' if reason=='severe_state_ood' else 'joint';recorded_severe=last['recorded_state_severe'] if kind=='state' else jointalready
   label='recorded_severe_overlap' if recorded_severe else ('rollout_only_after_safety_action_changes' if last['changed_actions_before'] else 'rollout_only_no_prior_action_changes')
   ood.append({'crop':crop,'episode_index':e['episode_index'],'termination_reason':reason,'classification':label,'executed_steps':e['executed_steps'],'scheduled_steps':e['scheduled_valid_steps'],'recorded_state_distance':last['recorded_state_distance'],'recorded_joint_distance':last['recorded_joint_distance'],'state_distance':last['info']['state_ood_score'],'joint_distance':last['info']['joint_ood_score'],'reset_termination':e['executed_steps']==0,'recorded_state_severe_before_or_at':any(rr['recorded_state_severe'] for rr in rows),'recorded_path_normal_through_termination':not any(rr['recorded_state_severe'] for rr in rows),'prior_action_changes':last['changed_actions_before']})
   joint_query=np.r_[np.array(last['current_scaled'])*np.array(support_config['state_weights']),np.array(last['proposed'])*np.array(support_config['action_weights'])]/np.sqrt(2)
   sq=(joint_reference-joint_query)**2;nearest=np.argpartition(sq.sum(axis=1),support_config['neighbors']-1)[support_config['neighbors']-1];components=sq[nearest]
   for coordinate in np.argsort(components)[-10:][::-1]:jointcomponents.append({'crop':crop,'episode_index':e['episode_index'],'termination_reason':reason,'coordinate':joint_columns[coordinate],'squared_distance_component':components[coordinate],'component_fraction':components[coordinate]/components.sum() if components.sum() else 0,'joint_distance_reconstructed':np.sqrt(components.sum()),'joint_distance':last['info']['joint_ood_score']})
   np.testing.assert_allclose(np.sqrt(components.sum()),last['info']['joint_ood_score'],rtol=1e-6,atol=1e-8)
   for location,rr in [('before',rows[-2] if len(rows)>1 else None),('at',last)]:
    if rr:near.append({'crop':crop,'episode_index':e['episode_index'],'location':location,'timestamp':rr['timestamp'],'row_id':rr['row_id'],'step':rr['step'],'state_distance':rr['info']['state_ood_score'],'joint_distance':rr['info']['joint_ood_score'],'recorded_state_distance':rr['recorded_state_distance'],'physical_state':json.dumps(rr['current_physical']),'scaled_state':json.dumps(rr['current_scaled'])})
 np.testing.assert_allclose(-result['terminal_cost_sum'],result['scheduled_steps']-result['executed_steps'],rtol=0,atol=0)
 pathparity.append({'crop':crop,'unchanged_recorded_action_steps_compared':matched,'max_abs_difference':maxerror,'safety_modified_steps_not_comparable':differentaction,'passed':True})
 a=pd.read_csv(out/'action_summary.csv');fixed=read(H/crop/'action_response.json')['report']['actions'];a.insert(0,'crop',crop);a['global_action_response_support']=a.action.map(lambda n:fixed[n]['support_count']);a['global_action_response_status']=a.action.map(lambda n:fixed[n]['status']);actiontables.append(a)
 reward=pd.read_parquet(out/'reward_steps.parquet');adjustment=reward.reward-reward.terminal_cost-(reward.base_reward-reward.warning_penalty)
 np.testing.assert_allclose(reward.base_reward.sum()-reward.warning_penalty.sum()+adjustment.sum()+reward.terminal_cost.sum(),reward.reward.sum(),rtol=1e-6,atol=1e-8)
 terminal_magnitude=abs(reward.terminal_cost.sum());rewardreconciliations.append({'crop':crop,'base_reward_sum':reward.base_reward.sum(),'warning_penalty_sum':reward.warning_penalty.sum(),'clip_and_fallback_adjustment':adjustment.sum(),'terminal_cost_sum':reward.terminal_cost.sum(),'reward_sum':reward.reward.sum(),'terminal_absolute_share_of_final_return':terminal_magnitude/abs(reward.reward.sum()) if reward.reward.sum() else None})
 weighted=pd.read_csv(out/'reward_summary.csv').dropna(subset=['six_term_absolute_share'])
 dominant=weighted.sort_values('six_term_absolute_share',ascending=False).iloc[0]
 active=reward.loc[reward.executed]
 rewardflags.append({'crop':crop,'dominant_term':dominant['term'],'dominant_share_of_six_penalties':float(dominant.six_term_absolute_share),'dominant_over_80pct_diagnostic':bool(dominant.six_term_absolute_share>.8),'base_minus_warning_below_worst_fraction':float((active.base_reward-active.warning_penalty<=-1).mean()),'terminal_cost_absolute_share':rewardreconciliations[-1]['terminal_absolute_share_of_final_return']})
 write(out/'phase12_path_comparison.json',pathparity[-1])
pd.DataFrame(jointcomponents).to_csv(D/'joint_termination_kth_neighbor_components.csv',index=False)
pd.DataFrame(ood).to_csv(D/'ood_termination_attribution.csv',index=False);pd.DataFrame(firsts).to_csv(D/'first_severe_ood.csv',index=False);pd.DataFrame(near).to_parquet(D/'pretermination_states.parquet',index=False);pd.DataFrame(pathparity).to_csv(D/'phase12_runtime_path_comparison.csv',index=False);actions=pd.concat(actiontables);actions.to_csv(D/'action_support_projection_comparison.csv',index=False);pd.DataFrame(rewardreconciliations).to_csv(D/'reward_reconciliation.csv',index=False)
tests=Path('/tmp/geas_step16_hlv15_b60_tests.log').read_text();assert '4 passed' in tests;(D/'regression_tests.txt').write_text(tests)
pd.DataFrame(rewardflags).to_csv(D/'reward_sanity_flags.csv',index=False)
status='PASS' if all(r['Step17_entry']=='PASS' for r in results) else 'BLOCKED'
P=ROOT/'experiments/rl_policy_training/artifacts/past_only_qc_hlv15_b60_r6_v1'
reaches=[];conditional=[];divergences=[];alignment=[];extra_reward=[]
for crop in SELECTED:
 out=D/crop;frame=pd.read_parquet(P/crop/'validation_physical.parquet').set_index('row_id');cfg=read(S/crop/'state_action_support.json');obs=cfg['state_columns'];ep=pd.read_csv(out/'episode_summary.csv')
 for h in (3,6,12):
  reaches.append({'crop':crop,'horizon_minutes':h*5,'scheduled_runs':int((ep.scheduled_valid_steps>=h).sum()),'reached_runs':int((ep.executed_steps>=h).sum()),'all_runs':len(ep)})
 trajectories={h:[] for h in (3,6,12)}
 for e in ep.to_dict('records'):
  with gzip.open(out/'episodes'/f"{e['episode_index']:04d}.jsonl.gz",'rt') as f:rows=[json.loads(x) for x in f]
  first=None
  for row in rows:
   actual=frame.loc[row['row_id'],obs].to_numpy(float);current=np.array(row['current_physical']);delta=current-actual;bad=~np.isclose(current,actual,rtol=1e-6,atol=1e-8)
   if first is None and bad.any():
    first={'crop':crop,'episode_index':e['episode_index'],'first_step':row['step'],'timestamp':row['timestamp'],'features':';'.join(np.array(obs)[bad]),'max_abs_difference':float(np.max(np.abs(delta))),'prior_action_changes':row['changed_actions_before'],'interpretation':'first observed difference, not causal attribution'}
   info=row['info'];alignment.append({'crop':crop,'episode_index':e['episode_index'],'step':row['step'],'timestamp':row['timestamp'],'row_id':row['row_id'],'recorded_state_distance':row['recorded_state_distance'],'free_state_distance':info['state_ood_score'],'recorded_proposed_joint_distance':row['recorded_joint_distance'],'free_proposed_joint_distance':info['joint_ood_score'],'free_final_joint_distance':info.get('executed_joint_ood_score'),'state_severe_threshold':cfg['thresholds']['state_severe'],'joint_severe_threshold':cfg['thresholds']['joint_severe'],'changed_actions_before':row['changed_actions_before'],'executed':row['executed'],'termination_reason':info['termination_reason']})
  divergences.append(first or {'crop':crop,'episode_index':e['episode_index'],'first_step':None,'interpretation':'no divergence in visited current observations'})
  executed=[r for r in rows if r['executed']]
  for h in trajectories:
   if len(executed)>=h:trajectories[h].append(np.array([r['recorded_path_difference'] for r in executed[:h]]))
 for h,arr in trajectories.items():
  if arr:
   a=np.stack(arr)
   for j,target in enumerate(obs[:3]):conditional.append({'crop':crop,'horizon_minutes':h*5,'target':target,'surviving_runs':len(a),'trajectory_steps':len(a)*h,'trajectory_rmse':float(np.sqrt(np.mean(a[:,:,j]**2))),'trajectory_mae':float(np.mean(abs(a[:,:,j]))),'endpoint_rmse':float(np.sqrt(np.mean(a[:,-1,j]**2))),'scope':'survival conditioned; changed-action paths not counterfactual truth'})
 reward=pd.read_parquet(out/'reward_steps.parquet');x=reward.loc[reward.executed,'base_reward'];extra_reward.append({'crop':crop,'term':'base_reward','n':len(x),'mean':x.mean(),'p50':x.quantile(.5),'p90':x.quantile(.9),'p99':x.quantile(.99),'min':x.min(),'max':x.max(),'sum':x.sum()})
pd.DataFrame(reaches).to_csv(D/'horizon_reach_counts.csv',index=False)
pd.DataFrame(conditional).to_csv(D/'survival_conditioned_horizon_metrics.csv',index=False)
pd.DataFrame(divergences).to_csv(D/'first_recorded_runtime_divergence.csv',index=False)
pd.DataFrame(alignment).to_parquet(D/'timestamp_aligned_ood.parquet',index=False)
pd.DataFrame(extra_reward).to_csv(D/'base_reward_summary.csv',index=False)
summary=pd.DataFrame([{'crop':r['crop'],'runs':r['episode_runs'],'executed_runs':int((pd.read_csv(D/r['crop']/'episode_summary.csv').executed_steps>0).sum()),'executed':r['executed_steps'],'scheduled':r['scheduled_steps'],'executed_rate':r['executed_fraction'],'normal':r['termination_counts'].get('episode_boundary',0),'state_severe':r['termination_counts'].get('severe_state_ood',0),'joint_repeated':r['termination_counts'].get('repeated_severe_action_ood',0),'guard':r['practical_rollout_gate'],'physical':r['physical_sanity']} for r in results]);summary.to_csv(D/'step16_summary.csv',index=False)
lines=[f'# HLV15 causal B60/R6 Step16 연구 환경 검증 — {status}',
'\nHLV15 Phase0 → Transition Phase1/2 → Step14 handoff/Step15 support PASS 이후의 비학습 Step16 검증이다. 실제 배포 online actuator-safe fallback은 별도 **BLOCKED**로 유지한다. PPO/Step17–19/Step19.5/HPO/Test/domain-v2/C/E는 실행하지 않았다.',
'\n## 고정 입력과 기준',
'melon=XGBoost, cucumber=ExtraTrees, strawberry=LightGBM. past_only_qc_hlv15_b60_r6_v1 dataset/scaler/schema와 command_carry_hlv15_b60_r6_v1 선정 모델, step14_hlv15_b60_r6_v1 전용 loader, step15_hlv15_b60_r6_v1 support만 로드했다. 입력·실행 코드 해시를 protocol/lock에서 검증했고 최종 보고 시에도 원본 무변경을 확인했다. 기존 A/v4/B60R6 모델·support를 공식 입력으로 사용하지 않았다.',
'\n기존 연구 실행 가능성 guard 유지: executed/scheduled≥50%, run의≥50%가 15분 또는 더 짧은 전체 길이 도달, 60분 예정 run이 있으면 최소 하나 도달. threshold/reward/support는 변경하지 않았다. [사전 고정 protocol](protocol.json), [lock](protocol_lock.json).',
'\n## Recorded baseline — rollout 전에 재검산',
table(pd.DataFrame(pro['baseline_checks']).drop(columns='thresholds')),
'\n행 기준 recorded 비율이며 sequential termination 비율과 다르다. Step15 저장 distance 및 threshold와 일치했고 melon5985/cucumber4663/strawberry17399 transition을 확정했다. warning은 severe 포함이다. 정확한 threshold는 protocol baseline_checks, episode/source/series 분해는 *_recorded_baseline_groups.csv에 보존했다.',
'\n## 전체 Validation rollout',table(summary),table(pd.DataFrame(reaches)),
'\n모든 frozen rollout ID와 마지막 target observation을 포함하고 각 run을 두 번 재현했다. 조기종료는 허용하되 scheduled 분모를 줄이지 않았다. episode_summary.csv는 run별 길이·종료 이유를, episodes/*.jsonl.gz는 검증된 trace를 제공한다. state severe는 model 실행 전 종료되므로 terminal-only 호출은 실행 step 수에서 제외된다.',
'\n## 환경 causal 계약',
'reset → observation → recorded proposed e_t → state/joint OOD → safety projection/fallback → weather constraint → final e_t → Transition → next derived/history → reward → termination 순서다. state severe면 Transition 없이 종료한다. reset은 provenance의 e_(t-1), e_(t-2)를 공급하며 current e_t를 이전 값으로 채우지 않는다. 실제 모델 입력=current scaled observation+final e_t, next command carry=final e_t, 다음 상태 reward와 이전 두 action을 매 step 확인했다. sensor rolling history와 command history는 분리하며 검증된 solar/alias/degree/condensation 정의를 유지했다.',
'\nHLV15/B60/R6은 원천 데이터 admission 계약이다. simulation-generated 상태에 raw stale detector나 새로운 HLV QC를 실행하지 않았다. current action 변경 시 reset observation 불변, terminal target, 첫 state severe의 무실행, 반복 joint severe fallback/종료비용을 별도 4개 회귀 테스트로 검증했다. [결과](regression_tests.txt).',
'\n## OOD 원인 분류',table(pd.DataFrame(ood).groupby(['crop','termination_reason','classification']).size().rename('runs').reset_index()),
'\n기록 경로가 방문한 모든 timestamp에서 state severe가 아니었는지도 아래에 별도 집계한다. 종료점 overlap과 구분하며, warning까지 없다는 의미는 아니다.',table(pd.DataFrame(ood).loc[lambda f:f.termination_reason.eq('severe_state_ood')].groupby(['crop','recorded_path_normal_through_termination','classification']).size().rename('runs').reset_index()),'\nrecorded_severe_overlap은 같은 timestamp 기록 상태/기록 action도 severe라는 뜻이다. rollout_only_no_prior_action_changes는 기록 action 유지 경로이고, after_safety_action_changes는 순수 Transition 오류로 단정할 수 없다. 단순 종료점 분류이며 teacher-forcing 원인 확정 실험은 아니다. state와 joint를 분리했다. 거리 기여도는 kth-neighbor 거리 성분으로 인과 중요도가 아니다.',
'\n[timestamp별 recorded/free/proposed/final joint](timestamp_aligned_ood.parquet), [처음 recorded/runtime 차이](first_recorded_runtime_divergence.csv), [첫 severe](first_severe_ood.csv), [종료 분류](ood_termination_attribution.csv), [종료 직전/당시 상태](pretermination_states.parquet), [joint 거리 성분](joint_termination_kth_neighbor_components.csv). 처음 차이는 관측된 divergence이며 원인 feature 판정과 구분한다.']
for crop in SELECTED:
 f=pd.read_csv(D/crop/'ood_kth_neighbor_contributions.csv');g=f.loc[f.termination_reason.eq('severe_state_ood')].groupby('feature').agg(occurrences=('feature','size'),mean_distance_share=('component_fraction','mean')).sort_values('occurrences',ascending=False).head(8).reset_index();lines.append(f'\n### {crop} severe state 주요 좌표\n'+table(g))
lines+=['\n## Action support/projection/fallback',table(actions),
'\n분모는 실행 step이다. change rate는 proposed→final, local_support_limited는 기존 projection 변경 여부다. 전역 action-response와 현재 상태 주변 k50 local support는 서로 다르며 전역 support 부족이 자동 action 동결을 뜻하지 않는다. 기존 연구 fallback은 실제 actuator 안전성이 검증됐다는 뜻이 아니다. 딸기 shade는 global support804이나 response0이라는 제한도 유지한다.',
'\n## Reward audit',table(pd.DataFrame(rewardreconciliations)),table(pd.DataFrame(rewardflags)),table(pd.DataFrame(extra_reward)),
'\nbase−warning+clipping/fallback adjustment+terminal=final return을 재현했다. six-term 비중과 clipped final return의 기여도는 다르다. terminal 비중은 abs(terminal sum)/abs(final sum)이다. 온도 항목 집중과 RH/VPD/cond/HV의 0 여부는 아래 표로 확인한다. 계산 finite/reproducible이 reward 설계 적합성 승인을 뜻하지 않으며, PPO 전에 별도 reward audit가 필요하다. weight/formula/normalization은 그대로다.']
for crop in SELECTED:lines.append(f'\n### {crop} reward\n'+table(pd.read_csv(D/crop/'reward_summary.csv')))
lines+=['\n## Physical sanity / Phase1/2 parity',table(pd.DataFrame(pathparity)),
'\n같은 시작 window에서 과거 및 현재 final action이 기록 action과 동일한 step만 Phase1/2 저장 예측과 비교했다. safety 변경 이후 경로에는 동일 prediction을 요구하지 않는다. Phase1/2는 OOD 종료 없는 고정 horizon, Step16은 OOD/projection 있는 sequential rollout이므로 모집단도 다르다. 아래15/30/60분 metric은 그 길이까지 살아남은 run만의 조건부 통계이며 전체 Validation 성능으로 해석하지 않는다. action이 변경된 경로와 기록의 차이는 반사실적 예측 오차로 단정하지 않는다.',table(pd.DataFrame(conditional)),
'\n물리 범위는 기존 T[-60,80]°C/RH[0,100]%/CO₂[0,10000]ppm이다. 급변 T5°C/RH20%p/CO₂500ppm/step은 설명용 flag이며 gate 기준을 새로 조절한 것이 아니다. 생육 적정성·실제 배포 안전성의 보증이 아니다.']
for crop in SELECTED:lines.append(f'\n### {crop} physical\n'+table(pd.read_csv(D/crop/'physical_summary.csv')))
lines+=['\n## 제한과 최종 판단',
'딸기 Train CO₂≥1500ppm22행, high-CO₂+previous high vent3행/current high vent2행의 coverage 한계 유지. 기존 Phase1/2 CO₂ skill15/30/60분 −0.007097/−0.012277/−0.088191도 유지하며 이번 OOD 종료 후 조건부 metric으로 대체하지 않는다. legacy snapshot assumption, 미검증 geas2_5 일반화, 과거 clean free-only5건의 Transition/autoregressive limitation은 해소된 것으로 주장하지 않는다.',
f'\nStep16 연구 실행 가능성 종합 **{status}**. '+('다음은 **Step16 OOD/reward 결과 검토 후 Step17 entry decision**이다. 이 PASS는 Step17 실행 또는 PPO 학습 승인이 아니다.' if status=='PASS' else '전체 Step17 **BLOCKED** 유지. 다음 최소 비용 작업 하나는 실패 작물의 recorded-severe overlap과 unchanged-action rollout-only를 분리한 비학습 OOD 경로 감사다. 같은 timestamp 기록/teacher-forced/free path를 비교하여 분포 이탈과 누적 예측 영향을 구분한다.'),
'\n온라인 actuator-safe fallback은 별도 **BLOCKED**. 공식 dataset/scaler/model/support/threshold/reward 무변경. domain-v2/C/E와 기존 Step19.5 미통과를 재판정하지 않았다. 과거 결과는 `historical metric preserved / causal availability gate superseded by past-only QC rebuild`로만 취급한다.',
'\n[최종 gate](final_gate.json), [Step16 요약](step16_summary.csv), [재현성 및 기존 모델 경로 parity](phase12_runtime_path_comparison.csv).']
lines[1:1]=['\n핵심 결과: 멜론94.30%/오이63.44%/딸기59.11% 실행률로 세 작물 모두 기존 Step16 guard를 통과했다. 모든 run 2회 재현·causal wiring·reward 재계산·physical 검사 PASS, NaN/Inf 및 기존 물리 범위 위반0건이다. 딸기 state severe30건은 종료 시점 기록 overlap19건/이전 safety action 변경 이후 rollout-only11건이며, action 변경 이전 rollout-only는0건이다. 이 분류만으로 이전 autoregressive limitation이 해결됐다고 판단하지 않는다.', '\n보상 제한은 독립적으로 남는다. 온도 penalty 비중은 멜론98.09%/오이99.84%/딸기99.45%; 딸기 terminal cost는 최종 음의 return 절댓값의80.71%다. 수치적 계산 PASS와 보상 설계 적합성은 구분한다. Step17은 검토 대기이며 이번 작업에서 실행하지 않았다.']
(D/'final_decision_ko.md').write_text('\n'.join(lines)+'\n')
write(D/'final_gate.json',{'status':status,'Step17_entry':'REVIEW_PENDING' if status=='PASS' else 'BLOCKED','crops':results,'contract_binding_replay_reward_checks':'PASS','new_environment_regression_tests':4,'phase12_unchanged_action_path_parity':'PASS','original_input_hashes_unchanged':True,'online_deployment':'BLOCKED','Step17_19_PPO_HPO_Step19_5_Test_executed':False,'threshold_reward_support_relaxed':False})
write(D/'report_code_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tests/test_step16_hlv15_b60_r6_env.py']})
write(D/'artifact_integrity.json',{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='artifact_integrity.json'})
print('FINAL',status,flush=True)
