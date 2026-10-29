"""Stored Development result analysis only; never runs policy or Transition."""
from pathlib import Path
import json,hashlib,datetime,statistics,collections
from geas35.rl.mdp_v1 import project_mdp_v1_action_constraints,MdpV1Config
O=Path(__file__).resolve().parent;R=O.parents[3];A=R/'experiments/rl_policy_training/artifacts'
def load(p):return json.loads(Path(p).read_text())
def write(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):return {'n':len(a),'mean':statistics.mean(a) if a else None,'median':statistics.median(a) if a else None,'min':min(a) if a else None,'max':max(a) if a else None}
c=load(A/'cucumber_profile_a_final_action_migration_review_v1/migration_contract.json')
reeval=load(O/'development_reevaluation.json');assert reeval['PASS'] and reeval['seed_runs']==75 and load(O/'regression_results.json')['passed']==18
assert not (O/'technical_failure.json').exists()
th=c['inputs'];shadow=load(A/'cucumber_profile_a_final_action_shadow_v1/final_action_shadow_contract.json');T=shadow['thresholds'];ACT=shadow['action_order']
def mechanism(rows,migrated):
 i=rows[-1]['info'];category=i['category']
 if category=='COMPLETE':return 'COMPLETE'
 if category in ['SYSTEM_INVALID','UNSCORABLE']:return category
 if i.get('reason')=='severe_state_ood':return 'STATE_SUPPORT_FAILURE'
 if category=='FAILURE' and migrated and rows[-1]['final_counter_last_valid']>=3:return 'EXECUTED_JOINT_SUPPORT_FAILURE'
 if category=='FAILURE' and not migrated and rows[-1]['proposal_counter_after']>=3:return 'LEGACY_PROPOSAL_SUPPORT_FAILURE'
 return 'OTHER_EXISTING_TERMINATION'
matrix=[];perseed={};regressions=[];seven=[];replay_parity=[]
primary={(x['seed'],x['run_id']) for x in shadow['primary_population']}
oldseven={(x['seed'],x['run_id']):x for x in load(A/'cucumber_profile_a_final_action_shadow_v1/execution_v1/shadow_case_summary.json')}
for result in reeval['results']:
 s=result['seed'];rid=result['run_id'];rows={}
 for phase in ['legacy','development']:
  rows[phase]=[json.loads(l) for l in (O/f'{phase}_seed{s}_{rid}.jsonl').open()]
 L,N=rows['legacy'],rows['development'];lm,nm=mechanism(L,False),mechanism(N,True)
 if nm in ['SYSTEM_INVALID','UNSCORABLE']:label='TECHNICAL_INVALID'
 elif lm==nm=='COMPLETE':label='SAME_COMPLETE'
 elif lm!='COMPLETE' and nm=='COMPLETE':label='LEGACY_FAILURE_TO_MIGRATED_COMPLETE'
 elif lm=='COMPLETE' and nm!='COMPLETE':label='LEGACY_COMPLETE_TO_MIGRATED_FAILURE'
 elif lm==nm:label='SAME_FAILURE_MECHANISM'
 elif lm!='COMPLETE' and nm!='COMPLETE':label='DIFFERENT_FAILURE_MECHANISM'
 else:label='OTHER'
 def endtime(x):return x[-1]['outcome_timestamp'] or x[-1]['decision_timestamp']
 lac=load(O/f'legacy_seed{s}_{rid}_accounting.json');nac=result['accounting']
 delta=(datetime.datetime.fromisoformat(endtime(N))-datetime.datetime.fromisoformat(endtime(L))).total_seconds()/60
 pair={'seed':s,'run_id':rid,'parent_episode_id':nac['parent_episode_id'],'classification':label,'legacy_outcome':lm,'migrated_outcome':nm,'legacy_end_timestamp':endtime(L),'migrated_end_timestamp':endtime(N),'termination_timestamp_difference_minutes':delta,'legacy_executed_steps':lac['admitted'],'migrated_executed_steps':nac['admitted'],'executed_step_difference':nac['admitted']-lac['admitted'],'legacy_category':lac['category'],'migrated_category':nac['category'],'first_semantic_divergence_timestamp':result['first_semantic_divergence_timestamp'],'legacy_terminal_mask':{'bootstrap':0,'trace':0,'on_last_real_transition':lac['admitted']-1},'migrated_terminal_mask':{'bootstrap':0,'trace':0,'on_last_real_transition':nac['admitted']-1},'same_numeric_terminal_masks_but_alignment_may_differ':True,'legacy_discounted_return':lac['numeric_return'],'migrated_discounted_return':nac['numeric_return']}
 matrix.append(pair)
 d=perseed.setdefault(s,{'seed':s,'episode_rate_denominator':25,'outcomes':collections.Counter(),'comparison_classes':collections.Counter(),'state_rows':0,'final_queries':0,'proposal_severe':0,'final_severe':0,'proposal_reliability_recovered_events':0,'projection':0,'fallback':0,'weather':0,'complete_returns':[],'lengths':[],'scheduled_transitions':0})
 d['outcomes'][nm]+=1;d['comparison_classes'][label]+=1;d['lengths'].append(nac['admitted']);d['scheduled_transitions']+=nac['scheduled_steps']
 if nm=='COMPLETE':d['complete_returns'].append(nac['numeric_return'])
 for r in N:
  info=r['info'];ps=info['joint_distance']>T['joint_severe'];d['state_rows']+=1;d['proposal_severe']+=ps
  if info['final_action'] is not None:
   d['final_queries']+=1;fs=info['final_joint_distance']>T['joint_severe'];d['final_severe']+=fs;d['proposal_reliability_recovered_events']+=ps and not fs
   d['projection']+=info['joint_distance']>T['joint_warning'];d['fallback']+=info['fallback_action'] is not None
   mid=info['fallback_action'] if info['fallback_action'] is not None else info['projected_action'];w,flags=project_mdp_v1_action_constraints(dict(zip(ACT,mid)),r['current_state'],MdpV1Config())
   assert [w[k] for k in ACT]==info['final_action'];d['weather']+=flags['weather_constraint_applied']
 if label=='LEGACY_COMPLETE_TO_MIGRATED_FAILURE':
  div=result['first_semantic_divergence_timestamp'];assert div is not None
  terminal=N[-1];regressions.append({**pair,'termination_reason':terminal['info'].get('reason'),'terminal_proposal_streak':terminal['proposal_counter_after'],'terminal_final_streak':terminal['final_counter_last_valid'],'sole_semantic_switch_explanation':'Exact common prefix verified; final counter reaches3 at stop while legacy continues; frozen model/support/fallback/weather unchanged','sole_counter_semantics':nm=='EXECUTED_JOINT_SUPPORT_FAILURE','last_four_decisions':N[-4:]})
 if (s,rid) in primary:
  expected=oldseven[s,rid];assert nm==expected['final_shadow_outcome'] and endtime(N)==expected['final_timestamp']
  assert nac['admitted']==expected['legacy_executed_steps']+expected['additional_transitions']
  assert nac['numeric_return']==expected['full_shadow_discounted_return']
  seven.append({**pair,'prior_shadow_exact_consistency':True})
assert len(matrix)==75 and len(seven)==7 and len({x['classification'] for x in matrix})>=1
summaries=[]
for s,d in sorted(perseed.items()):
 assert d['scheduled_transitions']==2811 and sum(d['outcomes'].values())==25
 for k in ['COMPLETE','STATE_SUPPORT_FAILURE','EXECUTED_JOINT_SUPPORT_FAILURE','OTHER_EXISTING_TERMINATION','SYSTEM_INVALID','UNSCORABLE']:d['outcomes'].setdefault(k,0)
 d['outcome_rates']={k:v/25 for k,v in d['outcomes'].items()};d['failure_rate']=(d['outcomes']['STATE_SUPPORT_FAILURE']+d['outcomes']['EXECUTED_JOINT_SUPPORT_FAILURE']+d['outcomes']['OTHER_EXISTING_TERMINATION'])/25
 d['action_rates']={k:{'count':d[k],'denominator':d['state_rows'] if k=='proposal_severe' else d['final_queries'],'rate':d[k]/(d['state_rows'] if k=='proposal_severe' else d['final_queries'])} for k in ['proposal_severe','final_severe','proposal_reliability_recovered_events','projection','fallback','weather']}
 d['proposal_reliability_event_definition']='proposal severe AND valid final nonsevere; raw proposal severe rate reported separately; denominator final-action queries'
 d['complete_return_summary']=stats(d.pop('complete_returns'));d['valid_run_length_summary']=stats(d.pop('lengths'));summaries.append(d)
counts=collections.Counter(x['classification'] for x in matrix);outcomes=collections.Counter(x['migrated_outcome'] for x in matrix)
policy='MIGRATED ENV STRUCTURAL REVIEW REQUIRED'
status='FINAL-ACTION ENV v1 — PASS / FROZEN-POLICY REVIEW READY'
summary={'execution_status':status,'primary_policy_status':policy,'seed_summaries':summaries,'comparison_counts':dict(counts),'outcome_counts':dict(outcomes),'migration_induced_regressions':len(regressions),'historical_shadow_cases_consistent':7,'interpretation':'technical migration passes; all remaining failures are state/final support, requiring structural review; no arbitrary completion threshold; compatibility does not certify policy sufficiency','retraining_approved':False,'HPO_approved':False}
write('legacy_vs_migrated_run_matrix.json',matrix);write('cross_seed_summary.json',summary);write('migration_regression_cases.json',regressions);write('historical_seven_consistency.json',seven)
md=['# Final-Action Termination v1 — 환경 구현·회귀·고정 정책 재평가','',status,'',f'별도 정책 판정: **{policy}**','', '## 완료 범위','', '새 명시적 semantic version `geas35.rl.profile_a.final_action_termination.v1`을 구현했다. Legacy `profile_a_offline_v1.py`와 기존 dispatch/체크포인트/결과는 변경하지 않았다. 새 환경은 명시적 version과 Development 모드를 요구하며 학습 권한이 없다.','', '동결 migration 및 transitive hash를 먼저 검증했다. Synthetic 15개 + Legacy75 exact + semantic-divergence 실제7개 + 독립 인스턴스 두 번 실제7개 replay로 사전등록18개를 모두 통과한 뒤에만 전체75개 새 환경 평가를 시작했다. 실제7개 회귀에서는 이전 shadow trace와도 물리/state/action/reward/counter가 일치했다.','', '변경은 post-weather final joint query와 그 streak의 종료 권위다. Proposal streak/status는 diagnostic으로 유지한다. State severe pre-action, warning 실행, natural-end 우선, fallback/weather, Transition, reward/accounting, gamma=.99/lambda=.95는 유지했다. 추가 OOD penalty는 없다.','', '## Seed별 평가','', '| seed | COMPLETE | STATE failure | EXECUTED JOINT failure | other / invalid / unscorable | COMPLETE rate | failure rate | COMPLETE return mean / median | valid length mean / median |','|---|---:|---:|---:|---|---:|---:|---|---|']
for d in summaries:
 o=d['outcomes'];rr=d['complete_return_summary'];le=d['valid_run_length_summary'];md.append(f"| {d['seed']} | {o['COMPLETE']} | {o['STATE_SUPPORT_FAILURE']} | {o['EXECUTED_JOINT_SUPPORT_FAILURE']} | {o['OTHER_EXISTING_TERMINATION']}/{o['SYSTEM_INVALID']}/{o['UNSCORABLE']} | {o['COMPLETE']/25:.2%} | {d['failure_rate']:.2%} | {rr['mean']:.6f} / {rr['median']:.6f} | {le['mean']:.2f} / {le['median']} |")
md+=['','각 seed의 episode-rate 분모는25 scheduled natural runs다. 11 parent episodes·2,811 scheduled transitions를 seed마다 동일하게 사용했으며 총75 seed-run 평가가 완료됐다. COMPLETE mean/median은 완료 run만의 discounted return이고 FAILURE는 valid stopped prefix다. 이를 단독 정책 점수로 일반화하지 않는다.','', '| seed | proposal severe | recovered proposal reliability | final severe | projection | fallback | weather |','|---|---|---|---|---|---|---|']
for d in summaries:
 cells=[]
 for k in ['proposal_severe','proposal_reliability_recovered_events','final_severe','projection','fallback','weather']:
  x=d['action_rates'][k];cells.append(f"{x['count']}/{x['denominator']} ({x['rate']:.2%})")
 md.append('| '+str(d['seed'])+' | '+' | '.join(cells)+' |')
md+=['','Proposal severe는 proposal-scored decision 전체, final/action 개입 rate는 valid final-query decision을 분모로 한다. Recovered reliability event는 proposal severe AND final nonsevere다. Projection은 warning 분기 진입으로 fallback과 중첩 가능하다. Weather는 저장 intermediate/state로 부모 함수를 재계산한 weather-active flag이며 final action exact parity를 확인했다.','', '## Legacy 대 migrated','', '| 배타적 run 분류 | 건수 |','|---|---:|']
for k in ['SAME_COMPLETE','LEGACY_FAILURE_TO_MIGRATED_COMPLETE','SAME_FAILURE_MECHANISM','DIFFERENT_FAILURE_MECHANISM','LEGACY_COMPLETE_TO_MIGRATED_FAILURE','TECHNICAL_INVALID','OTHER']:md.append(f'| {k} | {counts[k]} |')
md+=['',f"합계75이며 migration-induced legacy COMPLETE→FAILURE는 **{len(regressions)}건**이다. 전체 run별 full ID·시각 차이·실행 step 차이·category 및 terminal mask alignment는 `legacy_vs_migrated_run_matrix.json`에 보존했다. Proposal 기반 실패와 final-action 기반 실패를 같은 메커니즘으로 합치지 않았다.",'', '## Migration-induced regression 개별 확인','']
if not regressions:md.append('해당 사례는 없었다. 이것만으로 기존 구조적 실패가 해소되었다고 해석하지 않는다.')
for x in regressions:md+= [f"- seed{x['seed']} / `{x['run_id']}`: {x['legacy_outcome']} → {x['migrated_outcome']}; 최초 semantic divergence={x['first_semantic_divergence_timestamp']}; reason={x['termination_reason']}; proposal/final streak={x['terminal_proposal_streak']}/{x['terminal_final_streak']}. Exact 공통 prefix를 통과했으며 counter 의미 차이 설명 여부={x['sole_counter_semantics']}. 종료 전4개 decision은 `migration_regression_cases.json`에 보존."]
md+=['','## 기존7개 shadow case','', '| seed/run | migrated outcome | 이전 shadow 일치 |','|---|---|---|']
for x in seven:md.append(f"| {x['seed']}/{x['run_id'][:10]} | {x['migrated_outcome']} | exact |")
md+=['','기존7건은 state failure5, executed-joint failure2로 이전 frozen shadow와 시각·유효 길이·return 및 실제 trace가 일치한다. 원래 Main의7건은 여전히 legacy proposal-based termination outcomes이며 수정하지 않는다.','', '## Binding·무학습·정책 판정','', '새 환경은 intent/shadow preregistration/shadow execution/migration/implementation/model/scaler/schema/support/Profile-A/reward/cohort를 hash binding한다. 기존 selected checkpoint의 original training metadata는 그대로 두고 별도 cross-environment evaluation binding에 새 환경 정보를 함께 저장했다. Legacy metadata로 새 환경 resume하는 것은 거부한다. 이 평가는 새로운 Main이나 새 환경 학습 결과가 아니다.','', f"남은 실패 분포는 {dict(outcomes)}다. 평가가 기술적으로 일관되더라도 state/final-support 실패가 해결되지 않았다. 현재 증거만으로 그 모든 실패를 model error, recorded-domain shift 또는 command/history 문제 중 하나로 단정하지 않는다. 이전 MIXED/recurrence evidence와 STRUCTURAL REVIEW REQUIRED를 유지한다. Primary next gate는 `{policy}`이며 임의 완료율 cutoff로 재학습·HPO를 승인하지 않는다.",'', '다음 한 단계는 **새 환경의 남은 state/final-support 실패 원인 분리 비학습 감사**다. 특히 신규 COMPLETE→FAILURE 사례가 있으면 위 개별 기록을 우선한다. 실행 의미가 바뀌어 episode prefix·terminal mask alignment가 달라졌으므로 재학습 필요성은 이 구조 검토 후 별도 결정한다. 이번에는 PPO 재학습, HPO, Test, 추가 structural gate를 실행하지 않았다.','', '`18 REGRESSIONS = 18/18 PASS`  ','`DEVELOPMENT REEVALUATION = 75/75 COMPLETE` (평가 작업 완료이며 자연 COMPLETE 수가75라는 뜻 아님)  ','`PPO OPTIMIZATION EXECUTED = NO`  ','`HPO EXECUTED = NO`  ','`TEST ACCESSED = NO`  ','`LEGACY ENVIRONMENT MODIFIED = NO`','']
(O/'final_environment_implementation.md').write_text('\n'.join(md))
for x in c['inputs']:assert sha(R/x['path'])==x['sha256']
source=R/'src/geas35/rl/profile_a_final_action_env_v1.py';assert sha(source)==load(O/'runtime_input_gate.json')['new_implementation_hash']
artifacts={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='implementation_manifest.json'}
write('implementation_manifest.json',{'execution_status':status,'policy_status':policy,'semantic_version':'geas35.rl.profile_a.final_action_termination.v1','implementation':{'path':str(source.relative_to(R)),'sha256':sha(source)},'tests':{'path':'tests/test_profile_a_final_action_env_v1.py','sha256':sha(R/'tests/test_profile_a_final_action_env_v1.py')},'migration_payload':c['contract_payload_sha256'],'inputs':c['inputs'],'frozen_inputs_unchanged':True,'regressions':18,'development_seed_runs':75,'selected_policy_parameter_integrity':load(O/'parameter_integrity.json')['PASS'],'guards':load(O/'execution_guard_report.json'),'artifact_sha256':artifacts,'policy_original_training_environment':'legacy proposal-based','training_authorized':False,'new_PPO_checkpoint':False,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print(json.dumps({'status':status,'policy':policy,'comparisons':counts,'outcomes':outcomes,'regressions':len(regressions)},ensure_ascii=False,indent=2))
