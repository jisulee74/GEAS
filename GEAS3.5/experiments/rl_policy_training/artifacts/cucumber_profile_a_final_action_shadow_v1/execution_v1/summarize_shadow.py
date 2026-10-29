"""Aggregate stored traces only. No policy/model inference."""
from pathlib import Path
import json,datetime,hashlib,statistics,collections
from geas35.rl.mdp_v1 import project_mdp_v1_action_constraints,MdpV1Config
O=Path(__file__).resolve().parent;P=O.parent;R=P.parents[3]
def load(p):return json.loads(Path(p).read_text())
def write(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=load(P/'final_action_shadow_contract.json');assert load(O/'execution_gate.json')['PASS'] and not (O/'technical_failure.json').exists()
T=c['thresholds'];print('Thresholds',T)
# frozen threshold record is scalar mapping.
act=c['action_order']
def stat(v):return {'count':len(v),'min':min(v) if v else None,'median':statistics.median(v) if v else None,'mean':statistics.mean(v) if v else None,'max':max(v) if v else None}
def level(d,kind):return None if d is None else 'severe' if d>T[kind+'_severe'] else 'warning' if d>T[kind+'_warning'] else 'in_support'
def summarize(rows):
 legal=[r for r in rows if r['info'].get('final_action') is not None];valid=[r for r in rows if r['reward'] is not None]
 counts={'projection':sum(r['info']['joint_distance']>T['joint_warning'] for r in legal),'fallback':sum(r['info']['fallback_action'] is not None for r in legal),'weather':sum(r['weather_flags']['weather_constraint_applied'] for r in legal)}
 return {'decision_count':len(rows),'final_action_decisions':len(legal),'valid_executed_transitions':len(valid),'counts':counts,'rates':{k:v/len(legal) if legal else None for k,v in counts.items()},'max_proposal_streak':max([r['proposal_counter_after'] for r in rows],default=0),'max_final_streak':max([r['final_counter_last_valid'] for r in rows],default=0),'max_state_distance':max([r['info']['state_distance'] for r in rows],default=None),'max_proposal_joint_distance':max([r['info']['joint_distance'] for r in rows],default=None),'max_final_joint_distance':max([r['final_query']['joint_distance'] for r in legal],default=None),'undiscounted_reward_sum':sum(r['reward'] for r in valid),'quality':{label:stat([r['info']['quality'][field] for r in valid]) for label,field in [('R_quality','quality_reward'),('Q_T','temperature_quality'),('Q_RH','rh_quality')]}}
cases=[];pairchecks=[]
for case in c['primary_population']:
 branches={}
 for branch in ['L','S']:
  rows=[json.loads(l) for l in (O/f"{branch}_{case['case_id']}.jsonl").open()]
  with (O/f"{branch}_{case['case_id']}_annotated.jsonl").open('x') as f:
   for r in rows:
    i=r['info'];r['state_status']=level(i['state_distance'],'state');r['proposal_status']=level(i['joint_distance'],'joint');r['final_status']=level(r['final_query']['joint_distance'] if r['final_query'] else None,'joint')
    r['weather_flags']=None
    if i.get('final_action') is not None:
     inter=i['fallback_action'] if i['fallback_action'] is not None else i['projected_action']
     w,flags=project_mdp_v1_action_constraints(dict(zip(act,inter)),r['current_state'],MdpV1Config())
     assert [w[k] for k in act]==i['final_action'];r['weather_flags']=flags
    r['weather_flag_provenance']='pure frozen weather function on stored intermediate/state; exact stored final parity; no rollout'
    r['region']='post_stop' if r['step']>case['legacy_stop_step_zero_based'] else 'legacy_stop_transition' if r['step']==case['legacy_stop_step_zero_based'] else 'shared_prefix'
    f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
  branches[branch]=rows
 L,S=branches['L'],branches['S'];stop=case['legacy_stop_step_zero_based'];assert len(L)==stop+1
 for i in range(len(L)):
  a=json.loads(json.dumps(L[i]));b=json.loads(json.dumps(S[i]));a.pop('branch');b.pop('branch')
  if i==stop:a['info']['category']=None
  assert a==b,(case['case_id'],i)
 pairchecks.append({'case_id':case['case_id'],'exact_shared_decisions':len(L),'includes_stop_transition':True,'PASS':True})
 end=S[-1];info=end['info'];account=load(O/f"S_{case['case_id']}_accounting.json")
 if info['category']=='COMPLETE':outcome='COMPLETE'
 elif info.get('reason')=='severe_state_ood':outcome='STATE_SUPPORT_FAILURE'
 elif info['category']=='FAILURE' and end['final_counter_last_valid']>=3:outcome='EXECUTED_JOINT_SUPPORT_FAILURE'
 elif info['category'] in ['SYSTEM_INVALID','UNSCORABLE']:outcome=info['category']
 else:outcome='OTHER_EXISTING_TERMINATION'
 post=[r for r in S if r['step']>stop];valid=[r for r in post if r['reward'] is not None]
 recovered=any(r['state_status']!='severe' and r['final_status']!='severe' for r in valid)
 if post:assert post[0]['decision_timestamp']==case['legacy_stop_outcome_timestamp']
 elapsed=(datetime.datetime.fromisoformat(end['outcome_timestamp'] or end['decision_timestamp'])-datetime.datetime.fromisoformat(case['legacy_stop_outcome_timestamp'])).total_seconds()/60
 # pre-action state failure uses its decision; joint failure uses executed outcome.
 early=outcome in ['STATE_SUPPORT_FAILURE','EXECUTED_JOINT_SUPPORT_FAILURE'] and elapsed<=15
 full=summarize(S);ps=summarize(post)
 discounted=sum(.99**k*r['reward'] for k,r in enumerate([r for r in S if r['reward'] is not None]));assert discounted==account['numeric_return']
 x={**case,'legacy_terminal_reason':'repeated_joint_severe_termination','shadow_continuation_allowed':bool(post and post[0]['state_status']!='severe'),'continuation_start_timestamp':post[0]['decision_timestamp'] if post else None,'continued_at_least_one_step':bool(valid),'RECOVERY_SUPPORTED_CONTINUATION':recovered,'additional_transitions':len(valid),'additional_simulated_minutes':5*len(valid),'final_shadow_outcome':outcome,'final_decision_timestamp':end['decision_timestamp'],'final_outcome_timestamp':end['outcome_timestamp'],'final_timestamp':end['outcome_timestamp'] or end['decision_timestamp'],'termination_minutes_since_legacy_outcome':elapsed,'early_support_failure_within_15_minutes':early,'full':full,'post_stop':ps,'full_shadow_discounted_return':discounted,'legacy_discounted_return':load(O/f"L_{case['case_id']}_accounting.json")['numeric_return'],'legacy_post_stop_return':None,'outcome_is_counterfactual':True}
 cases.append(x)
 write(f"context_{case['case_id']}.json",[r for r in S if r['step']>=max(0,stop-3)])
counts=collections.Counter(x['final_shadow_outcome'] for x in cases);early=sum(x['early_support_failure_within_15_minutes'] for x in cases)
verdict='FINAL-ACTION SHADOW — STRUCTURAL FAILURE PERSISTS' if early>=4 else 'FINAL-ACTION SHADOW — MIXED' if len(counts)>1 else 'FINAL-ACTION SHADOW — CONTINUATION COHERENT'
agg={'verdict':verdict,'cases':7,'continued':sum(x['continued_at_least_one_step'] for x in cases),'recovery_supported':sum(x['RECOVERY_SUPPORTED_CONTINUATION'] for x in cases),'outcome_counts':dict(counts),'early_support_failures':early,'additional_simulated_minutes':stat([x['additional_simulated_minutes'] for x in cases]),'technical_invalid':0,'decision_rule':'BLOCKED first; >=4 early support failures; heterogeneous outcomes MIXED; otherwise COHERENT','no_generalization':True}
write('shadow_case_summary.json',cases);write('aggregate_shadow_summary.json',agg);write('shadow_parity_validation.json',{'PASS':True,'cases':pairchecks,'weather_flag_recomputation_exact':True,'no_extra_inference':True})
md=['# Final-Action Shadow Continuation 실행 결과','',verdict,'','정확히 사전등록된 7개 seed-run에 한해 기존 선택 정책을 그대로 사용한 counterfactual shadow 진단이다. 역사적 Main 결과와 정책 순위는 변경하지 않았다.','', '## 실행 gate','', '55개 동결 입력 hash와 사전등록 payload를 확인했다. Canonical GEAS3.5/.venv(Python 3.12.3)에서 수행했다. Legacy L 7건 모두 current state, info/action/support/prediction/reward/target binding, next56, proposal counter, 종료 시각 및 최종 accounting이 저장 결과와 exact 일치했다. 모든 L 통과 후 S를 실행했다.','', 'S는 각 run을 독립 reset/prehistory에서 시작했다. Legacy stop transition까지 L/S의 proposal/logprob, support, action, Transition62 입력, next4/56, history/carry, reward, target binding과 두 counter가 exact 일치했다. 그 stop transition의 category만 다르며 공통 구간에 포함했다. S는 추가 step에서 자체 정책·모델·history만 사용했다. 미래 실내값 주입·teacher forcing·기록 action 재사용은 없었다.','', '구현은 새 실행 디렉토리에서 frozen `_step`을 복제하여 pure final-support observer를 넣고 S의 종료 counter 참조만 `self.final_counter`로 바꿨다. 원본 executable은 수정하지 않았다. Final severe query는 weather cap 이후, prediction 이전이며 current state severe 종료가 우선이다. Natural end가 반복 counter보다 우선한다.','', '## 7건 결과','', '| seed/run 앞10자리 | 추가 step / 분 | 최종 S outcome | 최종 시각 | recovery-supported |','|---|---:|---|---|---|']
for x in cases:md.append(f"| {x['seed']}/{x['run_id'][:10]} | {x['additional_transitions']} / {x['additional_simulated_minutes']} | {x['final_shadow_outcome']} | {x['final_timestamp']} | {x['RECOVERY_SUPPORTED_CONTINUATION']} |")
md+=['',f"추가1step 이상 진행 {agg['continued']}/7, recovery-supported {agg['recovery_supported']}/7. 최종 분포: {dict(counts)}. 기술 invalid는0이다. 추가 분 median/min/max = {agg['additional_simulated_minutes']['median']} / {agg['additional_simulated_minutes']['min']} / {agg['additional_simulated_minutes']['max']}.",f"첫3개 예정 continuation decision/15분 이내 support 종료는 {early}/7이다. 사전등록 precedence를 그대로 적용하여 `{verdict}`로 판정했다. 별도 completion-rate threshold는 사용하지 않았다.",'', '추가 step은 legacy stop outcome 이후 실제 유효 transition만 센다. State severe pre-action 종료에는 가짜 transition/reward를 넣지 않는다. `RECOVERY_SUPPORTED_CONTINUATION`은 추가 유효 step의 current와 final joint가 nonsevere였다는 별도 flag이며 COMPLETE가 아니다.','', '## 거리·개입·reward','', '| seed/run | proposal/final streak 최대 | state / proposal joint / final joint 최대 | projection/fallback/weather (전체 final-action decision 분모) | post-stop reward합 | full discounted return |','|---|---|---|---|---:|---:|']
for x in cases:
 f=x['full'];co=f['counts'];md.append(f"| {x['seed']}/{x['run_id'][:10]} | {f['max_proposal_streak']}/{f['max_final_streak']} | {f['max_state_distance']:.6f} / {f['max_proposal_joint_distance']:.6f} / {f['max_final_joint_distance']:.6f} | {co['projection']}/{co['fallback']}/{co['weather']} (n={f['final_action_decisions']}) | {x['post_stop']['undiscounted_reward_sum']:.6f} | {x['full_shadow_discounted_return']:.6f} |")
md+=['','Projection count는 frozen warning 분기 진입 수이며 fallback과 중첩될 수 있다. Weather는 부모의 실제 weather-active flag 기준이다. 저장 intermediate/state로 순수 weather 함수를 재계산하고 stored final과 exact 일치함을 확인했다(추가 rollout/inference 없음). 전체와 post-stop rate·분모 및 R_quality/Q_T/Q_RH min/median/mean/max/count는 `shadow_case_summary.json`에 별도 보존했다.','', 'Full return은 reset부터 gamma=.99, post-stop 합은 undiscounted다. 유효 FAILURE return은 멈춘 prefix의 값이다. L에는 post-stop outcome/reward가 없으므로 이를0으로 채우거나 S와의 관측 개선량으로 비교하지 않는다.','', '## 보존·재현성·해석','', '원래 7건은 legacy proposal-based termination outcomes다. 새 S outcome은 별도의 counterfactual 증거로만 읽는다. 정책이 역사적으로 완주했거나 final-action 의미가 경험적으로 우월하거나 실제 온실 배포가 안전해졌다는 뜻이 아니다. MIXED 원인 진단과 STRUCTURAL REVIEW REQUIRED, 별도 online safety blocker를 유지한다.','', 'torch.no_grad 및 optimizer/gradient/checkpoint-write 차단, 실제 tensor digest 실행 전후 동일, 원본 55개 hash 재검증을 수행했다. 정책 evaluation은 Train Development 7건뿐이다. 기존 bound artifact 무결성 확인을 위한 hash 검증과 Validation 정책 평가는 구분되며 후자는 수행하지 않았다. Test/P0/live는 접근하지 않았다.','', '모든 원시 L/S trace, annotation, 주변 context, exact parity 및 parameter integrity를 실행 namespace에 보존했다. 동일 current state로 final query를 수행한 결과와 명령/history를 함께 남겼다.','', '## 다음 한 단계','', '**Shadow 결과에 대한 실행 환경 migration 여부 검토**만 다음 gate로 권고한다. 이 검토에서 비학습 회귀·평가 범위와 남은 state-domain 실패를 명시해야 한다. 이번 실행은 authoritative 환경 교체, policy 재평가/재학습, HPO, Main 재생성을 승인하지 않는다.','', '`LEGACY PARITY = PASS`  ','`NEW SHADOW ROLLOUT EXECUTED = YES`  ','`PPO OPTIMIZATION EXECUTED = NO`  ','`HPO EXECUTED = NO`  ','`TEST ACCESSED = NO`  ','`AUTHORITATIVE ENVIRONMENT MODIFIED = NO`','']
(O/'final_action_shadow_execution.md').write_text('\n'.join(md))
for r in c['authoritative_inputs']:assert sha(R/r['path'])==r['sha256']
files={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='execution_manifest.json'}
write('execution_manifest.json',{'verdict':verdict,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preregistration_payload_sha256':c['contract_payload_sha256'],'environment':c['environment'],'input_hashes':c['authoritative_inputs'],'all_input_hashes_unchanged':True,'legacy_parity':'PASS','shadow_parity':'PASS','parameter_hash_integrity':load(O/'parameter_hash_integrity.json'),'new_shadow_rollout_executed':True,'PPO_optimization_executed':False,'HPO_executed':False,'Test_accessed':False,'authoritative_environment_modified':False,'artifact_hashes':files,'aggregate':agg})
print(json.dumps(agg,ensure_ascii=False,indent=2))
