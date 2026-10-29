"""Report existing trace-only entry audit, no environment execution."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'experiments/rl_policy_training/artifacts/step17_entry_audit_hlv15_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(f):
 f=f.copy()
 for c in f:
  if pd.api.types.is_float_dtype(f[c]):f[c]=f[c].map(lambda x:f'{x:.6g}')
 return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(map(str,r))+' |' for r in f.itertuples(index=False,name=None)])
def csv(name):return pd.read_csv(O/(name+'.csv'))
summary=read(O/'analysis_summary.json');checks=read(O/'checks.json');assert all(v=='PASS' for k,v in checks.items() if k!='new_rollout_or_training')
state=pd.read_parquet(O/'reward_trigger_states.parquet');components=csv('reward_component_activity');sat=csv('temperature_saturation');sv=csv('survivability');od=csv('ood_summary');dyn=csv('reward_dynamics');rec=csv('reward_accounting');actions=csv('action_intervention');trigger=csv('reward_trigger_checks');hv=csv('heat_vent_trigger_details');lag=csv('action_reward_lag_summary')
contexts=state.groupby(['crop','obs_is_daytime','obs_current_target_temp_c','target_min','target_max']).agg(rows=('temp','size'),mean_temperature=('obs_indoor_temp_c','mean'),temperature_error_p50=('temperature_error','median'),saturated_rate=('temp_saturated','mean')).reset_index();contexts.to_csv(O/'temperature_target_context.csv',index=False)
reason={'rh':'A: observed RH never >90%; lower RH has no penalty in current contract. C: humidity objective is upper-limit only.', 'vpd':'A: observed VPD never <0.3kPa. C: vpd_max_kpa=None; upper VPD objective not represented.', 'cond':'A: observed T−dewpoint never <0.8°C. Current contract uses indoor air margin, not measured plant surface condensation.', 'hv':'A: no observed heat-on row exceeds Qventloss>0.25×Qheat; see heat-on/off split. B: heating gate and positive-excess threshold define activation.', 'temp':'active but dominating/saturated: error beyond target band ≥3°C gives fixed1/6; target±band is carried from next observation, not a z-score.', 'act':'active, limited informative signal: command reversals only; no direct heat/cool/fan energy-use cost.'}
verdicts=[]
for c in ['melon','cucumber','strawberry']:
 for t in ['temp','rh','vpd','cond','hv','act']:
  z=trigger.loc[(trigger.crop==c)&(trigger.term==t)].iloc[0]
  verdicts.append({'crop':c,'term':t,'implementation':'PASS','design_status':'active but dominating/saturated' if t=='temp' else ('active and informative (reversal only)' if t=='act' else 'effectively inactive in observed replay'),'trigger_rows':z.trigger_rows,'nonzero_rows':z.nonzero_rows,'cause':reason[t]})
 for t in ['warning','terminal']:
  z=components.loc[(components.crop==c)&(components.term==t)].iloc[0]
  verdicts.append({'crop':c,'term':t,'implementation':'PASS (Step16 accounting and trace)','design_status':'active; support-event signal, not climate trade-off','trigger_rows':z.nonzero_rows,'nonzero_rows':z.nonzero_rows,'cause':'fixed joint-distance warning adjustment' if t=='warning' else 'negative remaining scheduled steps at severe termination; run length dependent'})
terms=pd.DataFrame(verdicts);terms.to_csv(O/'reward_term_verdicts.csv',index=False)
lines=[f"# {summary['verdict']}",
'\n**환경은 실행 가능하지만, 현재 고정 보상으로 PPO를 시작하는 것은 보류한다.** Step16 executability PASS를 유지하며, 이번 감사는 Step17을 실행하지 않았다. 기존 dataset/scaler/model/support/threshold/reward와 Step16 trace를 변경하지 않았다.',
'\n| 판단 영역 | 판정 |\n|---|---|\n| OOD/environment | 기존 recorded replay에서 독립적인 진입 blocker를 입증하지 못함; 제한 유지 |\n| Reward implementation | PASS — 방문한 유한 상태에서 저장값·독립 수식·순수 보상 함수 일치 |\n| Reward informativeness | BLOCKED — 네 핵심 항목 비활성, 온도 포화 및 terminal 집중, 직접 에너지 비용 부재 |\n| Step17 entry | BLOCKED — REWARD |\n| Online deployment | BLOCKED — actuator-safe fallback 미검증 |',
'\n다음 최소 후속 작업 하나: **reward contract redesign preregistration**. 이번 작업에서 weight/threshold/normalization/term을 수정하거나 변경안을 구현하지 않았다.',
'\n## 범위·판정 원칙',
'기존 Step16의 저장 trace만 사용했다. 보상 함수만 재계산했으며 환경/Transition 모델을 instantiate하거나 rollout을 재실행하지 않았다. [사전 protocol](protocol.json)에 사용자 제공 판단 기준·기여도 분모·lag 정의를 기록한 뒤 집계했다. 별도 숫자 admission cutoff를 맞추지 않았다. 온도 saturation은 기존 error/3 clip 수식의 경계이며 tuning이 아니다. near-unique는 소수8자리 반올림한 설명용 집계다.',
'\n## OOD 종료 분류',table(od),
'\n분모 all_runs_rate는 작물별 전체 scheduled run, same_kind_ood_rate는 해당 state/joint 종료 수다. A는 같은 종료 timestamp의 recorded state 또는 state+recorded action도 severe인 경우다. B는 A가 아니면서 이전 safety/projection action 변경이 있었던 경로다. **B는 action-induced 인과 확정이 아니라 action이 달라진 경로**라는 뜻이다. C는 방문한 기록 경로에 해당 severe가 없고 prior action 변경도 없는 경우다. earlier recorded severe가 있는 mixed 사례는 별도로 보존한다.',
'\n딸기 state severe의 rollout_only_no_prior_action_changes=0을 압축 trace에서 다시 검증했다. 이 관측은 이번 고정 recorded replay에만 해당한다. 과거 clean5 autoregressive limitation의 해결, 새로운 PPO action의 안정성, geas2_5 일반화를 입증하지 않는다. 기록 분포 이탈과 safety 변경 경로는 잔여 제한이며 현재 통과한 실행 가능성 guard를 다시 낮추거나 OOD를 해제하지 않는다.',
'\n## Trajectory 확보와 safety 개입',table(sv[['crop','runs','executed_runs','executed_rate','reach_15min_rate_all_runs','reach_30min_rate_all_runs','reach_60min_rate_all_runs','normal_rate','ood_rate','length_steps_p10','length_steps_p50','length_steps_p90']]),
'\n길이는5분 step 수이며 시간은×5분이다. reach rate 분모는 짧은 run까지 포함한 전체 run이다. 기존 guard의 “15분 또는 정상 short-run 끝” 정의는 그대로 유지되며 여기의 순수15분 비율과 구분한다. OOD로0 step 종료한 run도 포함했다. 충분한 PPO trajectory를 보장하는 실제 policy 실험이 아니라, 기록 replay에서 unexplained severe 때문에 구조적으로 모든 학습 경로가 막혔는지 확인한 결과다.',table(actions),
'\n변경 비율의 분모는 Transition이 실제 실행된 step이다. melon curtain/fan, strawberry vent/fan/cool은 아래 실제 action별 수치로 비교한다. local support projection·weather·fallback은 정책 제안과 실제 적용 action을 다르게 만들 수 있어 action-credit 해석에 제한이 있다. 이 비율을 미래 PPO policy modification rate로 동일시하지 않는다. 전역 action-response support와 local support 범위를 혼동하지 않는다.',
'\n## 보상 활성도와 기여도',table(components),
'\nsix_absolute_share는 clipping 이전 six penalty 절댓값 합에 대한 비율이다. ratio_to_abs_final_return은 abs(component sum)/abs(final return)의 회계상 비교치이며, 각 penalty의 독립적인 최종 기여도가 아니다. clipping/fallback 조정이 존재하므로 아래 전체 항등식과 함께 읽어야 한다. penalty 값은 양수이나 base에는 음수로 들어가며 warning은 차감·terminal은 음수다.',table(rec),
'\nbase − warning + clipping/fallback adjustment + terminal = final return을 검증했다. base는 six 합의 음수다. 핵심 penalty 통계는 executed row, warning/terminal/adjustment는 terminal-only 호출을 포함한 전체 호출을 분모로 한다. 원시 센서/상태를 대체하거나 missing 값을 채운 적 없다.',
'\n## RH/VPD/cond/HV가 0인 이유',
'모든 실행 step에서 여섯 branch가 호출되고 raw→normalized→weighted 계산이 저장 trace와 일치했다. 해당 입력은 모두 finite였다. 따라서 관측된0은 branch 누락이나 NaN이0으로 감춰진 사례가 아니다. finite하지 않은 값을0 처리하는 코드 경로는 존재하지만 이번 trace에는 해당하지 않았다.',
'\n| 항목 | 실제 trigger / bounded penalty | 해석 |\n|---|---|\n| RH | max(RH−90,0)/10을[0,1]clip | 90% 초과만 제재; 건조/낮은 RH trade-off 없음 |\n| VPD | max(0.3−VPD,0)/0.3을clip | low VPD만 제재; upper limit 미설정 |\n| condensation | max(0.8−(T−dewpoint),0)/0.8을clip | 현재 공기 온도 기준 결로 여유 |\n| HV | heat>0.5, Qheat>5, max(Qvent−0.25Qheat,0)/Qheat/0.5를clip | 난방 중 기준 초과 환기 손실만 제재; 일반 energy cost 아님 |\n| act | 직전 두 명령과 현재 명령의 방향 반전, continuous/binary 가중 평균 | 소비량·명령 크기 자체에 대한 energy penalty 아님 |',table(trigger),table(hv),table(csv('reward_input_ranges')),
'\n원인 분류: RH/VPD/cond는 실제 방문 상태가 trigger를 만족하지 않은 A가 직접 원인이다. 한쪽 한계만 보는 목표 정의(C)가 넓은 humidity/VPD trade-off를 표현하지 못한다는 별도 설계 제한이 있다. HV는 heat gating 및 손실 threshold(B) 안에서 실제 conflict trigger가0(A)인지 위 heat-on/off 집계로 구분한다. “0이라 필요 없다” 또는 모든 PPO 경로에서도0이라고 주장하지 않는다.',
'\n## 온도 포화',table(sat),table(contexts),
'\n온도 error는 max(lower−T,0)+max(T−upper,0), penalty=min(error/3,1)×1/6이다. 따라서 목표 band 밖3°C 이상은 더 멀어져도0.1666667로 동일하다. 이는 가중치 합산 오류가 아니라 기존 clipping 정의이며, 해당 영역에서는 온도 개선/악화를 구분하는 local reward 차이가0이다. target·T는 물리°C이고 trace의 next observation target 범위 및 저장 metadata를 대조했다. day/night 기본24/12°C와 band±1°C가 실제 상태에 얼마나 맞는지는 별도 목표 계약 문제이며 이번에 수정하지 않았다.',
'\n## Terminal cost와 reward dynamic range',table(csv('terminal_by_reason')),table(dyn),
'\nterminal은 남은 scheduled step을−1씩 비용으로 부과한다. 정상 종료는0이다. mean/median/max magnitude와 전체 terminal share를 종료 사유별로 분해했다. 같은 상태의 비용도 잔여 길이에 의존하므로 OOD를 피하는 event 신호가 dense climate feedback보다 훨씬 클 수 있다. 종료 호출/비종료 실행 step의 평균·중앙 절댓값 scale ratio는 reward_accounting.csv에 있다. variance는 모집단 분산(ddof=0)이다.',
'\n## Action 변화와 이후 보상 정렬',
'event timestamp t의 보상은 next observation s_(t+1)과 final e_t로 계산된 r_t다. lag h는 r_(t+h)−r_t이며 미래 구간은 사후 진단으로만 읽는다. event 두 가지를 분리했다: proposed→final 변경(projection)과 final e_t→직전 실행 e_(t−1) 변화(command_delta). 같은 run의 이후1/3/6 executed step에서 각 penalty의 endpoint signed/absolute 변화와 그 안에서 한 번이라도 달라진 여부를 집계했다. 종료로 충분한 step이 없으면 right-censored로 표시하고0으로 채우지 않았다. 동시 여러 action·weather·모델 경로가 바뀌므로 인과 효과나 actuator별 학습 가능성 증명으로 쓰지 않는다.',
'[전체 action별 lag 요약](action_reward_lag_summary.csv), [timestamp별 정렬](action_reward_lag_events.parquet). RH/VPD/cond/HV가 이 replay에서 모두0이므로 이들 항목을 통한 actuator 구별 신호도 관측되지 않는다. 온도·act의 nonzero 변화가 일부 있어도 다목적 reward의 적합성이 자동으로 충족되지는 않는다.']
# Compact presentation: all actions, each lag, temperature and action-stability only.
for c in ['melon','cucumber','strawberry']:
 lines.append(f'\n### {c} action→reward 변화 (projection event)\n'+table(lag.loc[(lag.crop==c)&(lag.event=='projection')&lag.term.isin(['temp','act']),['action','lag','term','event_rows','observed_lag_rows','right_censored','mean_abs_change','any_change_within_h_rows']]))
lines+=['\n## 구현과 설계 판정',table(terms),
'\nImplementation PASS는 방문한 finite 입력과 코드 수식에 대한 재현/독립 계산 검증이다. 물리적 제어 목적이나 모든 미방문 action의 적합성 승인이 아니다. 온도·terminal 지배, RH/VPD/cond/HV 비활성, energy 소비를 직접 표현하지 않는 act 정의가 함께 있어 사용자 제공 PPO entry 기준에 따라 REWARD blocker로 판정한다. 환경 OOD 제한을 reward 설계 문제의 원인이라고 섞지 않는다.',
'\n## 유지할 제한과 다음 한 가지 작업',
'**reward contract redesign preregistration**만 추천한다. 사전에 온도 목표/허용범위와 포화, humidity/VPD/condensation 목표, energy와 reversal의 구분, warning/terminal의 목적·scale, 평가 모집단 및 승인 기준을 명시하는 설계 작업이다. 여기서 새 수치/weight를 선택하거나 실험을 실행하지 않았다.',
'\nHLV15 Phase0/Transition/Step14/15/16 PASS는 유지한다. Step17/PPO/Step17–19/Step19.5/HPO/Test는 미실행. domain-v2 C/E는 별도 unresolved로 유지한다. strawberry Train CO₂≥1500ppm22행, high-CO₂+previous high vent3행/current high vent2행,15/30/60분 CO₂ Persistence skill 음수의 제한은 그대로다. melon/strawberry legacy snapshot assumption, geas2_5 일반화와 과거 clean5 autoregressive limitation도 유지한다. online actuator-safe fallback **BLOCKED**는 별도다.',
'\n[판정 JSON](final_gate.json), [검증 결과](checks.json), [입력/분모 사전 고정](protocol.json).']
lines[2:2]=['\n핵심 수치: temperature penalty saturation 비율은 melon90.56%/cucumber79.78%/strawberry48.12%다. 딸기는 모든 dense reward가 상수인 것은 아니며 partial41.04%, zero10.84%도 존재한다. 그러나 four core terms는 세 작물 모두0이고, six penalty 중 온도 비중은98.09%/99.84%/99.45%다. 따라서 “신호가 전혀 없다”가 아니라 “현재 다목적 정책 목적을 충분히 구별하지 못한다”는 REWARD 판정이다.', '\nHV가0인 직접 근거: heat-on melon31행, cucumber0행, strawberry1590행이며, heat-on 중 Qventloss>0.25Qheat는 세 작물 모두0행이다. 난방이 꺼진 행의 환기 손실은 현재 conflict 계약상 penalty 대상이 아니다. terminal 중 state severe의 비중은 cucumber82.05%/strawberry70.14%이고, terminal이 최종 음의 return 절댓값에서 차지하는 비중은 각각78.39%/80.71%다.']
lines += ['\n## 코드 근거',f'[고정 config와 equal weights]({ROOT}/src/geas35/rl/mdp_v1.py:175), [action reversal]({ROOT}/src/geas35/rl/mdp_v1.py:575), [물리 HV]({ROOT}/src/geas35/rl/mdp_v1.py:649), [trigger와 clip]({ROOT}/src/geas35/rl/mdp_v1.py:730), [weighted sum]({ROOT}/src/geas35/rl/mdp_v1.py:1000), [OOD·warning·terminal 실행 계약]({ROOT}/src/geas35/rl/model_driven_env.py:340), [기록 target 갱신 whitelist]({ROOT}/src/geas35/models/transition/causal_state.py:93).']
(O/'final_decision_ko.md').write_text('\n'.join(lines)+'\n')
gate={**summary,'status':'BLOCKED','decision_category':'B','Step17_entry':'BLOCKED','blocker':'REWARD','Step16_executability':'PASS unchanged','reward_modified':False,'PPO_or_rollout_or_Test_executed':False,'domain_v2_C_E_executed':False,'original_artifacts_unchanged':True}
(O/'final_gate.json').write_text(json.dumps(gate,indent=2,ensure_ascii=False)+'\n')
(O/'report_code_manifest.json').write_text(json.dumps({str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},indent=2)+'\n')
(O/'artifact_integrity.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='artifact_integrity.json'},indent=2)+'\n')
print(summary['verdict'])
