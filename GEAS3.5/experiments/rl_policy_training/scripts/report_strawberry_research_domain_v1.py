"""Read-only verification and report of the already frozen domain candidate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import shadow_strawberry_research_domain_v1 as m
D=m.D
rule=m.check()
train=pd.read_parquet(m.P/'strawberry/train_physical.parquet',columns=m.META+rule['observation_columns']+rule['actions'])
val=pd.read_parquet(m.P/'strawberry/validation_physical.parquet',columns=m.META+rule['observation_columns']+rule['actions'])
# Exact deterministic reconstruction of Train evidence, without changing the frozen rule.
for group in ('state_cells','action_cells'):
 for name,spec in rule[group].items():
  assert m.occupancy(train,spec['columns'],rule['edges'])==spec['occupancy'],name
for col,edges in rule['edges'].items():
 assert np.array_equal(np.unique(np.quantile(train[col],[0,.1,.25,.5,.75,.9,1])),edges)
# Boundary and mutation tests check an unchanged contract, not tuned criteria.
probe=train.iloc[[0]].copy()
probe.loc[:,'obs_growth_stage_dat']=rule['ranges']['obs_growth_stage_dat'][1]+1
assert m.classify(probe,rule)[0][0]==2
assert m.binvalue([1,1.00001],[0.,1.]).tolist()==[2,3]
base=m.classify(train.iloc[:64],rule)
altered=train.iloc[:64].copy();altered['series_id']='unseen_probe';altered[rule['actions']]=.123456
mut=m.classify(altered,rule);assert np.array_equal(base[0],mut[0]) and base[1]==mut[1]
assert rule['observation_columns']==m.read(m.P/'schema.json')['observation_columns']
assert rule['observation_columns']==m.read(m.S/'state_action_support.json')['state_columns']
assert len(rule['observation_columns'])==55
side=pd.read_parquet(D/'Validation_shadow_sidecar.parquet');conditions=pd.read_parquet(D/'Validation_condition_evidence.parquet')
train_side=pd.read_parquet(D/'Train_domain_sidecar.parquet')
assert len(side)==len(val) and side.row_id.is_unique and np.array_equal(side.row_id,val.row_id)
for split,frame in [('Train',train),('Validation',val)]:
 assert np.isfinite(frame[rule['observation_columns']+rule['actions']].to_numpy(float)).all()
for field in ('row_domain','transition_domain','target_domain','action_domain'):
 assert side[field].isin(m.LEVEL).all()
# Detailed physical evidence, including clean starts and the retained reference-shortage cases.
clean=pd.read_csv(D/'clean5_start_domain.csv');cases=pd.read_csv(D/'recorded_severe39_domain.csv');ref=pd.read_csv(D/'reference_shortage16_domain.csv')
lookup=val.set_index('row_id');side_lookup=side.set_index('row_id')
case_sets=[('recorded39',cases,'row_id'),('reference16',ref,'row_id'),('clean5_start',clean,'start_row_id')]
items=[]
for tag,frame,key in case_sets:
 for _,case in frame.iterrows():
  rid=case[key];position=int(np.flatnonzero(val.row_id.eq(rid))[0]);state=lookup.loc[rid]
  for _,c in conditions.loc[conditions.position.eq(position)].iterrows():
   for col in rule['state_cells'][c.condition]['columns']:
    items.append({'set':tag,'episode_index':case.episode_index,'row_id':rid,'timestamp':state.reg_date,'series_id':state.series_id,'condition':c.condition,'condition_level':m.LEVEL[int(c.level)],'Train_cell_rows':int(c.Train_rows),'Train_cell_episodes':int(c.Train_episodes),'feature':col,'physical_value':float(state[col]),'Train_min':rule['ranges'][col][0],'Train_max':rule['ranges'][col][1],'row_domain':side_lookup.loc[rid,'row_domain']})
pd.DataFrame(items).to_csv(D/'case_physical_Train_evidence.csv',index=False)
# Train/source/series evidence, without categorical rules on series.
tr=train[m.META].merge(train_side[['row_id','domain']],on='row_id',validate='one_to_one')
tr.groupby(['source_confidence','series_id','domain']).size().rename('rows').reset_index().to_csv(D/'Train_source_series_domain.csv',index=False)
side.groupby(['source_confidence','series_id','row_domain']).size().rename('rows').reset_index().to_csv(D/'Validation_source_series_domain.csv',index=False)
# Per-source absence of coverage remains a limitation, not admission tuning.
coverage=[]
for split,f in [('Train',train),('Validation',val)]:
 for source,g in f.groupby('source_confidence'):
  coverage.append({'split':split,'source':source,'rows':len(g),'CO2_ge1500':int(g.obs_indoor_co2_ppm.ge(1500).sum()),'CO2_ge1500_prevvent_ge075':int((g.obs_indoor_co2_ppm.ge(1500)&g.obs_prev_vent_pct.ge(.75)).sum()),'CO2_ge1500_currentvent_ge075':int((g.obs_indoor_co2_ppm.ge(1500)&g.vent_pct.ge(.75)).sum())})
pd.DataFrame(coverage).to_csv(D/'source_high_CO2_coverage.csv',index=False)
runs=pd.read_csv(D/'Validation_Step16_outcome_shadow.csv')
for col in ['start_state_domain','whole_run_domain']:
 pd.crosstab(runs.termination_reason,runs[col]).reindex(columns=m.LEVEL,fill_value=0).to_csv(D/('Step16_outcomes_'+col+'.csv'))
summary=m.read(D/'shadow_summary.json')
verification={'frozen_Train_evidence_exact_rebuild':True,'quantile_edges_exact_rebuild':True,'max_edge_not_merged_with_exceedance':True,'outside_Train_DAT_is_OUT':True,'series_and_current_action_state_invariance':True,'identical_55schema_order':True,'all_eligible_rows_finite':True,'Validation_unique_complete_row_registry':True,'repeated_classification_identical':summary['deterministic_replay'],'official_bound_input_hashes_unchanged':True,'rule_hash_unchanged':m.read(D/'rule_lock.json')['sha256'],'Test_access':False,'model_fit':False,'official_mutation':False}
m.check();m.write(D/'verification.json',verification)
def table(frame):
 # No optional tabulate dependency.
 f=frame.copy().fillna('');cols=list(f.columns)
 def fmt(v):return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v).replace('|',' / ').replace('\n',' ')
 return '| '+' | '.join(map(str,cols))+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))
counts=[]
for name,key,total in [('current row','row_counts',summary['rows']),('transition (current+target+action)','transition_counts',summary['rows']),('run start state','run_start_counts',summary['runs']),('whole recorded run','run_counts',summary['runs'])]:
 counts.append({'모집단':name,'전체':total,**{level:f"{summary[key].get(level,0)} ({100*summary[key].get(level,0)/total:.2f}%)" for level in m.LEVEL}})
series=pd.read_csv(D/'Validation_series_domain.csv').pivot(index='series_id',columns='row_domain',values='rows').reset_index()
physical=pd.read_csv(D/'Validation_series_physical_reasons.csv').groupby('series',group_keys=False).head(6)
ref_in=ref.loc[ref.row_domain.eq('IN_DOMAIN'),['episode_index','row_id','series_id']]
clean_info=clean.merge(side[['row_id','state_reasons']],left_on='start_row_id',right_on='row_id').merge(val[['row_id','obs_growth_stage_dat','obs_indoor_temp_c','obs_prev_vent_pct']],on='row_id')
text=f'''# Strawberry Train-only research domain shadow v1

## 결론

**판정 B: Train-only domain 후보는 재현 가능하게 정의했지만, 전체 Validation 경로를 유지하는 범위가 좁고 기존 local OOD와도 일치하지 않아 공식 Step16 재검증 전 추가 coverage/support 설계가 필요하다.**

추천 후보는 이번에 사전 동결한 **저차원 Train coverage + 별도 action-pair admission** 하나다. 공식 적용 승인이나 안전성 승인이 아니다. 후보·bin·행 수 기준을 Validation 결과에 맞춰 바꾸지 않았다. strawberry Step16 및 전체 Step17은 **BLOCKED 유지**다.

전체 흐름은 B60/R6 Phase0 PASS → Transition 재선정 PASS → Step14/15 PASS → Step16 strawberry 실행률 39.60%로 BLOCKED → OOD 분리 → full Train/reference coverage 감사 → **현재 연구 domain 후보의 shadow 평가**다.

## 동결 및 증거 범위

- 기준 Train: eligible **44,097행**, 단일 `strawberry_geas2_0`, `legacy_command_snapshot_assumption`.
- 단일 후보 recipe를 먼저 저장하고 Train에서 bin/occupancy를 생성한 뒤 규칙 hash를 동결했다. `freeze`와 `shadow`를 별도 실행했다.
- frozen rule SHA256: `{summary['rule_sha256']}`.
- rule 입력은 Train physical/provenance/fit registry, 기존 scaler/55-schema/contract, 기존 support geometry 및 reference binding이다. 기존 scaler/support를 재생성하거나 재보정하지 않았다.
- 이번 rule 생성에는 Validation row, termination, reward, prediction metric을 읽지 않았다. 다만 이전 감사 결과와 사용자가 지정한 분석 축은 알려진 상태였다. **완전히 새로운 맹검 검증이라고 주장하지 않는다.**
- `rule_lock.json`, `recipe_before_Train_retention.json`, `frozen_domain_rule.json`에 규칙과 입력 hash를 남겼다. Validation 결과는 동결 후 별도 sidecar만 만들었다.

## 후보 계약: 물리적 안전이 아닌 연구 적용 근거

다음 조건들의 OUT 우선 / LOW 다음 순서로 분류한다.

| 수준 | 조건 |
| --- | --- |
| OUT_OF_DOMAIN | 55 observation 중 nonfinite, 감시하는 물리/context 축이 Train min/max 밖, DAT 30일 bin 미관측, 또는 지정 저차원 cell이 Train에 0행 |
| LOW_SUPPORT | cell/DAT bin은 관측됐지만 50행 미만 또는 2 episode 미만, 또는 Train과 다른 source confidence |
| IN_DOMAIN | 위 조건이 없고 모든 지정 cell에 50행 이상·2 episode 이상 근거 존재 |

행 수 50은 기존 support의 k=50과 수량상 일관성을 고려한 **후보 기준**이다. 50행이 같은 cell 안에 있다는 것이 기존 55차원 k50 거리 통과나 독립 표본 50개를 뜻하지 않는다. 2 episode는 단일 episode에만 의존하지 않기 위한 사전 고정 휴리스틱이며 통계적 신뢰도·안전성을 보증하지 않는다. Train 자기포함 occupancy이므로 Train retention도 일반화 성능이 아니다.

각 변수의 bin은 Train quantile `[0,.1,.25,.5,.75,.9,1]`의 중복 edge를 제거해 만든다. 실제 edge/범위는 frozen rule 및 `Train_axes_summary.csv`에 있다. 정확한 최댓값과 최댓값 초과는 별도 bin이다. 0과 최대값에 질량이 집중된 command의 bin도 이 정의를 따른다. 이 binning의 민감도를 이번 결과로 조정하지 않았다.

조건은 4묶음의 **14개 2~3차원 cell**이다. 55차원 조합을 암기하지 않지만, 14개 조건의 교집합이 강한 제한을 만든다는 한계는 있다.

| 묶음 | 관측 상태 조건 |
| --- | --- |
| 생육·weather | DAT×wind, DAT×DLI, wind×DLI, light×DLI |
| 생육·실내 | DAT×T, DAT×RH, DAT×CO₂ |
| 실내·이전 command | T×RH, RH×CO₂, T×previous vent, RH×previous vent, CO₂×previous vent |
| 환기 관련 | T×Tout×previous vent, vent-loss×previous vent |

`DAT×indoor`는 세 pair로 분해했다. 따라서 4차원 DAT/T/RH/CO₂ 전체 결합이나 모든 55변수 상호작용의 coverage를 보증하지 않는다. DLI·vent-loss 등은 기존 physical 정의 그대로이며 `obs_prev_vent_pct`의 저장 scale은 **0~1**이다(75%는 0.75). 기존 단위·scaler·weights는 변경하지 않았다.

**현재 action은 observation domain에 넣지 않는다.** 별도 action 적용 전 후보 검사로 6개 action 각각 `(이전 command, 현재 command)` cell을 같은 50행/2episode 기준으로 검사한다. series ID 역시 판정 입력이 아니다. 모든 current action과 series 라벨을 바꿔도 상태 판정과 사유가 같음을 검증했다.

보수적인 향후 사용안은 IN_DOMAIN 상태에서만 연구 admission을 검토하고 LOW/OUT은 별도 보류하는 것이다. 그 안에서도 **기존 Step15 local state/joint OOD, safety/projection은 그대로 적용**해야 한다. 이번에는 runtime을 바꾸지 않았다.

## Train-only DAT 및 유지 범위

Train 상태 IN_DOMAIN **43,452행(98.54%)**, LOW_SUPPORT **645행(1.46%)**, OUT 0행. 별도 action-pair는 IN **44,075행**, LOW **22행**, OUT 0행이다.

{table(pd.read_csv(D/'Train_DAT_coverage.csv'))}

DAT 1~176이 모두 매일 존재한다는 뜻은 아니다. 특히 180~209 bin은 **DAT 191의 97행·1episode만 존재**하여 LOW 근거다. 192 이상은 관측 min/max 밖이므로 OUT가 우선하며 210~239는 bin 자체 미관측이다. Bin 이내라도 joint cell에 없는 조건은 OUT가 된다. 빈 날짜를 interpolation하지 않았다.

## 동결 후 전체 Validation shadow

{table(pd.DataFrame(counts))}

transition은 현재 상태·기록 target 상태·현재/이전 action-pair 중 가장 제한적인 수준이다. **target 관측은 offline 기록 transition의 shadow 검사에만 사용**하며 online 현재 observation 판정에 미래 정보를 쓰지 않는다. whole-run은 기존 run 전체의 모든 transition이 만족해야 IN이다. start-state admission과 전체 기록 run의 결과는 서로 다르다. 실제 model rollout을 새로 실행하거나 잘린 run을 공식 episode로 만들지 않았다.

Row/transition IN 비율 약 32%/31%와 whole-run IN 1/72는 다른 분모다. 1개 run만 남는다는 값은 기존 run 전체를 보존할 때의 엄격한 shadow 지표이며 임의로 구간을 잘라 새 실행 가능성을 승인하지 않았다. run별 `longest_in_domain_sequence`도 별도 CSV에 보존했다.

### 기존 Step16 종료와 비교

전체 기록 run 수준:

{table(pd.read_csv(D/'Step16_outcomes_whole_run_domain.csv'))}

시작 상태 수준:

{table(pd.read_csv(D/'Step16_outcomes_start_state_domain.csv'))}

정상 종료 4개 중 2개도 전체 기록 경로 OUT이다. 이 규칙은 성공 run만 골라내는 완전 분류기가 아니다. 시작 상태 IN 23개 중 기존 정상 종료는 3개이고, 실패는 state severe 14개/joint severe 6개다. 기록 경로 shadow와 실제 free rollout 경로는 동일하지 않으므로 이를 새 후보의 실행 성공률로 해석하지 않는다.

### 기존 local support와 별도 역할

{table(pd.read_csv(D/'domain_vs_existing_support.csv'))}

IN_DOMAIN 중에도 frozen support severe **481/5,639행(8.53%)**이 있다. 저차원 적용 근거와 기존 55차원 local OOD가 다름을 보여준다. 반대로 OUT가 물리적으로 위험하다는 의미도 아니다. 기존 severe threshold **0.7523722704654124**를 그대로 유지하며 이 shadow에서 어떤 case도 PASS로 재승인하지 않았다.

## Series ID 대신 물리/context 조건

{table(series)}

새 series geas2_5에서도 **4,802/8,216행(58.45%)**이 IN이고, 기존 series geas2_0은 **837/9,464행(8.84%)**만 IN이다. 새 series라는 이유만으로 배제하지 않았다. geas2_5 LOW/OUT의 이유는 모두 사전 정의한 물리/context 조건으로 기록된다. 빈도는 중복 가능하다.

{table(physical)}

기존 series의 주된 제한은 late DAT이고, 새 series에서는 DAT×T, DAT×RH, T/Tout×previous vent, vent-loss 범위/조합 등이 나타난다. 두 split의 모든 source confidence는 동일한 legacy assumption이라 source 조건 자체로 탈락한 것은 아니다.

**Residual series effect는 미확인 limitation**이다. geas2_5가 Train에 없으며, 물리 축이 겹친 4,802행에도 series 특유의 미관측 운영·센서 조건이 있을 수 있다. 이 coverage audit만으로 series 효과가 없거나 모델이 일반화한다고 확인할 수 없다.

## 기존 39건, reference shortage 16건, clean 5건

| 대상 | IN_DOMAIN | LOW_SUPPORT | OUT_OF_DOMAIN |
| --- | --- | --- | --- |
| recorded-severe 39 | 7 | 13 | 19 |
| full Train radius coverage는 있으나 reference 부족 16 | 3 | 7 | 6 |
| clean free-only 5개 시작 상태 | 0 | 0 | 5 |

39건 전체와 16건 부분집합의 개별 사유·물리값·Train cell row/episode 수는 `case_physical_Train_evidence.csv`에 저장했다. 기존 39건 severe 판정은 모두 유지한다.

Reference shortage이면서 domain IN인 3건:

{table(ref_in)}

이 3건은 향후 reference sampling 검토 대상으로 분리할 근거가 있다. 그러나 나머지 full Train 부족/command·context 빈 영역까지 reference sampling만으로 해결할 수 있다는 뜻은 아니다. 이번에는 sampler를 변경하지 않았다.

Clean 5개 시작 상태:

{table(clean_info[['episode_index','series','obs_growth_stage_dat','obs_indoor_temp_c','obs_prev_vent_pct','start_state_domain']])}

24~27은 Train DAT 최대 191 초과와 관련 joint cell 미관측으로 OUT, 50은 DAT×T 및 T×previous vent/T×Tout×previous vent 미관측으로 OUT이다. 기존 local support에서는 정상 시작점이었지만 저차원 Train 근거는 다른 제한을 드러냈다. **5건을 shadow domain 밖으로 분류했다고 Transition one-step error/autoregressive accumulation이 해결된 것은 아니다.** 기존 teacher-normal/free-only severe 진단은 유지하며, IN 영역에서도 누적 안정성이 보장된다는 증거는 없다.

## High-CO₂ / command coverage의 분모

{table(pd.DataFrame(coverage))}

기존 high-CO₂/high-vent 제한은 current command 기준과 observation의 previous command 기준을 혼동하면 안 된다. 이 표는 둘을 별도 집계한다. 현재/이전 command 어느 정의에서도 고농도 학습 근거가 매우 희소하며, 이번 작업에서 이를 복구하지 않았다. Validation 수는 B60/R6 최종 eligible 모집단 기준이므로 stale QC 이전 202행과 같은 분모가 아니다.

## 검증 및 공식 제한사항

- Train quantile edge와 20개 state/action cell occupancy를 독립 재계산하여 동결 규칙과 정확히 일치했다.
- 동일 입력 반복 판정, 현재 action·series 변경 불변성, Train 최대 DAT 초과, max bin 경계, 55-column 순서, 입력 finite, Validation 17,680행 row registry 일치를 검증했다.
- 읽은 공식 Train/scaler/schema/support 입력 hash와 rule hash가 완료 후에도 일치했다. `verification.json` 참조.
- 상태·transition·run sidecar만 생성했다. 공식 dataset/scaler/model/support/weights/threshold/reward는 변경하지 않았다. fit/Test/PPO/Step17~19/Step19.5/HPO를 실행하지 않았다.
- legacy snapshot은 실제 raw command 전송 timestamp를 복원한 ground truth가 아니다.
- Train high-CO₂ 및 high-CO₂/high-vent 부족은 그대로다. 현재 관측 및 previous vent 기준 집계는 `source_high_CO2_coverage.csv`에 별도 저장했다. Domain을 좁히는 것이 dynamics 학습 coverage 복구는 아니다.
- clean5 Transition 누적 오차, geas2_5 미관측 series, online actuator-safe fallback BLOCKED, reward imbalance audit 미해결을 모두 유지한다.
- reward audit 후보: temp가 six-penalty 약 99%, RH/VPD/cond/HV 0, strawberry terminal cost 약 92.40%. 이번 domain 생성에는 reward를 사용하지 않았다.

## 다음 판단

추천 계약은 **현재 동결한 후보 하나를 추가 설계의 기준으로 보존**하는 것이다. IN만 연구 admission 검토 대상, LOW/OUT 보류, action-pair는 observation과 분리, 기존 local OOD는 유지한다. 공식 Step16에 즉시 적용하지 않는다.

최소 후속 작업은 **이 후보를 고정한 상태에서 Train의 episode/시기별 독립 유지 검증으로 coverage 근거의 안정성을 감사**하는 것이다. 50행/2episode, quantile cell 경계와 제한의 재현성이 충분한지 먼저 확인하고, 조건을 수정해야 한다면 별도 사전 정의 version으로 다룬다. 이번 Validation 결과에 cutoff를 재조정하지 않는다. 이후의 연구 admission/protocol 변경 승인, IN 영역의 rollout 안정성 검증, 별도 reward audit가 필요하다.

**strawberry Step16 BLOCKED / 전체 Step17 BLOCKED 유지. 온라인 배포 안전성 승인 없음.**
'''
(D/'final_decision_ko.md').write_text(text)
m.check()
artifacts=[p for p in sorted(D.iterdir()) if p.is_file() and p.name!='artifact_manifest.json']
m.write(D/'artifact_manifest.json',{'version':D.name,'judgment':'B','scope':'Train-only candidate freeze then Validation shadow; no official admission','files':[{'path':p.name,'sha256':m.sha(p)} for p in artifacts],'report_script_sha256':m.sha(Path(__file__)),'offline_strawberry_Step16':'BLOCKED','overall_Step17':'BLOCKED','online_deployment':'BLOCKED'})
print(json.dumps({'judgment':'B','verification':verification,'high_CO2':coverage,'report':str(D/'final_decision_ko.md')},ensure_ascii=False))
