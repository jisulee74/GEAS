from pathlib import Path
import json,hashlib
import pandas as pd
import audit_strawberry_domain_independence_v1 as a
D=a.D;a.verify_inputs()
s=a.read(D/'summary.json');p=a.read(D/'protocol.json');f=pd.read_csv(D/'fold_summary.csv');c=pd.read_csv(D/'cell_stability.csv');m=pd.read_csv(D/'method_summary.csv')
def table(df):
 def v(x):return f'{x:.4f}' if isinstance(x,float) else str(x).replace('|','/')
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(v(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
sm=m[['method','mature_only','folds','rows','state_IN_rate','state_LOW_rate','state_OUT_rate','state_median_IN_rate','state_min_IN_rate','state_drop_from_full_pp']].copy()
for col in sm.columns:
 if col.endswith('_rate'):sm[col]*=100
rank=c.loc[c.kind.eq('state')&~c.mature_only&c.method.isin(['episode','month','season','calendar14'])].pivot(index='cell',columns='method',values='supported_loss_row_rate').reset_index()
for col in rank.columns[1:]:rank[col]*=100
rank=rank.sort_values('month',ascending=False)
act=c.loc[c.kind.eq('action')&((c.method.eq('forward_month')&c.mature_only)|(~c.method.eq('forward_month')&~c.mature_only))].copy()
act['IN_pct']=100*act.heldout_supported_rows/act.evaluated_rows
act['LOW_pct']=100*act.heldout_sparse_rows/act.evaluated_rows
act['OUT_pct']=100*act.heldout_unobserved_rows/act.evaluated_rows
act=act[['method','cell','evaluated_rows','IN_pct','LOW_pct','OUT_pct']]
act.to_csv(D/'action_pair_rates.csv',index=False)
# Row-weighted threshold failure summaries, overlapping cells must not be summed as unique rows.
evidence=pd.read_parquet(D/'fold_cell_evidence.parquet')
worst=f.loc[f.method.eq('episode')].sort_values('state_IN_rate').head(10)
time=f.loc[f.method.isin(['month','season','forward_month']),['method','fold','rows','evidence_rows','state_IN_rate','state_LOW_rate','state_OUT_rate']].copy()
for col in ['state_IN_rate','state_LOW_rate','state_OUT_rate']:time[col]*=100
criteria=[]
def metric(method,key):return float(m.loc[m.method.eq(method)&m.mature_only.eq(method=='forward_month'),key].iloc[0])
for method,field,minimum in [('episode','state_IN_rate',.90),('episode','state_median_IN_rate',.90),('episode','state_min_IN_rate',.50),('month','state_IN_rate',.70),('month','state_median_IN_rate',.70),('month','state_min_IN_rate',.20),('calendar14','state_IN_rate',.70),('calendar14','state_median_IN_rate',.70),('calendar14','state_min_IN_rate',.20),('season','state_IN_rate',.50),('forward_month','state_IN_rate',.50),('episode','action_IN_rate',.95),('month','action_IN_rate',.80),('calendar14','action_IN_rate',.80),('forward_month','action_IN_rate',.80)]:
 value=metric(method,field);criteria.append({'method':method,'A_metric':field,'required_pct':100*minimum,'actual_pct':100*value,'met':value>=minimum})
ct=pd.DataFrame(criteria);ct.to_csv(D/'predeclared_A_criteria_results.csv',index=False)
text=f'''# Strawberry frozen domain v1 Train independence audit

**최종 판정 B.** Episode를 독립 제외하면 대부분 유지되지만, 계절·forward 시간 검증에서 coverage가 크게 감소한다. 현재 후보를 공식 admission으로 승인하지 않는다. **strawberry Step16 / 전체 Step17 BLOCKED 유지.**

## 범위와 사전 고정

B60/R6 → Step16 strawberry BLOCKED → coverage audit → research domain shadow v1 판정 B 이후, 이번 작업은 **Train 내부 occupancy 근거의 독립 유지 검증**이다. 44,097행·157 episode·단일 strawberry_geas2_0을 사용했다.

`protocol.json`과 `protocol_lock.json`을 fold 실행 전에 저장했다. Protocol SHA256 `{a.sha(D/'protocol.json')}`. 기존 v1 rule SHA256 `{p['rule_sha256']}`.

50행, 2episode, quantile edge, 14 state cells, 6 action-pair, DAT 30일 bin, 기존 range/source 기준을 변경하지 않았다. Heldout의 row와 같은 episode의 모든 row를 근거에서 제외하고 나머지 Train의 occupancy만 재계산했다. 이는 모델 fit이나 새로운 공식 domain/support 생성이 아니다.

사전 A 기준: episode weighted/median IN≥90%, min≥50%; month/calendar14 각각 weighted/median≥70%, min≥20%; season weighted≥50%; warm forward weighted≥50%; action episode≥95%, month/calendar14/forward≥80%. 모두 만족해야 A다.

사전 B 최소 기준: A 미충족이면서 episode weighted IN≥70%, month/calendar14 각각≥40%, action episode≥80%. 그 외는 C다. 이는 이번 감사를 위한 프로젝트 진단 기준이며 기존 공식 gate나 통계적 보증이 아니다. 결과를 보고 변경하지 않았다. DAT-bin holdout은 명시적 필수 bin을 통째로 비워 반드시 OUT가 되는 구조적 stress이므로 A/B/C 수치 판정에서 제외한다고 **사전에** 명시했다.

## 검증 설계와 한계

- episode: leave-one-episode-out 157 fold.
- month: YYYY-MM 7 fold, season: year-season 3 fold(DJF의 12월은 다음 연도에 소속).
- calendar14: 1970-01-01을 기준으로 고정한 연속 14일 block 14 fold. 데이터 양에 맞춰 경계를 이동하지 않았다.
- DAT30: floor(DAT/30) 7 fold.
- forward: 7개 월을 순차 평가하되 해당 월 이전 timestamp만 evidence로 사용. 첫 월은 cold-start로 모두 보고하고, warm 요약에서는 제외했다.
- 시간 block이 episode 일부만 포함해도 같은 episode의 나머지 행을 evidence에서 purge했다. fold별 제외/purge 수는 `fold_summary.csv`에 있다.
- **고정 bin edge/range/scaler는 전체 Train을 이미 본 geometry다.** 요청대로 이를 변경하지 않았으므로 forward 결과는 ‘고정 geometry에 조건부인 과거 occupancy 검증’이다. Future row는 occupancy/episode 수에 들어가지 않았지만, 완전한 prospective preprocessing/domain-discovery 독립성은 검증하지 못했다. 미래 정보가 전혀 없는 새 규칙이라고 주장하지 않는다.
- 전체 Train 자기포함 결과는 상태 IN 43,452/44,097=98.5373%, action IN 44,075/44,097=99.9501%다.

## 유지율 요약

아래 비율은 %다. weighted IN은 행 가중, median/min은 fold별 비율의 비가중 통계다. 감소는 전체 Train 98.5373% 대비 percentage point다. forward mature_only=True는 첫 cold-start 월 제외다.

{table(sm)}

Episode IN 96.43%는 full Train보다 2.11pp 감소하여 대부분의 episode는 근거가 독립적으로 존재한다. 그러나 calendar14 82.79%, month 74.28%, season 7.30%로 감소하며, warm forward IN/LOW/OUT는 **17.57% / 7.45% / 74.98%**다. 따라서 전체 Train을 retrospective하게 모아 본 coverage를 미관측 시기에도 유지된다고 일반화할 수 없다.

### 월·계절·forward 상세

{table(time)}

서로 다른 시기의 DAT·weather·indoor가 함께 변하므로 이 결과만으로 season과 DAT의 인과 효과를 분리할 수 없다. 시간적으로 인접한 episode를 제외하는 검사와 계절 전체를 제외하는 검사는 난이도가 다르다. 데이터 수 감소 및 범위 이동도 동시에 발생한다. Season은 SON/DJF/MAM 3개뿐이며 MAM은 1,425행이다. 여름 등 미관측 season으로 일반화하지 않는다.

### 최저 유지 episode

{table(worst[['fold','rows','DAT_min','DAT_max','state_IN','state_LOW','state_OUT','state_IN_rate']])}

마지막 표 IN rate는 0~1 값이다. 모든 157 episode의 IN/LOW/OUT 수와 비율은 `fold_summary.csv`에 보존했다. 최솟값이 0인 episode가 있어 전체 평균만으로 안정성을 승인할 수 없다.

## 14개 cell 안정성

표는 full Train에서 supported였으나 heldout evidence에서 sparse 또는 unobserved로 변한 **평가 행 비율(%)**이다. 한 행이 여러 cell에 포함되므로 cell별 비율을 합산해 전체 이탈률로 해석하지 않는다.

{table(rank)}

- Episode 독립 제거에서 손실이 큰 cell은 DAT×T(0.83%), T×Tout×previous vent(0.74%), T×previous vent(0.54%)다.
- Month에서는 DAT×RH(15.31%), DAT×DLI(14.79%), DAT×T(12.83%)가 상대적으로 불안정하다.
- Season에서는 DAT×RH(73.64%), DAT×CO₂(68.90%), DAT×DLI(68.49%) 등의 근거가 크게 무너진다.
- light×DLI는 전체 검사에서 supported→unsupported 손실 0으로 상대적으로 안정적이다. 하지만 기존 sparse cell까지 모두 안정적이라는 뜻은 아니다. CO₂×previous vent 및 RH×previous vent도 episode/month에서는 안정적이지만 season을 통째로 제외하면 손실이 발생한다.
- Warm forward에서도 DAT×T/DAT×RH 등과 명시적 DAT30 근거가 주요 제한이다. 어느 cell도 제거하지 않았다.

`cell_stability.csv`에는 supported→sparse, supported→unobserved, sparse→unobserved를 **평가 행 수/비율 및 unique fold-cell 수**로 모두 저장했다. `fold_cell_evidence.parquet`에는 각 fold/bin의 full count, retained count/fraction, full/retained episode 수를 보존했다. 따라서 50행 미만과 2episode 미만으로 각각 얼마나 자주 떨어졌는지 확인할 수 있다. 두 조건의 탈락은 겹칠 수 있다. 이미 sparse였던 cell이 50행 아래로 추가 감소하는 경우도 있어 threshold crossing 수와 supported-loss 수는 같지 않다.

예를 들어 month DAT×RH는 supported→sparse 2,346행, supported→unobserved 4,405행이다. 50행 경계 하락은 6,751행, 2episode 경계 하락은 5,670행이다. 이는 임계값을 낮춰야 한다는 근거가 아니라, 독립 시기에서 동일 근거를 유지하지 못한다는 진단이다.

## DAT191 경계

DAT180~209는 DAT191 97행/1episode만 존재한다. 해당 episode `iot97_strawberry_geas2_0_seg24_20240910`를 제외하면 **97→0행, 1→0episode**, DAT cell은 sparse→unobserved, 해당 97행은 OUT가 된다. month/season/calendar14에서도 같은 근거 소실이 확인된다. 날짜가 2024-09라고 해서 DAT가 작다고 추정하지 않았고 기록된 DAT191 그대로 사용했다.

DAT30 bin 전체를 제외하는 7 fold에서는 모든 행이 OUT다. 이는 명시적인 DAT30 admission 조건의 구조적 결과이며 독립적인 7회 성능 실패나 C 판정 근거로 중복 계산하지 않는다. DAT192 이상은 frozen Train 최대191 밖이라는 기존 제한을 유지했다. 이번 eligible Train에 없는 DAT192 이상의 가상 행을 새 표본으로 만들지 않았다.

## 별도 action-pair 독립성

{table(act.loc[act.method.isin(['episode','forward_month'])])}

전체 action admission은 episode/month/calendar14에서 IN 99.9501%로 full Train과 같지만 **LOW22가 LOW20+OUT2로 악화**된다. 같은 IN 비율이 모든 pair의 안정성을 뜻하지 않는다. episode 기준 shade curtain의 2행은 미관측 pair, vent의 20행은 sparse로 남는다. cool/fan/heat/thermal은 이 검사에서 supported를 유지했다.

DAT30 action IN 99.74%, season 98.98%, warm forward 97.54%다. 모든 6개 action×모든 방법의 상세 IN/LOW/OUT는 `action_pair_rates.csv`에 있다. Action pair 유지가 물리적 제어 안전성·효과·모델 반응을 보장하지 않는다. Observation 판정과 current action-pair 판정은 분리했으며 observation에 current action을 추가하지 않았다.

## 사전 판정 적용

{table(ct)}

A는 episode/calendar14 최솟값, month 최솟값, season, warm forward 등의 기준을 충족하지 못했다. B의 최소 기준은 모두 충족하므로 **B**로 판정했다. Season/forward의 큰 감소를 숨기거나 Validation 성능으로 판정을 보완하지 않았다.

## 검증과 보존

- 195 fold 모두 heldout episode와 evidence episode가 분리됨을 검사했다.
- Forward evidence timestamp는 해당 월 시작보다 엄격히 이전이다.
- Full Train occupancy/episode 수 및 상태/action 판정이 frozen v1과 정확히 일치했다.
- Episode 첫 fold와 마지막 forward fold의 21개 cell씩 총42개에 대해 별도 pandas groupby 구현으로 row/episode count를 교차 검증했다.
- Evidence 제거 후 count가 증가하거나 분류가 개선되지 않는 단조성을 확인했다.
- 각 방법에서 Train 44,097행이 정확히 한 번씩 heldout 판정됨을 확인했다.
- v1 전체 artifact는 byte hash로 작업 전후 동일함을 확인했다. 기존 Validation 파일을 평가·재계산하지 않았으며 보존 확인용 checksum만 계산했다.
- 기존 Validation shadow의 row IN 약31.9%, transition IN 약30.84%, whole-run IN 1/72, IN 내 severe 8.53%, clean5 시작 OUT 결과는 그대로다. 이번 결과를 그 규칙에 되먹임하지 않았다.
- geas2_0 내부 episode/time coverage만 감사했다. geas2_5 일반화, legacy snapshot 실물 전송 시각, high-CO₂/high-vent dynamics는 검증하지 않았다.
- Dataset/scaler/model/support/threshold/reward를 변경하지 않았고 fit/PPO/HPO/Test 및 공식 Step 재실행을 하지 않았다.

## 후속 작업 하나

**Train-only 시기 독립성을 목표로 하는 domain v2 사전 설계 문서 작성**만 추천한다. 이번에 확인된 retrospective coverage와 forward deployment-like coverage의 목적 차이, DAT/season 의존 근거를 어떻게 취급할지, 그리고 별도 Train 평가 분할·사전 gate를 먼저 정해야 한다. 이번 감사에서 v2 조건·bin·cutoff를 수정하거나 기존 Validation에 재적용하지 않았다. V1을 공식 적용하거나 50/2 기준을 낮추는 승인은 아니다.

기존 clean5 Transition/autoregressive 진단, reward imbalance audit, online actuator-safe fallback blocker도 유지한다. **strawberry Step16 / 전체 Step17 BLOCKED**이며 실제 배포 승인이 아니다.
'''
(D/'final_decision_ko.md').write_text(text)
a.verify_inputs()
a.write(D/'artifact_manifest.json',{'version':D.name,'judgment':s['judgment'],'files':[{'path':x.name,'sha256':a.sha(x)} for x in sorted(D.iterdir()) if x.is_file() and x.name!='artifact_manifest.json'],'report_script_sha256':a.sha(Path(__file__))})
print(D/'final_decision_ko.md')
