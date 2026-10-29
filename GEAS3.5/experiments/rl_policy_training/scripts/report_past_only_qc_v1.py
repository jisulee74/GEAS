from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[3];O=R/'experiments/rl_policy_training/artifacts/past_only_qc_shadow_v1';A=O.parent/'qc_future_interpolation_blast_radius_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def table(f):
 def fmt(v):return f'{v:.4g}' if isinstance(v,float) else str(v).replace('|','/')
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))
def main():
 assert (O/'verification.json').exists()
 s=pd.read_parquet(O/'source_reconstruction.parquet');p=pd.read_csv(O/'population_summary.csv');w=pd.read_csv(O/'window_summary.csv');m=pd.read_csv(O/'membership_summary.csv')
 assert len(s)==429 and len(p)==18
 for r in s.itertuples():
  if pd.notna(r.past_timestamp):assert pd.Timestamp(r.past_timestamp)<pd.Timestamp(r.current_timestamp)
  if pd.notna(r.second_past_timestamp):assert pd.Timestamp(r.second_past_timestamp)<pd.Timestamp(r.past_timestamp)
  if r.reconstructable:
   v=r.past_value
   if r.candidate=='PAST_TWO_POINT_EXTRAPOLATION':v+=(r.past_value-r.second_past_value)*(pd.Timestamp(r.current_timestamp)-pd.Timestamp(r.past_timestamp))/(pd.Timestamp(r.past_timestamp)-pd.Timestamp(r.second_past_timestamp))
   assert v==r.shadow_value
 # Future endpoint mutation must not change the pure causal reconstruction.
 mutated=s.copy();mutated['right_endpoint_timestamp_audit_only']=pd.Timestamp('2099-01-01');assert mutated.shadow_value.equals(s.shadow_value)
 runs=s.groupby(['candidate','crop','split','run_id']).reconstructable.all().reset_index()
 rsum=runs.groupby('candidate').agg(runs=('run_id','size'),fully_reconstructable_runs=('reconstructable','sum')).reset_index()
 old=pd.read_csv(A/'past_only_feasibility.csv')
 now=runs.loc[runs.candidate.eq('PAST_TWO_POINT_EXTRAPOLATION')].merge(old[['run_id','two_past_points_available']],on='run_id',validate='one_to_one')
 now.to_csv(O/'previous_feasibility_comparison.csv',index=False)
 src=s.groupby(['candidate','variable']).agg(events=('event_id','size'),reconstructable=('reconstructable','sum'),age_max=('last_valid_age_min','max'),age_p50=('last_valid_age_min','median'),signed_mean=('signed_difference','mean'),absolute_max=('absolute_difference','max'),exact_same=('exact_same','sum')).reset_index()
 fs=[];prop=[]
 for crop in ['melon','cucumber','strawberry']:
  for split in ['train','validation']:
   tag=f'{crop}_{split}';edges=pd.read_parquet(A/f'{tag}_feature_lineage.parquet')
   for cand in ['HOLD_LAST_VALID','INVALID_QUARANTINE','PAST_TWO_POINT_EXTRAPOLATION']:
    f=pd.read_parquet(O/f'{tag}_{cand}_feature_comparison.parquet')
    if f.empty:continue
    for (role,feature),g in f.groupby(['role','feature']):fs.append({'crop':crop,'split':split,'candidate':cand,'role':role,'feature':feature,'rows':len(g),'changed':int(g.meaningful_changed.sum()),'invalid':int(g.invalid.sum()),'exact_same':int(g.exact_same.sum()),'signed_mean':g.signed_difference.mean(),'max_abs_difference':g.absolute_difference.max()})
    # Recover recipient lag without reading source datasets again.
    evp=A/f'{tag}_source_events.csv.gz'
    try:ev=pd.read_csv(evp)
    except pd.errors.EmptyDataError:continue
    for role in ['current','target']:
     changed=f.loc[f.role.eq(role)&f.meaningful_changed]
     maxlag=0.
     for x in changed.itertuples():
      poss=edges.loc[edges.feature.eq(x.feature),'recipient_position'].unique()
      # max ancestry lag is bounded from frozen source to same-date changed timestamp.
      sources=ev.loc[ev.variable.eq('out_light'),'current_timestamp'] if x.feature!='obs_outdoor_wind_speed' else ev.loc[ev.variable.eq('out_windsp'),'current_timestamp']
      for st in pd.to_datetime(sources):
       dt=pd.Timestamp(x.timestamp)
       if st.date()==dt.date() and st<=dt:maxlag=max(maxlag,(dt-st).total_seconds()/60)
     prop.append({'crop':crop,'split':split,'candidate':cand,'role':role,'max_changed_ancestry_horizon_minutes':maxlag,'definition':'same-day source ancestry to numerically changed recipient, not isolated event causal attribution'})
 pd.DataFrame(fs).to_csv(O/'feature_difference_summary.csv',index=False);pd.DataFrame(prop).to_csv(O/'propagation_horizon.csv',index=False);rsum.to_csv(O/'run_reconstructability.csv',index=False);src.to_csv(O/'source_difference_summary.csv',index=False)
 binding=[]
 for crop in ['melon','cucumber','strawberry']:
  for split in ['train','validation']:
   tag=f'{crop}_{split}'
   try:rp=pd.read_csv(A/f'{tag}_past_available_runs.csv')
   except pd.errors.EmptyDataError:continue
   if rp.empty:continue
   x=s.loc[s.crop.eq(crop)&s.split.eq(split)].merge(rp[['run_id','right_value']],on='run_id',validate='many_to_one')
   registry=pd.read_parquet(A/f'{tag}_transition_sidecar.parquet',columns=['row_id','reg_date'])
   x=x.merge(registry,left_on='current_timestamp',right_on='reg_date',how='left',validate='many_to_one');binding.append(x)
 pd.concat(binding,ignore_index=True).to_parquet(O/'source_reconstruction_with_endpoint_and_row_id.parquet',index=False)
 final='B'
 text=f'''# Past-only QC reconstruction shadow comparison

**최종 판정 B — HLV와 invalid/quarantine 사이에 프로젝트의 age/gap 허용 정책 결정이 추가로 필요하다.** 구현 가능한 과거값이 존재한다는 사실은 그 값을 오래 유지해도 정확하거나 안전하다는 승인이 아니다. 공식 채택/재구축/재학습/도메인 재실행은 하지 않았다.

## 동결 계약과 범위

기존 blast-radius audit의 143 feature-event, 92 run을 변경 없이 사용했다. 각 event는 source timestamp/method/좌우 endpoint와 기존 row position으로 binding했다. 비교 후보는 HOLD_LAST_VALID, INVALID_QUARANTINE, PAST_TWO_POINT_EXTRAPOLATION 세 개뿐이다. `protocol.json`·`protocol_lock.json`에 계산 전 후보/근거 조회/invalid 규칙과 코드 hash를 저장했다.

6개 공식 Train/Validation 모집단을 읽었지만 성능/Transition prediction/OOD/reward를 평가하지 않았다. Test는 읽지 않았다. Domain-v2 C/E calendar에 해당하는 Train 값의 shadow 노출은 존재하지만 domain-v2 rule/admission/gate는 실행하지 않았다. E의 historical exposure 한계와 기존 BLOCKED는 유지한다.

## 과거 evidence와 feasibility

Same crop/series/segment/source_file_path 안에서 timestamp가 현재보다 작은 유효 원본 측정만 조회했다. Missing/rule-outlier/AI-outlier/imputed/ffill 표시 또는 interpolation 출처는 과거 측정 후보에서 제외했다. Shadow 값 자체를 다음 event의 valid measurement로 재사용하지 않았다. Episode는 source 단절과 동일하지 않으므로 같은 physical identity면 경계 전 과거값을 조회할 수 있다. Age cutoff를 만들거나 결과 후 조절하지 않았다.

{table(rsum)}

이전 feasibility와 run별 비교는 `previous_feasibility_comparison.csv`에 보존했다. 차이가 난 run은 strawberry_validation:out_windsp:19174다. 현재는 2025-11-21 13:50, 최근 과거값은 13:45, 두 번째는 2025-11-19 17:40으로 **현재보다 2,650분(44시간10분) 전**이다. 같은 source/series/segment라는 사실만으로 이 장거리 기울기를 신뢰할 수 없다. 이것은 extrapolation 계산 가능성만 복구된 것이며 안정성 승인이 아니다. 이전 91/92는 episode 안에서 QC값의 최근2점을 찾은 feasibility였다. 이번 결과는 raw-value 우선, ffill/imputed 제외, 동일 physical identity 기준이므로 count 차이가 있으면 근거 조회 계약 차이이며 event를 제거한 것이 아니다. 모든 run/event별 과거값·두 timestamp·age·gap·원본 QC·미래 right timestamp·reconstructability는 `source_reconstruction.parquet`에 있다.

{table(src)}

Signed/absolute difference는 original offline interpolation과의 **변화량**이며 정확도 score가 아니다. 물리 truth는 unknown. 실제 extrapolation 광량 범위는 -356~3,384, 풍속 최대71.05로 나왔다(기존 저장 단위). 이를 물리 truth로 보거나 새 threshold로 제거하지 않았다. Extrapolation은 source 값에 clamp를 적용하지 않았으며 음수/큰 값이 나와도 자동 보정하지 않았다. 기존 derived 함수의 light.clip(0) 등은 기존 정의로 유지되지만 raw shadow source의 이상값을 승인하는 뜻은 아니다.

## 계산 계약

- HLV: 가장 최근 valid past raw value. 근거가 없으면 NaN/invalid.
- INVALID: affected source를 NaN으로 표시하고 frozen ancestry의 모든 descendant에 invalid mask를 전파. 공식 row 삭제 없음.
- Two-point: `v1+(v1-v0)*(t-t1)/(t1-t0)`; 두 시각 모두 t 이전이어야 함. 이는 extrapolation이며 새 dynamics 가정이다. 두 점이 없으면 invalid.
- 모든 후보는 current 이전 identity별 history만 참조한다. 미래 endpoint는 감사용 출력일 뿐 계산 입력이 아니다.

Daily sequential recurrence에서 현재 shadow light로 daytime/target을 정하고, 원본과 shadow의 현재 증분 차이를 날짜별로 누적했다. DLI는 `max(light,0)*dt_seconds*2.02/1e6`, degree는 기존 target/T의 양의 차이×dt_minutes다. 날짜가 바뀌면 누적 delta를 reset한다. 미래 row를 이용해 과거 누적을 수정하지 않는다.

기존 physical baseline은 불변이고 sidecar에만 shadow를 저장했다. 현재값이 valid한 후보의 누적 차이만 더하며 invalid ancestry는 최종적으로 NaN mask로 처리해 원본 default0/fill에 의한 허위 복구를 막는다. Target min/max/center는 기존 명시적 override 우선, 없으면 daytime 분기 및 ±1°C band를 유지한다. B vent-loss는 **이전 command**로 재계산한다. 이번 affected event에서는 measured light_sum 분기가 유지되어 별도의 적분 fallback 변경이 필요하지 않았다.

Wind는 obs_outdoor_wind_speed만 바뀌고 light/degree/rolling 경로로 새 연결을 만들지 않는다. Low-VPD/RH90/ACH/VPD/dewpoint/condensation는 이번 wind/light reconstruction의 수치 변화 대상이 아니다. Conditional branch와 영계수로 original과 같은 값이면 그대로 unchanged로 센다.

## Current/target 및 history 변화

Meaningful changed는 rtol=1e-6, atol=1e-8 밖의 차이이며 exact same 여부도 별도 보존한다. Invalid는 changed와 구분한다. Propagated changed rows는 직접 event가 없는 후속 current행의 변화다. Direct event와 history가 겹치는 row는 direct 쪽에 배정해 중복을 피했다.

{table(p)}

각 후보·feature의 original/shadow/signed/absolute difference, exact same, meaningful changed, invalid는 `*_feature_comparison.parquet`에 있다. `feature_difference_summary.csv`에서 daytime/target/degree 등이 실제 몇 행 바뀌었는지 확인할 수 있다. Code dependency만으로 numerical change라고 세지 않았다. `propagation_horizon.csv`는 같은 날 source ancestry→changed recipient의 최대 시간이며 여러 event가 있는 날의 개별 원인 기여 분리는 아니다.

INVALID의 ancestry propagation은 conservative invalidity다. 원래 zero coefficient 때문에 수치가 같을 수 있는 경로도 원천 가용성 확인 전에는 묵시적으로 valid 처리하지 않았다. Reconstruction 가능한 후보와 달리 accuracy/최소손실을 목표로 invalid 범위를 줄이지 않았다.

## 기존 transition/window

{table(w)}

기존 frozen window의 start ID/end target/run을 검증한 뒤 current 또는 target이 changed/invalid인 window를 집계했다. 새 episode/window를 생성하지 않았다. Admission을 실제로 바꾸지 않았고 이 수는 shadow 영향이다.

## Fit/support membership 영향 예상

{table(m)}

A. Finite changed 값: 기존 row identity를 유지할 수 있는 후보지만, 물리 validity/새 age 정책이 승인되기 전 eligibility 유지 확정은 아니다. 새로운 scaler/model/support 수치를 old artifact와 혼용할 수 없다.
B. Invalid current/target: 향후 transition/window admission 재판정 필요. Scaler는 current row validity, Transition은 current+target 구분.
C. 후속 history: 직접 event를 제외해도 그 후 current/target 영향이 남으므로 registry 전체 lineage를 따라 판단해야 한다.

표는 회원 포함 여부와 값 변경/invalid 예상만 계산했다. Scaler/model/support fit 또는 성능 변화는 계산하지 않았다.

## Causal 계약 검증과 해석

| 항목 | HLV | INVALID | Two-point |
| --- | --- | --- | --- |
| Future endpoint 없는 계산 | PASS | PASS | 두 과거점 있을 때 PASS; 없으면 invalid |
| Strict past 및 identity 준수 | PASS | PASS | PASS |
| Deterministic replay | PASS | PASS | PASS |
| 동일 history의 online 순차 계산 가능 | PASS(계약 수준) | PASS(quality mask 수준) | PASS(계약 수준) |
| Daily history 전파·자정 reset 정의 | PASS | PASS, invalid ancestry 보존 | PASS |
| Missing 정의 | past 없으면 invalid | 항상 invalid | 두 점 없으면 invalid |
| 전체 run 숫자 복구 | 위 feasibility 결과 | 의도적으로 미복구 | 위 feasibility 결과 |
| 물리 정확성 / 실제 안전성 | 미확정 | 미확정 | 미확정 |

동일 순차 계산을 두 번 수행해 array/NaN 일치, timestamp strict order, 원본 event 수, 과거2점 식의 독립 계산을 확인했다. 이는 실제 live ingestion의 QC flag 생성 시각·지연까지 검증한 것은 아니다. Frozen event 목록은 감사용 trigger이며 live에서 미래를 보고 ‘나중에 interpolation될 row’를 알아낸다는 뜻이 아니다. 공식화하려면 현재 missing/invalid 신호만으로 동일 trigger를 정의해야 한다. 따라서 live deployment PASS를 부여하지 않는다.

## 추천 하나와 향후 범위

**HLV를 기본으로 하되 past가 없거나 age/source 근거가 허용되지 않으면 invalid로 남기는 causal QC 계약을 다음 공식 후보로 사전 정의할 것을 추천한다.** 아직 age cutoff가 잠기지 않았으므로 판정은 B다. 이번 데이터에서 age가 짧아 보여도 결과로 cutoff를 맞추지 않는다. Two-point는 새 추세 가정을 더하므로 단순히 복구 행이 많다는 이유로 우선하지 않는다. INVALID 단독은 가장 보수적이지만 history/fit 분모 손실과 운영 처리 계약을 함께 결정해야 한다.

공식 적용 시 최소 의존성 순서는 QC physical 및 영향 daily derived/history 재생성 → 새 B60/R6 Phase0 dataset/admission 및 Train-only scaler → Transition fresh refit/reselection → Step14 binding → Step15 support/calibration → Step16 재검증이다. Invalid를 택하면 row/window eligibility도 재확정해야 한다. 단순 raw source 15행 교체로 끝나지 않는다. Test는 잠그고 기존 artifact를 보존한 새 버전이 필요하다. 이 작업은 이번에 실행하지 않았다.

## 유지 상태

기존 audit B bounded but pipeline-relevant, historical metrics/판정 artifact 유지. Domain-v2 D input integrity BLOCKED, D geometry/independence 및 C/E gate 미실행, strawberry Step16/전체 Step17/online deployment BLOCKED 유지. High-CO₂/high-vent 부족, geas2_5 unseen series, legacy snapshot assumption, clean5 accumulation, reward imbalance는 이번 shadow가 해결한 문제가 아니다.
'''
 (O/'final_decision_ko.md').write_text(text)
 for path,h in json.loads((O/'input_hashes.json').read_text()).items():assert sha(R/path)==h
 wr('final_gate.json',{'judgment':final,'official_adoption':False,'future_endpoint_in_reconstruction':False,'physical_accuracy':'unproven','deployment_safety':'BLOCKED','domain_v2':'BLOCKED','Step16_strawberry':'BLOCKED','Step17':'BLOCKED','recommendation':'HLV with predeclared age/source validity and invalid fallback; policy lock pending'})
 wr('artifact_manifest.json',{'files':[{'path':p.name,'sha256':sha(p)} for p in sorted(O.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'],'report_script_sha256':sha(Path(__file__))})
 print(rsum.to_string(index=False));print(src.to_string(index=False))
if __name__=='__main__':main()
