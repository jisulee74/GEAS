"""Final read-only audit reporting and complete dependency inventory."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[3];O=R/'experiments/rl_policy_training/artifacts/qc_future_interpolation_blast_radius_v1';B=R/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,default=str)+'\n')
def table(f):
 def fmt(v):return f'{v:.3f}' if isinstance(v,float) else str(v).replace('|','/')
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))
def main():
 pop=pd.read_csv(O/'population_summary.csv');line=pd.read_csv(O/'artifact_lineage.csv');win=pd.read_csv(O/'window_summary.csv');feat=pd.read_csv(O/'feature_summary.csv');series=pd.read_csv(O/'source_series_summary.csv');trace=pd.read_csv(O/'Step16_trace_membership.csv');look=pd.read_csv(O/'variable_lookahead_summary.csv')
 assert pop.formula_mismatch.sum()==0 and feat.value_mismatches.sum()==0
 assert (pd.read_csv(O/'DLI_integral_verification.csv').eval('DLI_recorded_integral_rows == matches')).all()
 for pcol in ['eligible_direct','eligible_propagated_only','eligible_any','current_still_future','transition_any']:pop[pcol+'_pct']=100*pop[pcol]/pop.eligible_rows
 for col in ['direct','propagated','current_any','transition_any']:line[col+'_pct']=100*line[col]/line.rows
 pop.to_csv(O/'population_summary_with_rates.csv',index=False);line.to_csv(O/'artifact_lineage_with_rates.csv',index=False)
 obs=json.loads((B/'schema.json').read_text())['observation_columns'];graph=[]
 light_now={'obs_outdoor_light','obs_is_daytime','obs_current_target_temp_c','obs_current_target_temp_min_c','obs_current_target_temp_max_c','obs_derived_vent_loss_proxy'}
 light_hist={'obs_derived_dli_mol_m2','obs_derived_heating_degree_minutes','obs_derived_cooling_degree_minutes'}
 for c in obs:
  if c=='obs_outdoor_wind_speed':dep='wind direct; no stored history descendant among55'
  elif c in light_now:dep='light direct/control dependency; target only if explicit current override absent; zero coefficient may mask numeric effect'
  elif c in light_hist:dep='light same-day ancestry; degrees via selected target fallback; not proof of nonzero effect'
  elif c in ['obs_derived_light_sum','obs_derived_light_sum_ratio']:dep='light history ONLY if recorded measured cumulative absent; source out_light_sum direct if interpolated; no such affected branch observed here'
  else:dep='not dependent on current/past wind-light interpolation in inspected pipeline'
  graph.append({'feature':c,'wind_light_dependency':dep,'actual_current_affected_rows':int(feat.loc[feat.role.eq('current')&feat.feature.eq(c),'unique_rows'].sum()),'support_column':True})
 pd.DataFrame(graph).to_csv(O/'all55_dependency_inventory.csv',index=False)
 (O/'dependency_graph.mmd').write_text('''flowchart TD
  WIND[wind two-sided QC] --> OW[obs_outdoor_wind_speed]
  LIGHT[light two-sided QC] --> OL[obs_outdoor_light]
  LIGHT --> DAY[obs_is_daytime]
  DAY -->|only absent explicit current target| TARGET[current target min/max/center]
  TARGET --> DEG[daily degree-minute history]
  LIGHT --> DLI[daily DLI integral]
  LIGHT --> VENT[vent-loss with previous command and T/Tout]
  LIGHT -->|only absent recorded cumulative| SUM[daily light sum and ratio]
  OW --> SUPPORT[55-state support and Transition inputs]
  DLI --> SUPPORT
  DEG --> SUPPORT
  VENT --> SUPPORT
  SUM --> SUPPORT
  TARGET --> SUPPORT
  OL --> SUPPORT
  DAY --> SUPPORT
  TRH[T/RH without wind-light path] --> VPD[VPD dewpoint condensation]
  TRH --> ROLL[low-VPD RH90 rolling and ACH]
''')
 # Audit availability at run beginnings without constructing any replacement value.
 rawsummary=[];past=[];extra_hashes={}
 for crop in ['melon','cucumber','strawberry']:
  for split in ['train','validation']:
   tag=f'{crop}_{split}'
   try:e=pd.read_csv(O/f'{tag}_source_events.csv.gz')
   except pd.errors.EmptyDataError:continue
   if e.empty:continue
   qpath=B/crop/f'{split}_prepared_qc.parquet';extra_hashes[str(qpath.relative_to(R))]=sha(qpath)
   q=pd.read_parquet(qpath);q.reg_date=pd.to_datetime(q.reg_date)
   for (series_id,var,method),g in e.groupby(['series_id','variable','method']):rawsummary.append({'crop':crop,'split':split,'series':series_id,'source_confidence':'api_transmitted_command' if crop=='cucumber' else 'legacy_command_snapshot_assumption','sensor_source':'prepared QC source_file_path; command confidence is NOT sensor truth','variable':var,'method':method,'QC_feature_rows':len(g),'eligible_feature_rows':int(g.eligible_current.sum()),'lookahead_max_min':g.lookahead_min.max()})
   for rid,g in e.groupby('run_id'):
    x=g.iloc[0];var=x.variable;t=pd.Timestamp(x.current_timestamp);qg=q.loc[q.episode_id.eq(x.episode_id)&q.reg_date.lt(t)].copy();mc=var+'_imputation_method'
    valid=pd.to_numeric(qg[var],errors='coerce').notna()
    for suf in ['_missing_flag','_rule_outlier_flag','_ai_outlier_flag']:
     if var+suf in qg:valid &= pd.to_numeric(qg[var+suf],errors='coerce').fillna(0).eq(0)
    if mc in qg:valid &= ~qg[mc].fillna('').astype(str).str.contains('interpolation',regex=False)
    prev=qg.loc[valid].tail(2)
    past.append({'run_id':rid,'crop':crop,'split':split,'variable':var,'run_start':t,'run_end':g.current_timestamp.max(),'gap_rows':len(g),'last_valid_past_timestamp':str(prev.reg_date.iloc[-1]) if len(prev) else None,'last_valid_past_value':float(prev[var].iloc[-1]) if len(prev) else None,'previous_valid_past_timestamp':str(prev.reg_date.iloc[-2]) if len(prev)>1 else None,'previous_valid_past_value':float(prev[var].iloc[-2]) if len(prev)>1 else None,'hold_last_available':bool(len(prev)),'two_past_points_available':bool(len(prev)>1),'missing_rows':int(pd.to_numeric(g.missing_flag,errors='coerce').fillna(0).ne(0).sum()),'outlier_rows':int(pd.to_numeric(g.rule_outlier_flag,errors='coerce').fillna(0).ne(0).sum()),'gap_minutes_to_last':(t-prev.reg_date.iloc[-1]).total_seconds()/60 if len(prev) else None,'sensor_source_file':str(q.loc[int(x.position)].get('source_file_path','unknown')),'no_reconstruction_applied':True})
 pd.DataFrame(rawsummary).to_csv(O/'QC_source_series_variable_summary.csv',index=False);pd.DataFrame(past).to_csv(O/'past_only_feasibility.csv',index=False)
 # Source scan evidence hashes + actual application metadata; no pickle load.
 sourcefiles=['offline_dataset_preparation/scripts/01_resample.py','offline_dataset_preparation/scripts/03_control_quality.py','src/geas35/preprocessing/3_missing_outliers_handling.py','src/geas35/features/derived.py','src/geas35/features/context.py','src/geas35/rl/mdp_v1.py','src/geas35/rl/vent_command_memory_b_v3.py','src/geas35/rl/sensor_history_env_v4.py','src/geas35/rl/recorded_light_env_v1.py','src/geas35/models/quality/modern_tcn.py','src/geas35/models/quality/patch_tst.py','src/geas35/models/quality/deep/common.py','experiments/rl_policy_training/scripts/build_stale_qc_b60_r6_v1.py','experiments/transition_model_selection/scripts/run_stale_qc_transition_v1.py','src/geas35/models/transition/command_carry_b_v1.py']
 for name in sourcefiles:extra_hashes[name]=sha(R/name)
 applications=[]
 for crop,model in [('melon','modern_tcn'),('cucumber','patch_tst'),('strawberry','modern_tcn')]:
  p=R/f'experiments/quality_control_model_selection/artifacts/{crop}/{model}/quality_model_application.json';extra_hashes[str(p.relative_to(R))]=sha(p);ap=json.loads(p.read_text());applications.append({'crop':crop,'model':ap['model_name'],'path':str(p.relative_to(R)),'observation_columns':ap['observation_columns'],'window':'trailing ending at current; inherited reconstruction for PatchTST'})
 write('quality_application_metadata.json',applications);write('source_review_hashes.json',extra_hashes)
 nr=len(past);pf=pd.DataFrame(past)
 selected=pop[['crop','split','QC_rows','eligible_rows','QC_future_raw_rows','eligible_direct','eligible_propagated_only','eligible_any','eligible_any_pct','current_still_future','target_any','transition_any']]
 wshow=win.copy();wshow['trajectory_pct']=100*wshow.trajectory_any/wshow.total
 text=f'''# QC future interpolation causal leakage blast-radius audit

**최종 범위 판정 B — bounded but pipeline-relevant.** 직접 미래 보간은 소수지만 세 작물 Train의 daily history와 scaler/model/support 입력으로 이어졌다. 과거 metric/판정 artifact는 보존하고 **현재시점 가용성에 관한 causal 주장은 재검증 필요**로 분리한다. 이번에 causal dataset rebuild 필요 범위를 확정하거나 값을 바꾸지 않았다.

## 범위와 용어

공식 B60/R6 3작물 × Train/Validation 6개 모집단을 전수 검사했다. Prepared QC는 Phase0 eligible 이전 population이고 최종 physical은 eligibility 이후다. 원천 센서 물리 오류 여부는 판정하지 않는다. Test 접근, model fit, scaler/support 생성, Step16/domain-v2/C/E gate 실행, reward 변경은 없다.

- **Direct**: 같은 timestamp의 interpolated source가 current 55 observation 또는 같은 행의 derived/control branch에 들어감.
- **Propagated**: 이전 interpolated source의 출처가 이후 누적 feature에 남음. Direct와 겹칠 수 있어 propagated-only도 따로 표시.
- **Ancestry/provenance 영향**: 위 두 유형의 합집합. Endpoint가 이미 도착한 뒤에도 과거 온라인 계산값과 다를 수 있는 offline 이력 출처다.
- **Still-future**: 해당 recipient timestamp보다 right endpoint가 늦음. 이것만이 그 recipient에서 여전히 미래 측정치를 요구하는 경우다.
- Dependency는 값이 달라졌다는 counterfactual 증명이 아니다. Day threshold 결과가 그대로이거나 vent=0이면 수치 효과가 0일 수 있다. 이번에 대체값으로 성능/효과를 계산하지 않았다.

## 1. 코드 전수 조사와 실제 사용

실제 경로는 01_resample → 02_split → 03_control_quality(대표 sensor/QC) → MDP physical/derived → B command carry v4 → B60/R6 admission이다. B60/R6 rebuild는 기존 physical 값을 유지하고 stale/recovery admission을 적용하므로 QC 보간 출처가 사라지지 않는다.

| Method/경로 | 변수 | 좌/우 endpoint | 미래 가능/상한 | 공식 6모집단 사용 |
| --- | --- | --- | --- | --- |
| short_time_linear_interpolation | out_temp, out_hum, out_windsp, out_rainfall, out_airpress | 과거 left + 미래 right | 최대3 invalid행; 연속5분이면 첫행 최대15분. 시간 gap이면 코드에 절대분 상한은 별도 없음 | **wind만** 실제: strawberry Train9/QC·eligible6, Validation13/QC·eligible12 |
| short_circular_time_interpolation | out_winddirec | 양끝 방향의 원형 차이 | 동일 | 0건; wind direction은55열에 없음 |
| short_solar_cycle_time_interpolation | out_light | 과거 left + 미래 right | 동일 | melon Train1/eligible0, cucumber Train1/eligible0, strawberry Train53/eligible10·Validation66/eligible8 |
| short_solar_accumulation_interpolation | out_light_sum | 양끝 누적값 | 동일 | 0건 |
| short_previous_state_persistence | out_rain, etc_plc_norm | 과거만 | 미래 endpoint 없음 | 전체 method label usage는 method_usage.csv; 미래 보간 count에는 제외 |
| 01 causal ffill / 누적광량 적분·추세 외삽 | inserted sensor/command, out_light_sum | 이전 실제 관측/이전2점 | 미래값 미사용 | 기존 경로 유지; 미래 보간과 혼동하지 않음 |
| ModernTCN / PatchTST reconstruction | 내부 T/RH/CO₂ 등8변수 | 현재에서 끝나는 trailing window | 샘플 입력상 미래 row 없음. PatchTST는 동일 reconstruct 상속 | 실제 application metadata와 imputed/대표source 사용량 별도 보존 |
| centered convolution/attention 내부 | 위 trailing window의 hidden representation | window 내부 | window 끝이 현재이므로 미래 관측이 아님 | 새로운 interpolation source로 세지 않음 |
| derived rolling / condensation regression | T/RH 기반 | trailing60/30분 | 10분 forecast는 과거 regression의 외삽, 실제 미래 sensor 주입 아님 | wind/light와 직접 연결 없음 |
| light_eta clear-sky inverse | 날짜·위치·clear sky curve | 하루 뒤쪽 curve/timestamp 검색 가능 | 미래 실측 광량 사용과 다름. offline timestamp/curve 가용성은 별도 검토 한계 | interpolation 전파 count에 포함하지 않음 |

직접 미래 method는 코드의 `_short_linear_interpolation`(03_control_quality.py:269~318)과 분기/label(378~425)에서 확인했다. Active 경로에 추가 bfill/centered rolling/two-sided observed smoothing은 발견하지 못했다. 이는 inspected generation path에 관한 결론이며 저장소 모든 무관한 실험 코드나 외부 producer의 보장은 아니다.

AI model의 학습 파라미터/평균 통계가 전체 Train을 봤는지는 별도 chronological-fit 문제다. 이번 결론은 prediction-time future endpoint에 관한 것이며, AI가 trailing 입력이라는 이유로 모든 historical training-time causality를 승인하지 않는다. 외부 weather 보간은 AI 내부 sensor 계산 **이후** 수행되므로 이번 wind/light 보간이 AI를 통해 T/RH history로 되먹임되는 경로는 없다.

## 2. 전체 직접/전파 영향

{table(selected)}

총 QC future-dependent unique 행은 {int(pop.QC_future_raw_rows.sum())}행, feature-row는 {int(pop.QC_future_feature_rows.sum())}건이다. 최종 eligible direct current는 딸기 Train15/Validation20뿐이다. Ancestry current 합집합은 {int(pop.eligible_any.sum())}행이며 이 중 recipient에서도 right endpoint가 미래인 행은 {int(pop.current_still_future.sum())}행이다. 나머지를 모두 ‘그 시점의 미래 유출’이라고 부르지 않는다.

각 작물/split/source/series 비율은 `source_series_summary.csv`, QC 이전 source/series/variable은 `QC_source_series_variable_summary.csv`다. 여기서 api/legacy는 **command provenance 수준**이며 sensor source의 물리적 진실값과 동일하지 않다. 실제 sensor source_file_path는 past feasibility registry에 함께 보존했다.

{table(series[['crop','split','source','series','total','direct','propagated_only','current_any','current_still_future']])}

### Look-ahead

{table(look)}

모든 source 보간 수식 재현 일치, physical direct mapping mismatch0. `*_source_events.csv.gz`에는 raw value, missing/outlier flag, 좌/우 timestamp와 값, 공식 eligible 포함 여부를 보존했다. No-event split의 빈 event CSV는 영향0을 뜻하며 absent evidence 오류가 아니다.

## 3. 55 observation dependency graph

`all55_dependency_inventory.csv`는 정확한 55개 열 전체에 대해 wind/light 의존 여부를 표시한다. `dependency_graph.mmd`는 아래 관계를 시각화한다.

- Wind → obs_outdoor_wind_speed → 기존55 state/model/support 입력. Wind를 누적하는 rolling feature는 없다. Runtime safety/reward가 wind를 사용할 수 있지만 이번 reward 동작은 재실행하지 않았다.
- Light → obs_outdoor_light → obs_is_daytime. **Solar-period one-hot은 날짜/위치 및 기존 period label 경로**, daylight-condition one-hot은 기존 sky label/sunshine ratio 경로이므로 raw out_light threshold와 구분한다.
- Light → DLI: 현재 light×dt의 날짜별 누적. 6모집단 eligible 전체에서 기록 적분식과 저장 DLI가 일치했다.
- Light → current target min/max/center: explicit current target override가 없을 때만 day/night target 선택 branch에 참여한다. 그 target → heating/cooling degree의 같은 날짜 누적.
- Light → B vent-loss: previous vent×max(T−Tout,0)×(1−clip(light/1000,0,1)). Zero coefficient일 때도 dependency ancestry와 수치 효과를 구분한다.
- Light → light_sum은 **기록 out_light_sum 등이 없을 때만** 적분 fallback; 이번 보간 event 전파에서는 해당 branch 영향이 관측되지 않았다. Raw cumulative가 존재하면 0도 유효값이므로 light 적분으로 임의 교체하지 않는다. Ratio도 그 선택 branch를 따른다.
- Clear-sky sum/light_eta: timestamp/위치/clear-sky/target path이며 실제 interpolated light의 history 경로가 아니다. 미래 timestamp grid 의존 가능성은 interpolation과 별개로 한계 표시.
- **VPD/dewpoint/condensation margin/10분 forecast, low-VPD/RH90, ACH day/night: wind/light not dependent.** T/RH 기반이며 rolling/forecast history에 raw light를 넣지 않는다. Sensor-history buffer에 전체 row가 있어도 이 feature들이 모든 column을 사용하는 것은 아니다.
- Temp diff는 T−Tout로 wind/light not dependent(다만 코드상 out_temp 미래 보간이 있었다면 직접 영향 가능; 이번 실제0). Degree는 target 선택을 통해 light에 조건부 의존한다.
- Support는 55열 전체를 그대로 사용한다. 이번 감사가 feature를 추가/삭제하지 않았다.

각 affected current/target observation별 QC source value, physical mapped value, feature 값, current/recipient/source/right timestamp, direct/history lag는 `*_observation_value_lineage.parquet`에 보존했다. 모든 {int(feat.value_comparisons.sum())}건 값 비교 mismatch0이며 B vent-loss는 previous-command 식으로 확인했다. Degree 등에서 저장 prepared 값과 physical 같다는 것은 대체조건에서 값이 바뀐다는 뜻이 아니다.

## 4. History propagation과 availability

{table(pop[['crop','split','eligible_direct','eligible_propagated','eligible_propagated_only','max_propagation_minutes','max_eligible_propagation_minutes']])}

멜론 2025-05-28 10:05 light1행은 직접 eligible에서 빠졌어도 같은 날짜 DLI/target-degree ancestry가 Train161행에 남았다. 오이 2026-04-22 07:15 light1행도 이후 Train81행에 남았다. 이 recipient들에서 right endpoint는 이미 가용하며 still-future0이다.

딸기 Train 최대 ancestry 전파는 QC 전체 기준1,390분, eligible current 기준1,385분이다. Validation은 QC1,350분/eligible1,125분. Daily cumulative는 자정에 끊긴다. Actual dataset derived는 날짜별 누적이므로 임의 episode reset을 추가해 전파를 줄이지 않았다. 세부 edge에는 lag=0,5,10…별 recipient와 영향 feature가 있어 event별로 추적할 수 있다.

상한은 **실제 자료의 관측 범위**이며 일반적인 영구 horizon이 아니다. 같은 날짜 이력이 계속된다는 구조적 lineage와 미래 endpoint의 도착 시각은 서로 다르다. Low-VPD/RH90/condensation forecast로의 wind/light 전파 count는0이다.

## 5. 기존 transition/window 영향

{table(wshow[['crop','split','horizon','total','current_path_any','target_path_any','trajectory_any','trajectory_pct','trajectory_still_future']])}

원래 3/6/12-step registry start/end와 동일 run을 검증해서만 집계했다. 새로운 window 생성/episode 연결은 없다. Current/target은 trajectory의 모든 observation을 포함하고 양쪽 합집합은 중복 제거했다. 이 target은 **next 55 observation**이며 T/RH/CO₂ 예측 target 자체가 오염됐다는 뜻이 아니다. 이번 실제 external interpolation은 indoor target 값을 직접 바꾸는 경로가 없다.

## 6. Scaler / model / support / Step16 lineage

{table(line[['crop','split','artifact','rows','direct','propagated','current_any','current_any_pct','transition_any']])}

선정 model fit completed metadata의 row 수를 검증하고 B60/R6 runner의 입력 binding 및 `model.fit(train, validation_dataset=None)` 경로를 확인했다. 모델 weight를 로드/fit/predict하지 않았다. Fit inclusion과 성능의 material change는 별개다. Scaler는 fit registry, support는 frozen reference/calibration role registry로 join했다.

Validation one-step은 전체 eligible row, multistep은 위 기존 window registry다. 모델 예측값이나 RMSE를 다시 계산하지 않았다. 학습 데이터에 영향이 있으면 깨끗한 Validation의 예측에도 파라미터를 통해 간접 영향이 가능하지만 그 크기/영향 행 수는 이번 membership만으로 계산할 수 없다.

### 실제 저장 Step16 trace

{table(trace)}

초기 state는 recorded input이므로 직접 lineage를 연결할 수 있다. Free rollout의 endogenous state는 기록값과 다르므로 `recorded_current_any`를 모델 생성 state의 오염 확정 수로 바꾸지 않는다. 저장된 current_physical에서 ancestry feature와 같은 값이 나타나는 행 수를 별도로 보존했다. 딸기 initial6/72, 전체 pretermination 포함11/7,053, 실제 executed current5/7,001에 동일 값 feature가 있었다. 멜론/오이는 Validation 입력의 이 보간 경로는0이지만 fit/scaler/support 간접 lineage는 남는다.

## 7. 기존 PASS 해석과 재검증 범위

| 단계 | 확인된 포함 범위 | 기존 숫자/판정 보존 | causal claim에 대한 판단 |
| --- | --- | --- | --- |
| B60/R6 Phase0 | Current ancestry melon161, cucumber81, strawberry Train1072/Val510; direct strawberry15/20 | 기존 dataset/테스트 결과 보존 | 기존 teacher forcing parity는 유지되는 사실. 당시 이용가능한 관측이라는 claim은 **causal revalidation 필요** |
| Transition Phase1/2 | Fit current 위 Train counts; Val strawberry516 transition union 및 위 window counts | Historical metric preserved, 기존 선정 이력 유지 | 새로운 과거-only 입력 계약이 생기면 입력/fit/evaluation 영향 재검증 필요. 현재 수치 효과 미측정 |
| Step14 | 위 fit으로 만든 bound model/scaler를 인계 | Binding/reload/runtime parity의 historical 사실 보존 | Input causal provenance는 upstream 영향 상속, handoff causal 범위 재검증 필요 |
| Step15 | Reference melon63/cucumber68/strawberry340, calibration0/0/17 current ancestry | 기존 threshold/reference/거리 유지 | fit input availability 검증 필요. Reference 변경/threshold 완화 승인 아님 |
| Step16 melon | Val 직접/전파0, upstream scaler/model/reference 영향 존재 | 기존 연구 실행가능성 PASS 이력·수치 보존 | 새 input contract 후 upstream 영향 범위에 따라 재검증; 현재 실행 안 함 |
| Step16 cucumber | Val 직접/전파0, upstream fit/support 영향 | 기존 연구 PASS 이력 보존 | 위와 같음; 장치 안전성 승인 아님 |
| Step16 strawberry | Recorded Val current510, stored initial6/72 및 일부 trace lineage | 기존 BLOCKED와 historical 결과 유지 | 기존 OOD 원인이 전부 이 보간 때문이라고 재분류하지 않음. Causal input 재검증 필요 |

여기서 ‘재검증 필요’는 과거 PASS 파일을 이번에 FAIL로 덮어썼다는 뜻이 아니다. 정확한 표현은 **historical metric preserved / causal availability gate requires revalidation**이다.

## 8. Past-only 재구성 가능성 — 방법 선택 안 함

`past_only_feasibility.csv`에 {nr}개 affected run의 last/previous valid 같은 센서값, timestamp, past age, gap row 수, missing/outlier 구분, sensor source 파일을 보존했다. Last-valid available {int(pf.hold_last_available.sum())}/{nr}, two past points available {int(pf.two_past_points_available.sum())}/{nr}. 실제 값 대입은 하지 않았다.

| 후보 | 구현 가능성 | 남는 판단 |
| --- | --- | --- |
| Past-only hold-last-valid | 위 past value가 있는 run에서 계산 가능 | 허용 age/gap/source boundary, validity/불확실성을 사전 정의해야 함. 물리 정확성 승인 아님 |
| Current invalid/quarantine | Sidecar로 표시 가능 | 직접 row만 지우면 DLI/degree와 target/window/fit lineage가 남으므로 propagation 계약 필요 |
| 두 past point 기반 one-sided 추세 | 두 과거점이 있으면 계산 가능 | 현재 이후를 추정하는 것은 **interpolation이 아니라 extrapolation**. 새 공식 방법으로 채택하지 않음 |
| 기존 causal 경로 재사용 | 01 causal ffill, 누적광량 과거 적분/외삽, short_previous_state_persistence 구현 존재 | 해당 변수/주기에 그대로 적합하다는 승인은 아님 |

Sensor2/평균/상수로 교체하지 않았다. Future endpoint가 실제 당시 알려져 있었다고 주장하지 않는다. 물리 sensor 고장/정상 판정도 하지 않는다.

## 9. E/C 봉인과 audit exposure

이번 감사는 사용자 승인에 따라 **strawberry 전체 Train QC를 포함**하므로 v2 C/E 기간과 겹치는 source·lineage 값을 읽었다. 이를 ‘해당 기간 어떤 값도 읽지 않았다’고 보고하지 않는다. 별도 namespace/access ledger에 이 추가 exposure를 명시했다.

반면 **domain-v2 C/E recipe/admission/gate는 전혀 실행하지 않았다.** 기존 v2 result/lock/registry 파일 byte hash는 동일하며 E final rule lock도 만들지 않았다. Domain-v2 E 평가 횟수는 여전히0이다. 향후 E를 historically unseen이라고 부를 수 없다는 기존 제한이 유지·강화된다. 이번 Validation 읽기는 contamination membership 감사에만 사용했고 성능 tuning은 없다.

## 10. 최종 범위와 다음 조치 하나

**B: bounded but pipeline-relevant.** Active interpolation은 실제2변수·총 {int(pop.QC_future_raw_rows.sum())} QC unique행으로 제한되고, 최종 current ancestry 비율은 melon Train0.56%, cucumber Train0.57%, strawberry Train2.43%/Validation2.88%다. Daily history까지 이어지므로 A처럼 직접 몇 행에만 국한됐다고 할 수 없다. 다만 이번 증거만으로 모든 시기·모든 history에 광범위한 systemic contamination이나 전면 rebuild가 반드시 필요하다고 C를 확정하지 않는다.

**추천 최소 후속 작업 하나: 광량·풍속의 past-only reconstruction shadow 계약 실험.** 과거에 실제 가용했던 정보만으로 가능한 후보를 별도 사전 정의하고, 원본을 유지한 shadow에서 current뿐 아니라 DLI/target-degree/history 가용성을 비교하는 것이 우선이다. 이번 결과로 어떤 대체법을 공식 선택하거나 성능이 좋아지는 방법을 고르지 않는다. 재구축·재학습 여부는 해당 causal 계약과 영향 범위를 확정한 후 판단한다.

DOMAIN-ONLY BLOCKED(D input integrity), D geometry/independence NOT RUN, C NOT RUN, E gate 미개봉, strawberry Step16/전체 Step17/online deployment BLOCKED를 유지한다. High-CO₂/high-vent 부족, unseen series, legacy assumption, clean5 accumulation, reward imbalance도 유지한다.

## 산출물·검증 제한

Source event/endpoint 수식 재현, current/target physical mapping, DLI 적분 전체 일치, 기존 window boundary, fit/support membership, 입력·v2 원본 hash 보존을 확인했다. 보강 단계의 영향0 split 빈 CSV 처리 오류는 `enrichment_attempt_1`에 보존하고 loader만 수정했다. 공식 데이터·recipe 변경은 없다.

본 전수 감사의 정확한 범위는 frozen prepared QC의 method/flag와 현재 확인된 생성 코드의 일치다. 외부 producer 원본 timestamp의 진실성, 학습된 QC 모델의 모든 parameter-time lineage, unknown upstream feature 생성은 독립 보장이 아니다. Dependency count는 conservative branch ancestry이며 모델 성능 영향/비0 counterfactual 효과가 아니다. 이 한계를 감춘 blanket causal PASS는 부여하지 않는다.
'''
 (O/'final_decision_ko.md').write_text(text)
 # Do not modify any official artifact. Verify prior source hashes again.
 for n in ['input_hashes.json','enrichment_input_hashes.json','source_review_hashes.json']:
  for path,h in json.loads((O/n).read_text()).items():assert sha(R/path)==h
 for x in json.loads((O/'v2_original_hashes.json').read_text()):assert sha(R/x['path'])==x['sha256']
 write('final_gate.json',{'audit_classification':'B bounded but pipeline-relevant','historical_metrics_preserved':True,'causal_availability_revalidation_required':True,'domain_v2_reexecuted':False,'C_E_gates_executed':False,'audit_source_exposure_overlaps_C_E_dates':True,'Test_accessed':False,'official_artifacts_modified':False,'recommendation':'light/wind past-only reconstruction shadow contract experiment','all_current_blockers_maintained':True})
 write('artifact_manifest.json',{'report_script_sha256':sha(Path(__file__)),'files':[{'path':str(p.relative_to(O)),'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json']})
 print('REPORT COMPLETE',nr,'runs',int(feat.value_comparisons.sum()),'mapped checks')
if __name__=='__main__':main()
