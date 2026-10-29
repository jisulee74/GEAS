from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/transition_model_selection/artifacts/strawberry_co2_diagnosis_b_v1';P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4/strawberry';A=ROOT/'experiments/transition_model_selection/artifacts/command_carry_b_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str,allow_nan=True)+'\n')
h=pd.read_csv(D/'horizon_metrics.csv');ep=pd.read_csv(D/'episode_metrics.csv');g=pd.read_csv(D/'group_metrics.csv');steps=pd.read_csv(D/'step_metrics.csv');oracle=pd.read_csv(D/'feedback_oracle_metrics.csv');w=pd.read_parquet(D/'window_decomposition.parquet');t=pd.read_parquet(D/'step_trace.parquet');actions=pd.read_parquet(D/'action_response_rows.parquet');coverage=pd.read_csv(D/'population_group_counts.csv')
b=pd.read_parquet(P/'validation_physical.parquet');q=pd.read_parquet(P/'validation_prepared_qc.parquet');pred=pd.read_parquet(A/'strawberry/lightgbm/fit/validation_predictions.parquet').obs_indoor_co2_ppm.to_numpy();truth=b.target_next_indoor_co2_ppm.to_numpy();cur=b.obs_indoor_co2_ppm.to_numpy();error=pred-truth
one={'rows':len(b),'model_rmse':float(np.sqrt(np.mean(error**2))),'persistence_rmse':float(np.sqrt(np.mean((cur-truth)**2))),'model_mae':float(np.abs(error).mean()),'persistence_mae':float(np.abs(cur-truth).mean()),'model_bias':float(error.mean())}
reg=pd.DataFrame({'co2_group':pd.cut(cur,[-np.inf,400,600,800,1000,1500,np.inf],right=False).astype(str),'stable_step_abs_delta_lt5':np.abs(truth-cur)<5,'actual_delta':truth-cur,'predicted_delta':pred-cur,'error':error})
rows=[]
for (co2,stable),z in reg.groupby(['co2_group','stable_step_abs_delta_lt5']):rows.append({'co2_group':co2,'stable_step_abs_delta_lt5':bool(stable),'rows':len(z),'actual_delta_mean':z.actual_delta.mean(),'predicted_delta_mean':z.predicted_delta.mean(),'bias':z.error.mean(),'rmse':float(np.sqrt(np.mean(z.error**2)))})
pd.DataFrame(rows).to_csv(D/'one_step_regime_bias.csv',index=False)
# Distinguish endpoint-stable from truly stable over every intermediate timestamp.
last=t.loc[t.horizon_minutes.eq(60)].copy();last['true_change_abs']=(last.truth_co2-last.initial_co2).abs();last['teacher_persistence_error']=last.recorded_current_co2-last.truth_co2
stability=last.groupby('window_id').true_change_abs.max().rename('max_true_change').reset_index();ww=w.loc[w.horizon.eq(60)].merge(stability,on='window_id',validate='one_to_one');ww['entire_trajectory_stable_lt5']=ww.max_true_change.lt(5)
true_stable=[]
for stable,z in ww.groupby('entire_trajectory_stable_lt5'):
 true_stable.append({'entire_trajectory_stable_lt5':bool(stable),'windows':len(z),'model_rmse':float(np.sqrt(z.model_sse.sum()/z.steps.sum())),'persistence_rmse':float(np.sqrt(z.persistence_sse.sum()/z.steps.sum())),'model_sse_share':float(z.model_sse.sum()/ww.model_sse.sum()),'bias':float(z.model_error_sum.sum()/z.steps.sum())})
write(D/'trajectory_stability.json',true_stable)
z=ep.loc[ep.horizon.eq(60)].sort_values('excess_sse',ascending=False);top=z.head(2);total=h.loc[h.horizon.eq(60)].iloc[0]
summary={'one_step':one,'top2_episodes':top.episode.tolist(),'top2_window_share':float(top.windows.sum()/z.windows.sum()),'top2_model_sse_share':float(top.model_sse.sum()/z.model_sse.sum()),'global_bias_square_fraction_of_mse':float(total.bias**2/total.model_rmse**2),'teacher_persistence_60min_rmse':float(np.sqrt(np.mean(last.teacher_persistence_error**2))), 'overall_status':'BLOCKED unchanged','new_fit':False,'feature_change':False,'Test_accessed':False}
# Original values/flags only from frozen Train/Validation; no data quality correction.
qcrows=[];joint=[];plateaus=[]
for split in ('train','validation'):
 qc=pd.read_parquet(P/f'{split}_prepared_qc.parquet');pr=pd.read_parquet(P/f'{split}_provenance.parquet');high=qc.obs_indoor_co2_ppm.ge(1500)
 for pop,mask in [('eligible',pr.eligible),('excluded',~pr.eligible)]:
  sel=high&mask;joint.append({'split':split,'population':pop,'high_co2_rows':int(sel.sum()),'high_co2_high_vent_rows':int((sel&pr.current_vent_pct.ge(.75)).sum()),'high_co2_command_unknown_rows':int((sel&pr.current_vent_pct.isna()).sum())})
 if split=='validation':
  for episode in top.episode:
   block=qc.loc[qc.episode_id.eq(episode)].copy();cols=['reg_date','episode_id','obs_indoor_co2_ppm','obs_outdoor_light','in_co2_raw_value','in_co2_2_raw_value','in_co2_representative_source']+[c for c in qc if 'co2' in c and 'flag' in c];qcrows.append(block[cols])
   change=block.obs_indoor_co2_ppm.ne(block.obs_indoor_co2_ppm.shift())|(block.reg_date.diff()!=pd.Timedelta(minutes=5));rid=change.cumsum()
   for _,segment in block.groupby(rid):
    if len(segment)>=3:plateaus.append({'episode':episode,'start':str(segment.reg_date.iloc[0]),'end':str(segment.reg_date.iloc[-1]),'rows':len(segment),'co2':float(segment.obs_indoor_co2_ppm.iloc[0]),'all_raw_sensor1_equal_representative':bool(segment.in_co2_raw_value.eq(segment.obs_indoor_co2_ppm).all()),'sensor1_imputed_rows':int(segment.in_co2_imputed_flag.fillna(0).astype(bool).sum()),'light_unique_values':int(segment.obs_outdoor_light.nunique())})
pd.concat(qcrows).to_csv(D/'worst_episode_qc_evidence.csv.gz',index=False,compression='gzip');write(D/'high_co2_joint_coverage.json',joint);write(D/'worst_episode_plateaus.json',plateaus);write(D/'diagnosis_summary.json',summary)
classification=[
 {'cause':'학습 coverage 부족','status':'확인됨','evidence':'Train >=1500ppm eligible16 vs excluded2251; Validation eligible202. 고농도+vent>=75% Train0 vs Validation185.','limit':'A에서도 고농도32행뿐이며 B에서16행으로 감소. QC 고농도 제외2251행 중2235행은 이미 A에 없었다. B 축소만의 성능 인과 효과는 미확인.'},
 {'cause':'특정 episode/시간대 분포 문제','status':'확인됨','evidence':f"두 episode가 60분 window {summary['top2_window_share']:.2%}에서 모델 SSE {summary['top2_model_sse_share']:.2%}를 차지.",'limit':'기록 plateau는 확인했지만 실제 안정 상태/수집 정체/품질 문제는 구분되지 않음. day/night은 동결 obs_is_daytime 정의.'},
 {'cause':'action-response 부족','status':'확인됨','evidence':'전체 Train vent 변화 >=10pp 1916건으로 전역적으로 부족하지 않음. >=1500ppm에서는 닫힘2, 열림0, 고농도+대개도 표본0.','limit':'국소 표본 부족을 확인했으며 실제 환기 인과 효과는 명령 snapshot/미관측 제어·생육 요인과 혼재.'},
 {'cause':'one-step 모델 bias','status':'확인됨','evidence':'전체 one-step CO2 RMSE 41.270 vs Persistence41.000; 고농도 안정 구간의 하락 예측과 저농도 일부 구간의 상승 예측.','limit':'global mean bias만으로 전체 SSE를 설명할 수 없음; regime별 반대 bias가 상쇄됨.'},
 {'cause':'autoregressive error accumulation','status':'확인됨','evidence':'CO2 oracle feedback에서 60분 RMSE 31.490, T/RH oracle에서는101.177, free101.201.','limit':'oracle은 미래 관측 주입 진단이므로 배포 성능이나 대체 모델 점수가 아님. feedback이 항상 증폭한다는 뜻도 아님.'},
 {'cause':'현재 feature/model 구조 한계','status':'유력','evidence':'고농도 Validation202행이 75개 tree에서 동일 leaf 경로, Δ=-124.713ppm. 해당 leaf Train count41~44, min_child_samples39, 고농도 Train16행.','limit':'LightGBM 구조 자체의 불가피한 한계와 학습 coverage/목표 정의를 분리할 근거 부족; feature 변경 효과도 미검증.'},
 {'cause':'원천 CO2 plateau 물리적 의미 및 누락 제어','status':'근거 부족','evidence':'2025-11-11 원천 보존 sensor1=1942, sensor2=2730 및 광량715 반복, 해당 구간 보간 flag0.','limit':'원천값 반복이 실제 안정인지 장치/수집 정체인지 저장 자료만으로 확정하지 않음; 임의 삭제/보정 금지.'}]
write(D/'cause_classification.json',classification)
lines=['# Strawberry B CO₂ 비학습 원인 진단',
'**진단 완료. 기존 strawberry/전체 Phase 1/2 BLOCKED는 유지한다.** 새 모델·HPO·feature 변경·기준 변경·episode 제외 없이 고정 예측과 Train/Validation 자료만 분석했다.\n',
'## 핵심 판단\n',
f"단순한 전역 평균 bias 하나보다 **희소 고농도/대개도 구간과 두 episode에 집중된 잔차의 누적**이 주된 관측이다. 2025-11-10/11 두 episode는 60분 window의 {summary['top2_window_share']:.2%}이지만 LightGBM CO₂ 제곱오차의 {summary['top2_model_sse_share']:.2%}를 차지한다. Train에서 CO₂≥1500ppm은 B eligible16행, QC excluded2251행이며 고농도+vent≥75%는 eligible0행이다. 기존 A에도 고농도는32행뿐이었다. Validation eligible에는 각각202/185행이 있다.\n",
'고정 후보군에서는 이미 LightGBM이 best eligible이며 통과 모델이 없었다. 동일 설정 반복이나 후보 재선택만으로 해결됐다는 근거는 없다. 다만 이것이 새 모델 family/feature 추가가 반드시 필요하다는 증거도 아니다. 원천 기록과 coverage 문제를 먼저 검증하고, 필요하면 별도의 사전 정의 데이터/학습 목표 또는 model/feature 실험으로 다뤄야 한다. 이번에는 실행하지 않았다.\n',
'## 1. one-step과 누적 예측\n',
f"전체 eligible Validation {one['rows']:,}행의 CO₂ one-step RMSE는 LightGBM {one['model_rmse']:.3f}, Persistence {one['persistence_rmse']:.3f}ppm이다. MAE는 각각 {one['model_mae']:.3f}/{one['persistence_mae']:.3f}ppm이다. 따라서 CO₂ one-step이 완전히 양호하다고 말할 수는 없으며 이미 약간 열세다.\n",
'| horizon | windows | free RMSE | Persistence RMSE | skill | teacher-forced RMSE | free bias | endpoint RMSE |\n|---|---:|---:|---:|---:|---:|---:|---:|']
for _,x in h.iterrows():lines.append(f"| {int(x.horizon)}분 | {int(x.windows):,} | {x.model_rmse:.3f} | {x.persistence_rmse:.3f} | {x.skill:.6f} | {x.teacher_rmse:.3f} | {x.bias:.3f} | {x.endpoint_rmse:.3f} |")
lines += [f"\n동일한 60분 모집단 내 첫 step RMSE 37.714 → 마지막144.503ppm. 마지막 teacher-forced RMSE30.934ppm이다. 60분 전체 평균 bias -4.613ppm의 제곱은 전체 MSE의 {summary['global_bias_square_fraction_of_mse']:.2%}뿐이다. 작은 평균 bias의 선형 누적만으로 실패를 설명할 수 없고 특정 구간의 큰 일관된 오차와 반대 방향 bias 상쇄를 봐야 한다.\n",
'저장된 full one-step 예측을 동일 window의 각 기록 상태에 정렬해 teacher forcing을 만들었다. teacher-forced는 매번 정답 상태를 공급하므로 실제 60분 예측 성능과 동등 비교 대상이 아니다. step_trace의 summed_teacher_error는 각 기록 상태에서 예측한 ΔCO₂를 누적한 진단값이며 누적 one-step error와 정확히 일치함을 검사했다.\n',
'| diagnostic oracle | 60분 RMSE | endpoint RMSE |\n|---|---:|---:|']
for _,x in oracle.loc[oracle.horizon.eq(60)].iterrows():lines.append(f"| {x['mode']} | {x.rmse:.3f} | {x.endpoint_rmse:.3f} |")
lines += ['\nCO₂ 입력만 기록값으로 되돌리면 오차 대부분이 사라지고 T/RH만 되돌려서는 거의 바뀌지 않는다. 따라서 이 실패는 T/RH 오차의 간접 영향보다 CO₂ 자체의 반복 예측 경로와 더 밀접하다. 주입 전 예측을 평가했으며 모델/feature를 바꾸지 않았다. 기록 상태 Δ를 누적했을 때 마지막 RMSE169.036ppm보다 free endpoint144.503ppm이 오히려 작다. 이는 feedback이 무조건 오차를 증폭한다는 설명도 부정하며, 일부 상태 의존 보정은 오차를 줄이고 있음을 뜻한다.\n',
'## 2. episode 집중도\n','| episode | windows | model RMSE | Persistence RMSE | bias | 모델 SSE 비중 |\n|---|---:|---:|---:|---:|---:|']
for _,x in z.head(5).iterrows():lines.append(f"| {x.episode} | {int(x.windows)} | {x.model_rmse:.3f} | {x.persistence_rmse:.3f} | {x.bias:.3f} | {x.model_sse/z.model_sse.sum():.2%} |")
lines += ['\n모든66개 episode를 그대로 유지했다. episode_concentration.json에는 top1/3/5/10 기여도가 있다. net excess SSE 비중은 다른 episode의 개선분이 상쇄하므로100%를 넘을 수 있다. 독립 표본 수나 인과 기여율로 해석하지 않는다. sliding window의 겹침을 그대로 유지했으므로 반복 관측의 상관이 있다.\n',
'## 3. 시간대·생육·초기 상태\n','| dimension / group | windows | model RMSE | Persistence RMSE | skill |\n|---|---:|---:|---:|---:|']
for _,x in g.loc[g.horizon.eq(60)&g.dimension.isin(['day_night','dat','initial_co2'])].iterrows():lines.append(f"| {x.dimension}: {x['group']} | {int(x.windows)} | {x.model_rmse:.3f} | {x.persistence_rmse:.3f} | {x.skill:.4f} |")
lines += ['\nhour/date/light/vent/실제 변화량별 전체 수치는 group_metrics.csv에 있다. day/night은 시계시간이 아니라 기존 obs_is_daytime(동결 광량 기준)을 사용한다. 특히 worst 기록에는 자정에도 광량715가 반복되므로 이를 실제 주간 농업 현상으로 단정하면 안 된다. 변화량 그룹은 사후 정답 기반 진단이며 학습/선정에 사용하지 않았다.\n',
'## 4. 학습 모집단과 action-response\n',
'기존 A Train90,861 → B44,102행. 이번 excluded49,409행은 QC 전체93,511행을 분모로 한 집합이며 A→B 감소46,759행과 동일 집합이라고 주장하지 않는다. CO₂≥1500 구간 QC Train2,267행 중16행만 eligible(약0.71%)로 남는다. 이는 QC 전체47.16% retention보다 훨씬 낮다. 다만 실제 A/B 행 key를 대조하면 A 고농도32행 중16행이 B에서 유지됐고16행이 추가 제외됐다. QC 제외 고농도2251행 중2235행은 이미 A에도 없었다. 따라서 B 계약이 고농도 부족을 처음 만들었다는 주장은 틀리며, 기존 희소성이 B에서 더 줄어든 것으로 구분한다. A scaled CO₂를 기존 scaler로 역변환해 동일 QC 물리값과 일치함도 검증했다. A_B_train_membership.json/parquet에 실제 교집합과 감소 행을 보존했다. excluded 데이터를 재투입하지 않았다.\n',
'고농도/대개도 공동 상태는 Train0, Validation185행이다. Train vent 변화10pp 이상은 열림969/닫힘947, 합1,916건(eligible의 약4.34%)이므로 전체 action support가 없다는 설명은 맞지 않는다. 하지만 고농도에서는 열림0/닫힘2건으로 매우 부족하다. action_response_groups.csv는 hour/day-night/DAT/CO₂/light별 빈도와 command 변화 전후5분 CO₂ 변화량을 보존한다. 명령은 실제 개도 센서가 아니고 통제되지 않은 관측이므로 환기의 인과 효과 추정으로 해석하지 않는다.\n',
'eligible/excluded의 CO₂·ΔCO₂·광량·T/RH·vent·DAT 분포 및 missing 수는 population_distribution.csv, 그룹별 표본수는 population_group_counts.csv, 오차와 Train 표본수 결합은 coverage_error_overlap.csv에 있다. unknown command는 0/현재 값으로 채우지 않았다. 일부 그룹의 0/희소 표본은 단변량 또는 고농도+대개도 공동 coverage 부족의 직접 증거지만, 제외가 오차를 얼마나 증가시켰는지는 재학습 대조 없이 확정하지 않는다.\n',
'## 5. Persistence가 유리한 이유와 기록 plateau\n',
'고정 CO₂ LightGBM의 75개 tree 경로를 확인한 결과, Validation 고농도202행은 모두 같은 leaf 경로이고 예측 ΔCO₂는 정확히 -124.713416ppm으로 같았다. 그중 안정184행의 실제 ΔCO₂는0이다. 해당 경로의 Train leaf count는41~44이며 min_child_samples39, Train 고농도는16행이다. 이는 이 fitted 모델에서 고농도 지속 상태를 별도 구분하지 못한다는 직접 증거이나 LightGBM family의 본질적 한계를 입증하지 않는다. high_co2_leaf_audit.json에 보존했다.\n',
'고농도 기록이 거의 유지되는 window에서 Persistence는 그 수준을 유지하지만 LightGBM은 하락 잔차를 반복한다. 낮은 CO₂ 안정 episode에서도 작은 상승 bias가 누적되어 Persistence가 유리하다. 60분 내내 초기값 대비 변화가5ppm 미만인7771개 window에서 모델 RMSE127.871ppm, Persistence1.816ppm이며 모델 전체 SSE의72.50%를 차지한다. 이 집계에는 희소 고농도 plateau가 포함돼 있으므로 일반적인 모든 안정 상태가 그만큼 나쁘다는 뜻은 아니다. 전체 경로 안정성은 endpoint가 돌아오는 경우와 구분해 trajectory_stability.json에 max|CO₂(t+k)−CO₂(t)|<5ppm으로 별도 기록했다.\n',
'worst 두 episode의 raw-value/품질 flag를 보존했다. 2025-11-11 00:00~01:00에서 sensor1 원천 보존값1942, sensor2 2730, representative source=sensor1, 광량715가 반복되며 해당 구간의 missing/ffill/imputed flag는0이다. 따라서 단순 QC 보간으로 생겼다고 단정할 수 없고, 실제 안정 상태인지 수집 정체인지도 이 자료만으로 구분되지 않는다. 원천 기록을 임의 수정하거나 episode를 제거하지 않았다.\n',
'## 6. 원인 분류\n','| 분류 | 상태 | 근거 / 제한 |\n|---|---|---|']
for x in classification:lines.append(f"| {x['cause']} | {x['status']} | {x['evidence']} {x['limit']} |")
lines += ['\n## 7. 최소 비용 후속 순서\n',
'1. **비학습 원천 감사:** 두 worst 날짜와 고농도 Train 제외 구간의 원본 관측 갱신 timestamp/수집 주기/센서 선택/QC 경로를 확인한다. plateau의 물리적 안정과 원천 갱신 정체를 구분한다. 동결 자료만으로 확정 불가한 부분은 명시하며 임의 수정하지 않는다.\n',
'2. **비학습 국소 진단:** 같은 고정 모델의 고농도+개도 조건, DAT/series, 지속 plateau/급변 전후를 나눠 예측 ΔCO₂와 기록 ΔCO₂를 대조한다. 이미 생성한 trace·coverage 표를 우선 사용하며 새 모델 탐색은 하지 않는다.\n',
'3. **별도 사전 정의 실험 필요성 결정:** 원천/eligibility 결함이 실제 확인되면 계약을 지키는 데이터 재구축·Phase0 검증을 먼저 설계한다. 결함 없이 실제 희소 regime라면 데이터 coverage 보강 또는 현 모델 학습 목표/다중 step 훈련 대조를 별도 계획으로 고정한다. excluded 명령을 임의 복원해 재투입하는 방안은 아니다.\n',
'4. **구조 실험은 마지막:** 위 대조 뒤에도 지속되는 경우에만 별도 사전 정의 model/feature 실험이 필요한지 결정한다. 현재 증거로 특정 새 모델·feature가 해결한다고 약속할 수 없다. 현재 고정 설정/후보군 재선택만으로 PASS를 만드는 해법은 확인되지 않았다. 이번에는 어떤 추가 학습도 실행하지 않았다.\n',
'## 산출물과 검증\n',
'step_trace.parquet/csv.gz: 모든 window의 step별 초기값·기록값·LightGBM/Persistence/teacher 예측·bias/절대오차·endpoint 표시. window_decomposition/episode_metrics/group_metrics: 모집단 전체 분해. 원래 CO₂ RMSE 재현, window registry 수, teacher error 누적 항등식, 입력 hash 유지 검증 완료. oracle 결과는 선택 점수/모델로 사용하지 않았다.\n',
'![진단 비교](diagnosis_overview.png)\n',
'기존 B Phase0 PASS, strawberry/전체 Phase1/2 BLOCKED 및 Step19.5 미통과를 유지한다. Test, fit, HPO, 새 후보/feature, support 생성, Step14/15, PPO를 수행하지 않았다.\n']
(D/'final_decision_ko.md').write_text('\n'.join(lines))
# Standard standalone scientific plot.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(15,4))
s=steps.loc[steps.horizon.eq(60)];ax[0].plot(s.step*5,s.model_error_rmse,label='Free rollout');ax[0].plot(s.step*5,s.persistence_error_rmse,label='Persistence');ax[0].plot(s.step*5,s.teacher_error_rmse,label='Teacher forced');ax[0].set(xlabel='Minutes',ylabel='CO2 RMSE (ppm)',title='Same 60-min window population');ax[0].legend()
ax[1].bar(['Top 2 episodes','All other episodes'],[summary['top2_model_sse_share']*100,(1-summary['top2_model_sse_share'])*100]);ax[1].set(ylabel='Model squared-error share (%)',title='Top 2: 3.20% of windows')
ax[2].bar(['Train high CO2','Val high CO2','Train high + vent','Val high + vent'],[16,202,0,185]);ax[2].tick_params(axis='x',rotation=30);ax[2].set(ylabel='Eligible rows',title='CO2 >=1500; vent >=75%')
fig.tight_layout();fig.savefig(D/'diagnosis_overview.png',dpi=160);plt.close(fig)

assert not t.duplicated(['horizon_minutes','window_id','step']).any()
assert t.is_endpoint.sum()==len(w)
for hh in (15,30,60):
 subset=t.loc[t.horizon_minutes.eq(hh)];assert subset.groupby('window_id').size().eq(hh//5).all()
 assert subset[['truth_co2','lightgbm_co2','persistence_co2','teacher_forced_co2']].notna().all().all()
leaf=json.loads((D/'high_co2_leaf_audit.json').read_text());assert leaf['unique_high_leaf_paths']==1 and leaf['validation_high_rows']==202
write(D/'final_verification.json',{'diagnosis_complete':True,'all_horizon_registry_windows':len(w),'trace_rows':len(t),'unique_horizon_window_step':True,'one_endpoint_per_window':True,'stored_predictions_finite':True,'oracle_experiments':'diagnostic only, not new candidates or scoring changes','model_leaf_audit_complete':True,'original_BLOCKED_unchanged':True})

for x in json.loads((D/'input_manifest.json').read_text()):assert sha(ROOT/x['path'])==x['sha256']
for x in json.loads((A/'artifact_integrity.json').read_text()):assert sha(A/x['path'])==x['sha256']
scripts=sorted(set((ROOT/'experiments/transition_model_selection/scripts').glob('*strawberry_co2*b_v1.py'))|set((ROOT/'experiments/transition_model_selection/scripts').glob('*strawberry_co2*_v1.py')));write(D/'code_manifest.json',[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in scripts]);write(D/'artifact_integrity.json',[{'path':str(p.relative_to(D)),'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file() and p.name!='artifact_integrity.json'])
print(json.dumps(summary,indent=2));print('report complete')
