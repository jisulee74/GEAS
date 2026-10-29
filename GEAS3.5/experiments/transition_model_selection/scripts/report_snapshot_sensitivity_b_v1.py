from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/transition_model_selection/artifacts/stale_snapshot_sensitivity_b_v1'
def table(df):
 cols=list(df.columns);lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
 for row in df.itertuples(index=False,name=None):lines.append('| '+' | '.join(str(v) for v in row)+' |')
 return '\n'.join(lines)+'\n'
s=pd.read_csv(D/'duration_sensitivity.csv');m=pd.read_csv(D/'metric_sensitivity.csv');scores=pd.read_csv(D/'weighted_score_sensitivity.csv');c=pd.read_csv(D/'high_train_attribution_with_unknown.csv');short=pd.read_csv(D/'short_repeat_reference.csv');dis=pd.read_csv(D/'sensor_disagreement_overlap.csv');error=pd.read_csv(D/'masked_error_concentration.csv')
text=['# Raw repeated-snapshot 비학습 QC sensitivity — 최종 판단\n',
'''**이번 판단: A — 일반적인 raw 반복 규칙이 문제 구간을 식별하므로, 별도 stale-snapshot QC 계약 실험을 설계할 근거가 충분하다.** 이는 실제 센서/수집 정체를 확정했다는 뜻이 아니다. 물리 원인에 관한 앞선 source audit **C(undetermined)**는 유지한다. 정상/고장 ground truth가 없어 specificity나 실제 오탐률을 확정할 수 없다.

**고정 상태:** `vent_command_carry_b_v4` 및 딸기 Phase0 PASS 유지, 딸기/전체 Transition Phase1/2 BLOCKED 유지. 기존 LightGBM/Persistence 예측·점수·선정 결과 그대로다. 외부 문의, 모델 load/predict/fit, Test, HPO, feature 변경, dataset/scaler 재생성, episode 삭제, Step14/15·PPO·Step19.5 실행은 하지 않았다.

**규칙을 결과 전에 고정한 방식**

[protocol.json](protocol.json)과 hash lock을 생성한 뒤, 세 작물 Train/Validation의 기존 raw 파일을 동결 QC timestamp/series와 정확히 연결했다. 값 비교에는 원천 숫자만 사용했고 QC 대표값·episode ID·특정 날짜·모델 오차는 사용하지 않았다.

- signature A: raw CO₂1 + CO₂2 exact repeat.
- signature B: raw CO₂1 + CO₂2 + T1 + RH1 + light exact repeat.
- 같은 crop/series/segment/split, 원천 timestamp 간격이 정확히5분인 경우만 연결한다. gap에서 끊고 결측/NaN을 동일 값 반복으로 세지 않는다. resampling/ffill로 삽입된 행은 탐지 입력에서 제외한다. 날짜/일별 episode 경계 자체는 끊지 않는다.
- duration은 마지막−첫 raw timestamp이며 cutoff 이상(≥)을 표시한다. 30분은7개 raw row가 필요하다. **30/60/120/180분 모두 진단하고 공식 threshold를 선택하지 않았다.** split/관측 gap에서 잘린 run은 관측 가능한 지속시간만 사용한다.
- 완료된 run 전체를 표시하는 **사후 진단**이다. run 시작 시점에는 이후 지속시간을 알 수 없으므로 이 mask를 실시간 causal QC로 바로 쓸 수 없다. 실시간 탐지 지연과 과거 행 처리 방침은 후속 계약 실험의 별도 항목이다.

**세 작물 탐지 분포**

모수는 동결 QC Train/Validation 전체다. 괄호의 raw 모수는 실제 원천 row가 존재하는 행만 센 것이다. 표의 고농도 수는 QC 대표 CO₂≥1500이며, raw sensor1 기준 고농도 수는 CSV에 별도로 있다. 둘은 QC sensor 선택 때문에 같지 않을 수 있다.

| 작물/split | QC 모수 | raw 모수 |
|---|---:|---:|
| melon Train | 29,088 | 29,085 |
| melon Validation | 6,048 | 6,048 |
| cucumber Train | 23,616 | 23,615 |
| cucumber Validation | 4,896 | 4,896 |
| strawberry Train | 93,511 | 69,559 |
| strawberry Validation | 19,872 | 19,870 |

멜론 Train은 A/B 모든 cutoff에서 **1 run·47행(0.1616%), 고농도47행**이며 이미 B eligible에는0행이다. 멜론 Validation 및 오이 Train/Validation은 A/B 모든 cutoff에서0 run·0행이다. 정상이라는 인증이 아니라 이 exact-repeat 규칙의 미검출이다.

딸기 결과:
''']
t=s.loc[s.crop.eq('strawberry'),['split','signature','cutoff_minutes','runs','flagged_rows','fraction_QC','flagged_high_QC_co2','flagged_B_eligible']].copy();t['fraction_QC']=t.fraction_QC.map(lambda v:f'{v*100:.4f}%');t.columns=['split','signature','분','run','raw 반복 행','QC 대비','고농도 행','B eligible 중 반복 행'];text.append(table(t))
text.append('''전체48조합 및 raw 모수 대비 비율: [duration_sensitivity.csv](duration_sensitivity.csv). 모든 run의 시작/종료/지속시간: [raw_runs.csv.gz](raw_runs.csv.gz). 전체 행별 tag는 작물/split별 `*_row_tags.parquet`에 보존했다. 이 파일은 진단 sidecar이며 새 학습 dataset이 아니다.

**기존 문제 구간 검출 및 짧은 반복**

| 사후 확인용 구간 | 전체 행 | signature A 검출 | signature B 검출 | cutoff |
|---|---:|---:|---:|---|
| Validation 2025-11-10 18:45–11-11 09:35 |179|179|172|30/60/120/180 모두|
| Train 2024-10-01 00:45–10-02 10:15 |403|403|403|30/60/120/180 모두|
| Train 2024-10-20 03:15–10-21 10:20 |374|374|374|30/60/120/180 모두|

B는 첫 문제 구간에서 T1/RH1/light까지 고정된19:20부터 검출하므로 앞7행은 잡지 않는다. 이것은 오류를 보고 경계를 이동한 결과가 아니라 signature 차이의 결과다. Train 두 구간은 기존 A/B에서 이미 제외돼 있었다. 날짜는 탐지 완료 후 이 확인표에만 사용했다.

**30분 미만**의 짧은 exact-repeat 구간은 모든 cutoff에서0행 검출된다. 다음은 최소2행 반복인 짧은 구간의 규모다. 이 구간을 모두 정상이라고 보증하거나, 미검출을 근거로 진단 성능을 과장하지 않는다.
''')
t=short.loc[short.duration_bin.eq('lt30'),['crop','split','signature','runs','rows']];text.append(table(t))
text.append('''30분 cutoff는 딸기 Train에서 A만 CO₂1351의55분12행을 추가 검출한다(B는 미검출). Validation에서는 A/B 모두 CO₂424의30분7행, CO₂574의80분17행도 검출한다. 후자의 두 run은 여러 센서가 함께 반복된 구간이지만 정상/정체 label은 없으므로 **오탐이라고 확정하지도, 정상 plateau라고 제외하지도 않는다.** 또한 B는 낮은 CO₂392의405분82행도 검출한다. 따라서 이 규칙은 고농도만 골라내는 규칙이 아니다. [detected_runs_context.csv](detected_runs_context.csv)

**기존 예측의 Validation 집계 민감도 — 공식 평가와 분리**

저장된 one-step 예측과 rollout NPZ만 읽었다. one-step은 현재 또는 target 상태가 flagged이면 집계에서 분리한다. multi-step은 초기 상태부터 각 intermediate/endpoint까지 하나라도 flagged이면 **window 전체**를 분리한다. 모델별로 동일한 잔존 모집단을 사용하고, 중간 step만 골라내거나 window를 다시 생성하지 않았다. 공식 weighted score와 같은 세 target(T/RH/CO₂) 평균 nRMSE·drift 정의, 기존 full eligible Validation 표준편차 분모 및 가중치(0.10/0.10/0.25/0.45/0.10)를 유지했다. 무마스킹 결과는 기존 점수·RMSE와 tolerance 내 재현됐다.

다음 표의 각 horizon 칸은 **LightGBM RMSE / Persistence RMSE / skill**, 단위는ppm이며 skill=1−L/P다. 5분은 one-step, 나머지는 전체 trajectory RMSE다.
''')
rows=[]
for rule,g in m.loc[m.model.eq('lightgbm')].groupby('rule',sort=False):
 row={'rule':rule}
 for _,r in g.iterrows():row[str(int(r.horizon_minutes))+'분']=f'{r.co2_rmse:.3f} / {r.persistence_co2_rmse:.3f} / {r.CO2_skill_vs_persistence:+.4f}'
 rows.append(row)
text.append(table(pd.DataFrame(rows)))
text.append('\nweighted score는 CO₂ 단독 점수가 아니라 기존 세 target 점수다. CO₂ 단독 가중 analog는 이름을 명시해 CSV에 별도로 제공하며 공식 점수가 아니다.\n')
t=scores.pivot(index='rule',columns='model',values='all_target_weighted_score').reindex([r['rule'] for r in rows]).reset_index();t['lightgbm']=t.lightgbm.map(lambda x:f'{x:.6f}');t['persistence']=t.persistence.map(lambda x:f'{x:.6f}');text.append(table(t))
text.append('''마스킹에 따라 LightGBM의 상대 점수가 좋아지지만 **공식 통과를 뜻하지 않는다.** 단일 threshold를 최적값으로 선택하지 않았고, 공식 Validation에서 실제 행을 삭제하지 않았다. 잔존 모집단에 대한 조건부 성능이며 전체 온실 제어 성능/향후 일반화를 추정하는 점수가 아니다. window들은 서로 겹쳐 독립 표본도 아니다.

60분 전체17,112 windows에서 A 규칙은177–225개(1.03–1.31%)에 걸리며 기존 LightGBM CO₂ SSE의83.42–83.73%가 그 mask에 집중된다. B는170–218개(0.99–1.27%), SSE77.72–78.03%다. 즉 **날짜·모델 오차 없이 탐지한 raw 반복 패턴이 기존 오차 집중 구간과 겹친다**는 증거다. B의60분 LightGBM RMSE는101.201→47.736–48.010으로 변한다. 오차 집중도는 영향 분석이며 정체의 인과 증명이 아니다.

각 평가의 잔존/전체 sample·window 수는 [metric_sensitivity.csv](metric_sensitivity.csv), 실제 window mask는 [diagnostic_window_masks.parquet](diagnostic_window_masks.parquet), SSE 비중은 [masked_error_concentration.csv](masked_error_concentration.csv)에 있다.

**고농도 Train 2,267행: 반복과 풍속/광량 QC를 함께 분해**

2,267행 전체에 raw 존재 여부, signature별 run ID/duration/cutoff, A/B membership과 B reason을 부착했다: [train_high_2267_tags.csv](train_high_2267_tags.csv). 2,235행이 이미 A에서 제외됐고, A32행 중 B에서16행 추가 제외·16행 잔존이라는 공식 경로는 그대로다.

다음은 A 제외2,235행의 배타적 분해다. 원천 row가 없는645행은 규칙 미검출이지만 **실제 non-repeated라는 증거가 없으므로 unknown**으로 분리한다.
''')
text.append(table(c.rename(columns={'signature':'signature','cutoff_minutes':'분','A_excluded_high':'A 제외','repeated_and_weather':'반복+weather QC','observed_raw_not_flagged_and_weather':'raw 있음·미검출+weather QC','raw_absent_weather_repeat_unknown':'raw 없음·반복 unknown+weather QC','other':'weather QC 외'})))
text.append('''A 반복 규칙은881행(39.42%), B는777–810행(34.77–36.24%)에서 겹친다. **이 반복 행도 전부 풍속/광량 QC 결측과 겹친다.** 따라서 snapshot과 weather QC를 상호 배타적 인과 원인처럼 더할 수 없다. 표는 반복 여부를 먼저 분리한 관측상 분해다. 별도의 exclusive A 제외 reason은 기존처럼 필수 상태 결측2,227·경계7·시간 불연속1이며, 경계/시간불연속8행에도 weather 결측이 함께 있었다. 표의 `weather QC 외=0`이 경계 reason이 없다는 뜻은 아니다.

B signature30/60/120 기준으로 보면 **810 반복+weather / 780 raw 미검출+weather / 645 raw 부재·unknown+weather**다. 따라서 Train 고농도 coverage 부족이 반복 snapshot만의 문제는 아니다. 원천값의 풍속 단위/허용범위, 광량 범위 및 미해결 QC의 영향이 별도로 남는다. 이번에는 임의 단위 환산·범위 완화·보간·재투입을 하지 않았다.

A에서 남은32행 중 signature A의 고농도 반복5행은 B 추가 제외1·B 잔존4로 나뉜다. signature B는 A 잔존32행에서0행 검출이다. B 잔존16행을 지금 삭제하거나 사용 가능 수를 변경하지 않는다.

**센서 disagreement: 분포만 진단**

차이는 raw CO₂1−CO₂2 절댓값이다. 50/100/250/500/1000ppm 구간과 각각의 중복을 저장했으며 새 QC threshold가 아니다. 아래는 설명용500ppm 이상과60분 반복의 중복이다. 다른 cutoff 전체는 CSV에 있다.
''')
t=dis.loc[dis.difference_ge_ppm.eq(500)&dis.cutoff_minutes.eq(60),['crop','split','signature','finite_sensor_pair_rows','large_difference_rows','overlap_flagged_rows']];text.append(table(t))
text.append('''딸기 Validation에서 차이≥500ppm1,980행 중 A60 반복1,839행, B60 반복1,695행으로 중복이 크지만, 차이 자체로 어느 센서가 잘못됐는지 판정할 수 없다. source audit에서 확인한 sensor1 우선 정책·센서 평균 미사용·센서 교체 없음·disagreement flag 미적용을 유지했다. [sensor_disagreement_distribution.csv](sensor_disagreement_distribution.csv), [sensor_disagreement_overlap.csv](sensor_disagreement_overlap.csv).

**최종 판단과 다음 한 가지 조치**

- **A(이번 sensitivity 결론): 지지됨.** 날짜/모델을 쓰지 않은 반복 규칙이 세 문제 구간을 네 cutoff 모두에서 잡고, 짧은 반복 대부분과 구별되는 별도 run을 만든다. 따라서 stale-snapshot **후보 QC 계약 실험을 설계할 근거**가 있다. 정상/고장 자동 분류의 검증 완료는 아니다.
- **B의 한계는 남음:** 정상 여부의 외부 label이 없으므로 정체의 물리적 사실 및 오탐률은 미확정이다. 이전 source audit의 C를 유지한다. 이는 이번 A 결론과 다른 질문이다.
- **weather QC/coverage는 병존:** A 제외의34.77–39.42%는 반복과 겹치지만, 관측 raw 미검출과 raw 부재 구간도 크다. 전체 coverage 부족을 stale만으로 해결한다고 결론내릴 수 없다. 반복보다 weather QC가 항상 더 핵심이라는 단일 C 결론 역시 이번 집계만으로 단정하지 않는다.

**다음에는 `stale-snapshot QC 계약 실험` 하나를 먼저 권고한다.** 새 학습 없이, 이번 두 signature와 지속시간 후보에 대해 sensor-history의 online elapsed 탐지/오프라인 whole-run 표시 차이, 탐지 지연, gap/source 경계, unknown 및 보존·quarantine provenance를 사전 정의해 shadow 검증하는 것이 최소 비용이다. cutoff는 모델 점수로 고르지 않고, 이 제안을 공식 dataset 변경 승인으로 간주하지 않는다. 외부 추가 문의는 진행하지 않는다. 이번 단계에서 이 후속 실험 자체를 실행하지 않았다.

**검증과 보존**

무마스킹 점수 및 rollout RMSE 기존 결과 재현, 동일 window ID 대조, raw 존재 여부, B⊆A, cutoff mask 포함관계, gap/series/segment·episode·NaN·0값·elapsed cutoff·날짜/모델오차 불변성 검사를 통과했다. [verification.json](verification.json), [boundary_regression_checks.json](boundary_regression_checks.json). 공식 입력·model 파일 hash 유지: [input_manifest.json](input_manifest.json). 코드/출력 hash: [artifact_integrity.json](artifact_integrity.json).

새 namespace는 `stale_snapshot_sensitivity_b_v1`이다. **Phase0 PASS 및 전체 Phase1/2 BLOCKED 유지. 공식 모델 selection/Step14·15 진입 승인 없음.**
''')
(D/'final_decision_ko.md').write_text('\n'.join(text))
# Exportable overview of conditional performance; no threshold selection indicator.
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
for sig in ('A','B'):
 g=m.loc[m.model.eq('lightgbm')&m.horizon_minutes.eq(60)&m.rule.str.startswith(sig+'_')].copy();g['cutoff']=g.rule.str.split('_').str[1].astype(int);ax[0].plot(g.cutoff,g.co2_rmse,marker='o',label='signature '+sig)
ax[0].axhline(101.201020,color='black',linestyle='--',label='official unmasked');ax[0].set_xlabel('Diagnostic duration cutoff (minutes)');ax[0].set_ylabel('60-min LightGBM CO2 RMSE (ppm)');ax[0].legend();ax[0].set_title('Conditional retained-window metrics only')
for sig in ('A','B'):
 g=c.loc[c.signature.eq(sig)];ax[1].plot(g.cutoff_minutes,g.repeated_and_weather,marker='o',label='signature '+sig)
ax[1].set_xlabel('Diagnostic duration cutoff (minutes)');ax[1].set_ylabel('Repeated AND weather-QC excluded rows');ax[1].set_title('Of 2,235 already A-excluded high-CO2 rows');ax[1].legend()
for a in ax:a.grid(alpha=.25)
fig.tight_layout();fig.savefig(D/'sensitivity_overview.png',dpi=150);plt.close(fig)
(D/'decision.json').write_text(json.dumps({'sensitivity_conclusion':'A: evidence sufficient to design a stale-snapshot QC contract experiment','physical_source_audit':'C / undetermined unchanged','recommended_single_next_action':'non-learning stale-snapshot QC contract shadow experiment','official_cutoff_selected':None,'Phase0':'PASS unchanged','Phase1_2':'BLOCKED unchanged','official_dataset_model_edited':False},indent=2,ensure_ascii=False)+'\n')
inputs=json.loads((D/'input_manifest.json').read_text())
for p,h in inputs.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
scripts=list((ROOT/'experiments/transition_model_selection/scripts').glob('*snapshot_sensitivity_b_v1.py'))
(D/'code_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},indent=2)+'\n')
(D/'artifact_integrity.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.iterdir()) if p.is_file() and p.name!='artifact_integrity.json'},indent=2)+'\n')
print('Report complete; original hashes verified.')
