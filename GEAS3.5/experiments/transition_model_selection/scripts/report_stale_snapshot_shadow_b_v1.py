"""Finalize shadow-only report and integrity manifests; no production mutation."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/transition_model_selection/artifacts/stale_snapshot_shadow_b_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(f):
 return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(map(str,r))+' |' for r in f.itertuples(index=False,name=None)])+'\n'
for name in ('input_manifest.json','metric_input_manifest.json'):
 for p,h in json.loads((D/name).read_text()).items():assert sha(ROOT/p)==h
assert sha(D/'protocol.json')==json.loads((D/'protocol_lock.json').read_text())['sha256']
s=pd.read_csv(D/'impact_summary.csv');k=pd.read_csv(D/'known_interval_detection.csv');m=pd.read_csv(D/'conditional_metrics.csv');sc=pd.read_csv(D/'conditional_scores.csv');r=pd.read_csv(D/'detected_run_provenance.csv')
assert len(s)==96
assert pd.read_csv(D/'previous_sensitivity_mask_parity.csv').different_rows.sum()==0
text=['# Stale-snapshot QC 후보 계약 shadow 검증\n',
'''**최종 판정: B — causal 구현은 가능하지만 공식 threshold와 처리 정책을 정할 근거는 아직 부족하다.** 기술적 구현 가능성(A)은 검증했다. 물리적 정상/정체 여부에 관한 source audit C(undetermined), 일반 반복 규칙의 문제 구간 식별을 지지하는 sensitivity A를 모두 유지한다.

공식 `vent_command_carry_b_v4` 및 strawberry Phase0 PASS, strawberry/전체 Transition Phase1/2 BLOCKED는 그대로다. 공식 dataset/scaler/model/Validation 및 LightGBM/Persistence 결과를 변경하지 않았다. Test·fit·새 predict·HPO·feature 변경·support·dataset 재생성·episode 삭제·Step14/15·Step16–19·Step19.5·PPO를 실행하지 않았다. 외부 문의도 하지 않았다.

## 사전 고정 계약과 causal detector

signature A는 raw CO₂1/CO₂2, B는 raw CO₂1/CO₂2/T1/RH1/light의 동시 exact repeat이다. 30/60/120/180분 후보를 모두 유지했고 날짜·episode·예측·오차·QC AI flag를 탐지 조건으로 사용하지 않았다. [protocol.json](protocol.json)과 detector hash를 실제 실행 전에 고정했다.

`Detector.update()`는 현재 event 한 개와 이전 내부 상태만 사용한다. 첫 valid row/값 변경에서 elapsed=0, 같은 signature의 연속 raw timestamp가 정확히5분씩 이어질 때 elapsed=현재−시작이다. **elapsed≥cutoff에 도달한 현재 행부터** online stale=True다. 30분이면7번째 raw row에서 처음 판단한다. 미래로 과거 flag를 변경하지 않는다.

crop/series/segment/raw source 변경, timestamp gap·중복·역순, 실제 raw row 부재, signature NaN/Inf/숫자 변환 실패, explicit raw-invalid, source identity 불명확 시 run을 끊는다. source는 `data_silla_enc + 원천 source_file_path + source_segment_id`다. 일별 RL episode/day 자체는 입력으로 받지 않고, 물리적으로 연속하면 history를 유지한다. Train/Validation은 별도 stream이라 split 이전의 미확인 이력을 추정하지 않는다.

raw-invalid는 `input_valid=False, stale=False`이며 **정상 판정이 아니라 unknown**이다. 실제 raw 테이블에 센서 health flag가 없으므로 이번 replay에서는 원천 행 존재와 숫자 finite를 사용했다. 새로운 물리 범위/AI threshold는 적용하지 않았다. 값 변경은 반복 종료일 뿐 센서 복구 인증은 아니다. 운영 복귀 대기/hysteresis는 미정이다.

원천 timestamp는 DB `reg_date`이며 실제 measurement 갱신 시각이라는 증거는 없다. detection delay는 관측된 반복 시작 기준이다. detector는 실제 raw 수집 stream의 supervisor 후보이며, **Transition Model의 예측 상태나 scaled observation에 실행하면 안 된다.** 예측값 반복은 센서 정체 증거가 아니다. 생산 환경/controller에는 연결하지 않았다.

## 두 offline 처리 후보

|항목|CF: causal-forward|RR: confirmed-run retrospective|
|---|---|---|
|online 판단|cutoff 도달 후부터|동일하게 cutoff 도달 후부터|
|offline mask|detection부터 run 끝|확정된 run 전체|
|cutoff 이전 행|유지|offline 감사에서만 나중에 표시|
|정보 의미|그 시점까지의 정보|이후 확인 정보가 포함된 사후 annotation|
|현재 적용|sidecar만|sidecar만|

RR의 earliest confirmation은 start+cutoff, 개별 row의 판정 가능 시점은 `max(row timestamp, confirmation)`이다. final run end는 사후 provenance다. 이를 과거 online observation의 causal flag로 쓰거나 공식 Validation을 삭제하는 근거로 쓰지 않는다. source/run/raw ID/elapsed/reason과 두 mask를 보존했고, annotation availability도 별도 sidecar에 기록했다.

## 검증

미래 suffix를 바꿔도 prefix 판단이 동일했다. 0값·NaN/Inf·명시적 invalid·raw 부재·source/crop/series/segment·gap/역순·자정·값 변경·정확한 cutoff를 검사했다. detector JSON 저장/복원 후 실제 여섯 stream을 재처리하여 **354,062회 업데이트**의 elapsed/flag가 정확히 일치했다. 모든 최초 detection=start+cutoff, CF⊆RR, raw run duration=(행 수−1)×5분을 확인했다.

RR mask는 이전 sensitivity의48개 조합과 행 차이0이었다. 강화된 source 경계를 확인해도 현재 데이터의 결과가 같았다는 뜻이다. 실제 day/episode 연속 경계 보존 횟수와 break reason은 [boundary_summary.json](boundary_summary.json)에 있다. [causality_boundary_verification.json](causality_boundary_verification.json), [previous_sensitivity_mask_parity.csv](previous_sensitivity_mask_parity.csv).

transition은 현재 또는 target state가 표시되면 영향, window는 t0부터th까지 하나라도 표시되면 전체 영향으로 계산했다. exact target/3·6·12-step registry를 확인했다. target을 본 뒤 학습 sample을 보류하는 것은 가능하지만 t시점 관측 flag로 소급하지 않는다.

## 세 작물 영향

각 signature×cutoff×mode×crop/split의96조합을 [impact_summary.csv](impact_summary.csv)에 기록했다. QC/raw 모수, run·row·고농도·eligible 교집합과 transition 및3/6/12-step 전체/영향 수를 포함한다. **모든 탐지 run의 정상 여부는 undetermined**이므로 정상 여부 불명확 run 수=탐지 run 수다. raw가 없는 행은 unknown으로 run을 끊으며 stale mask로 자동 대체하지 않는다.

멜론 Train은 A/B 모든 cutoff에서1 run·RR47행(고농도47). CF30/60/120/180분은41/35/23/11행이다. 이미 B eligible 밖이라 기존 transition/window 영향0이다. 멜론 Validation과 오이 Train/Validation은 모든 조합에서0 run·0 row·0 transition/window다. 이 결과는 정상 인증이 아니다.

딸기 표의 각 칸은 **CF/RR**이다. eligible은 시작 observation과의 교집합이라 target도 보는 transition 영향과 다르다.
''']
for split in ('train','validation'):
 rows=[]
 for (sig,cut),g in s.loc[s.crop.eq('strawberry')&s.split.eq(split)].groupby(['signature','cutoff_minutes']):
  cf=g.loc[g['mode'].eq('causal_forward')].iloc[0];rr=g.loc[g['mode'].eq('confirmed_run_retrospective')].iloc[0]
  z={'signature':sig,'분':cut,'run/불명확':cf.stale_runs}
  for col,label in [('masked_raw_rows','raw'),('B_eligible_overlap','eligible'),('high_CO2_overlap','고농도'),('affected_transitions','transition'),('affected_windows_3','15분 window'),('affected_windows_6','30분 window'),('affected_windows_12','60분 window')]:z[label]=f'{cf[col]}/{rr[col]}'
  rows.append(z)
 text.append(f'### Strawberry {split}\n\n'+table(pd.DataFrame(rows)))
text.append('''모든 mask는 [transition_masks.parquet](transition_masks.parquet), [window_masks.parquet](window_masks.parquet), 여섯 `*_row_sidecar.parquet`에 분리했다. 새 학습 dataset이 아니며 기존 registry를 변경하지 않았다.

## 문제 구간 사후 확인

날짜는 아래 확인에만 사용했다. RR 범위는 run 시작–종료, CF는 각 detection–종료다. delay는 각 signature run 시작 기준30/60/120/180분이다.
''')
rows=[]
for (split,sig,start,end),g in k.groupby(['split','signature','start','end'],sort=False):
 z={'split/sig':split+'/'+sig,'run 시작':start,'RR 종료':end}
 for _,v in g.sort_values('cutoff_minutes').iterrows():z[f'{int(v.cutoff_minutes)}분 detection']=v.first_online_detection
 rows.append(z)
text.append(table(pd.DataFrame(rows)))
text.append('''10/20–21 문제 구간의 A run은 실제로10/19 20:25부터 이어진다. 요청 날짜로 run 시작을 잘라내지 않았다. B는10/20 03:15부터다. Validation A는11/10 18:45, B는19:20 시작이어서 동일 cutoff에서 B가35분 늦다.

멜론 Train은2025-05-28 06:10–10:00 run이며, A/B 최초 detection은06:40/07:10/08:10/09:10이다. source·raw ID 포함 [melon_train_run_provenance.csv](melon_train_run_provenance.csv), [known_interval_detection.csv](known_interval_detection.csv), [detected_run_provenance.csv](detected_run_provenance.csv).

## Runtime 행동 후보 — 적용하지 않음

|후보|정보 흐름/인과성|남은 제한|
|---|---|---|
|observation invalid→safe controller/fallback|raw 현재/과거→validity supervisor→cutoff 이후 정책 입력 차단|기존 fallback이 같은 stale T/RH를 사용한다면 안전성을 보장하지 못함. 별도 검증된 안전 처리가 필요|
|transition/sample exclusion|sidecar→현재/target 유효성→학습 admission 보류|CF는 online과 동일, RR은 offline 사후 annotation. episode 전체 삭제 불필요|
|flag만 기록·기존 값 유지|causal detector→모델 외부 quality metadata|잘못된 값으로 정책이 계속 동작할 위험이 남음. flag를 model feature로 넣는 것은 별도 schema/학습 변경|

센서값을 sensor2/평균/ffill로 바꾸지 않았다. sidecar flag는 모델 feature가 아니다. 반복 종료와 운영 복귀 승인은 분리해야 한다.

## 조건부 Validation metric — 진단 전용

저장된 strawberry LightGBM/Persistence 예측만 사용했다. 기존 full eligible Validation 표준편차·세 target nRMSE·drift·가중치0.10/0.10/0.25/0.45/0.10을 유지했다. 동일 mask를 두 모델에 적용했고 무마스킹 score/RMSE를 재현했다. 기존60분 CO₂ L/P=101.201/68.873ppm, skill=−0.469386,17,112 windows다.
''')
rows=[]
for (sig,cut),g in m.loc[m.model.eq('lightgbm')&m.horizon_minutes.eq(60)&m['mode'].ne('unmasked')].groupby(['signature','cutoff_minutes']):
 z={'sig':sig,'분':cut}
 for mode,label in [('causal_forward','CF'),('confirmed_run_retrospective','RR')]:
  v=g.loc[g['mode'].eq(mode)].iloc[0];z[label+' L/P RMSE']=f'{v.CO2_RMSE:.3f}/{v.Persistence_CO2_RMSE:.3f}';z[label+' skill']=f'{v.CO2_skill:+.6f}';z[label+' 잔존']=int(v.kept)
 rows.append(z)
text.append(table(pd.DataFrame(rows)))
text.append('기존 세 target weighted score는 L/P=0.231561/0.214950이며 아래는 조건부 점수다.\n')
rows=[]
for (sig,cut),g in sc.loc[sc['mode'].ne('unmasked')].groupby(['signature','cutoff_minutes']):
 z={'sig':sig,'분':cut}
 for mode,label in [('causal_forward','CF L/P'),('confirmed_run_retrospective','RR L/P')]:
  v=g.loc[g['mode'].eq(mode)].set_index('model');z[label]=f"{v.loc['lightgbm','all_target_weighted_score']:.6f}/{v.loc['persistence','all_target_weighted_score']:.6f}"
 rows.append(z)
text.append(table(pd.DataFrame(rows)))
text.append('''one-step·15/30/60분 RMSE/skill/잔존 수/SSE 비중은 [conditional_metrics.csv](conditional_metrics.csv), score는 [conditional_scores.csv](conditional_scores.csv). CO₂ 단독 analog는 공식 score와 구분했다. CF B180의60분 skill은−0.039283, RR은+0.213191이므로 retrospective 평가를 online 성능으로 취급할 수 없다. 겹치는 window는 독립 표본이 아니며 조건부 모집단의 결과다. 어떤 숫자도 공식 PASS·threshold 선택에 사용하지 않았다.

## 최종 판단과 보수적 후보

**B: 구현 가능하지만 공식 threshold/처리 정책 근거 부족.** 실제 정상/정체 ground truth, 오탐 차단 비용, 탐지 지연 비용, fallback의 stale 센서 의존성 및 복귀 정책이 미확정이다. 네 cutoff 중 하나를 선택하지 않았고, 두 offline 정책도 공식 채택하지 않았다.

후속 공식 QC 실험의 후보 하나로 **signature B + causal-forward quarantine + observation validity supervisor**를 권고한다. 다중 센서 동시 반복으로 후보 범위를 좁히고 cutoff 이전 행을 소급 삭제하지 않으며, offline transition/window도 같은 causal flag의 현재/target 기준으로 보류하는 조합이다. Runtime에서는 invalid를 모델 외부 supervisor로 전달하고 센서 정체에 대해 검증된 안전 처리만 허용하는 계약을 검토한다. 이는 현재 fallback 안전성 승인이나 실제 적용 지시가 아니다. RR은 감사 근거로만 보존하는 것이 보수적이다.

이 후보는 불필요한 소급 삭제와 단일 센서 정체 오판을 줄이는 관점에서 보수적이며, 모든 위험을 최소화한다고 입증된 정책은 아니다. 적용 전 탐지 지연·오탐 허용 기준, 안전 처리의 입력 의존성, 복귀 조건을 먼저 정해야 한다. **threshold는 미선정**이며 이번에는 해당 후속 정책 실험·공식 dataset 재구축·재학습을 실행하지 않았다.

새 namespace는 `stale_snapshot_shadow_b_v1`. 공식 입력 hash는 [input_manifest.json](input_manifest.json), [metric_input_manifest.json](metric_input_manifest.json), 검증은 [verification.json](verification.json), [metric_verification.json](metric_verification.json)에 있다. 코드/출력 hash는 [code_manifest.json](code_manifest.json), [artifact_integrity.json](artifact_integrity.json)에 보존했다.

**Phase0 PASS, strawberry/전체 Phase1/2 BLOCKED, 물리 source audit C 유지. 공식 데이터·모델 변경 및 다음 공식 단계 진입 승인은 없다.**
''')
(D/'final_decision_ko.md').write_text('\n'.join(text))
for path in sorted(D.glob('*_row_sidecar.parquet')):
 f=pd.read_parquet(path);o=f[['crop','split','series_id','segment_id','reg_date','raw_idx','raw_source']].copy()
 for sig in ('A','B'):
  for cut in (30,60,120,180):
   confirm=f[sig+'_run_start']+pd.Timedelta(minutes=cut);hit=f[f'{sig}_{cut}_confirmed_run_retrospective']
   o[f'{sig}_{cut}_retrospective_available_at']=pd.concat([f.reg_date,confirm],axis=1).max(axis=1).where(hit)
   assert (o.loc[hit,f'{sig}_{cut}_retrospective_available_at']>=f.loc[hit,'reg_date']).all()
 o.to_parquet(D/path.name.replace('_row_sidecar','_annotation_availability'),index=False)
(D/'decision.json').write_text(json.dumps({'final':'B','causal_implementation_verified':True,'physical_cause':'C undetermined unchanged','sensitivity_A':'supported unchanged','candidate':'signature B + causal-forward + validity supervisor','official_cutoff':None,'official_policy_adopted':False,'Phase0':'PASS unchanged','Phase1_2':'BLOCKED unchanged'},indent=2)+'\n')
scripts=list((ROOT/'experiments/transition_model_selection/scripts').glob('*stale_snapshot*shadow*b_v1.py'))+[ROOT/'experiments/transition_model_selection/scripts/stale_snapshot_detector_shadow_v1.py']
(D/'code_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in scripts},indent=2)+'\n')
(D/'final_verification.json').write_text(json.dumps({'official_input_hashes_unchanged':True,'protocol_hash_unchanged':True,'impact_combinations':len(s),'previous_retrospective_mask_differences':0,'annotation_availability_not_before_row':True,'report_complete':True},indent=2)+'\n')
(D/'artifact_integrity.json').write_text(json.dumps({p.name:sha(p) for p in sorted(D.iterdir()) if p.is_file() and p.name!='artifact_integrity.json'},indent=2)+'\n')
print('Report saved; input hashes verified; 96 shadow combinations complete.')
