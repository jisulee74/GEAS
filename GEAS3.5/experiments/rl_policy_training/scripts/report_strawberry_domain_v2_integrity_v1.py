"""Corroborate discovery-only integrity blocker; never construct domain geometry."""
from pathlib import Path
import json,hashlib,datetime
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parents[3];O=R/'experiments/rl_policy_training/artifacts/strawberry_domain_v2_execution_v1';B=R/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1/strawberry';L=R/'experiments/rl_policy_training/artifacts/strawberry_domain_v2_execution_lock_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def main():
 assert not (O/'final_decision_ko.md').exists()
 write('integrity_report_code_lock.json',{'sha256':sha(Path(__file__)),'scope':'D-only blocker corroboration; no C/E features or geometry','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 ledger=json.loads((O/'access_ledger.json').read_text());allowed=set(json.loads((O/'input_allowlist.json').read_text())['paths'])
 def read(name,cols):
  p=B/name;assert str(p) in allowed
  filters=[('reg_date','>=',pd.Timestamp('2024-09-01')),('reg_date','<',pd.Timestamp('2024-12-01'))]
  ledger.append({'path':str(p),'columns':cols,'filters':str(filters),'kind':'D_only_integrity_corroboration','E_feature_access':False,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()});write('access_ledger.json',ledger)
  return pd.read_parquet(p,columns=cols,filters=filters)
 hits=pd.read_csv(O/'D_future_interpolation_evidence.csv',parse_dates=['reg_date'])
 q=read('train_prepared_qc.parquet',['episode_id','reg_date','out_windsp','out_light','out_windsp_imputation_method','out_light_imputation_method'])
 physical=read('train_physical.parquet',['row_id','reg_date','obs_outdoor_wind_speed','obs_outdoor_light'])
 p=read('train_provenance.parquet',['row_id','legacy_previous_timestamp','legacy_previous_previous_timestamp','history_clean','eligible'])
 q.reg_date=pd.to_datetime(q.reg_date);ph=physical.set_index('row_id');records=[]
 for _,h in hits.iterrows():
  g=q.loc[q.episode_id.eq(h.episode_id)].sort_values('reg_date').reset_index(drop=True);j=int(g.index[g.reg_date.eq(h.reg_date)][0]);col=h['column'];raw=col.removesuffix('_imputation_method');obs='obs_outdoor_wind_speed' if raw=='out_windsp' else 'obs_outdoor_light'
  left=j-1;right=j+1
  while left>=0 and str(g.loc[left,col])==h.method:left-=1
  while right<len(g) and str(g.loc[right,col])==h.method:right+=1
  assert left>=0 and right<len(g)
  lt,rt=g.loc[left,'reg_date'],g.loc[right,'reg_date'];assert lt<h.reg_date<rt
  frac=(h.reg_date-lt)/(rt-lt);expected=float(g.loc[left,raw])+(float(g.loc[right,raw])-float(g.loc[left,raw]))*frac
  if raw=='out_light':expected=max(expected,0.)
  stored=float(g.loc[j,raw]);actual=float(ph.loc[h.row_id,obs])
  assert np.isclose(actual,stored,rtol=1e-6,atol=1e-8)
  records.append({'row_id':h.row_id,'episode_id':h.episode_id,'current_timestamp':h.reg_date,'raw_variable':raw,'observation_feature':obs,'method':h.method,'left_timestamp':lt,'right_timestamp':rt,'future_minutes':(rt-h.reg_date).total_seconds()/60,'left_value':g.loc[left,raw],'right_value':g.loc[right,raw],'qc_value':stored,'physical_value':actual,'formula_expected':expected,'formula_match':bool(np.isclose(expected,stored,rtol=1e-6,atol=1e-8)),'timestamp_status':'right_endpoint_reconstructed_from_contiguous_method_run; current-feature lower bound, not full history maximum'})
 e=pd.DataFrame(records);e.to_csv(O/'D_future_endpoint_physical_evidence.csv',index=False)
 reg=pd.read_parquet(O/'split_metadata_registry.parquet');reg['lineage_status']='not_audited_due_to_early_stop'
 prev=p.set_index('row_id');dm=reg.stage.eq('D');known=dm&reg.row_id.isin(prev.index)
 reg.loc[known,'command_history_max_timestamp']=pd.to_datetime(reg.loc[known,'row_id'].map(prev.legacy_previous_timestamp)).to_numpy()
 reg.loc[known,'command_history_status']='stored_legacy_previous_timestamp; transmitted timestamp assumption retained'
 assert (pd.to_datetime(reg.loc[known,'command_history_max_timestamp'])<reg.loc[known,'reg_date']).all()
 # Never fabricate a full sensor maximum from current timestamp.
 lower=e.groupby('row_id').right_timestamp.max();reg['sensor_history_future_endpoint_lower_bound']=reg.row_id.map(lower)
 reg.loc[reg.row_id.isin(lower.index),'lineage_status']='confirmed_current_feature_future_endpoint'
 reg.loc[dm&~reg.row_id.isin(lower.index),'lineage_status']='full_sensor_history_lineage_not_completed_due_to_integrity_stop'
 reg.to_parquet(O/'lineage_registry_at_stop.parquet',index=False)
 summary=json.loads((O/'lineage_preflight_summary.json').read_text());den=json.loads((O/'preliminary_denominator_ledger.json').read_text())
 result={'status':'DOMAIN-ONLY BLOCKED','stop_stage':'D_input_lineage_integrity_before_geometry','not_C_gate_failure':True,'D_future_input_rows':int(e.row_id.nunique()),'D_affected_feature_rows':len(e),'formula_matches':int(e.formula_match.sum()),'physical_QC_matches':len(e),'D_geometry':'NOT_RUN','D_independence':'NOT_RUN','C_gate':'NOT_RUN','E_gate':'NOT_RUN','E_feature_access':0,'E_evaluation_count':0,'C_feature_access':0,'Validation_Test_access':0,'D_rule_sha256':None,'final_rule_lock_sha256':None,'domain_local_support_prospective':'NOT EVALUABLE / BLOCKED','strawberry_Step16':'BLOCKED','overall_Step17':'BLOCKED','online_deployment':'BLOCKED','recipe_changed':False,'official_artifacts_changed':False}
 write('execution_result.json',result)
 den['evaluated_domain_rows']={'D':None,'C':None,'E':None};den['physical_growth_action_gate_numerators']='NOT_EVALUATED: integrity stop before geometry; never substitute zero';write('denominator_ledger_at_stop.json',den)
 write('integrity_verification.json',{'protocol_hash_unchanged':sha(L/'preregistration_lock_ko.md')==json.loads((O/'protocol_lock.json').read_text())['sha256'],'QC_physical_feature_match':len(e),'interpolation_formula_matches':int(e.formula_match.sum()),'formula_rows':len(e),'D_command_history_strictly_before_current':bool((pd.to_datetime(reg.loc[known,'command_history_max_timestamp'])<reg.loc[known,'reg_date']).all()),'full_sensor_history_max_not_fabricated':True,'runtime_recipe_engine_not_completed_after_integrity_block':True,'no_D_geometry_C_or_E_evaluation':True})
 examples=e[['current_timestamp','observation_feature','left_timestamp','right_timestamp','future_minutes','physical_value','formula_match']].head(6)
 def table(f):return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in f.itertuples(index=False,name=None))
 report=f'''# Strawberry domain v2 実행 — D integrity BLOCKED

**DOMAIN-ONLY BLOCKED. 중지 위치는 D 입력 lineage 검사이며, C gate 실패나 E 성능 실패가 아니다.**

실행 흐름: protocol exact SHA256 검증 → preflight 코드/dependency/allowlist lock → Train metadata registry 및 D lineage 검사 → **current feature 미래 보간 확인 → 중지**.

요청의 Case A(C FAIL)/Case B(C PASS→E)보다 앞선 integrity 중지 사례다. Geometry나 D independence를 수행했다고 보고하지 않으며 C/E 수치를 만들어 채우지 않는다.

## 실제 원인

Quarantine 후 D 후보 **18,447행 중 15개 unique row**에서 현재 observation의 풍속/광량에 미래 endpoint를 사용하는 QC interpolation 이력이 확인됐다.

- `out_windsp`: 6행, `short_time_linear_interpolation`.
- `out_light`: 10행, `short_solar_cycle_time_interpolation`.
- 두 변수 중복 1행으로 unique row는 15행. 영향 feature-row는 16개.
- QC→physical 값 일치 **{len(e)}/{len(e)}**, 저장된 양 끝점으로 보간식 재구성 일치 **{int(e.formula_match.sum())}/{len(e)}** (`rtol=1e-6, atol=1e-8`).

이는 bin의 미래 노출 문제가 아니라 **D 원래 current feature 자체가 현재보다 뒤의 측정값으로 보간된 문제**다. 해당 endpoint가 D 기간 안에 있어도 현재시점에 가용하지 않았으므로 causal 조건 위반이다. Boundary-crossing quarantine과 구분한다.

코드 근거: `offline_dataset_preparation/scripts/03_control_quality.py`의 `_short_linear_interpolation`은 invalid run 뒤 `right_value=group_values.iloc[end]`, `right_time=group_times.iloc[end]`를 사용한다(269~318행). 광량/풍속 method label을 저장하는 분기는 378~425행이다. `src/geas35/rl/mdp_v1.py`의 physical mapping은 현재 `out_light`와 `out_windsp`를 observation으로 옮긴다. 이번에는 실제 D row에서 저장된 QC 값과 physical 값도 교차 확인했다.

{table(examples)}

전체 16 feature-row의 좌/우 timestamp·값·future minutes·공식 row ID·formula match는 `D_future_endpoint_physical_evidence.csv`에 있다. 좌우 endpoint는 같은 episode의 연속 method-run을 통해 복원했으며 수식 일치를 검증했다. 이는 해당 변수의 미래 dependency 증거이지 모든 센서/파생변수의 전체 history maximum을 완성했다는 뜻이 아니다.

## 중지 근거와 변경하지 않은 항목

잠긴 protocol §4는 “원천 자체에 current-feature future evidence가 있으면 integrity violation으로 BLOCKED하며 quarantine으로 누출을 숨기지 않는다”고 규정한다. 따라서 15행만 임의 삭제하고 계속 진행하지 않았다. Sensor-history/derived 경로의 추가 영향 전체도 아직 산정하지 않았으며 15행은 이번 직접 current-value 검사에서 확인된 범위다.

Recipe의 9 cells, S2 growth, 50행/2episode, quantile, 기간, C/E gate는 수정하지 않았다. Full experiment engine과 회귀 검증은 이 필수 integrity blocker 이후 진행하지 않았다. 현재 구현된 것은 lock/preflight/metadata-lineage 및 blocker 근거 검증 코드다. READY였던 문서의 논리적 실행 가능성과 실제 입력의 causal 무결성은 별개다.

## 모집단과 미실행 분모

| 분모 | 행 수 |
| --- | ---: |
| 기존 eligible Train metadata | 44,097 |
| D calendar membership | 18,447 |
| C calendar membership | 8,835 |
| E calendar membership | 16,815 |
| Period/episode/target boundary quarantine | 0 |

이 수는 metadata-based split이며 geometry 기반 admission 결과가 아니다. Sensor-history lineage 전체 검증이 완료되지 않았으므로 최종 D/C/E 적격 모집단을 승인하지 않는다. Split metadata registry와 별도 `lineage_registry_at_stop.parquet`를 보존했다. D의 command history는 저장된 legacy previous timestamp로 현재보다 이전임을 확인했다. 이것이 실제 장치 송신시간의 독립 측정이라는 뜻은 아니다.

확인하지 못한 sensor-history 최대 timestamp는 NaT로 남겼다. Current timestamp로 임의 채우지 않았다. 확인된 미래 endpoint는 `sensor_history_future_endpoint_lower_bound`라는 별도 열에 보존했다. 아직 검사하지 않은 C/E lineage는 미검증이라고 표시했다.

Physical IN/LOW/OUT, growth SUPPORTED/LOW/UNSEEN, action admission, episode/block rate는 **NOT_EVALUATED/null**이다. 0% 또는 gate 실패 수치로 대체하지 않는다. D_rule hash와 final_rule_lock hash도 생성되지 않았다.

## E 봉인과 접근 증거

- E: **sealed internal temporal evaluation after v2 lock**.
- E feature access **0**, E evaluation count **0**, C feature access **0**.
- Validation/Test 접근 **0**. Support/model/scaler 데이터도 로드하지 않았다.
- Registry/footer schema는 전체 Train의 metadata만 읽었다. D의 QC·command provenance와 D의 풍속/광량 physical column만 predicate/column 제한으로 추가 확인했다.
- `access_ledger.json`에 읽은 경로·column·filter·목적·시간을 기록했다. E feature 값을 materialize하거나 평가하지 않았다. 이는 application-level read ledger이며 물리 저장장치 byte 단위 감사 주장은 아니다.
- `protocol.sha256`과 `protocol_lock.json`은 기존 lock과 일치한다. 단계별 코드 hash 및 numpy/pandas/pyarrow/Python 버전을 데이터 metadata 접근 전에 보존했다.

기존 E 집계/Validation shadow를 과거 감사에서 본 exposure 한계는 그대로다. 이번 중지 후 E를 추가로 열어 원인을 찾거나 규칙을 조정하지 않았다.

## 최종 상태와 다음 최소 조치

| 항목 | 상태 |
| --- | --- |
| Domain-only prospective internal gate | **BLOCKED — D input integrity** |
| D geometry / independence | NOT RUN |
| C go/no-go | NOT RUN |
| E final evaluation | NOT RUN / 미개봉 |
| Domain + local-support prospective | **NOT EVALUABLE / BLOCKED** |
| Strawberry Step16 | **BLOCKED** |
| 전체 Step17 | **BLOCKED** |
| Online deployment | **BLOCKED** |

최소 후속 조치는 **기존 Train QC의 시간 양방향 보간 및 그 derived/history 전파 범위를 별도 비학습 감사하는 것**이다. 그 후 과거-only 입력 계약을 어떻게 공급할지 프로젝트 결정을 받아야 한다. 이번에 보간값 교체·row 삭제·dataset 재생성·threshold 완화나 새 실험 재실행을 하지 않았다.

High-CO₂/high-vent 부족, geas2_5 unseen series, legacy snapshot assumption, clean5 Transition/autoregressive accumulation, reward imbalance, actuator-safe fallback blocker를 모두 유지한다. 이 integrity 발견만으로 과거 모든 모델 성능 결과가 무효라고 확대 단정하지 않지만, 본 causal v2 gate의 입력으로는 승인할 수 없다.
'''
 (O/'final_decision_ko.md').write_text(report.replace('実행','실행'))
 # Hash read-only source inputs and executable/source evidence; no Validation/Test traversal.
 names=['schema.json','runtime_contract.json','strawberry/train_registry.parquet','strawberry/train_physical.parquet','strawberry/train_prepared_qc.parquet','strawberry/train_provenance.parquet']
 write('input_content_hashes_at_stop.json',[{'path':str((B.parent/n).relative_to(R)),'sha256':sha(B.parent/n)} for n in names])
 write('artifact_manifest.json',{'status':'BLOCKED','files':[{'path':p.name,'sha256':sha(p)} for p in sorted(O.iterdir()) if p.is_file() and p.name!='artifact_manifest.json']})
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
