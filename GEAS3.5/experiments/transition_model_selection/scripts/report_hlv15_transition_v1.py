"""Report fixed HLV15 Transition selection, all targets/actions; no fit."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'src'))
from geas35.rl.vent_command_memory_b_v3 import OBS,ACTIONS
D=R/'experiments/transition_model_selection/artifacts/command_carry_hlv15_b60_r6_v1';P=R/'experiments/rl_policy_training/artifacts/past_only_qc_hlv15_b60_r6_v1';OLD=D.parent/'command_carry_stale_b60_r6_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def skill(x,ref):return 1-x/ref if np.isfinite(ref) and ref>0 else np.nan

def table(f):
 def cell(v):
  if isinstance(v,(float,np.floating)):return (f'{v:.6e}' if abs(v)>=1e6 else f'{v:.6f}') if np.isfinite(v) else 'NA'
  return str(v)
 return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in f.itertuples(index=False,name=None)])+'\n'

def main():
 pro=read(D/'protocol.json');binding=read(D/'execution_binding.json');ph=sha(D/'protocol.json')
 assert ph==binding['protocol_sha256']==sha(OLD/'protocol.json')
 assert sha(D/'execution_binding.json')==read(D/'execution_lock.json')['sha256']
 for p,h in binding['input_hashes'].items():assert sha(R/p)==h,p
 for p,h in binding['code_hashes'].items():assert sha(R/p)==h,p
 for x in pro['configuration_hashes']:assert sha(R/x['path'])==x['sha256']
 assert read(D/'batch_runtime_parity.json')['passed']
 rows=[];targets=[];actions=[];selected={};models=[];references=[];algorithms=[]
 for crop in ('melon','cucumber','strawberry'):
  assert (D/crop/'execution_complete.json').exists()
  sel=read(D/crop/'selection.json');selected[crop]=sel
  baseline=read(D/crop/'persistence/rollout/completed.json');bf=read(D/crop/'persistence/fit/completed.json')
  for name,spec in pro['candidates'][crop].items():
   eligible=[];all_evaluated=[]
   for seed in spec['seeds']:
    label=name+(f'_seed_{seed}' if name=='mlp' else '');dest=D/crop/label
    if not (dest/'rollout/completed.json').exists():
     failures=[read(p) for p in (dest/'fit/failure.json',dest/'rollout/failure.json') if p.exists()]
     assert failures;rows.append(dict(crop=crop,candidate=label,algorithm=name,fit_success=(dest/'fit/completed.json').exists(),evaluation_success=False,failure=failures,final_eligible=False));continue
    f=read(dest/'fit/completed.json');r=read(dest/'rollout/completed.json')
    assert f['from_scratch'] and f['reload_pass'] and f['protocol_sha256']==r['protocol_sha256']==ph
    assert sha(dest/'fit/model.pkl')==f['model_sha256']==r['model_sha256']
    assert f['scaler_sha256']==sha(P/crop/'observation_scaler.json')
    pop={v['split']:v for v in binding['populations'] if v['crop']==crop}
    assert f['train_rows']==pop['train']['rows'] and f['validation_rows']==pop['validation']['rows']
    expected=json.loads(json.dumps(spec['params']))
    if name=='mlp':expected['random_state']=seed
    assert f['params']==expected
    finite=f['one_step_nan_inf']==0 and all(m['nan_inf_count']==0 for m in r['rollout'].values()) and np.isfinite(r['score'])
    physical=f['one_step_physical_violation_rate']==0 and all(m['physical_violation_rate']==0 for m in r['rollout'].values())
    improves=r['score']<baseline['score']
    row=dict(crop=crop,candidate=label,algorithm=name,seed=f['seed'],fit_success=True,evaluation_success=True,one_step_nrmse=f['one_step']['aggregate_metrics']['normalized_mean_rmse'],
      drift=r['components']['drift'],weighted_score=r['score'],weighted_persistence_skill=skill(r['score'],baseline['score']),finite_gate=finite,physical_gate=physical,action_response_eligible=f['action_response']['eligible'],
      quality_eligible=r['eligible'],persistence_improved=improves,final_eligible=bool(r['eligible'] and improves),selected=sel.get('selected')==label,fit_seconds=f['fit_seconds'],rollout_seconds=r['seconds'])
    for h,m in r['rollout'].items():
     assert m['window_count']==pop['validation']['windows'][str(int(h[:-3])//5)]
     row[h+'_nrmse']=m['trajectory_nrmse'];row[h+'_nan_inf']=m['nan_inf_count'];row[h+'_physical_violation_rate']=m['physical_violation_rate']
     for target,v in m['targets'].items():targets.append(dict(crop=crop,candidate=label,horizon=h,target=target,**v,persistence_skill=skill(v['rmse'],baseline['rollout'][h]['targets'][target]['rmse']),sample_count=m['window_count']))
    pred=pd.read_parquet(dest/'fit/validation_predictions.parquet')
    for j,(target,v) in enumerate(f['one_step']['target_metrics'].items()):
     lo,hi=[(-60,80),(0,100),(0,10000)][j];x=pred[target].to_numpy(float)
     targets.append(dict(crop=crop,candidate=label,horizon='one-step',target=target,**v,persistence_skill=skill(v['rmse'],bf['one_step']['target_metrics'][target]['rmse']),sample_count=len(x),nan_inf_count=int((~np.isfinite(x)).sum()),physical_violation_rate=float(((x<lo)|(x>hi)|~np.isfinite(x)).mean())))
    for action,v in f['action_response']['actions'].items():
     sensitivity=v['counterfactual_mean_absolute_sensitivity']
     actions.append(dict(crop=crop,candidate=label,action=action,support_count=v['support_count'],support_eligible=v['status']=='eligible',status=v['status'],response_magnitude=sensitivity,nonzero_finite_response=bool(np.isfinite(sensitivity) and sensitivity>1e-12),candidate_action_gate=f['action_response']['eligible'],population='new eligible Validation, adjacent same episode/admission run',counterfactual_low=v['counterfactual_low'],counterfactual_high=v['counterfactual_high']))
    modelbinding=dict(crop=crop,candidate=label,phase0_namespace=P.name,train_registry_sha256=sha(P/crop/'train_registry.parquet'),validation_registry_sha256=sha(P/crop/'validation_registry.parquet'),validation_windows_sha256=sha(P/crop/'validation_windows.parquet'),schema_sha256=sha(P/'schema.json'),scaler_sha256=sha(P/crop/'observation_scaler.json'),model_sha256=f['model_sha256'],protocol_sha256=ph,execution_binding_sha256=sha(D/'execution_binding.json'),observation_order=OBS,action_order=ACTIONS,target_order=OBS[:3],target_units=['degC','percent','ppm'],from_scratch=True,model_path=str((dest/'fit/model.pkl').relative_to(R)),Step14_handoff=False)
    write(dest/'model_input_binding.json',modelbinding)
    if sel.get('selected')==label:
     assert finite and physical and r['eligible'] and improves and sel['status']=='PASS'
     modelbinding['repeat_model_sha256']=sha(dest/'repeat/model.pkl');modelbinding['repeat_reload']=sel['reproducibility'];models.append(modelbinding)
    rows.append(row);all_evaluated.append(row)
    if r['eligible'] or name=='persistence':eligible.append(row)
   if name!='mlp' and all_evaluated:
    algorithms.append({**all_evaluated[0],'aggregation':'single frozen candidate; failed gates remain visible'})
   elif eligible:
    eligible.sort(key=lambda x:(x['weighted_score'],x['seed'] or 0));agg=eligible[len(eligible)//2] if name=='mlp' else eligible[0]
    algorithms.append({**agg,'aggregation':'median eligible seed, no new seed filtering' if name=='mlp' else 'single frozen candidate'})
   else:algorithms.append(dict(crop=crop,algorithm=name,candidate='none eligible',quality_eligible=False,final_eligible=False))
  oldsel=read(OLD/crop/'selection.json');oldr=read(OLD/crop/oldsel['selected']/'rollout/completed.json')
  references.append(dict(crop=crop,old_selected=oldsel['selected'],old_score=oldr['score'],new_selected=sel.get('selected'),new_leader=sel.get('leader'),new_score=next((x['weighted_score'] for x in rows if x['crop']==crop and x.get('candidate')==sel.get('selected')),None),note='historical only; populations/scaler/target std differ; no old weights reused'))
 assert len(rows)==30 and len(algorithms)==24
 pd.DataFrame(rows).to_csv(D/'candidate_summary.csv',index=False);pd.DataFrame(algorithms).to_csv(D/'algorithm_summary.csv',index=False)
 pd.DataFrame(targets).to_csv(D/'target_metrics_and_skill.csv',index=False);pd.DataFrame(actions).to_csv(D/'action_response.csv',index=False);pd.DataFrame(references).to_csv(D/'historical_comparison.csv',index=False)
 write(D/'selected_candidate_bindings.json',models)
 overall='PASS' if all(x['status']=='PASS' for x in selected.values()) else 'BLOCKED'
 write(D/'final_gate.json',dict(status=overall,crops=selected,protocol_sha256=ph,all_input_hashes_unchanged=True,fresh_candidate_runs=30,Step14_15_executed=False,support_created=False,Test_accessed=False,old_models_scalers_reused=False,current_BLOCKED=['strawberry Step16','overall Step17','online deployment']))
 texts=[f'# HLV15 causal B60/R6 Transition Phase1/2 — {overall}\n','작성일: 2026-10-08 KST\n',f'**전체 Transition Phase1/2 {overall}.** 이번 작업은 fresh 학습·재선정까지만 수행했다. Step14/15와 이후 단계는 실행하지 않았다.\n','## 입력과 사전 고정\n',
  f'기존 protocol을 byte-identical하게 보존했다. SHA256 `{ph}`. 기존 8 candidates, seed/config, 가중치 one-step/15m/30m/60m/drift = 0.10/0.10/0.25/0.45/0.10, target std(ddof=0), pooled trajectory nRMSE, window-mean drift 및 q90/CVaR90을 유지했다. 기존 winner/prior_selected는 선택이나 실행 우선순위로 사용하지 않았다.\n',
  'protocol에 남아 있는 이전 phase0/input_hashes/prior_selected/execution 필드는 역사 provenance다. 새 실행의 실제 입력은 [execution_binding.json](execution_binding.json)으로만 결속했으며 로더 경로를 새 Phase0로 고정했다. 원본 protocol 자체는 수정하지 않았다. 모든 crop는 새 Train만 fresh fit했고 Validation fit/old warm start는 없다. MLP early stopping의 내부 분할은 기존 설정의 Train 내부 분할이다.\n',
  table(pd.DataFrame([dict(crop=x['crop'],split=x['split'],rows=x['rows'],windows_3=x['windows']['3'],windows_6=x['windows']['6'],windows_12=x['windows']['12']) for x in binding['populations']])),
  '사용자가 지정한 3/6/12 window 수는 Train registry와 정확히 일치했다. 성능은 위 Validation registry 전체로 평가했다. stride/subsampling, RL safety projection/fallback/OOD 종료를 사용하지 않았다. 입력 scaler/schema/row/window/model binding은 후보별 `model_input_binding.json`에 기록했다.\n',
  '## 고정 gate와 선정\n','finite·physical violation 0·action-response support≥100/nonzero response·Persistence improvement를 기존 방식으로 적용했다. action-response support는 기존 구현대로 새 Validation의 연속 episode/admission-run 내 action 변화 수이며, Train support/OOD 생성이 아니다. 지원 충분 action 중 반응하는 action이 있는지의 기존 candidate gate를 유지했고 모든 6 action이 지원 충분해야 한다는 새 조건을 추가하지 않았다. per-target 음수 skill은 공개하며 새 rejection rule로 사용하지 않았다.\n',
  '작물별 MLP는 고정 seeds 42/43/44 중 quality-eligible seed의 score 중앙값을 대표로 사용했다. 그 뒤 동일 score/name tie-break와 Persistence guard를 적용했다. 후보별 10개 seed 실행을 수행했으며 아래 표에는 알고리즘별 대표 8개를 표시한다. 모든 seed 결과는 [candidate_summary.csv](candidate_summary.csv)에 있다.\n']
 alg=pd.DataFrame(algorithms);ts=pd.DataFrame(targets);act=pd.DataFrame(actions)
 for crop,sel in selected.items():
  texts.append(f"## {crop} — {sel['status']} / 선정: {sel.get('selected') or '없음'}\n")
  cols=['algorithm','candidate','one_step_nrmse','15min_nrmse','30min_nrmse','60min_nrmse','drift','weighted_score','finite_gate','physical_gate','action_response_eligible','persistence_improved','final_eligible']
  texts.append(table(alg.loc[alg.crop.eq(crop)].reindex(columns=cols)))
  if crop=='strawberry':
   texts.append('MLP는 세 seed 모두 quality gate 미통과이며, 대표 score를 임의 선택하지 않는다. 실패한 seed도 아래에 모두 공개한다.\n')
   texts.append(table(pd.DataFrame(rows).loc[lambda f:f.crop.eq(crop)&f.algorithm.eq('mlp')].reindex(columns=['candidate','one_step_nrmse','15min_nrmse','30min_nrmse','60min_nrmse','drift','weighted_score','finite_gate','physical_gate','final_eligible'])))
  texts.append(f"Persistence weighted score={read(D/crop/'persistence/rollout/completed.json')['score']:.6f}. 판단: {sel.get('approval_reason')}.\n")
  label=sel.get('selected') or sel.get('leader')
  if label:
   texts.append(table(ts.loc[ts.crop.eq(crop)&ts.candidate.eq(label),['horizon','target','mae','rmse','nrmse','persistence_skill','sample_count','physical_violation_rate','nan_inf_count']]))
   texts.append(table(act.loc[act.crop.eq(crop)&act.candidate.eq(label),['action','support_count','support_eligible','response_magnitude','nonzero_finite_response']]))
  if sel['status']=='PASS':texts.append('fresh repeat/reload 및 전체 window prediction parity: '+str(sel['reproducibility'])+'\n')
 texts+=['## 전체 target·action 공개\n','모든 후보/seed/target의 one-step·15/30/60분 MAE/RMSE/R²/nRMSE/tail/endpoint drift/physical/nonfinite/Persistence skill은 [target_metrics_and_skill.csv](target_metrics_and_skill.csv)에 있다. 각 action support count/eligibility/response magnitude/nonzero는 [action_response.csv](action_response.csv)에 있다. sensitivity는 기존 구현의 T/RH/CO₂ 물리 단위 평균 절댓값이며 실제 인과 효과 보증은 아니다.\n','## Historical 비교\n',table(pd.DataFrame(references)),
  '`historical metric preserved / causal availability gate superseded by past-only QC rebuild`\n',
  '같은 winner면 causal QC 변경 후에도 selection이 재현됐다는 제한적 해석이며, 바뀌었다면 새 causal input population의 fresh selection 변화다. 어느 경우에도 HLV reconstruction의 물리 정확성 증명은 아니다. 이전 score와는 모집단/scaler/normalization 분모가 달라 동일 모집단 성능 향상으로 해석하지 않는다.\n','## 유지되는 제한\n',
  '선정 모델의 CO₂ skill: melon XGBoost 15/30/60분 −0.055799/−0.072234/−0.081595, strawberry LightGBM −0.007097/−0.012277/−0.088191이다. 종합 score 개선과 기존 gate로 PASS했으며 모든 target에서 Persistence보다 우수하다는 뜻이 아니다. strawberry shade curtain의 counterfactual response는 0이다. 기존 gate는 지원 충분 action 전체가 무반응인 후보를 제외하는 규칙이므로 이를 새로운 단일-action 탈락 조건으로 바꾸지 않았다.\n',
  'strawberry Train CO₂≥1500은22행, previous vent≥0.75와 동시3행, current vent≥0.75와 동시2행이다. PASS여도 high-CO₂ dynamics 학습 또는 high-CO₂/high-vent 일반화 확인이라고 주장하지 않는다. geas2_5 unseen series, melon/strawberry legacy_command_snapshot_assumption, clean5 Transition/autoregressive accumulation historical limitation, reward imbalance, online actuator-safe fallback BLOCKED는 유지한다. cucumber source는 api_transmitted_command이며 근거 수준을 합쳐 해석하지 않는다. domain-v2 D integrity 재개 전 upstream causal stack 재검증이 필요하다.\n',
  '## 후속 작업\n',('다음 작업 하나: **새 selected models/scalers/schema를 이용한 Step14 handoff + Step15 support rebuild**. 이번에는 실행하지 않았다.' if overall=='PASS' else '전체 Transition BLOCKED이므로 Step14 진입을 중단한다. 위 crop/candidate/gate 실패에 대해서만 최소 원인 진단을 먼저 해야 한다.'),
  '\nstrawberry Step16 / 전체 Step17 / online deployment BLOCKED를 유지한다. Test, HPO, reward 변경, support 생성, Step14/15/16, domain-v2/C/E, PPO, Step17–19, Step19.5는 실행하지 않았다.']
 (D/'final_decision_ko.md').write_text('\n'.join(texts)+'\n')
 write(D/'artifact_integrity.json',{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name not in ('artifact_integrity.json','job.lock')})
 print('FINAL',overall,{c:(s['status'],s.get('selected')) for c,s in selected.items()},flush=True)
if __name__=='__main__':main()
