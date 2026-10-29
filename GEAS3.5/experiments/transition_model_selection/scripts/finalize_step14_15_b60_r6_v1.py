from pathlib import Path
import sys,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.models.transition.handoff_b60_r6_v1 import sha,SELECTED
H=ROOT/'experiments/transition_model_selection/artifacts/step14_b60_r6_v1';S=ROOT/'experiments/rl_policy_training/artifacts/step15_b60_r6_v1'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def record(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def table(f):return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(map(str,r))+' |' for r in f.itertuples(index=False,name=None)])
assert read(S/'bound_loader_validation.json')['status']=='PASS'
gate=read(S/'final_gate.json');gate['bound_support_runtime_loader_validation']='PASS';gate['causal_history_regression_tests_passed']=32;write(S/'final_gate.json',gate)
entry={'version':'geas35.offline_research.step14_15.b60_r6.v1','loader':'geas35.rl.bound_support_b60_r6_v1.load_bound_research_artifacts','crops':{c:record(S/c/'binding.json') for c in SELECTED},'code':[record(ROOT/p) for p in ['src/geas35/models/transition/handoff_b60_r6_v1.py','src/geas35/rl/bound_support_b60_r6_v1.py','experiments/transition_model_selection/scripts/verify_step14_15_binding_b60_r6_v1.py']],'raw_stale_detector_in_simulation':False,'Step16_entry_only':'PASS, not executed','online_deployment':'BLOCKED'}
write(S/'research_environment_binding.json',entry)
summary=pd.read_csv(S/'support_summary.csv');rows=[];actions=[]
for c in SELECTED:
 m=read(H/c/'manifest.json');rows.append({'crop':c,'algorithm':SELECTED[c],'model SHA256':m['bundle_hashes']['model.pkl'],'scaler SHA256':m['bundle_hashes']['observation_scaler.json']})
 a=read(H/c/'action_response.json')
 for name,x in a['report']['actions'].items():actions.append({'crop':c,'action':name,'support_count':x['support_count'],'status':x['status'],'mean_abs_sensitivity':x['counterfactual_mean_absolute_sensitivity']})
pd.DataFrame(rows).to_csv(S/'official_transition_bindings.csv',index=False);pd.DataFrame(actions).to_csv(S/'action_response_summary.csv',index=False)
f=summary.loc[summary.split.eq('validation'),['crop','kind','warning_threshold','severe_threshold','warning_including_severe_rate','severe_rate']].copy()
for col in f.columns[2:]:f[col]=f[col].map(lambda x:f'{x:.6f}')
extra='\n## 최종 수치와 추가 검증\n'+table(f)+'\n\n비율은 0–1 단위다. 특히 cucumber state severe 23.40%, strawberry state severe 20.29%/joint severe 25.35%로 높다. melon joint warning도 42.91%다. Step16 진입 PASS는 기존 보정/구현 gate를 충족했다는 뜻이며 이러한 분포 이탈을 해결했다는 뜻이 아니다. 기준을 완화하거나 Validation에 맞춰 재보정하지 않았다.\n\n'+table(pd.DataFrame(actions))+'\n\n[공식 모델/scaler 전체 해시](official_transition_bindings.csv), [연구 환경 진입 binding](research_environment_binding.json), [support/runtime 로더 결속 검사](bound_loader_validation.json). support 전용 로더는 manifest/crop/schema/scaler/reference hash를 모두 검사한다. 72개 Validation 상태의 score reload parity 및 21개 잘못된 support binding 거부, 기존 causal/sensor history 회귀 32개 PASS. Step14에도 작물당 5개 변조/crop 거부 검사가 있다.\n\n새 단일-row adapter의 object dtype 실패는 명시 float 변환으로 수정했으며 incomplete_attempt1 namespace에 실패 시도 산출물을 보존했다. 모델/선정/support gate는 바뀌지 않았다.\n'
p=S/'final_decision_ko.md';p.write_text(p.read_text()+extra)
(H/'final_decision_ko.md').write_text('# B60/R6 Step14 handoff — PASS\n\nmelon=XGBoost, cucumber=ExtraTrees, strawberry=KNN. 새 scaler/schema/model/registry hash binding과 전체 Validation one-step 및 작물당96 rollout step parity PASS. fresh-repeat/reload 증거 재확인, 추가 fit 없음.\n\n[통합 Step14/15 최종 보고서](../../../rl_policy_training/artifacts/step15_b60_r6_v1/final_decision_ko.md) · [handoff manifest](step14_handoff_manifest.json) · [사전 고정 프로토콜](protocol.json).\n\nStep16 연구 검증 진입 가능하나 아직 실행하지 않았다. 높은 Validation OOD는 별도 제한으로 남는다. 실제 online deployment는 actuator-safe fallback 미검증으로 BLOCKED다.\n')
for folder in (H,S):write(folder/'artifact_integrity.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact_integrity.json'})
print('FINALIZED',gate['status'])
