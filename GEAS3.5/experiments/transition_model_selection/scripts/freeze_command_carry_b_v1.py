"""Freeze B Phase 1/2 before any B model fit or performance inspection."""
from pathlib import Path
import hashlib,json,sys,yaml,importlib.metadata
ROOT=Path(__file__).resolve().parents[3]
D=ROOT/'experiments/transition_model_selection/artifacts/command_carry_b_v1'
P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4'
A=ROOT/'experiments/transition_model_selection/artifacts/step12_v1_4_root_cause/official_rerun/deduplicated_without_quality_flags'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':
 D.mkdir(exist_ok=False)
 assert json.loads((P/'gate_results.json').read_text())['status']=='PASS'
 for x in json.loads((P/'artifact_integrity.json').read_text()):assert sha(P/x['path'])==x['sha256']
 decisions=json.loads((ROOT/'experiments/transition_model_selection/artifacts/step14_v1_4/researcher_decision.json').read_text())['decisions']
 inventory={};inputs=[];configs=[]
 names=['persistence','linear_regression','linear_svr','knn','extra_trees','lightgbm','xgboost','mlp']
 for crop in ('melon','cucumber','strawberry'):
  cp=ROOT/f'experiments/transition_model_selection/configs/{crop}_v1_4_without_flags.yaml';cfg=yaml.safe_load(cp.read_text());inventory[crop]={}
  assert [x['name'] for x in cfg['models']['candidates']]==names
  for name in names:
   bp=A/crop/crop/name/'best_config.json'
   if bp.exists():params=json.loads(bp.read_text())['best_config'];source=bp;reason='stored final A protocol hyperparameters; no search'
   else:params=next(x['params'] for x in cfg['models']['candidates'] if x['name']==name);source=cp;reason='no eligible saved final config; existing protocol base settings retained, not tuned'
   inventory[crop][name]={'params':params,'source':str(source.relative_to(ROOT)),'sha256':sha(source),'reason':reason,'seeds':([42,43,44] if name=='mlp' else [params.get('random_state')])}
   configs.append({'path':str(source.relative_to(ROOT)),'sha256':sha(source)})
  for p in sorted((P/crop).glob('*')):
   if p.is_file():inputs.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
 for name in ('schema.json','gate_protocol.json','gate_results.json','runtime_contract.json','artifact_integrity.json'):
  p=P/name;inputs.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
 protocol={'version':'transition.command_carry_b.v1','frozen_before_fit':True,'date_kst':'2026-10-07','phase0_namespace':str(P.relative_to(ROOT)),
 'input_contract':'55 scaled observations + 6 current transmitted commands -> 3 physical residual targets',
 'feature_variant':'deduplicated_without_quality_flags','feature_scope_reason':'all three official A selections and frozen B v4 schema use 55 observations; 57-feature historical QC ablation is inventoried but not a new B schema/scaler',
 'prior_selected':{c:decisions[c]['candidate_name'] for c in decisions},'candidates':inventory,'input_hashes':inputs,'configuration_hashes':configs,
 'weights':{'one_step':.1,'15min':.1,'30min':.25,'60min':.45,'drift':.1},
 'metrics':'existing Validation target std ddof=0 normalization; pooled trajectory nRMSE; window-mean drift; q90/CVaR90',
 'windows':'all exact frozen 3/6/12-step Validation registry entries, no stride/subsample; shorter horizons retain their own full populations',
 'runtime':'SensorHistoryStateUpdater v4; no safety projection, fallback, OOD termination or prediction clipping',
 'eligibility':'existing finite objective and one-step metrics, zero NaN/Inf and zero physical violations at every horizon, action-response support>=100 and nonzero response; persistence reference only',
 'selection':'minimum existing weighted score among eligible candidates; MLP uses median eligible seed among fixed 42/43/44; name tie-break; failed fits remain failures',
 'approval_guard':'user-requested Persistence comparison: selected weighted score must improve Persistence; negative per-target skill disclosed, not newly made an automatic rejection (legacy protocol permitted it)',
 'reproducibility':'selected config/seed fresh repeat fit; reload prediction equality rtol=1e-6 atol=1e-8; hashes recorded; failed repeat blocks approval',
 'forbidden':['Test','HPO','support generation','PPO','Step16-19'],
 'old_A_reference':decisions,'dependencies':{n:importlib.metadata.version(n) for n in ('numpy','pandas','scikit-learn','lightgbm','xgboost','scipy')},
 'execution':'old selected algorithm/config first per crop, followed by fixed bank; unchanged n_jobs in saved settings; BLAS threads capped for execution only; completion files bound to protocol hash'}
 write(D/'protocol.json',protocol);write(D/'protocol_lock.json',{'sha256':sha(D/'protocol.json')})
 (D/'protocol_ko.md').write_text('''# B Transition Phase 1/2 사전 동결

기존 공식 선정은 멜론 XGBoost, 오이/딸기 LightGBM이며 모두 55 observation without_quality_flags이다. 동일 8 알고리즘의 저장된 최종 하이퍼파라미터를 사용하고, 과거 실패로 최종 설정이 없는 딸기 Linear Regression은 기존 YAML 기본 설정만 사용한다. 57-feature QC ablation은 기존 별도 schema 실험으로 기록하되 B v4의 고정 schema/scaler를 바꾸지 않는다. 기존 A 가중치는 로드하지 않는다.

MLP는 기존 42/43/44 restart 및 eligible seed의 중앙 score 선택을 유지한다. 나머지는 저장 설정 seed를 유지한다. 새 HPO/설정 탐색은 없다. 기존 one-step 0.10, 15분 0.10, 30분 0.25, 60분 0.45, drift 0.10 점수와 물리 위반 0/action-response gate를 적용한다. 기존 최종 결정은 연구자 판단이었으므로, 이번 사용자 위임에 따라 eligible 중 최저 점수로 결정한다. Persistence보다도 가중 점수가 나쁘면 승인 보류하는 추가 보수적 조건을 결과 전에 명시한다. 기존에 허용된 target별 음의 skill은 숨기지 않고 제한사항으로 보고한다.

각 horizon은 Phase 0 registry 전체를 평가한다. 기존 A의 stride=12보다 모집단이 크므로 A 저장 score는 참고일 뿐 직접 동등 비교가 아니다. 계산 가속 구현은 v4 scalar runtime과 독립 parity 확인 후에만 모델 평가에 사용한다. 결과를 본 뒤 후보/선정 규칙/모집단을 바꾸지 않는다. Test/support/PPO/HPO/Step16~19는 실행하지 않는다.
''')
 print('frozen',sha(D/'protocol.json'))
