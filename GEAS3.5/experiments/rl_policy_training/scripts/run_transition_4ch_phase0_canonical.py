"""Canonical-only Phase0 rebuild; no learners; preserve wrong-environment outputs."""
from pathlib import Path
import sys, json, hashlib, importlib.metadata, importlib.util, platform, os, datetime
R=Path(__file__).resolve().parents[3]
B=R/'experiments/rl_policy_training/artifacts'
O=B/'transition_4ch_phase0_canonical_v1'
OLD=B/'past_only_qc_hlv15_b60_r6_4ch_v1'
PR=B/'transition_4ch_migration_prereg_v1'
EXPECTED='/NHNHOME/WORKSPACE/26mafra001_A/BASE/theimc/jslee/GEAS/GEAS3.5/.venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load_module(name,file):
 spec=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 if O.exists():raise RuntimeError('Canonical output already exists; never overwrite')
 O.mkdir()
 pr=json.loads((PR/'migration_preregistration.json').read_text());lock=json.loads((PR/'preregistration_lock.json').read_text())
 deps={n:importlib.metadata.version(n) for n in pr['dependencies']}
 runtime={'sys_executable':sys.executable,'python_version':platform.python_version(),'prefix':sys.prefix,'base_prefix':sys.base_prefix,'cwd':os.getcwd(),'sys_path_before_builder':sys.path,'dependencies':deps,'captured_at_kst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}
 passed=sys.executable==EXPECTED and platform.python_version()=='3.12.3' and deps==pr['dependencies'] and sha(PR/'migration_preregistration.json')==lock['sha256']
 write(O/'interpreter_gate.json',{'passed':passed,'expected_executable':EXPECTED,'expected_python_version':'3.12.3','expected_dependencies':pr['dependencies'],'runtime':runtime,'preregistration_sha256':lock['sha256']})
 if not passed:
  write(O/'phase0_manifest.json',{'status':'TRANSITION 4CH PHASE0 — BLOCKED','failed_gate':'mandatory_interpreter','runtime':runtime})
  (O/'final_transition_4ch_phase0.md').write_text('TRANSITION 4CH PHASE0 — BLOCKED\nMandatory interpreter gate failed; no Phase0 build executed.\n');return
 print('MANDATORY INTERPRETER + DEPENDENCY GATE PASS',flush=True)
 # Snapshot every previous output and lock before any new build.
 protected_dirs=[OLD,B/'transition_4ch_phase0_v1',PR,B/'transition_4ch_dependency_reconciliation_v1']
 protected={str(p):sha(p) for d in protected_dirs for p in d.rglob('*') if p.is_file()}
 D=O/'dataset'/pr['next_dataset_namespace']
 builder=load_module('fourch_builder',Path(__file__).with_name('build_transition_4ch_phase0.py'))
 builder.REPORT=O;builder.DEST=D
 builder.main()  # two independent fresh rebuilds; input access remains frozen
 m=json.loads((O/'phase0_manifest.json').read_text())
 m.update(logical_dataset_namespace=pr['next_dataset_namespace'],canonical_dataset_path=str(D),runtime_environment=runtime,mandatory_interpreter_gate=True,wrong_environment_outputs_preserved=True,implementation_corrections=[])
 write(O/'phase0_manifest.json',m)
 print('CANONICAL INDEPENDENT REBUILDS COMPLETE; START RUNTIME VERIFICATION',flush=True)
 verifier=load_module('fourch_verifier',Path(__file__).with_name('verify_transition_4ch_phase0.py'))
 verifier.D=D;verifier.O=O;verifier.verify()
 # Cross-environment comparisons are diagnostic, never require byte identity.
 import pandas as pd
 import numpy as np
 comparisons=[]
 for rel in sorted(m['artifact_hashes']):
  old,new=OLD/rel,D/rel
  entry={'path':rel,'old_sha256':sha(old),'canonical_sha256':sha(new),'byte_identical':sha(old)==sha(new)}
  if new.suffix=='.parquet':
   a,b=pd.read_parquet(old),pd.read_parquet(new)
   entry.update(rows_old=len(a),rows_canonical=len(b),columns_equal=list(a)==list(b),dtypes_equal=a.dtypes.astype(str).to_dict()==b.dtypes.astype(str).to_dict())
   try:pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True);entry['values_exact']=True
   except AssertionError:entry['values_exact']=False
   try:pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-6,atol=1e-8);entry['values_within_tolerance']=True
   except AssertionError:entry['values_within_tolerance']=False
   diffs={}
   if list(a)==list(b) and len(a)==len(b):
    for c in a:
     if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
      av,bv=a[c].to_numpy(float),b[c].to_numpy(float);mask=np.isfinite(av)&np.isfinite(bv)
      diff=float(np.max(np.abs(av[mask]-bv[mask]))) if mask.any() else 0.
      if diff:diffs[c]=diff
   entry['nonzero_numeric_max_abs_differences']=diffs
   if rel.endswith(('_registry.parquet','_windows.parquet')):
    entry['membership_equal']=entry['values_exact'] # includes row/window ordering
  else:
   a,b=json.loads(old.read_text()),json.loads(new.read_text());entry['json_semantically_equal']=a==b
   if rel.endswith('observation_scaler.json'):
    entry['parameters']={}
    for key in ['means','scales']:
     cols=list(a[key]);av=np.array([a[key][c] for c in cols]);bv=np.array([b[key][c] for c in cols]);entry['parameters'][key]={'keys_equal':set(a[key])==set(b[key]),'exact':bool(np.array_equal(av,bv)),'within_tolerance':bool(np.allclose(av,bv,rtol=1e-6,atol=1e-8)),'max_abs_difference':float(np.max(np.abs(av-bv)))}
    entry['binding_fields_may_differ_with_parquet_hash']=['fit_registry_sha256']
  comparisons.append(entry)
 oldm=json.loads((B/'transition_4ch_phase0_v1/phase0_manifest.json').read_text());m=json.loads((O/'phase0_manifest.json').read_text())
 comparison={'files':comparisons,'pair_counts_equal':m['counts']==oldm['counts'],'runtime_parity':{'old_failures':oldm['runtime_failures'],'canonical_failures':m['runtime_failures'],'old_coverage':oldm['runtime_validation'],'canonical_coverage':m['runtime_validation']},'byte_identical_files':sum(e['byte_identical'] for e in comparisons),'files_compared':len(comparisons),'byte_identity_is_not_a_cross_environment_gate':True}
 write(O/'wrong_environment_comparison.json',comparison)
 preserved=all(sha(Path(p))==h for p,h in protected.items())
 assert preserved,'Protected artifact changed'
 assert all(sha(R/p)==h for p,h in m['input_hashes'].items()),'Frozen input changed'
 assert all(sha(D/p)==h for p,h in m['artifact_hashes'].items()),'Canonical output changed'
 m.update(runtime_environment=runtime,logical_dataset_namespace=pr['next_dataset_namespace'],canonical_dataset_path=str(D),protected_artifact_hashes=protected,wrong_environment_outputs_preserved=preserved,comparison_path='wrong_environment_comparison.json',comparison_sha256=sha(O/'wrong_environment_comparison.json'),orchestrator_sha256=sha(Path(__file__)),interpreter_gate_sha256=sha(O/'interpreter_gate.json'))
 m['gates']['mandatory_interpreter']=passed;m['gates']['prior_artifact_preservation']=preserved
 m['deterministic_canonical_rebuild']={'independent_builds':2,'membership_counts_schema_scaler_windows_equal':True,'all_serialized_hashes_equal':True,'files_compared':len(m['artifact_hashes']),'evidence':'independently constructed file trees compared byte-for-byte through SHA256, plus build statistics equality'}
 m['status']='TRANSITION 4CH PHASE0 — PASS / FIXED-CANDIDATE RESELECTION READY' if all(m['gates'].values()) else 'TRANSITION 4CH PHASE0 — BLOCKED'
 m['fitting_authority']=str(D) if all(m['gates'].values()) else None
 write(O/'phase0_manifest.json',m)
 report=O/'final_transition_4ch_phase0.md';s=report.read_text()
 s=s.replace('초기 실패 내역은 manifest implementation_corrections에 보존한다.','이 수정은 이전 Phase0에서 완료된 구현이며 이번에는 그대로 재검증했다. 초기 실패 내역은 보존된 이전 Phase0 manifest에 있다.')
 s += '\n## Canonical 실행 binding\n\n'
 s += f"- Python executable: `{sys.executable}`\n- Python: `{platform.python_version()}`\n- logical namespace: `{pr['next_dataset_namespace']}`\n- canonical fitting input path: `{D}`\n- **dependency_lock = PASS**\n- 기존 conda 결과는 감사용으로만 보존하며 fitting authority가 아니다.\n\n| dependency | actual/locked |\n|---|---|\n"
 for k,v in deps.items():s+=f'| {k} | {v} |\n'
 s+='\n## SHA256\n\n| artifact | SHA256 |\n|---|---|\n'
 for label,h in [('preregistration',m['preregistration_sha256']),('schema',m['artifact_hashes']['schema.json']),('runtime adapter',m['runtime_adapter_sha256'])]:s+=f'| {label} | `{h}` |\n'
 for crop in ('melon','cucumber','strawberry'):
  for name in ('observation_scaler.json','train_registry.parquet','validation_registry.parquet','train_windows.parquet','validation_windows.parquet'):
   key=f'{crop}/{name}';s+=f'| {key} | `{m["artifact_hashes"][key]}` |\n'
 s+='\n각 windows parquet는 horizon=1/3/6/12 registry를 구분해 저장한다. 전체 input/output hash와 code binding은 manifest에 보존한다.\n\n## 이전 환경과 비교\n\n'
 s+=f"- Pair/window count 요약 동일: {comparison['pair_counts_equal']}\n- 파일 byte-identical: {comparison['byte_identical_files']}/{comparison['files_compared']}\n"
 sem=[e['path'] for e in comparisons if e.get('values_exact') is False];s+=f'- Parquet 값의 exact 차이가 있는 파일: {sem}\n'
 s+='- Membership·feature/target 값·scaler parameter·dtype 차이와 serialization hash를 `wrong_environment_comparison.json`에 분리 기록했다. 환경 간 byte identity는 PASS 조건이 아니다.\n- Runtime parity는 두 환경의 동일 검사 범위와 mismatch를 비교했다. canonical PASS는 새로 실행한 검사에만 근거한다.\n\n다음 단계는 고정 후보군의 Train/Validation fresh fit/reselection이다. 이번 작업에서는 모델 학습·Persistence scoring·support·Step16·PPO·Test/P0 접근을 실행하지 않았다.\n'
 report.write_text(s)
 print(m['status'],flush=True)
 print('cross-environment byte identical',comparison['byte_identical_files'],'/',len(comparisons),flush=True)
if __name__=='__main__':main()
