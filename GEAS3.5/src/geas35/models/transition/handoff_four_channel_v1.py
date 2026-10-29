"""Immutable research-only 4CH handoff; no fitting or controller execution."""
from pathlib import Path
from dataclasses import asdict
import json,pickle,sys,platform,importlib.metadata
import numpy as np
from geas35.models.transition.four_channel_phase0 import FourChannelRuntime,sha,physical_gate,TARGETS
VERSION='geas35.step14.hlv15_b60_r6.4ch.v1'
CONTRACT='experiments/rl_policy_training/artifacts/transition_4ch_handoff_support_prereg_v1/handoff_support_contract.json'
CONTRACT_HASH='c82132755b02a258d084013b767f2e7d6ed7a509fb3890b5c7c419298d7d5c9f'

def checked(root,record):
    p=(Path(root)/record['path']).resolve()
    if not p.is_relative_to(Path(root).resolve()) or sha(p)!=record['sha256']:raise ValueError('Unbound path/hash: '+record['path'])
    return p

def contract(root):
    p=Path(root)/CONTRACT
    if sha(p)!=CONTRACT_HASH:raise ValueError('Contract hash mismatch')
    c=json.loads(p.read_text());e=c['environment']
    if sys.executable!=e['sys_executable'] or platform.python_version()!=e['python']:raise ValueError('Wrong interpreter')
    if {k:importlib.metadata.version(k) for k in e['dependencies']}!=e['dependencies']:raise ValueError('Dependency mismatch')
    return c

class BoundFourChannelRuntime:
    def __init__(self,directory,*,crop,manifest_sha256,root):
        self.root=Path(root);self.directory=Path(directory);c=contract(root);p=self.directory/'manifest.json'
        if sha(p)!=manifest_sha256:raise ValueError('Manifest hash mismatch')
        m=json.loads(p.read_text());self.manifest=m
        if crop not in c['selected']:raise ValueError('Unknown crop')
        b=c['selected'][crop];s=c['schema']
        if m['version']!=VERSION or m['crop']!=crop or m['algorithm']!=b['algorithm'] or m['contract_sha256']!=CONTRACT_HASH:raise ValueError('Crop/algorithm/version mismatch')
        if m['dependencies']!=c['environment'] or m['schema_contract']!=s:raise ValueError('Dependency/schema order mismatch')
        if m['authority']!=c['authoritative'] or m['selected_binding']!=b:raise ValueError('Authority substitution')
        for record in list(c['authoritative'].values())+list(b['phase0_files'].values())+[b['selected_manifest']]:
            if isinstance(record,dict) and 'path' in record:checked(root,record)
        for name,key in [('model.pkl','model'),('schema.json','schema'),('observation_scaler.json','observation_scaler')]:
            if m['bundle_hashes'][name]!=b[key]['sha256'] or sha(self.directory/name)!=b[key]['sha256']:raise ValueError('Bundle changed: '+name)
            if m['bundle_paths'][name]!=name:raise ValueError('Unbound bundle path')
        for record in m['implementation_bindings']:checked(root,record)
        self.adapter=FourChannelRuntime(self.directory/'schema.json',self.directory/'observation_scaler.json',schema_hash=b['schema']['sha256'],scaler_hash=b['observation_scaler']['sha256'])
        self.obs,self.actions=self.adapter.obs,self.adapter.actions;self.scaler=self.adapter.scaler
        with (self.directory/'model.pkl').open('rb') as f:self.model=pickle.load(f)
        if list(self.model.input_columns_)!=s['input_order'] or list(self.model.target_columns_)!=TARGETS:raise ValueError('Fitted column order mismatch')
        if self.model.target_formulation!='residual_physical_delta_v1':raise ValueError('Own-channel residual required')
        if asdict(self.model.observation_scaler)!=asdict(self.scaler):raise ValueError('Embedded scaler mismatch')
    def predict(self,physical,actions):
        frame=self.scaler.transform_frame(physical.copy());a=np.asarray(actions,dtype=float)
        if a.shape!=(len(frame),6) or not np.isfinite(a).all() or ((a<0)|(a>1)).any() or not np.isin(a[:,3:],[0,1]).all():raise ValueError('Invalid six-action input')
        frame.loc[:,self.actions]=a
        if not np.isfinite(frame[self.obs].to_numpy(float)).all():raise ValueError('Invalid state')
        result=self.model.predict(frame[self.obs+self.actions].astype(float)).next_observation
        physical_gate(result.to_numpy(float));return result
    def reset(self,initial,prehistory):
        self.adapter=FourChannelRuntime(self.directory/'schema.json',self.directory/'observation_scaler.json',schema_hash=sha(self.directory/'schema.json'),scaler_hash=sha(self.directory/'observation_scaler.json'),prehistory=prehistory)
        return self.adapter.reset(initial)
    def advance(self,action,exogenous):
        x=self.adapter.model_input(action).to_frame().T
        pred=self.model.predict(x).next_observation.iloc[0];physical_gate(pred.to_numpy(float))
        out=self.adapter.advance(prediction=pred.to_dict(),action=action,exogenous=exogenous)
        return pred,out
