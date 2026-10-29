"""Hash-bound HLV15 causal B60/R6 research Transition runtime; no raw stale detection."""
from pathlib import Path
import hashlib
import json
from dataclasses import asdict
import numpy as np
import pandas as pd
from geas35.models.transition.inference import load_transition_model_from_artifact_dir
from geas35.rl.mdp_v1 import MdpV1ObservationScaler
from geas35.rl.sensor_history_env_v4 import SensorHistoryStateUpdater

VERSION = 'geas35.step14.hlv15_b60_r6.v1'
SELECTED = {'melon':'xgboost','cucumber':'extra_trees','strawberry':'lightgbm'}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()
class Identity:
    def transform_frame(self,f): return f.copy()
    def inverse_transform_frame(self,f): return f.copy()
class BoundTransitionRuntime:
    """Requires caller-pinned manifest hash; uses physical history and bound scaling.

    recorded_next supplies frozen exogenous inputs only. Endogenous state, aliases,
    histories and command carry are updated by SensorHistoryStateUpdater. Admission
    masks belong to dataset/raw supervisor, never to model-generated state.
    """
    def __init__(self, directory, *, crop, manifest_sha256, root):
        self.directory=Path(directory); root=Path(root)
        path=self.directory/'manifest.json'
        if sha(path)!=manifest_sha256: raise ValueError('Unbound manifest')
        m=json.loads(path.read_text()); self.manifest=m
        if m['version']!=VERSION or m['crop']!=crop or m['model_name']!=SELECTED[crop]:
            raise ValueError('Crop/algorithm/version mismatch')
        if m['dataset_version']!='past_only_qc_hlv15_b60_r6_v1': raise ValueError('Dataset namespace mismatch')
        for name, digest in m['bundle_hashes'].items():
            if sha(self.directory/name)!=digest: raise ValueError('Bundle hash mismatch: '+name)
        for record in m['input_bindings']:
            if sha(root/record['path'])!=record['sha256']: raise ValueError('Input hash mismatch: '+record['path'])
        schema=json.loads((self.directory/'schema.json').read_text())
        self.obs=schema['observation_columns']; self.actions=schema['actions']
        if len(self.obs)!=55 or len(self.actions)!=6 or m['input_order']!=self.obs+self.actions or m['action_order']!=self.actions or m['target_order']!=self.obs[:3]:
            raise ValueError('Input schema mismatch')
        self.scaler=MdpV1ObservationScaler(**json.loads((self.directory/'observation_scaler.json').read_text()))
        self.loaded=load_transition_model_from_artifact_dir(self.directory, expected_target_formulation=m['target_formulation'])
        self.model=self.loaded.model
        if list(self.model.target_columns_)!=m['target_order'] or list(self.model.input_columns_)!=m['input_order']:
            raise ValueError('Fitted schema mismatch')
        if asdict(self.model.observation_scaler)!=asdict(self.scaler): raise ValueError('Embedded scaler mismatch')
    def predict(self, physical, action):
        frame=physical.to_frame().T if isinstance(physical,pd.Series) else physical.copy()
        frame=self.scaler.transform_frame(frame)
        if not np.isfinite(frame[self.obs].to_numpy(float)).all(): raise ValueError('Invalid observation outside admitted domain')
        frame.loc[:,self.actions]=np.asarray(action,dtype=float).reshape(-1,len(self.actions))
        return self.model.predict(frame[self.obs+self.actions].astype(float)).next_observation
    def reset(self, initial, prehistory):
        self.updater=SensorHistoryStateUpdater(Identity(),prehistory)
        self.updater.reset(initial)
        return self.updater.current_physical.copy()
    def advance(self, action, recorded_next):
        current=self.updater.current_physical.copy()
        prediction=self.predict(current,action).iloc[0]
        self.updater.advance(current_scaled=current,recorded_next_scaled=recorded_next,
            prediction_physical=prediction.to_dict(),action=dict(zip(self.actions,action)))
        return prediction, self.updater.current_physical.copy()
