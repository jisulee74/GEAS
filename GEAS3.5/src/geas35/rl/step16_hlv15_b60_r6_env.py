"""B60/R6 research wrapper around unchanged Step16 safety/reward step logic."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from geas35.rl.model_driven_env import GeasModelDrivenEnv
from geas35.rl.bound_support_hlv15_b60_r6_v1 import load_bound_research_artifacts
from geas35.models.transition.handoff_hlv15_b60_r6_v1 import sha
from geas35.rl.sensor_history_env_v4 import SensorHistoryStateUpdater

VERSION='geas35.rl.step16.hlv15_b60_r6.v1'
class PredictionAudit:
    def __init__(self,model):self.model=model;self.last_input=None;self.last_prediction=None
    def __getattr__(self,name):return getattr(self.model,name)
    def predict(self,frame):
        self.last_input=frame.copy();result=self.model.predict(frame);self.last_prediction=result.next_observation.copy();return result
class B60R6ResearchEnv(GeasModelDrivenEnv):
    def __init__(self,*,root,crop):
        root=Path(root);s=root/'experiments/rl_policy_training/artifacts/step15_hlv15_b60_r6_v1'
        entry_path=root/'experiments/rl_policy_training/artifacts/step16_hlv15_b60_r6_v1/protocol.json'
        lock=json.loads(entry_path.with_name('protocol_lock.json').read_text())
        if sha(entry_path)!=lock['sha256']:raise ValueError('Unbound Step16 protocol')
        entry=json.loads(entry_path.read_text());bound=entry['research_environment_bindings'][crop]
        rt,support=load_bound_research_artifacts(root/bound['path'],binding_sha256=bound['sha256'],crop=crop,root=root)
        p=root/'experiments/rl_policy_training/artifacts/past_only_qc_hlv15_b60_r6_v1'/crop
        physical=pd.read_parquet(p/'validation_physical.parquet');self.physical_source=physical
        self.qc=pd.read_parquet(p/'validation_prepared_qc.parquet')
        self.provenance=pd.read_parquet(p/'validation_provenance.parquet').set_index('row_id',verify_integrity=True)
        pieces=[];episodes=[];offset=0
        for rollout,group in physical.groupby('mdp_v1_rollout_id',sort=False):
            group=group.copy();n=len(group);first=group.iloc[0];last=group.iloc[-1]
            assert group.episode_id.nunique()==1
            assert (pd.to_datetime(group.target_timestamp)-pd.to_datetime(group.reg_date)).eq(pd.Timedelta(minutes=5)).all()
            if n>1:assert np.array_equal(group.target_timestamp.iloc[:-1],group.reg_date.iloc[1:])
            target=last.copy();target[rt.obs]=last[['next_'+c for c in rt.obs]].to_numpy(float);target['reg_date']=last.target_timestamp;target['row_id']='target:'+str(last.row_id);target['rl_valid_transition']=0
            pieces.append(pd.concat([group,target.to_frame().T],ignore_index=True))
            episodes.append({'start':offset,'final_observation':offset+n,'scheduled_valid_steps':n,'episode_id':str(first.episode_id),'series_id':str(first.series_id),'segment_id':str(first.segment_id),'rollout_id':str(rollout),'date':str(pd.Timestamp(first.reg_date).date())});offset+=n+1
        frame=rt.scaler.transform_frame(pd.concat(pieces,ignore_index=True))
        super().__init__(crop=crop,split='validation',frame=frame,transition_bundle=rt.loaded,observation_scaler=rt.scaler,support_model=support)
        self.bound_runtime=rt;self.episodes=episodes;self.model=PredictionAudit(rt.model)
        self.state_updater=SensorHistoryStateUpdater(self.scaler,self.qc)
        assert sum(e['scheduled_valid_steps'] for e in episodes)==len(physical)
    def reset(self,*,seed=None,options=None):
        _,info=super().reset(seed=seed,options=options)
        row=self.frame.iloc[self._position];pro=self.provenance.loc[row.row_id]
        self._last_action={a:float(pro['previous_'+a]) for a in self.action_columns}
        self._last_last_action={a:float(pro['previous_previous_'+a]) for a in self.action_columns}
        assert np.isfinite(list(self._last_action.values())+list(self._last_last_action.values())).all()
        np.testing.assert_allclose(self._current_physical[['obs_prev_'+a for a in self.action_columns]].to_numpy(float),list(self._last_action.values()),rtol=1e-6,atol=1e-8)
        self._current_scaled=self.scaler.transform_frame(self._current_physical.to_frame().T).iloc[0]
        np.testing.assert_allclose(self._current_scaled[list(self.observation_columns)].to_numpy(float),row[list(self.observation_columns)].to_numpy(float),rtol=1e-6,atol=1e-8)
        self.model.last_input=None;self.model.last_prediction=None
        info['environment_version']=VERSION
        return self._observation(),info
    def step(self,action):
        result=super().step(action);result[-1]['environment_version']=VERSION;return result
