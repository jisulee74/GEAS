"""HLV15 causal B60/R6 only; branch injection tests never mutate official artifacts."""
import numpy as np
import pytest
from pathlib import Path
from geas35.rl.step16_hlv15_b60_r6_env import B60R6ResearchEnv
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def env():return B60R6ResearchEnv(root=ROOT,crop='melon')
def test_all_exact_transitions_including_terminal_target(env):
 assert sum(e['scheduled_valid_steps'] for e in env.episodes)==len(env.physical_source)
 for e in env.episodes:
  final=env.frame.iloc[e['final_observation']]
  previous=env.frame.iloc[e['final_observation']-1]
  assert final.reg_date==previous.target_timestamp
  np.testing.assert_allclose(final[list(env.observation_columns)].to_numpy(float),env.scaler.transform_frame(env.physical_source.loc[env.physical_source.row_id.eq(previous.row_id)].rename(columns={c:'unused_'+c for c in env.observation_columns}).rename(columns={'next_'+c:c for c in env.observation_columns}))[list(env.observation_columns)].to_numpy(float)[0],rtol=1e-6,atol=1e-8)
def test_reset_current_action_does_not_supply_previous_history(env):
 before,_=env.reset(seed=42,options={'episode_index':0});p=env._position;actions=list(env.action_columns);saved=env.frame.loc[p,actions].copy();prev=env._last_action.copy();pp=env._last_last_action.copy()
 try:
  env.frame.loc[p,actions]=1-saved.to_numpy(float)
  after,_=env.reset(seed=42,options={'episode_index':0})
  np.testing.assert_array_equal(before,after);assert env._last_action==prev and env._last_last_action==pp
 finally:env.frame.loc[p,actions]=saved

def test_first_state_ood_does_not_predict_or_execute(env,monkeypatch):
 env.reset(seed=42,options={'episode_index':0});threshold=env.support.config['thresholds']['state_severe']
 monkeypatch.setattr(env.support,'scores',lambda state,action:{'state':np.array([threshold+1]),'joint':np.array([0.])})
 def forbidden(*args,**kwargs):raise AssertionError('State severe must terminate before Transition')
 monkeypatch.setattr(env.model,'predict',forbidden)
 _,reward,terminated,truncated,info=env.step(np.zeros(6))
 assert terminated and not truncated and env._executed_steps==0
 assert info['termination_reason']=='severe_state_ood' and info['executed_action'] is None
 assert reward==-env._episode['scheduled_valid_steps']

def test_three_joint_severe_events_terminate_and_charge_remaining(env,monkeypatch):
 env.reset(seed=42,options={'episode_index':0});threshold=env.support.config['thresholds']['joint_severe']
 monkeypatch.setattr(env.support,'scores',lambda state,action:{'state':np.array([0.]),'joint':np.array([threshold+1])})
 for i in range(3):
  _,reward,terminated,truncated,info=env.step(np.zeros(6));assert info['safety_fallback_applied'];assert np.isfinite(reward)
  assert terminated==(i==2)
 assert info['termination_reason']=='repeated_severe_action_ood'
 assert info['pessimistic_terminal_cost']==-(env._episode['scheduled_valid_steps']-3)
 assert reward==-1+info['pessimistic_terminal_cost']
