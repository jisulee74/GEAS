"""Non-training synthetic semantic boundary tests; no dataset/model access."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from geas35.rl import profile_a_final_action_env_v1 as new
from geas35.rl.profile_a_offline_v1 import EpisodeStaging, OfflineCollector, ACT

class Support:
    config={'thresholds':{'state_warning':.5,'state_severe':.6,'joint_warning':.4,'joint_severe':.5}}
    def __init__(self,props,finals,state=0.):self.props=props;self.finals=finals;self.state=state;self.queries=[]
    def scores(self,s,a):
        i=len(self.queries);self.queries.append((s.copy(),a.copy()))
        values=self.props if i%2==0 else self.finals
        return {'state':np.array([self.state]),'joint':np.array([values[min(i//2,len(values)-1)]])}
    def local_action_support(self,s):
        return [{'continuous':{k:[0.,1.] for k in ACT[:3]},'binary':{k:[0,1] for k in ACT[3:]}}]

class Reward:
    def bind(self,i):return ({'x':str(i)},None)
    def evaluate(self,b,p):return {'evaluation_status':'VALID','reward':.375,'quality_reward':.375,'temperature_quality':.5,'rh_quality':.25}

def env(props=(.2,),finals=(.2,),state=0.,n=8):
    e=new.FinalActionProfileAEnv.__new__(new.FinalActionProfileAEnv)
    e.done=False;e.counter=0;e.final_counter=0;e.cursor=0;e.identity='synthetic';e.obs=['x']
    current=pd.Series({'x':0.,'obs_indoor_temp1_c':25.,'obs_indoor_humidity1_pct':65.,'obs_current_target_temp_min_c':23.,'obs_current_target_temp_max_c':25.,'obs_is_daytime':1.,'obs_current_vpd_kpa':1.,'obs_outdoor_wind_speed':0.,'obs_rain_flag':0.})
    e.calls=[];e.runtime=SimpleNamespace(adapter=SimpleNamespace(current=current,exogenous=lambda x:x))
    def advance(action,exo):
        e.calls.append(action.copy());return pd.Series({'T1':25.,'T2':24.,'RH1':65.,'RH2':66.}),current
    e.runtime.advance=advance;e.observation=lambda:np.array([0.]);e.last_valid_observation=np.array([0.])
    e.support=Support(list(props),list(finals),state);e.reward_adapter=Reward();e.staging=EpisodeStaging(e.identity)
    e.run=pd.DataFrame({'x':[0.]*n,'next_x':[0.]*n,'target_timestamp':pd.date_range('2026-01-01 00:05',periods=n,freq='5min')})
    return e

def step(e,a=None):return OfflineCollector(e).step(np.array([.8,.5,.5,0,0,1]) if a is None else a,-1.,0.,lambda o:0.)

def test_proposal_diagnostic():
 e=env([.7],[.2]);[step(e) for _ in range(4)];assert e.counter==4 and e.final_counter==0 and not e.done

def test_final_authority():
 e=env([.2],[.7]);r=[step(e) for _ in range(3)];assert [x[2] for x in r]==[False,False,True];assert r[-1][4]['reason']=='executed_joint_support_failure' and len(e.calls)==3

def test_final_reset():
 e=env([.2],[.7,.7,.2,.7]);cs=[]
 for _ in range(4):step(e);cs.append(e.final_counter)
 assert cs==[1,2,0,1] and not e.done

def test_final_increment():
 e=env([.2],[.7]);r=step(e);assert e.counter==0 and r[4]['final_severe_streak']==1

def test_warning_executable():
 e=env([.2],[.7,.45]);step(e);r=step(e);assert e.final_counter==0 and r[4]['final_joint_status']=='warning' and r[1]==.375 and not e.done

def test_state_preaction():
 e=env(state=.7)
 with patch.object(new,'fallback',side_effect=AssertionError('fallback prohibited')):
  o,reward,done,_,i=step(e)
 assert done and reward is None and len(e.calls)==0 and i['final_action'] is None and i['final_joint_distance'] is None and len(e.staging.rows)==0

def test_natural_priority():
 e=env([.2],[.7],n=3);[step(e) for _ in range(3)];assert e.staging.category=='COMPLETE' and e.final_counter==3

def test_fallback_parity():
 e=env([.7],[.2]);s=e.runtime.adapter.current;local=e.support.local_action_support(None)[0];raw,expected,bad=new.legacy.fallback(s,local);i=step(e)[4]
 assert not bad and i['fallback_action']==expected.tolist()

def test_weather_parity():
 e=env();e.runtime.adapter.current['obs_rain_flag']=1.;i=step(e)[4]
 expected,flags=new.legacy.project_mdp_v1_action_constraints(dict(zip(ACT,i['intermediate_action'])),e.runtime.adapter.current.to_dict(),new.CFG)
 assert i['final_action']==[expected[k] for k in ACT] and flags['weather_constraint_applied']

def test_post_weather_query():
 e=env();e.runtime.adapter.current['obs_rain_flag']=1.;i=step(e)[4];a,b=e.support.queries
 assert np.array_equal(a[0].to_numpy(),b[0].to_numpy())
 assert b[1].iloc[0].tolist()==i['final_action'] and i['intermediate_action']!=i['final_action']
 assert list(e.calls[0].values())==i['final_action']

def test_reward_exact():
 e=env([.7],[.2]);r=step(e);assert r[1]==.375 and r[4]['quality']==Reward().evaluate(None,None)

def test_transition_input_exact():
 e=env();i=step(e)[4];assert list(e.calls[0])==ACT and list(e.calls[0].values())==i['final_action']

def test_no_ood_penalty():
 rewards=[]
 for p,f in [(.2,.2),(.45,.45),(.7,.2),(.7,.7)]:rewards.append(step(env([p],[f]))[1])
 assert rewards==[.375]*4

def test_invalid_accounting():
 e=env();step(e)
 e.support.finals=[float('nan')];r=step(e);assert r[4]['category']=='SYSTEM_INVALID' and e.staging.rows==[] and r[4]['final_joint_distance'] is None
 e=env();step(e)
 e.reward_adapter.bind=lambda i: (_ for _ in ()).throw(ValueError('invalid binding'))
 r=step(e);assert r[4]['category']=='UNSCORABLE' and e.staging.rows==[]
 e=env([.2],[.7]);[step(e) for _ in range(3)];assert len(e.staging.rows)==3 and e.staging.rows[-1]['bootstrap_mask']==e.staging.rows[-1]['trace_mask']==0
 e=env();e.support.local_action_support=lambda s:[{'continuous':{k:[] for k in ACT[:3]},'binary':{k:[] for k in ACT[3:]}}];e.support.props=[.7]
 r=step(e);assert r[4]['reason']=='unsupported_local_support_failure' and len(e.calls)==0

def test_checkpoint_env_binding():
 e=env()
 with pytest.raises(ValueError,match='SEMANTIC_MISMATCH'):e.validate_resume({'implementation_hash':'legacy'})
 with pytest.raises(ValueError,match='EXPLICIT_ENVIRONMENT_VERSION_REQUIRED'):new.FinalActionProfileAEnv('/unused',semantic_version='legacy')
 with pytest.raises(RuntimeError,match='PREREGISTRATION_REQUIRED'):e.authorize_training()
