"""Non-training regressions. Fixtures never authorize optimizer execution."""
import ast
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from geas35.rl import profile_a_offline_v1 as m

ROOT = Path(__file__).resolve().parents[1]
TRACES = []


def row(identity='e', step=0, reward=1., next_value=7.):
    return dict(identity=identity, next_identity=identity, step=step, reward=reward, value=0., next_value=next_value,
                observation=np.zeros(56), next_observation=np.ones(56), policy_action=np.array([.8,.3,.2,0,0,1]),
                policy_log_prob=-2., bootstrap_mask=1, trace_mask=1)


class StagingTests(unittest.TestCase):
    def test_complete_masks(self):
        s=m.EpisodeStaging('e');s.append(row());s.finalize('COMPLETE');a=s.arrays()
        self.assertEqual(a['bootstrap_mask'].tolist(),[0]);self.assertEqual(a['trace_mask'].tolist(),[0]);self.assertEqual(s.numeric_return,1.)

    def test_truncated_masks(self):
        s=m.EpisodeStaging('e');s.append(row());s.truncate(np.ones(56),'e')
        self.assertEqual((s.rows[-1]['bootstrap_mask'],s.rows[-1]['trace_mask']),(1,0))
        with self.assertRaises(ValueError):s.arrays()

    def test_old_bug_and_new_reset_isolation(self):
        tree=ast.parse((ROOT/'src/geas35/rl/hybrid_ppo.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_gae')
        scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated_old_gae>','exec'),scope)
        old=scope['compute_gae'](np.array([1,100]),np.zeros(2),np.zeros(2),np.array([False,True]),gamma=1,gae_lambda=1)[0]
        self.assertEqual(old.tolist(),[101,100])
        for second in [100,10000]:
            a,_=m.masked_gae([1,second],[0,0],[7,0],[1,0],[0,0],['a','b'])
            self.assertAlmostEqual(a[0],1+.99*7)

    def test_reset_observation_rejected(self):
        s=m.EpisodeStaging('e');s.append(row())
        for obs,identity in [(np.zeros(56),'e'),(np.ones(56),'new_episode')]:
            with self.assertRaises(ValueError):s.truncate(obs,identity)

    def test_failure_prefix_no_synthetic_transition(self):
        s=m.EpisodeStaging('e');s.append(row());s.finalize('FAILURE')
        self.assertEqual(len(s.rows),1);self.assertEqual(s.arrays()['returns'].tolist(),[1.]);self.assertEqual(s.numeric_return,1.)

    def test_empty_failure(self):
        s=m.EpisodeStaging('e');s.finalize('FAILURE');self.assertIsNone(s.arrays());self.assertIsNone(s.numeric_return)

    def test_system_invalid_rollback(self):
        s=m.EpisodeStaging('e');s.append(row());s.finalize('SYSTEM_INVALID');self.assertIsNone(s.arrays());self.assertEqual(s.rows,[])

    def test_unscorable_rollback(self):
        s=m.EpisodeStaging('e');s.append(row());s.finalize('UNSCORABLE');self.assertIsNone(s.arrays());self.assertIsNone(s.numeric_return)

    def test_late_invalid_rolls_back_all_fragments(self):
        for category in ['SYSTEM_INVALID','UNSCORABLE']:
            s=m.EpisodeStaging('e');s.append(row());s.truncate(np.ones(56),'e');s.append(row(step=1));s.finalize(category)
            self.assertEqual(s.rows,[]);self.assertIsNone(s.arrays())

    def test_chunk_then_failure_replaces_bootstrap(self):
        s=m.EpisodeStaging('e');s.append(row());s.truncate(np.ones(56),'e');s.finalize('FAILURE')
        self.assertEqual(s.arrays()['returns'].tolist(),[1.]);self.assertEqual(s.rows[0]['bootstrap_mask'],0)

    def test_cross_identity_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):m.masked_gae([1,2],[0,0],[0,0],[1,0],[1,0],['e','f'])
        for bad in [None,float('nan'),float('inf')]:
            s=m.EpisodeStaging('e')
            with self.assertRaises(ValueError):s.append(row(reward=bad))

    def test_proposed_logprob_storage(self):
        s=m.EpisodeStaging('e');r=row();r['info']={'final_action':[0]*6};s.append(r);s.finalize('COMPLETE');a=s.arrays()
        np.testing.assert_array_equal(a['actions'][0],r['policy_action']);self.assertEqual(a['log_probs'][0],-2.)

    def test_complete_none_ranking(self):
        self.assertLess(m.development_rank(0,0,0,.5,2),m.development_rank(0,0,0,None,1))
        self.assertLess(m.development_rank(0,0,0,None,1),m.development_rank(0,0,0,None,2))


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=m.ProfileAOfflineEnv(ROOT,'train');cls.dev=m.ProfileAOfflineEnv(ROOT,'development')
        cls.short=min(cls.env.cohort.runs,key=lambda k:len(cls.env.cohort.runs[k]))

    def setUp(self):
        self.env.done=True
        self.env.reset(self.short)

    def normal_scores(self, state, action):
        return {'state':np.array([0.]),'joint':np.array([0.])}

    def run_fixture(self, kind):
        env=self.env;env.done=True;env.reset(self.short)
        collector=m.OfflineCollector(env)
        proposal=np.array([.8,.4,.3,1.,0.,1.]);thr=env.support.config['thresholds']
        def scores(state,actions):
            return {'state':np.array([thr['state_severe']+1 if kind=='empty_failure' else 0.]),
                    'joint':np.array([thr['joint_severe']+1 if kind=='fallback' else ((thr['joint_warning']+thr['joint_severe'])/2 if kind=='projection' else 0.)])}
        local={'continuous':{c:(0.,.1) for c in m.ACT[:3]},'binary':{c:[0] for c in m.ACT[3:]}}
        with patch.object(env.support,'scores',side_effect=scores),patch.object(env.support,'local_action_support',return_value=[local]),patch('geas35.rl.mdp_v1.compute_mdp_v1_reward',side_effect=AssertionError('LEGACY_REWARD_FORBIDDEN')):
            if kind=='invalid':
                with patch.object(env.runtime,'advance',side_effect=ValueError('invalid model fixture')):
                    result=collector.step(proposal,-3.,0.,lambda o:0.)
            elif kind=='unscorable':
                with patch.object(env.reward_adapter,'evaluate',return_value={'evaluation_status':'UNSCORABLE','reward':None}):
                    result=collector.step(proposal,-3.,0.,lambda o:0.)
            else:
                result=collector.step(proposal,-3.,0.,lambda o:0.)
        TRACES.append({'case':kind,'fixture':'scores/local support controlled; real frozen Transition/recurrence unless invalid fixture',
                       'reward':result[1],'done':result[2],'info':result[4]})
        return collector,result

    def test_profile_bound_reset_56(self):
        self.assertEqual(self.env.observation().shape,(56,));self.assertEqual(self.env.reward_adapter.journal.identity,self.env.identity)
        self.assertEqual(self.env.fixed_metadata['profile_hash'],m.PROFILE_HASH)

    def test_no_automatic_switch(self):
        h=self.env.reward_adapter.binding_identity
        self.env.runtime.adapter.current['obs_growth_stage_dat']=191
        self.env.runtime.adapter.current['obs_is_daytime']=0
        self.assertEqual(h,self.env.reward_adapter.binding_identity)
        bound=self.env.reward_adapter.bind(0)
        self.assertEqual(self.env.reward_adapter.journal.lookup(bound[0]['temperature']).data['reference_value'],30.)

    def test_exact_channels_and_frozen_reward(self):
        reward=self.env.reward_adapter;bound=reward.bind(0)
        p=dict(zip(m.TARGETS,[30.,39.1,90.,69.53]));result=reward.evaluate(bound,p)
        self.assertEqual(result['evaluation_status'],'VALID');self.assertAlmostEqual(result['reward'],2/3)
        self.assertAlmostEqual(result['normalized_error_T1'],0);self.assertAlmostEqual(result['normalized_error_T2'],1)
        p.pop(m.TARGETS[1]);self.assertEqual(reward.evaluate(bound,p)['evaluation_status'],'UNSCORABLE')

    def test_projection_pipeline_and_logprob(self):
        c,result=self.run_fixture('projection');info=result[4]
        self.assertNotEqual(info['proposed'],info['final_action']);self.assertEqual(c.env.staging.rows[0]['policy_log_prob'],-3.)
        self.assertEqual(info['order'],['proposal','joint_ood','intervention','weather_cap','transition','derived_history_carry','quality','accounting'])

    def test_fallback_pipeline(self):
        _,r=self.run_fixture('fallback');self.assertEqual(r[4]['intervention'],'fallback');self.assertIsNotNone(r[4]['fallback_action'])

    def test_system_invalid_fixture(self):
        c,r=self.run_fixture('invalid');self.assertEqual(r[4]['category'],'SYSTEM_INVALID');self.assertIsNone(c.env.staging.arrays())

    def test_unscorable_fixture(self):
        c,r=self.run_fixture('unscorable');self.assertEqual(r[4]['category'],'UNSCORABLE');self.assertIsNone(c.env.staging.arrays())

    def test_failure_before_action_fixture(self):
        c,r=self.run_fixture('empty_failure');self.assertEqual(r[4]['category'],'FAILURE');self.assertIsNone(r[1]);self.assertEqual(len(c.env.staging.rows),0)

    def test_complete_real_model_fixture(self):
        e=self.env;c=m.OfflineCollector(e);count=0
        with patch.object(e.support,'scores',side_effect=self.normal_scores),patch('geas35.rl.mdp_v1.compute_mdp_v1_reward',side_effect=AssertionError('LEGACY')) as sentinel:
            while not e.done:
                r=c.step(np.array([.2,0,0,0,0,0]),-1.,0.,lambda o:0.);count+=1
                self.assertLessEqual(count,len(e.run))
            self.assertEqual(sentinel.call_count,0)
        self.assertEqual(r[4]['category'],'COMPLETE');self.assertEqual(len(e.staging.rows),len(e.run));self.assertIsNotNone(e.staging.arrays())
        TRACES.append({'case':'COMPLETE','fixture':'support scores normal; actual model/recurrence; natural shortest Train run','steps':count,'return':e.staging.numeric_return})

    def test_truncated_continuation_fixture(self):
        e=self.env;long=next(k for k,g in e.cohort.runs.items() if len(g)>3);e.done=True;e.reset(long);c=m.OfflineCollector(e)
        with patch.object(e.support,'scores',side_effect=self.normal_scores):
            r=c.step(np.array([.2,0,0,0,0,0]),-1,0,lambda o:7.)
            self.assertFalse(r[2]);c.truncate(r[0],e.identity)
            with self.assertRaises(ValueError):e.staging.arrays()
            with self.assertRaises(ValueError):c.truncate(np.zeros(56),'wrong')
            r=c.step(np.array([.2,0,0,0,0,0]),-1,0,lambda o:7.)
            self.assertEqual(len(e.staging.rows),2)
        e.staging.finalize('FAILURE');e.done=True
        self.assertEqual(e.staging.rows[-1]['bootstrap_mask'],0)
        TRACES.append({'case':'TRUNCATED then FAILURE','fixture':'explicit administrative cut and terminal accounting','masks':[[r['bootstrap_mask'],r['trace_mask']] for r in e.staging.rows]})

    def test_single_worker_and_multi_reject(self):
        self.assertIsInstance(m.OfflineCollector(self.env,1),m.OfflineCollector)
        for n in [0,2,4]:
            with self.assertRaises(ValueError):m.OfflineCollector(self.env,n)
            with self.assertRaises(ValueError):m.ProfileAOfflineEnv(ROOT,num_workers=n)

    def test_exact_cohorts(self):
        train=self.env.cohort;dev=self.dev.cohort
        self.assertEqual(sum(len(g) for g in train.runs.values()),11380);self.assertEqual(sum(len(g) for g in dev.runs.values()),2811)
        self.assertEqual(len(train.registry['train_episode_ids']),43);self.assertEqual(len(dev.registry['development_episode_ids']),11)
        self.assertFalse(set(train.registry['train_episode_ids'])&set(dev.registry['development_episode_ids']))

    def test_wrong_mode_reset_rejected(self):
        self.env.done=True
        with self.assertRaises(ValueError):self.env.reset(next(iter(self.dev.cohort.runs)))

    def test_validation_and_test_forbidden_before_io(self):
        with patch.object(pd,'read_parquet',side_effect=AssertionError('MUST_NOT_READ')):
            for mode in ['validation','test','p0','future_f']:
                with self.assertRaises(ValueError):m.Cohort(ROOT,mode)

    def test_no_future_indoor_in_exogenous(self):
        r=self.env.run.iloc[0].copy();r[self.env.obs]=r[['next_'+c for c in self.env.obs]].to_numpy(float);r['reg_date']=r.target_timestamp
        ex=self.env.runtime.adapter.exogenous(r)
        self.assertFalse(any(c in ex for c in m.TARGETS))
        a=dict(zip(m.ACT,[0]*6));x=self.env.runtime.adapter.model_input(a)
        r[m.TARGETS]=99999
        pd.testing.assert_series_equal(ex,self.env.runtime.adapter.exogenous(r))
        pd.testing.assert_series_equal(x,self.env.runtime.adapter.model_input(a))

    def test_checkpoint_binding_and_budget(self):
        meta=self.env.checkpoint_metadata();self.assertEqual(meta['gamma'],.99);self.assertEqual(meta['gae_lambda'],.95)
        self.assertIsNone(meta['budget_protocol_hash']);self.assertFalse(meta['training_authorized'])
        self.env.done=True
        for key in meta:
            bad=dict(meta);bad[key]='tampered'
            with self.assertRaises(ValueError):self.env.validate_resume(bad)
        self.assertTrue(self.env.validate_resume(meta))
        with self.assertRaises(RuntimeError):self.env.authorize_training()

    def test_deterministic_real_support_trace(self):
        results=[]
        for _ in range(2):
            self.env.done=True;o,info=self.env.reset(self.short,42);c=m.OfflineCollector(self.env)
            r=c.step(np.array([.2,0,0,0,0,0]),-1,0,lambda o:0.)
            results.append((o,r[0],r[1],r[4]['category'],r[4]['final_action']))
        for i in [0,1]:np.testing.assert_array_equal(results[0][i],results[1][i])
        self.assertEqual(results[0][2:],results[1][2:])
        TRACES.append({'case':'real frozen support repeat','fixture':False,'category':results[0][3],'reward':results[0][2],'final':results[0][4]})

    def test_parent_project_fallback_parity(self):
        src=ROOT/'experiments/rl_policy_training/scripts/validate_step16_four_channel_v1.py';tree=ast.parse(src.read_text())
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['project','fallback']]
        scope={'np':np,'ACT':m.ACT,'CFG':m.CFG,'TARGETS':m.TARGETS,'I':{k:i for i,k in enumerate(self.env.obs)}}
        exec(compile(ast.Module(body=funcs,type_ignores=[]),'<frozen parent functions>','exec'),scope)
        local={'continuous':{c:(.1,.7) for c in m.ACT[:3]},'binary':{c:[0,1] for c in m.ACT[3:]}}
        s=self.env.runtime.adapter.current.copy()
        for temp in [5,40]:
            for rh in [30,95]:
                s[m.TARGETS[0]]=temp;s[m.TARGETS[2]]=rh
                old=scope['fallback'](s[self.env.obs].to_numpy(float),local);new=m.fallback(s,local)
                np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1]);self.assertEqual(old[2],new[2])
        local['binary'][m.ACT[-1]]=[];self.assertTrue(m.project(np.zeros(6),local)[1])

    def test_actual_support_bounded_trace(self):
        e=self.env
        starts=pd.DataFrame([g.iloc[0][e.obs] for g in e.cohort.runs.values()])
        scaled=e.runtime.scaler.transform_frame(starts)
        scores=e.support.scores(scaled,pd.DataFrame(np.zeros((len(starts),6)),columns=m.ACT))
        candidates=[k for k,score in zip(e.cohort.runs,scores['state']) if score<=e.support.config['thresholds']['state_warning'] and len(e.cohort.runs[k])>=3]
        self.assertTrue(candidates)
        e.done=True;e.reset(candidates[0]);c=m.OfflineCollector(e);executed=0
        with patch('geas35.rl.mdp_v1.compute_mdp_v1_reward',side_effect=AssertionError('LEGACY')) as sentinel:
            for _ in range(3):
                proposal=e.run.iloc[e.cursor][m.ACT].to_numpy(float) # fixed recorded fixture, not an imitation objective
                result=c.step(proposal,-1.,0.,lambda obs:0.)
                executed+=int(result[1] is not None)
                TRACES.append({'case':'actual Train three-step bounded trace','fixture':'recorded proposals only; no scores/model override','reward':result[1],'info':result[4]})
                if result[2]:break
            self.assertEqual(sentinel.call_count,0)
        self.assertGreater(executed,0)
        if not e.done:c.truncate(result[0],e.identity)

    def test_real_nonfinite_output_quarantine(self):
        e=self.env
        with patch.object(e.support,'scores',side_effect=self.normal_scores),patch.object(e.runtime.model,'predict') as pred:
            pred.return_value.next_observation=pd.DataFrame([[float('nan'),25,70,70]],columns=m.TARGETS)
            result=m.OfflineCollector(e).step(np.zeros(6),-1,0,lambda obs:0)
            self.assertEqual(result[4]['category'],'SYSTEM_INVALID');self.assertIsNone(e.staging.arrays())

    def test_missing_binding_unscorable(self):
        e=self.env
        with patch.object(e.support,'scores',side_effect=self.normal_scores),patch.object(e.reward_adapter,'bind',side_effect=ValueError('missing binding')):
            r=m.OfflineCollector(e).step(np.zeros(6),-1,0,lambda obs:0)
            self.assertEqual(r[4]['category'],'UNSCORABLE');self.assertIsNone(e.staging.numeric_return)

    def test_null_and_nonfinite_no_zero(self):
        for bad,category in [(None,'UNSCORABLE'),(float('nan'),'SYSTEM_INVALID')]:
            e=self.env;e.done=True;e.reset(self.short)
            with patch.object(e.support,'scores',side_effect=self.normal_scores),patch.object(e.reward_adapter,'evaluate',return_value={'evaluation_status':'VALID','reward':bad}):
                r=m.OfflineCollector(e).step(np.zeros(6),-1,0,lambda obs:0)
                self.assertIsNone(r[1]);self.assertEqual(r[4]['category'],category)

    def test_policy_nonfinite_quarantine(self):
        r=m.OfflineCollector(self.env).step(np.zeros(6),float('nan'),0,lambda obs:0)
        self.assertEqual(r[4]['category'],'SYSTEM_INVALID');self.assertIsNone(self.env.staging.arrays())

    def test_actor_structure_without_training(self):
        import torch
        from geas35.rl.hybrid_ppo import HybridActorCritic,HybridPPOConfig,FeasibleActionSpec
        torch.manual_seed(42)
        model=HybridActorCritic(HybridPPOConfig(observation_dim=56,gamma=.99,gae_lambda=.95))
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        feasible=FeasibleActionSpec.unconstrained().batched(1,device='cpu')
        with torch.no_grad(),patch.object(torch.optim.Optimizer,'step',side_effect=AssertionError('OPTIMIZER_FORBIDDEN')):
            obs=torch.zeros((1,56));a=model.act(obs,feasible)
            rec=model.evaluate_actions(obs,a['action'],feasible)
            torch.testing.assert_close(a['log_prob'],rec['log_prob'])
            altered=a['action'].clone();altered[:,:3]=.05
            changed=model.evaluate_actions(obs,altered,feasible)
            self.assertFalse(torch.allclose(a['log_prob'],changed['log_prob']))
            self.assertTrue(torch.all((a['action'][:,:3]>=0)&(a['action'][:,:3]<=1)))
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
