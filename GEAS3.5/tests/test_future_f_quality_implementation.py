import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from geas35.objectives.quality import QualityEvaluator,S_T,S_RH,SCALE_HASH,SCALAR_HASH
from geas35.objectives.binding import FixtureJournal,TargetContext
from test_future_f_target_binding import make,T1,T2,T3
from test_future_f_quality_prereg import reference_spec,ART

class QualityImplementationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='scalar_fixture_');self.j=FixtureJournal(self.tmp.name)
        self.q=QualityEvaluator.from_file(ART);self.bindings={};self.hashes={};self.ctx={}
        for c,u,v in [('temperature','°C',17.25),('RH','%RH',47.5)]:
            b=make(component=c,unit=u,reference_value=v,component_mask=['temperature','RH'])
            self.j.commit(b,T1);self.bindings[c]=b;self.hashes[c]=b.target_hash;self.ctx[c]=TargetContext.from_binding(b)
        self.j.action('synthetic-A',list(self.hashes.values()),T2)
        self.outcome={'action_id':'synthetic-A','observed_at':T3,'binding_hashes':self.hashes.copy()}
    def tearDown(self):self.j.close();self.tmp.cleanup()
    def runq(self,errors=(0,0,0,0),**kw):
        obs=dict(zip(('T1','T2','RH1','RH2'),[17.25+errors[0],17.25+errors[1],47.5+errors[2],47.5+errors[3]]))
        args=dict(journal=self.j,binding_hashes=self.hashes,observations=obs,outcome=self.outcome,policy_contexts=self.ctx,environment_contexts=self.ctx)
        args.update(kw);return self.q.evaluate(**args)
    def test_frozen_identity(self):
        o=self.runq();self.assertEqual(S_T,9.100000000000001);self.assertEqual(S_RH,20.470000000000006)
        self.assertEqual(o['scale_hash'],SCALE_HASH);self.assertEqual(o['scalarization_hash'],SCALAR_HASH)
    def test_scale_hash_tamper(self):
        x=json.loads(ART.read_text());x['calibration']['scale_payload']['scales']['temperature']['scale']=1
        with self.assertRaisesRegex(ValueError,'SCALE_HASH'):QualityEvaluator(json.dumps(x))
    def test_scalar_hash_tamper(self):
        x=json.loads(ART.read_text());x['scalarization_contract']['formula']='changed'
        with self.assertRaisesRegex(ValueError,'SCALARIZATION_HASH'):QualityEvaluator(json.dumps(x))
    def test_zero(self):
        o=self.runq();self.assertEqual([o[k] for k in ('temperature_quality','rh_quality','quality_reward')],[1,1,1])
    def test_monotonic_all_channels(self):
        for ch in range(4):
            values=[]
            for v in (0,1,5,20):
                e=[0]*4;e[ch]=v;values.append(self.runq(e)['reward'])
            self.assertEqual(values,sorted(values,reverse=True));self.assertEqual(len(set(values)),4)
    def test_permutations(self):
        a=self.runq((1,7,3,15))['reward']
        self.assertEqual(a,self.runq((7,1,3,15))['reward']);self.assertEqual(a,self.runq((1,7,15,3))['reward'])
    def test_exact_fusion_weights_transform(self):
        o=self.runq((1,7,3,15));et=((1/S_T)+(7/S_T))/2;er=((3/S_RH)+(15/S_RH))/2
        self.assertEqual(o['fused_temperature_error'],et);self.assertEqual(o['fused_rh_error'],er)
        self.assertEqual(o['reward'],.5/(1+et)+.5/(1+er))
    def test_oracle_grid(self):
        for t in (0,1,9,100,1e308):
            for h in (-40,0,5,40):
                o=self.runq((t,t,h,-h));oracle=reference_spec(o['channels'],{'temperature':S_T,'RH':S_RH})
                self.assertTrue(math.isclose(o['reward'],oracle,rel_tol=1e-12,abs_tol=1e-15))
                self.assertTrue(0<=o['reward']<=1 and math.isfinite(o['reward']))
    def test_deterministic_serialization(self):
        a=json.dumps(self.runq((1,2,3,4)),sort_keys=True,allow_nan=False);b=json.dumps(self.runq((1,2,3,4)),sort_keys=True,allow_nan=False);self.assertEqual(a,b)
    def test_missing_each_channel(self):
        for ch in ('T1','T2','RH1','RH2'):
            obs={'T1':17.25,'T2':17.25,'RH1':47.5,'RH2':47.5};del obs[ch];o=self.runq(observations=obs)
            self.assertEqual(o['evaluation_status'],'UNSCORABLE');self.assertIsNone(o['reward']);self.assertIsNone(o['quality_reward'])
            self.assertTrue(any(r['reason']=='MISSING_SENSOR_VALUE' for r in o['reasons']))
    def test_nonfinite(self):
        o=self.runq(observations={'T1':float('nan'),'T2':1,'RH1':1,'RH2':1});self.assertIsNone(o['reward']);json.dumps(o,allow_nan=False)
    def test_no_binding(self):
        o=self.runq(binding_hashes={});self.assertIsNone(o['reward']);self.assertTrue(any(r['reason']=='TARGET_NOT_BOUND' for r in o['reasons']))
    def test_parity_mismatch(self):
        ctx=self.ctx.copy();ctx['RH']=replace(ctx['RH'],target_hash='wrong');o=self.runq(policy_contexts=ctx);self.assertIsNone(o['reward']);self.assertTrue(any(r['reason']=='TARGET_CONTEXT_MISMATCH' for r in o['reasons']))
    def test_ambiguous_and_hash_link(self):
        for changes in ({'ambiguous':True},{'action_id':'wrong'},{'binding_hashes':{'temperature':'wrong','RH':self.hashes['RH']}}):
            o=self.runq(outcome=dict(self.outcome,**changes));self.assertIsNone(o['reward']);self.assertTrue(any(r['reason']=='AMBIGUOUS_ACTION_LINKAGE' for r in o['reasons']))
    def test_excluded_terms_do_not_change_reward(self):
        a=self.runq();obs={'T1':17.25,'T2':17.25,'RH1':47.5,'RH2':47.5,'VPD':999,'heat':1,'vent':100,'switching':999,'terminal_cost':999};self.assertEqual(a['reward'],self.runq(observations=obs)['reward'])
    def test_late_commit_error_preserved(self):
        for field,reason in [('available_at','TARGET_NOT_AVAILABLE_PRE_ACTION'),('effective_from','TARGET_NOT_EFFECTIVE')]:
            b=make(action_id='synthetic-late-'+field,**{field:T2})
            try:self.j.commit(b,T1)
            except ValueError as e:o=self.q.unavailable([{'reason':str(e)}])
            else:self.fail('late commit accepted')
            self.assertIsNone(o['reward']);self.assertEqual(o['reasons'][0]['reason'],reason)
    def test_extreme_all_four_discrepancies(self):
        with tempfile.TemporaryDirectory() as path:
            j=FixtureJournal(path);hs={};ctx={}
            for c,u in [('temperature','°C'),('RH','%RH')]:
                b=make(component=c,unit=u,reference_value=-1e308,component_mask=['temperature','RH']);j.commit(b,T1);hs[c]=b.target_hash;ctx[c]=TargetContext.from_binding(b)
            j.action('synthetic-A',list(hs.values()),T2)
            o=self.q.evaluate(j,hs,{'T1':0.,'T2':0.,'RH1':0.,'RH2':0.},dict(self.outcome,binding_hashes=hs),ctx,ctx)
            self.assertTrue(math.isfinite(o['reward']) and 0<=o['reward']<=1)
            self.assertTrue(math.isclose(o['reward'],reference_spec(o['channels'],{'temperature':S_T,'RH':S_RH}),rel_tol=1e-12,abs_tol=0))
            json.dumps(o,allow_nan=False);j.close()
    def test_upstream_failure_envelope(self):
        for reason in ('TARGET_NOT_EFFECTIVE','TARGET_NOT_AVAILABLE_PRE_ACTION','NOT_ACTIVE_CONTEXT_UNRESOLVED','UNSUPPORTED_UNIT'):
            o=self.q.unavailable([{'reason':reason}]);self.assertIsNone(o['reward']);self.assertEqual(o['evaluation_status'],'UNSCORABLE');self.assertEqual(o['reasons'][0]['reason'],reason)
    def test_real_inactive_and_unit_paths(self):
        for mode in ('inactive','unit'):
            with tempfile.TemporaryDirectory() as path:
                j=FixtureJournal(path);ctx={};hs={}
                for c,u in [('temperature','°C'),('RH','%RH')]:
                    mask=['temperature'] if mode=='inactive' else ['temperature','RH']
                    changes={}
                    if c=='RH' and mode=='inactive':changes=dict(active=False,inactive_reason='NOT_ACTIVE_CONTEXT_UNRESOLVED',activation_context={},reference_value=None)
                    b=make(component=c,unit='bad' if mode=='unit' and c=='RH' else u,component_mask=mask,**changes)
                    j.commit(b,T1);hs[c]=b.target_hash;ctx[c]=TargetContext.from_binding(b)
                j.action('synthetic-A',list(hs.values()),T2)
                o=self.q.evaluate(j,hs,{'T1':18,'T2':18,'RH1':50,'RH2':50},dict(self.outcome,binding_hashes=hs),ctx,ctx)
                self.assertIsNone(o['reward']);self.assertTrue(any(r['reason']==('UNSUPPORTED_UNIT' if mode=='unit' else 'NOT_ACTIVE_CONTEXT_UNRESOLVED') for r in o['reasons']));j.close()

if __name__=='__main__':unittest.main()
