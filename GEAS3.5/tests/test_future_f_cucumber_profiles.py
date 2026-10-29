import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from geas35.objectives.profiles import CandidateProfile,load_profiles,schedule_profile,FROZEN_HASHES
from geas35.objectives.binding import canonical,digest
from geas35.objectives.live import LivePath
from test_future_f_live_path import at,event
from test_future_f_quality_prereg import ART

CATALOG=ART.parent.parent/'future_f_cucumber72_target_profiles_v1/cucumber72_target_profiles.json'

class CucumberProfileTests(unittest.TestCase):
    def setUp(self):
        self.profiles=load_profiles(CATALOG);self.tmp=tempfile.TemporaryDirectory(prefix='profile_fixture_');self.minute=0
        self.path=LivePath(self.tmp.name,ART,'synthetic-p0',synthetic=True,clock=lambda:at(self.minute))
    def tearDown(self):self.path.close();self.tmp.cleanup()
    def schedule(self,name='a',**changes):
        p=self.profiles['cucumber72_interview_'+name]
        args=dict(activation_id='synthetic-activation',activation_context=dict(context_label=p.data['context_label'],selection_method='EXPLICIT_PROJECT_DECLARATION',asserted_by='TEST_FIXTURE_ONLY',evidence_source='TEST_FIXTURE_ONLY explicit context'),effective_from=at(2),declared_by='TEST_FIXTURE_ONLY',declaration_basis='isolated synthetic activation test')
        args.update(changes);return schedule_profile(self.path,p,**args)
    def test_a_hash(self):
        p=self.profiles['cucumber72_interview_a'];self.assertEqual(CandidateProfile(p.payload).data['profile_hash'],FROZEN_HASHES[p.data['profile_id']])
    def test_b_hash(self):
        p=self.profiles['cucumber72_interview_b'];self.assertEqual(CandidateProfile(p.payload).data['profile_hash'],FROZEN_HASHES[p.data['profile_id']])
    def test_numbers_only(self):
        self.assertEqual([(p.data['temperature']['reference_value'],p.data['rh']['reference_value']) for p in self.profiles.values()],[(30,90),(23,60)])
    def test_approximate_and_no_band(self):
        for p in self.profiles.values():
            for c in ('temperature','rh'):
                self.assertIs(p.data[c]['approximate'],True)
                self.assertFalse(set(p.data[c])&{'tolerance','band','severe_threshold','interval'})
    def test_claim_and_pairing(self):
        for p in self.profiles.values():
            self.assertEqual(p.data['claim'],'PROJECT_OPERATIONAL_EVALUATION_TARGET');self.assertIn('not a grower-stated joint optimum',p.data['project_pairing'])
    def test_no_inference_inputs(self):
        params=set(inspect.signature(schedule_profile).parameters)
        self.assertFalse(params&{'DAT','stage','clock','observations','reward','outcome','weather'})
        for method in ('DAT','CLOCK_ONLY','MAX_REWARD'):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_MATCHING'):
                self.schedule(activation_context=dict(context_label='fruit_enlargement + morning',selection_method=method,asserted_by='x',evidence_source='x'))
    def test_future_explicit(self):
        x=self.schedule();self.assertEqual(x['status'],'SCHEDULED_NOT_EFFECTIVE');self.assertIsNone(self.path.get('start'))
        with self.assertRaisesRegex(ValueError,'NOT_EFFECTIVE'):self.path.prepare(x['declaration_hash'],'pending',event(),'explicit')
    def test_retrospective_rejected(self):
        self.minute=3
        for t in (at(0),at(3)):
            with self.assertRaisesRegex(ValueError,'PROSPECTIVE'):self.schedule(effective_from=t)
    def test_profile_rewrite(self):
        for name in self.profiles:
            d=self.profiles[name].data;d['temperature']['reference_value']=28;d['profile_hash']=digest({k:v for k,v in d.items() if k!='profile_hash'})
            with self.assertRaisesRegex(ValueError,'INTEGRITY'):CandidateProfile(canonical(d))
    def test_activation_rewrite(self):
        self.schedule()
        with self.assertRaisesRegex(ValueError,'IMMUTABLE'):self.schedule()
    def test_unsuitable_examples_absent(self):
        self.assertEqual(len(self.profiles),2)
        for p in self.profiles.values():
            self.assertNotIn(p.data['temperature']['reference_value'],(13,14,15));self.assertNotEqual(p.data['rh']['reference_value'],100)
            self.assertIn('fruit_enlargement',p.data['context_label'])
    def test_no_real_activation(self):
        for p in self.profiles.values():
            self.assertEqual(p.data['status'],'DECLARED_NOT_ACTIVE');self.assertIsNone(p.data['effective_from'])
        self.assertIsNone(self.path.get('start'))
    def test_context_mismatch(self):
        with self.assertRaisesRegex(ValueError,'MATCHING'):
            self.schedule(activation_context=dict(context_label='unknown',selection_method='EXPLICIT_PROJECT_DECLARATION',asserted_by='test',evidence_source='test'))
    def test_evidence_hash(self):
        import hashlib
        root=Path(__file__).resolve().parents[1]
        for p in self.profiles.values():
            e=p.data['temperature']['interview_evidence'];self.assertEqual(hashlib.sha256((root/e['artifact']).read_bytes()).hexdigest(),e['sha256'])
    def test_both_end_to_end(self):
        from geas35.objectives.quality import S_T,S_RH
        from test_future_f_quality_prereg import reference_spec
        for i,name in enumerate(('a','b')):
            # Each scenario has an isolated journal and clock.
            with tempfile.TemporaryDirectory() as tmp:
                p=LivePath(tmp,ART,'synthetic-p0',synthetic=True,clock=lambda:at(self.minute))
                try:
                    self.minute=0
                    candidate=self.profiles['cucumber72_interview_'+name]
                    context=dict(context_label=candidate.data['context_label'],selection_method='EXPLICIT_PROJECT_DECLARATION',asserted_by='TEST_FIXTURE_ONLY',evidence_source='TEST_FIXTURE_ONLY')
                    x=schedule_profile(p,candidate,activation_id='isolated',activation_context=context,effective_from=at(2),declared_by='TEST_FIXTURE_ONLY',declaration_basis='synthetic-only')
                    self.minute=3;r=event();before=copy.deepcopy(r);ctx=p.prepare(x['declaration_hash'],'future',r,'TEST_FIXTURE_ONLY explicit future identity');self.assertEqual(before,r)
                    self.assertEqual(json.loads(ctx['payload']['target_declaration']['declaration_basis'])['activation_context'],context)
                    self.minute=5;sent=event('sent_commands',4)
                    link=dict(method='EXPLICIT_ID',evidence_source='TEST_FIXTURE_ONLY',action_id='future',event_hash=sent['entry_hash'],ambiguous=False)
                    self.assertEqual(p.observe_action('future',sent,link,ctx)['evaluation_status'],'ACTION_BOUND')
                    self.minute=7;outcome=event(minute=6,idx=2);link.update(event_hash=outcome['entry_hash'],intervening_override=False,profile_hash=x['declaration_hash'])
                    result=p.evaluate('future',outcome,link,ctx,ctx)
                    self.assertEqual(result['evaluation_status'],'VALID');self.assertEqual(result['policy_context_hash'],result['environment_evaluation_context_hash']);self.assertFalse(result['live_activation'])
                    self.assertAlmostEqual(result['reward'],reference_spec(result['channels'],{'temperature':S_T,'RH':S_RH}),places=14)
                finally:p.close();self.minute=0

if __name__=='__main__':unittest.main()
