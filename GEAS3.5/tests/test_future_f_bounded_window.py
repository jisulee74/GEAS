import unittest
from geas35.objectives.binding import digest
import test_future_f_cucumber_profiles as fixtures
from test_future_f_live_path import at,event

class BoundedWindowTests(unittest.TestCase):
    setUp=fixtures.CucumberProfileTests.setUp
    tearDown=fixtures.CucumberProfileTests.tearDown
    schedule=fixtures.CucumberProfileTests.schedule
    def test_activation_hash(self):
        x=self.schedule(effective_until=at(6));a=x['activation_record']
        self.assertEqual(a['activation_hash'],digest({k:v for k,v in a.items() if k!='activation_hash'}))
        self.assertEqual(a['context_source'],'EXPLICIT_PROJECT_EVALUATION_CONTEXT')
    def test_pre_and_post_window_prepare(self):
        x=self.schedule(effective_until=at(6))
        for minute,reason in [(1,'NOT_EFFECTIVE'),(6,'EXPIRED'),(7,'EXPIRED')]:
            self.minute=minute
            with self.assertRaisesRegex(ValueError,reason):self.path.prepare(x['declaration_hash'],'action',event(),'explicit')
    def test_post_window_action(self):
        x=self.schedule(effective_until=at(6));self.minute=3
        ctx=self.path.prepare(x['declaration_hash'],'a',event(),'explicit');self.minute=7;s=event('sent_commands',6)
        o=self.path.observe_action('a',s,dict(method='EXPLICIT_ID',evidence_source='test',action_id='a',event_hash=s['entry_hash'],ambiguous=False),ctx)
        self.assertEqual(o['reasons'][0]['reason'],'TARGET_EXPIRED');self.assertIsNone(self.path.get('start'))
    def test_post_window_outcome(self):
        x=self.schedule(effective_until=at(6));self.minute=3
        ctx=self.path.prepare(x['declaration_hash'],'a',event(),'explicit');self.minute=5;s=event('sent_commands',4)
        self.path.observe_action('a',s,dict(method='EXPLICIT_ID',evidence_source='test',action_id='a',event_hash=s['entry_hash'],ambiguous=False),ctx)
        self.minute=7;o=event(minute=6,idx=2)
        result=self.path.evaluate('a',o,{},ctx,ctx)
        self.assertIsNone(result['reward']);self.assertEqual(result['reasons'][0]['reason'],'TARGET_EXPIRED')
    def test_pre_window_action(self):
        x=self.schedule(effective_until=at(6));self.minute=3
        ctx=self.path.prepare(x['declaration_hash'],'a',event(),'explicit');self.minute=5;s=event('sent_commands',1)
        o=self.path.observe_action('a',s,dict(method='EXPLICIT_ID',evidence_source='test',action_id='a',event_hash=s['entry_hash'],ambiguous=False),ctx)
        self.assertEqual(o['reasons'][0]['reason'],'TARGET_NOT_EFFECTIVE')
    def test_worker_bounds(self):
        from geas35.objectives.window import within
        from geas35.objectives.binding import timestamp
        a={'effective_from':at(2),'effective_until':at(6)}
        for t,expected in [(1,False),(2,True),(5,True),(6,False),(7,False)]:
            self.assertEqual(within(timestamp(at(t)),timestamp(at(t)),a),expected)
        self.assertFalse(within(timestamp(at(5)),timestamp(at(7)),a))
    def test_no_premature_final_verdict(self):
        from geas35.objectives.window import summarize
        a=self.schedule(effective_until=at(6))['activation_record']
        self.assertIsNone(summarize(self.path,a)['verdict'])
        self.assertEqual(summarize(self.path,a,completed=True)['verdict'],'CUCUMBER72 PROFILE-A SHADOW EVALUATION — NO SCORABLE ACTION')
    def test_invalid_interval(self):
        with self.assertRaisesRegex(ValueError,'INVALID_WINDOW'):self.schedule(effective_until=at(1))

if __name__=='__main__':unittest.main()
