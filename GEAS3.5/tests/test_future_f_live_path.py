import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from geas35.objectives.live import Declaration,LivePath,SEMANTICS,_EvaluationView
from geas35.objectives.binding import digest,TargetContext
from test_future_f_quality_prereg import ART


def at(n):return f'2032-01-01T00:{n:02d}:00+00:00'

def declaration(**changes):
    d=dict(target_profile_id='synthetic-profile',target_version='v1',crop='cucumber',crop_cycle=72,source=97,temperature=dict(active=True,reference_value=17.25,unit='degC',semantics=SEMANTICS,evidence_source='TEST_FIXTURE_ONLY'),rh=dict(active=True,reference_value=47.5,unit='percentage_points',semantics=SEMANTICS,evidence_source='TEST_FIXTURE_ONLY'),effective_from=at(0),available_at=at(0),declared_by='TEST_FIXTURE_ONLY',declaration_basis='synthetic non-learning test',claim='PROJECT_OPERATIONAL_EVALUATION_TARGET')
    d.update(changes);return Declaration.create(**d)


def event(stream='observations',minute=2,idx=1,**sensor):
    raw=dict(idx=idx,reg_date=at(minute))
    if stream=='observations':raw.update(iot_data_idx='97',in_temp='18.25',in_temp2='19.25',in_hum='48.5',in_hum2='49.5',**sensor)
    else:raw.update(channel='1',ratio='50',operation='OPEN')
    x=dict(stream=stream,session_id='synthetic-p0',raw=raw,source_row_id=idx,source_timestamp=raw['reg_date'],receipt_at_utc=at(minute),source_table='data_silla_enc' if stream=='observations' else 'send_ds_control_api_history',source_hash=digest(raw),reward_eligible=False,phase='P0',attribution='UNKNOWN',cross_table_causal_order='AMBIGUOUS')
    return dict(x,entry_hash=digest(x))


class LivePathTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='live_path_fixture_');self.minute=0
        self.p=LivePath(self.tmp.name,ART,'synthetic-p0',synthetic=True,clock=lambda:at(self.minute))
        self.d=declaration();self.h=self.p.declare(self.d);self.minute=3
        self.ctx=self.p.prepare(self.h,'future-cycle-1',event(), 'TEST_FIXTURE_ONLY explicit cycle key')
    def tearDown(self):self.p.close();self.tmp.cleanup()
    def action(self,**changes):
        self.minute=5;r=event('sent_commands',4)
        link=dict(method='EXPLICIT_ID',evidence_source='TEST_FIXTURE_ONLY',action_id='future-cycle-1',event_hash=r['entry_hash'],ambiguous=False)
        args=dict(action_id='future-cycle-1',sent_receipt=r,link=link,policy_context=self.ctx);args.update(changes)
        return self.p.observe_action(**args)
    def outcome(self,**changes):
        self.minute=7;r=event(minute=6,idx=2)
        link=dict(method='EXPLICIT_ID',evidence_source='TEST_FIXTURE_ONLY',action_id='future-cycle-1',event_hash=r['entry_hash'],ambiguous=False,intervening_override=False,profile_hash=self.h)
        args=dict(action_id='future-cycle-1',outcome_receipt=r,link=link,policy_context=self.ctx,environment_context=self.ctx);args.update(changes)
        return self.p.evaluate(**args)
    def test_declaration_and_deterministic_hash(self):self.assertEqual(self.d.payload,declaration().payload)
    def test_schema_claim_hash(self):
        for change in ({'crop_cycle':71},{'claim':'UNIVERSAL_AGRONOMIC_OPTIMUM'},{'declared_by':''}):
            with self.assertRaises(ValueError):declaration(**change)
        d=self.d.data;d['temperature']['reference_value']=99
        with self.assertRaisesRegex(ValueError,'HASH'):Declaration(json.dumps(d))
    def test_future_available(self):
        with self.assertRaisesRegex(ValueError,'NOT_AVAILABLE'):self.p.declare(declaration(available_at=at(9)))
    def test_future_effective(self):
        d=declaration(target_version='future',effective_from=at(9));h=self.p.declare(d)
        with self.assertRaisesRegex(ValueError,'NOT_EFFECTIVE'):self.p.prepare(h,'x',event(),'explicit')
    def test_old_input(self):
        with self.assertRaisesRegex(ValueError,'NONPROSPECTIVE'):self.p.prepare(self.h,'x',event(minute=0),'explicit')
    def test_commit_after_source_action(self):
        r=event('sent_commands',2);o=self.action(sent_receipt=r,link=dict(method='EXPLICIT_ID',evidence_source='test',action_id='future-cycle-1',event_hash=r['entry_hash'],ambiguous=False))
        self.assertIsNone(o['reward']);self.assertEqual(o['reasons'][0]['reason'],'BINDING_AFTER_ACTION');self.assertIsNone(self.p.get('start'))
    def test_valid_path(self):
        self.assertEqual(self.action()['evaluation_status'],'ACTION_BOUND');o=self.outcome()
        self.assertEqual(o['evaluation_status'],'VALID');self.assertTrue(0<o['reward']<=1);self.assertEqual(o['policy_context_hash'],o['environment_evaluation_context_hash']);self.assertIsNone(o['episode_return']);self.assertFalse(o['live_activation'])
        self.assertIn('SYNTHETIC_OBJECTIVE',self.p.get('start')[2])
    def test_exact_quality_parity(self):
        self.action();o=self.outcome();hs=self.ctx['payload']['binding_hashes'];c={k:TargetContext.from_binding(self.p.lookup(h)) for k,h in hs.items()}
        ref=self.p.quality.evaluate(_EvaluationView(self.p,at(4),'future-cycle-1'),hs,dict(T1=18.25,T2=19.25,RH1=48.5,RH2=49.5),dict(action_id='future-cycle-1',observed_at=at(6),binding_hashes=hs),c,c)
        for k in ref:
            if k not in ('live_activation',):self.assertEqual(ref[k],o[k])
    def test_action_parity(self):
        o=self.action(policy_context={});self.assertEqual(o['reasons'][0]['reason'],'TARGET_CONTEXT_MISMATCH');self.assertIsNone(self.p.get('start'))
    def test_outcome_parity(self):
        self.action();o=self.outcome(environment_context={});self.assertEqual(o['evaluation_status'],'UNSCORABLE');self.assertIsNone(o['reward'])
    def test_ambiguous_action(self):
        o=self.action(link={});self.assertEqual(o['reasons'][0]['reason'],'ACTION_ATTRIBUTION_UNRESOLVED')
    def test_ambiguous_outcome(self):
        self.action();o=self.outcome(link={});self.assertEqual(o['reasons'][0]['reason'],'ACTION_ATTRIBUTION_UNRESOLVED')
    def test_old_outcome(self):
        self.action();o=self.outcome(outcome_receipt=event(minute=1));self.assertEqual(o['reasons'][0]['reason'],'OUTCOME_NOT_AFTER_ACTION')
    def test_missing_four(self):
        self.action()
        for i,k in enumerate(('in_temp','in_temp2','in_hum','in_hum2')):
            r=event(minute=6,idx=20+i);r['raw'][k]=None;r['source_hash']=digest(r['raw']);r['entry_hash']=digest({k:v for k,v in r.items() if k!='entry_hash'})
            link=dict(method='EXPLICIT_ID',evidence_source='test',action_id='future-cycle-1',event_hash=r['entry_hash'],ambiguous=False,intervening_override=False,profile_hash=self.h)
            o=self.outcome(outcome_receipt=r,link=link);self.assertIsNone(o['reward']);self.assertTrue(any(x['reason']=='MISSING_SENSOR_VALUE' for x in o['reasons']))
    def test_version_immutable(self):
        self.action();h=self.p.declare(declaration(target_version='v2'))
        with self.assertRaisesRegex(ValueError,'BINDING_AFTER_ACTION'):self.p.prepare(h,'future-cycle-1',event(minute=5),'explicit')
        o=self.outcome();self.assertEqual(o['target_hash'],self.ctx['payload']['binding_hashes']);self.assertEqual(o['target_version']['temperature'],'v1')
    def test_same_version_cannot_change(self):
        with self.assertRaisesRegex(ValueError,'IMMUTABLE'):self.p.declare(declaration())
    def test_restart_binding(self):
        self.action();self.p.close();self.p=LivePath(self.tmp.name,ART,'synthetic-p0',synthetic=True,clock=lambda:at(self.minute))
        self.assertEqual(self.p.context('future-cycle-1'),self.ctx);self.assertEqual(self.outcome()['evaluation_status'],'VALID')
    def test_system_invalid_separate(self):
        self.action();o=self.outcome(accounting_status='SYSTEM_INVALID');self.assertEqual(o['evaluation_status'],'SYSTEM_INVALID');self.assertIsNone(o['reward'])
    def test_ood_no_penalty(self):
        self.action();o=self.outcome(accounting_status='TRUNCATED',support_termination='STATE_OOD');self.assertIsNotNone(o['reward']);self.assertEqual(o['support_termination'],'STATE_OOD');self.assertIsNone(o['episode_return'])
    def test_no_binding(self):
        o=self.p.evaluate('unbound',event(),{}, {},{});self.assertIsNone(o['reward']);self.assertEqual(o['reasons'][0]['reason'],'TARGET_NOT_BOUND')
    def test_p0_never_mutated(self):
        r=event();before=copy.deepcopy(r);self.p.prepare(self.h,'new',r,'explicit');self.assertEqual(r,before);self.assertFalse(r['reward_eligible'])
        with self.assertRaisesRegex(ValueError,'P0_STORAGE'):LivePath(Path(self.tmp.name)/'prospective_collection',ART,'s')
    def test_no_live_clock_injection(self):
        with self.assertRaisesRegex(ValueError,'CLOCK_OVERRIDE'):LivePath(self.tmp.name,ART,'s',clock=lambda:at(0))
    def test_no_network_controller_or_db_client_import(self):
        import geas35.objectives.live as m
        tree=ast.parse(Path(m.__file__).read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        for name in imports:self.assertFalse(any(x in name for x in ('socket','requests','pymysql','controller','urllib','subprocess')))
    def test_tampered_receipt(self):
        r=event();r['raw']['in_temp']=99
        with self.assertRaisesRegex(ValueError,'HASH'):self.p.prepare(self.h,'bad',r,'explicit')
    def test_real_adapter_rejects_synthetic_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=LivePath(tmp,ART,'not-activated')
            try:
                with self.assertRaisesRegex(ValueError,'SYNTHETIC_DECLARATION'):p.declare(self.d)
                self.assertIsNone(p.get('start'))
            finally:p.close()
    def test_inactive_cannot_start(self):
        d=declaration(target_version='inactive',rh=dict(active=False,reference_value=None,unit='percentage_points',semantics=SEMANTICS,evidence_source='TEST_FIXTURE_ONLY'))
        h=self.p.declare(d);ctx=self.p.prepare(h,'inactive',event(),'explicit');r=event('sent_commands',4);self.minute=5
        o=self.p.observe_action('inactive',r,dict(method='EXPLICIT_ID',evidence_source='test',action_id='inactive',event_hash=r['entry_hash'],ambiguous=False),ctx)
        self.assertEqual(o['reasons'][0]['reason'],'NOT_ACTIVE_CONTEXT_UNRESOLVED');self.assertIsNone(self.p.get('start'))
    def test_one_sent_event_cannot_bind_two_actions(self):
        ctx=self.p.prepare(self.h,'another',event(),'explicit');self.action();r=event('sent_commands',4)
        o=self.p.observe_action('another',r,dict(method='EXPLICIT_ID',evidence_source='test',action_id='another',event_hash=r['entry_hash'],ambiguous=False),ctx)
        self.assertEqual(o['reasons'][0]['reason'],'ACTION_ATTRIBUTION_UNRESOLVED')
    def test_durable_log_rejects_rewrite(self):
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):self.p.db.execute("UPDATE events SET payload='{}' WHERE key='session'")
        self.p.db.rollback()
    def test_actual_policy_payload_has_current_channels_and_targets(self):
        self.assertEqual(self.ctx['payload']['observation_channels']['T1'],'18.25')
        self.assertEqual(self.ctx['payload']['bindings']['temperature']['reference_value'],17.25)
        self.assertEqual(self.ctx['context_hash'],digest(self.ctx['payload']))
    def test_no_target_selection_from_outcome(self):
        self.action();link=dict(method='EXPLICIT_ID',evidence_source='test',action_id='future-cycle-1',event_hash=event(minute=6,idx=2)['entry_hash'],ambiguous=False,intervening_override=False,profile_hash='better-target')
        self.assertEqual(self.outcome(link=link)['reasons'][0]['reason'],'TARGET_CONTEXT_MISMATCH')

if __name__=='__main__':unittest.main()
