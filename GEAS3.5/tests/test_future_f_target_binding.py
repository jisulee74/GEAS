import ast
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from geas35.objectives.binding import Binding, TargetContext, FixtureJournal, evaluate, SCOPE, VERSION

T0='2031-01-01T00:00:00+00:00'
T1='2031-01-01T00:01:00+00:00'
T2='2031-01-01T00:02:00+00:00'
T3='2031-01-01T00:03:00+00:00'

def make(**updates):
    d=dict(objective_version='fixture-objective',target_version='fixture-target',component='temperature',active=True,
           inactive_reason='',reference_value=17.25,unit='°C',reference_semantics='SYNTHETIC_POINT_DISCREPANCY',
           activation_context={'scope':SCOPE,'condition':'synthetic-explicit'},activation_rule_version='fixture-explicit.v1',
           source=SCOPE,provenance_class=SCOPE,available_at=T0,effective_from=T0,decision_time=T1,action_id='synthetic-A',
           action_time=T2,component_mask=['temperature'],evaluator_version=VERSION,scope=SCOPE)
    d.update(updates);return Binding.create(**d)

class BindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='target_fixture_');self.path=Path(self.tmp.name);self.j=FixtureJournal(self.path)
    def tearDown(self):
        self.j.close();self.tmp.cleanup()
    def bound(self,**changes):
        b=make(**changes);self.j.commit(b,T1);self.j.action(b.data['action_id'],[b.target_hash],b.data['action_time']);return b
    def run_eval(self,b,obs=None,outcome=None,policy=None,environment=None):
        context=TargetContext.from_binding(b)
        return evaluate(self.j,b.target_hash,b.data['component'],obs or {'T1':18.25,'T2':15.25},
                        outcome or {'action_id':b.data['action_id'],'binding_hash':b.target_hash,'observed_at':T3},
                        policy or context,environment or context)
    def test_valid_binding(self):
        b=self.bound();self.assertEqual([x['status'] for x in self.run_eval(b)],['VALID','VALID'])
    def test_late_available_rejected(self):
        with self.assertRaisesRegex(ValueError,'TARGET_NOT_AVAILABLE_PRE_ACTION'):self.j.commit(make(available_at=T2),T1)
        self.assertIsNone(self.j.lookup(make(available_at=T2).target_hash))
    def test_late_effective_rejected(self):
        with self.assertRaisesRegex(ValueError,'TARGET_NOT_EFFECTIVE'):self.j.commit(make(effective_from=T2),T1)
    def test_immutable_object_and_nested_context(self):
        b=make()
        with self.assertRaises(FrozenInstanceError):b.payload='x'
        d=b.data;d['activation_context']['condition']='changed';self.assertNotEqual(d,b.data)
    def test_conflicting_rewrite(self):
        self.bound()
        with self.assertRaisesRegex(ValueError,'IMMUTABLE'):self.j.commit(make(reference_value=18.75),T1)
    def test_idempotent_commit(self):
        b=self.bound();n=self.j.db.execute('SELECT COUNT(*) FROM events').fetchone()[0];self.j.commit(b,T1)
        self.assertEqual(n,self.j.db.execute('SELECT COUNT(*) FROM events').fetchone()[0])
    def test_canonical_hash(self):
        b=make();d=dict(reversed(list(b.data.items())));d['available_at']='2031-01-01T09:00:00+09:00'
        self.assertEqual(b.target_hash,Binding.create(**d).target_hash)
    def test_parity(self):
        b=self.bound();self.assertEqual(TargetContext.from_binding(b),TargetContext.from_binding(Binding(b.payload)))
    def test_parity_mismatch_each_field(self):
        b=self.bound();c=TargetContext.from_binding(b)
        for change in ({'target_hash':'wrong'},{'target_version':'wrong'},{'component_mask':()},{'evaluator_version':'wrong'}):
            out=self.run_eval(b,policy=replace(c,**change));self.assertTrue(all(x['status']=='TARGET_CONTEXT_MISMATCH' and x['raw_difference'] is None for x in out))
    def test_temperature_channels(self):
        rows=self.run_eval(self.bound());self.assertEqual([r['raw_difference'] for r in rows],[1,-2]);self.assertEqual([r['absolute_difference'] for r in rows],[1,2])
    def test_rh_channels(self):
        b=self.bound(component='RH',unit='%RH',reference_value=47.5,component_mask=['RH'])
        rows=self.run_eval(b,{'RH1':49.5,'RH2':44.5});self.assertEqual([r['raw_difference'] for r in rows],[2,-3]);self.assertEqual(rows[0]['discrepancy_unit'],'percentage_points')
    def test_missing_t1_no_fallback(self):
        rows=self.run_eval(self.bound(),{'T2':18});self.assertEqual(rows[0]['status'],'MISSING_SENSOR_VALUE');self.assertIsNone(rows[0]['raw_difference']);self.assertEqual(rows[1]['status'],'VALID')
    def test_missing_rh1_no_fallback(self):
        b=self.bound(component='RH',unit='%RH',reference_value=47.5,component_mask=['RH']);rows=self.run_eval(b,{'RH2':55});self.assertEqual(rows[0]['status'],'MISSING_SENSOR_VALUE');self.assertEqual(rows[1]['raw_difference'],7.5)
    def test_unresolved_context(self):
        b=self.bound(active=False,inactive_reason='NOT_ACTIVE_CONTEXT_UNRESOLVED',activation_context={},reference_value=None,component_mask=[])
        self.assertTrue(all(x['status']=='NOT_ACTIVE_CONTEXT_UNRESOLVED' and x['score'] is None and x['raw_difference'] is None for x in self.run_eval(b)))
    def test_no_target(self):
        rows=evaluate(self.j,'missing','temperature',{'T1':1},{},None,None);self.assertTrue(all(r['status']=='TARGET_NOT_BOUND' and r['score'] is None for r in rows))
    def test_invalid_nonfinite(self):
        b=self.bound()
        for v in (float('nan'),float('inf'),float('-inf'),True,'18',-274):
            self.assertEqual(self.run_eval(b,{'T1':v})[0]['status'],'INVALID_SENSOR_VALUE')
    def test_rh_physical_invalid(self):
        b=self.bound(component='RH',unit='%RH',component_mask=['RH']);self.assertEqual(self.run_eval(b,{'RH1':101})[0]['status'],'INVALID_SENSOR_VALUE')
    def test_unsupported_unit(self):
        rows=self.run_eval(self.bound(unit='kelvin'));self.assertTrue(all(r['status']=='UNSUPPORTED_UNIT' and r['raw_difference'] is None for r in rows))
    def test_unsupported_semantics(self):
        self.assertEqual(self.run_eval(self.bound(reference_semantics='approximate-practice'))[0]['status'],'UNSUPPORTED_REFERENCE_SEMANTICS')
    def test_original_binding_survives_later_version(self):
        b=self.bound();new=make(target_version='fixture-v2',reference_value=99,action_id='synthetic-B',decision_time=T2,action_time=T3)
        self.j.commit(new,T2);self.j.action('synthetic-B',[new.target_hash],T3)
        self.assertEqual(self.run_eval(b)[0]['raw_difference'],1)
    def test_ambiguous_link(self):
        b=self.bound()
        for o in ({'action_id':'wrong','binding_hash':b.target_hash,'observed_at':T3}, {'action_id':b.data['action_id'],'binding_hash':'wrong','observed_at':T3}, {'action_id':b.data['action_id'],'binding_hash':b.target_hash,'observed_at':T3,'ambiguous':True}):
            self.assertEqual(self.run_eval(b,outcome=o)[0]['status'],'AMBIGUOUS_ACTION_LINKAGE')
    def test_outcome_before_action(self):
        b=self.bound();o={'action_id':b.data['action_id'],'binding_hash':b.target_hash,'observed_at':T1};self.assertEqual(self.run_eval(b,outcome=o)[0]['status'],'OUTCOME_NOT_AFTER_ACTION')
    def test_equal_timestamp_sequence_proof(self):
        b=make(action_time=T1);self.j.commit(b,T1);seq=self.j.action(b.data['action_id'],[b.target_hash],T1)
        o={'action_id':b.data['action_id'],'binding_hash':b.target_hash,'observed_at':T1}
        self.assertEqual(self.run_eval(b,outcome=o)[0]['status'],'OUTCOME_NOT_AFTER_ACTION')
        o.update(order_stream='fixture_journal',sequence=seq+1);self.assertEqual(self.run_eval(b,outcome=o)[0]['status'],'VALID')
    def test_restart(self):
        b=self.bound();self.j.close();self.j=FixtureJournal(self.path);self.assertEqual(self.j.lookup(b.target_hash),b);self.assertEqual(self.run_eval(b)[0]['raw_difference'],1)
    def test_no_scalar_no_vpd(self):
        rows=self.run_eval(self.bound());self.assertIsInstance(rows,list);self.assertTrue(all(r['score'] is None and 'reward' not in r for r in rows))
        with self.assertRaisesRegex(ValueError,'EXCLUDED'):evaluate(self.j,'x','VPD',{}, {},None,None)
    def test_live_path_rejected(self):
        with self.assertRaisesRegex(ValueError,'LIVE'):FixtureJournal(self.path/'prospective_collection'/'pilot_iot97_cycle72')
        self.assertFalse((self.path/'prospective_collection').exists())
    def test_nonfixture_target_rejected(self):
        with self.assertRaisesRegex(ValueError,'FIXTURE_ONLY'):make(scope='LIVE')
    def test_binding_after_action_rejected(self):
        self.bound()
        with self.assertRaisesRegex(ValueError,'BINDING_AFTER_ACTION'):self.j.commit(make(component='RH',unit='%RH',component_mask=['RH']),T1)
    def test_action_requires_exact_binding(self):
        b=make();self.j.commit(b,T1)
        with self.assertRaisesRegex(ValueError,'MISMATCH'):self.j.action(b.data['action_id'],['wrong'],T2)
    def test_mask_complete_before_action(self):
        b=make(component_mask=['temperature','RH']);self.j.commit(b,T1)
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'):self.j.action(b.data['action_id'],[b.target_hash],T2)
    def test_sql_updates_forbidden(self):
        self.bound()
        with self.assertRaisesRegex(sqlite3.IntegrityError,'IMMUTABLE'):self.j.db.execute('DELETE FROM events')
    def test_persistence_failure_not_committed(self):
        from unittest.mock import patch
        b=make()
        with patch.object(self.j,'_append',side_effect=OSError('fixture disk failure')):
            with self.assertRaises(OSError):self.j.commit(b,T1)
        self.assertIsNone(self.j.lookup(b.target_hash))
        with self.assertRaises(ValueError):self.j.action(b.data['action_id'],[b.target_hash],T2)
    def test_tampered_journal_rejected_on_restart(self):
        self.bound();self.j.close()
        with sqlite3.connect(self.path/'fixture_bindings.sqlite') as db:
            db.execute('DROP TRIGGER immutable_UPDATE')
            db.execute("UPDATE events SET hash='wrong' WHERE seq=1")
        with self.assertRaisesRegex(ValueError,'INTEGRITY'):FixtureJournal(self.path)
        self.j=FixtureJournal(self.path/'clean')
    def test_missing_action_unavailable(self):
        b=make();self.j.commit(b,T1);self.assertEqual(self.run_eval(b)[0]['status'],'AMBIGUOUS_ACTION_LINKAGE')
    def test_retro_commit_rejected(self):
        with self.assertRaisesRegex(ValueError,'INVALID_COMMIT_ORDER'):self.j.commit(make(),T3)
    def test_nonfinite_difference_rejected(self):
        b=self.bound(reference_value=-1e308)
        self.assertIsNone(self.run_eval(b,{'T1':1e308})[0]['raw_difference'])
    def test_no_control_dependencies(self):
        from geas35.objectives import binding
        tree=ast.parse(Path(binding.__file__).read_text())
        names=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        names += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(n and any(s in n for s in ('control','pymysql','requests','collection')) for n in names))

if __name__=='__main__':unittest.main()
