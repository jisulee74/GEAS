import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from geas35.objectives.correlation import Plan,correlate,evaluate_mock,uid
from geas35.objectives.binding import digest,TargetContext
from geas35.objectives.live import LivePath,_EvaluationView
from test_future_f_live_path import at,event,declaration
from test_future_f_quality_prereg import ART


class CorrelationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='correlation_mock_')
        self.minute=0
        self.p=LivePath(self.tmp.name,ART,'synthetic-p0',synthetic=True,clock=lambda:at(self.minute))
        self.profile=self.p.declare(declaration());self.minute=3
        self.actions=[dict(n_idx='1',channel=c,operation='OPEN',ratio=v,duration='') for c,v in [('1','25'),('2','75')]]
        self.plan=Plan.prepare(self.p,self.profile,event(),self.actions,['13'],['1','2','13'],dict(controller_name='GEAS2.5',controller_version='audited',runtime_crop='strawberry',actual_crop='cucumber',controller_code_hash='mock-code-hash'),42,300)
        self.d=self.plan.data;self.ctx=self.p.context(self.d['evaluation_cycle_id'])
        self.attempts=[self.attempt(i,100+i,4) for i in range(2)]
        self.completion=self.complete(4)
        self.outcome=dict(evaluation_cycle_id=self.d['evaluation_cycle_id'],ticket=self.d['outcome_ticket'],sequence=42,receipt=event(minute=6,idx=10),acquired_start=at(6),acquired_end=at(6),continuous_sensor_stream=True)
        self.coverage=self.cover();self.minute=7
    def tearDown(self):self.p.close();self.tmp.cleanup()
    def attempt(self,i,pk,minute,ordinal=1):
        r=event('sent_commands',minute,pk);r['raw'].update(self.actions[i]);r['source_hash']=digest(r['raw']);r['entry_hash']=digest({k:v for k,v in r.items() if k!='entry_hash'})
        return dict(evaluation_cycle_id=self.d['evaluation_cycle_id'],command_id=self.d['command_ids'][i],channel=self.actions[i]['channel'],attempt_id=uid(),attempt_ordinal=ordinal,payload_hash=digest(self.actions[i]),send_started_at=at(minute),response_status=200,committed=True,committed_history_pk=pk,persistence_confirmed_at=at(minute),sent_receipt=r)
    def complete(self,minute):
        return dict(evaluation_cycle_id=self.d['evaluation_cycle_id'],attempt_ids=[a['attempt_id'] for a in self.attempts],all_attempts_closed=True,completed_at=at(minute))
    def cover(self):
        return dict(evaluation_cycle_id=self.d['evaluation_cycle_id'],complete=True,authority='MOCK_ONLY all-send supervisor',**{'from':at(4),'through':at(6)},events=[dict(origin='THIS_CYCLE',cycle_id=a['evaluation_cycle_id'],attempt_id=a['attempt_id'],pk=a['committed_history_pk'],at=a['send_started_at']) for a in self.attempts])
    def result(self):return correlate(self.p,self.plan,self.attempts,self.completion,self.outcome,self.coverage)
    def evaluate(self):return evaluate_mock(self.p,self.plan,self.attempts,self.completion,self.outcome,self.coverage,self.ctx,self.ctx)
    def rejected(self,status):
        self.assertEqual(self.result()['correlation_status'],status)
        r=self.evaluate();self.assertIsNone(r['reward']);self.assertIsNone(r['quality']['reward']);self.assertIsNone(self.p.get('start'))
    def test_A_complete_and_exact_frozen_quality_parity(self):
        r=self.evaluate();self.assertEqual(r['correlation']['correlation_status'],'VALID');self.assertGreater(r['reward'],0)
        hs=self.ctx['payload']['binding_hashes'];cs={k:TargetContext.from_binding(self.p.lookup(h)) for k,h in hs.items()}
        ref=self.p.quality.evaluate(_EvaluationView(self.p,at(4),self.d['evaluation_cycle_id']),hs,dict(T1=18.25,T2=19.25,RH1=48.5,RH2=49.5),dict(action_id=self.d['evaluation_cycle_id'],observed_at=at(6),binding_hashes=hs),cs,cs)
        for k in ('reward','temperature_quality','rh_quality','normalized_error_T1','normalized_error_T2','normalized_error_RH1','normalized_error_RH2'):self.assertEqual(r['quality'][k],ref[k])
        self.assertFalse(r['quality']['live_activation'])
    def test_B_partial(self):
        self.attempts.pop();self.rejected('ACTION_INCOMPLETE')
    def test_B_missing_commit(self):
        self.attempts[0]['committed']=False;self.rejected('ACTION_INCOMPLETE')
    def test_C_unknown(self):
        self.coverage['events'].append(dict(origin='UNKNOWN',at=at(5),pk=999));self.rejected('OVERRIDE_UNKNOWN')
    def test_D_override(self):
        self.coverage['events'].append(dict(origin='CONFIRMED_OTHER',at=at(5),pk=999,evidence='mock producer origin'));self.rejected('OVERRIDE_DETECTED')
    def test_E_gap(self):
        self.outcome['continuous_sensor_stream']=False;self.rejected('SENSOR_GAP')
    def test_F_repeated_attempts(self):
        self.attempts.append(self.attempt(0,102,5,2));self.completion=self.complete(5);self.coverage=self.cover()
        self.assertEqual(self.result()['sent_history_row_ids'],[100,101,102]);self.assertIsNotNone(self.evaluate()['reward'])
    def test_G_restart(self):
        cycle=self.d['evaluation_cycle_id'];before=self.plan.hash;self.p.close()
        self.p=LivePath(self.tmp.name,ART,'synthetic-p0',synthetic=True,clock=lambda:at(self.minute))
        self.plan=Plan.reload(self.p,cycle);self.assertEqual(self.plan.hash,before);self.assertIsNotNone(self.evaluate()['reward'])
    def test_unique_cycle_not_clock_or_pid(self):
        self.minute=3
        p=Plan.prepare(self.p,self.profile,event(),self.actions,['13'],['1','2','13'],self.d['provenance'],42,300)
        self.assertNotEqual(p.data['evaluation_cycle_id'],self.d['evaluation_cycle_id'])
    def test_wrong_ticket(self):
        self.outcome['ticket']='nearest-row';self.rejected('OUTCOME_UNRESOLVED')
    def test_no_outcome(self):
        self.outcome=None;self.rejected('OUTCOME_UNRESOLVED')
    def test_sensor_sequence(self):
        self.outcome['sequence']=43;self.rejected('SENSOR_GAP')
    def test_acquisition_before_action_completion(self):
        self.outcome['acquired_start']=at(4);self.rejected('OUTCOME_UNRESOLVED')
    def test_stream_missing(self):
        self.coverage['complete']=False;self.rejected('OVERRIDE_UNKNOWN')
    def test_stream_end_insufficient(self):
        self.coverage['through']=at(5);self.rejected('OVERRIDE_UNKNOWN')
    def test_stream_missing_own_send(self):
        self.coverage['events'].pop();self.rejected('OVERRIDE_UNKNOWN')
    def test_pk_not_payload_timestamp_match(self):
        self.attempts[0]['committed_history_pk']=123456;self.rejected('SYSTEM_INVALID')
    def test_reused_pk(self):
        self.attempts[1]['committed_history_pk']=100;self.rejected('SYSTEM_INVALID')
    def test_tampered_receipt(self):
        self.attempts[0]['sent_receipt']['raw']['ratio']='99';self.rejected('SYSTEM_INVALID')
    def test_payload_changed_with_lineage(self):
        self.attempts[0]['payload_hash']=digest({});self.rejected('SYSTEM_INVALID')
    def test_before_prebinding(self):
        self.attempts[0]['send_started_at']=at(3);self.rejected('SYSTEM_INVALID')
    def test_completion_manifest(self):
        self.completion['attempt_ids']=[];self.rejected('ACTION_INCOMPLETE')
    def test_omission_no_fill(self):
        self.assertEqual(self.d['omitted'],['13']);self.assertEqual(self.d['expected_channel_set'],['1','2']);self.assertEqual(len(self.d['actuators']),2)
    def test_changed_plan(self):
        d=self.plan.data;d['actuators'][0]['ratio']='99';self.plan=Plan(json.dumps(d));self.rejected('SYSTEM_INVALID')
    def test_existing_parity_gate_preserved(self):
        r=evaluate_mock(self.p,self.plan,self.attempts,self.completion,self.outcome,self.coverage,{},self.ctx)
        self.assertIsNone(r['reward']);self.assertIsNone(self.p.get('start'))
    def test_existing_four_channel_gate_preserved(self):
        r=self.outcome['receipt'];r['raw']['in_temp2']=None;r['source_hash']=digest(r['raw']);r['entry_hash']=digest({k:v for k,v in r.items() if k!='entry_hash'})
        self.assertIsNone(self.evaluate()['reward'])
    def test_no_live_adapter(self):
        self.p.synthetic=False
        with self.assertRaisesRegex(ValueError,'MOCK_ONLY'):self.result()
        self.p.synthetic=True
    def test_pk_cross_cycle_reuse(self):
        self.p.append('correlation_pk:100',dict(cycle_id='another-cycle'))
        self.rejected('SYSTEM_INVALID')
    def test_event_clock_matches_exact_attempt(self):
        self.coverage['events'][0]['at']=at(5);self.rejected('SYSTEM_INVALID')
    def test_evidence_immutable_after_gate(self):
        self.evaluate();self.coverage['authority']='changed-attestation'
        r=self.evaluate();self.assertEqual(r['correlation']['correlation_status'],'SYSTEM_INVALID');self.assertIsNone(r['reward'])
    def test_log_timestamp_not_send_identity(self):
        self.attempts[1]['send_started_at']=at(5);self.attempts[1]['persistence_confirmed_at']=at(5)
        self.rejected('SYSTEM_INVALID')
    def test_no_network_imports(self):
        import geas35.objectives.correlation as m
        tree=ast.parse(Path(m.__file__).read_text());names=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):names.extend(x.name for x in n.names)
            if isinstance(n,ast.ImportFrom):names.append(n.module or '')
        for n in names:self.assertFalse(any(x in n for x in ('requests','socket','pymysql','subprocess','controller')))

if __name__=='__main__':unittest.main()
