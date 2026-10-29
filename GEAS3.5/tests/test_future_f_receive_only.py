import ast
import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from geas35.collection import receive_only as m

CONFIG=json.loads(Path(m.__file__).with_name('pilot_iot97_cycle72_v1.json').read_text())
NOW=dt.datetime(2026,10,22,5,0,tzinfo=dt.timezone.utc)
def observation(idx=1,**kw):
    return dict(idx=idx,iot_data_idx=97,reg_date='2026-10-22 14:00:00',in_temp='25',in_temp2='26',in_hum='70',in_hum2='71',out_temp='20',out_hum='60',out_light=100,**kw)
class Fixture:
    def __init__(self): self.rows={s:[] for s in m.STREAMS};self.interrupt=False
    def begin(self):
        if self.interrupt:raise ConnectionError()
    def binding(self):pass
    def page(self,s,after,ceiling=None):return [r for r in sorted(self.rows[s],key=lambda r:r['idx']) if r['idx']>after and (ceiling is None or r['idx']<=ceiling)][:2]
class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'fixture';self.store=m.Store(self.path,CONFIG,{'status':'FIXTURE'},create=True)
    def tearDown(self):
        if not self.store.lock.closed:self.store.close()
        self.tmp.cleanup()
    def reopen(self):self.store.close();self.store=m.Store(self.path,CONFIG,{},create=False)
    def records(self,s='observations'):return [json.loads(l) for l in (self.path/(s+'.jsonl')).read_text().splitlines()]
    def starts(self):return [r for r in self.records('collector_events') if r.get('event')=='PILOT_COLLECTION_START']
    def test_launch_empty_and_resume_no_start(self):
        self.store.append('collector_events',{'event':'SESSION_INITIALIZED'})
        m.poll(Fixture(),self.store);self.reopen()
        self.assertFalse(self.store.pilot_collection_started);self.assertIsNone(self.store.pilot_collection_start);self.assertEqual(self.starts(),[])
    def test_exit_before_first_record(self):
        self.store.append('collector_events',{'event':'SESSION_INITIALIZED'})
        self.reopen()
        self.assertFalse(self.store.pilot_collection_started);self.assertEqual(self.starts(),[])
    def test_authorization_gate_retained(self):
        text=Path(m.__file__).read_text()
        self.assertIn("auth.get('authorize_collection') is not True",text)
        self.assertIn("'approved_by','owner_role','approved_at','scope_confirmed','retention_approved','storage_approved'",text)
        self.assertLess(text.index("raise ValueError('explicit human collection authorization required')"),text.index('conn=pymysql.connect'))
    def test_db_failure_before_first_no_start(self):
        f=Fixture();f.interrupt=True
        with self.assertRaises(ConnectionError):m.poll(f,self.store)
        self.assertFalse(self.store.pilot_collection_started);self.assertEqual(self.starts(),[])
    def test_write_failure_no_start(self):
        with patch.object(m.os,'open',side_effect=OSError('fixture write failure')):
            with self.assertRaises(OSError):self.store.ingest('observations',observation(),NOW)
        self.assertFalse(self.store.pilot_collection_started);self.assertEqual(self.starts(),[])
    def test_fsync_failure_no_start(self):
        with patch.object(m.os,'fsync',side_effect=OSError('fixture fsync failure')):
            with self.assertRaises(OSError):self.store.ingest('observations',observation(),NOW)
        self.assertFalse(self.store.pilot_collection_started);self.assertEqual(self.starts(),[])
    def test_success_start_once_and_restart(self):
        self.store.ingest('observations',observation(),NOW)
        self.assertEqual(self.store.pilot_collection_start,NOW.isoformat())
        self.assertEqual(self.starts()[0]['collector_sequence'],2)
        self.assertEqual(self.starts()[0]['session_id'],self.path.name)
        self.reopen();self.store.ingest('observations',observation(2),NOW+dt.timedelta(minutes=5))
        self.assertEqual(len(self.starts()),1);self.assertEqual(self.store.pilot_collection_start,NOW.isoformat())
    def test_crash_after_data_fsync_before_start(self):
        with patch.object(self.store,'_ensure_start',side_effect=RuntimeError('fixture crash')):
            with self.assertRaises(RuntimeError):self.store.ingest('observations',observation(),NOW)
        original=(self.path/'observations.jsonl').read_bytes()
        self.assertEqual(self.starts(),[]);self.reopen()
        self.assertEqual(len(self.starts()),1);self.assertEqual(self.store.pilot_collection_start,NOW.isoformat())
        self.assertEqual((self.path/'observations.jsonl').read_bytes(),original)
        self.reopen();self.assertEqual(len(self.starts()),1)
    def test_config(self):
        for k,v in [('phase','P1'),('facility_id',98),('poll_seconds',0),('output_relative','/tmp/x')]:
            with self.assertRaises(ValueError):m.validate_config(dict(CONFIG,**{k:v}))
    def test_source_filter(self):
        x=observation();x['iot_data_idx']=98
        with self.assertRaises(ValueError):self.store.ingest('observations',x,NOW)
    def test_receipt_raw_sequence(self):
        self.store.ingest('observations',observation(),NOW);self.store.ingest('observations',observation(2),NOW)
        x=self.records();self.assertEqual([r['collector_sequence'] for r in x],[1,3]);self.assertEqual(x[0]['receipt_at_utc'],NOW.isoformat());self.assertIn('+09:00',x[0]['receipt_at_kst']);self.assertEqual(x[0]['raw']['in_temp2'],'26')
    def test_dedup(self):
        self.assertTrue(self.store.ingest('observations',observation(),NOW));self.assertFalse(self.store.ingest('observations',observation(),NOW));self.assertEqual(len(self.records()),1)
    def test_restart_checkpoint(self):
        self.store.ingest('observations',observation(),NOW);self.store.checkpoint();self.reopen();self.assertFalse(self.store.ingest('observations',observation(),NOW));self.store.ingest('observations',observation(2),NOW);self.assertEqual(self.records()[-1]['collector_sequence'],3)
    def test_crash_after_append_before_checkpoint(self):
        self.store.checkpoint();self.store.ingest('observations',observation(),NOW);self.reopen();self.assertEqual(self.store.seq,2);self.assertFalse(self.store.ingest('observations',observation(),NOW))
    def test_late_low_id_replay_and_interruption(self):
        f=Fixture();f.rows['observations']=[observation(2),observation(4)];m.poll(f,self.store)
        x=observation(3);x['reg_date']='2026-10-21 12:00:00';f.rows['observations'].append(x)
        f.interrupt=True
        with self.assertRaises(ConnectionError):m.poll(f,self.store)
        self.reopen();f.interrupt=False
        for _ in range(4):m.poll(f,self.store)
        self.assertEqual(sorted(r['source_row_id'] for r in self.records()),[2,3,4]);self.assertTrue(next(r for r in self.records() if r['source_row_id']==3)['qc']['late_arrival'])
    def test_offgrid_actor(self):
        x={'idx':7,'reg_date':'2026-10-22 13:59:07','channel':'1','ratio':'50','n_idx':'unknown','operation':'OPEN'}
        self.store.ingest('sent_commands',x,NOW);r=self.records('sent_commands')[0];self.assertEqual(r['source_timestamp'],x['reg_date']);self.assertEqual(r['actor'],'UNKNOWN');self.assertEqual(r['attribution'],'UNKNOWN')
    def test_append_only_and_revision(self):
        self.store.ingest('observations',observation(),NOW);before=(self.path/'observations.jsonl').read_bytes();x=observation();x['in_temp']='27';self.store.ingest('observations',x,NOW);self.assertTrue((self.path/'observations.jsonl').read_bytes().startswith(before));self.assertTrue(self.records()[-1]['source_revision'])
    def test_revision_reversion_preserved(self):
        a=observation();b=dict(a,in_temp='27')
        for x in (a,b,a):self.assertTrue(self.store.ingest('observations',x,NOW))
        self.reopen();self.assertFalse(self.store.ingest('observations',a,NOW));self.assertEqual(len(self.records()),3)
    def test_checkpoint_manifest_binding(self):
        self.store.checkpoint();p=self.path/'checkpoint.json';x=json.loads(p.read_text());x['payload']['manifest_hash']='wrong';x['hash']=m.digest(x['payload']);p.write_text(json.dumps(x));self.store.close()
        with self.assertRaises(ValueError):m.Store(self.path,CONFIG,{})
    def test_symlink_parent_rejected(self):
        link=Path(self.tmp.name)/'link';link.symlink_to(self.path,target_is_directory=True)
        with self.assertRaises(ValueError):m.Store(link/'nested',CONFIG,{},create=True)
    def test_manifest_tamper(self):
        p=self.path/'session_manifest.json';x=json.loads(p.read_text());x['payload']['phase']='P1';p.write_text(json.dumps(x));self.store.close()
        with self.assertRaises(ValueError):m.Store(self.path,CONFIG,{})
    def test_log_tamper_and_partial(self):
        self.store.ingest('observations',observation(),NOW);p=self.path/'observations.jsonl';p.write_bytes(p.read_bytes()+b'{');self.store.close()
        with self.assertRaises(ValueError):m.Store(self.path,CONFIG,{})
    def test_qc(self):
        x=observation();x.update(in_temp=None,in_temp2='NaN',in_hum='101',out_light='bad',reg_date='2026-10-21 00:00:00');self.store.ingest('observations',x,NOW);q=self.records()[0]['qc'];self.assertIn('in_temp',q['missing']);self.assertIn('in_temp2',q['nonfinite']);self.assertIn('in_hum',q['physical_invalid']);self.assertIn('out_light',q['unit_schema_mismatch']);self.assertTrue(q['stale']);self.assertFalse(q['imputed'])
    def test_gap_regression(self):
        self.store.ingest('observations',observation(),NOW)
        x=observation(2);x['reg_date']='2026-10-22 14:15:00';self.store.ingest('observations',x,NOW+dt.timedelta(minutes=15));self.assertTrue(self.records()[-1]['qc']['gap'])
        x=observation(3);self.store.ingest('observations',x,NOW+dt.timedelta(minutes=16));self.assertTrue(self.records()[-1]['qc']['timestamp_regression'])
    def test_p0_ineligible_no_fake_bindings(self):
        self.store.ingest('observations',observation(),NOW);x=self.records()[0];self.assertFalse(x['reward_eligible']);self.assertEqual(x['objective_status'],'OBJECTIVE_TARGET_MISSING');self.assertEqual(self.records('decision_bindings'),[])
    def test_fixture_network_disabled(self):
        with patch.object(socket,'socket',side_effect=AssertionError('NETWORK FORBIDDEN')):
            f=Fixture();f.rows['observations']=[observation()];m.poll(f,self.store)
    def test_static_no_send(self):
        tree=ast.parse(Path(m.__file__).read_text());imports=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):imports.extend(x.name for x in n.names)
            elif isinstance(n,ast.ImportFrom):imports.append(n.module or '')
        self.assertFalse(any('control' in x or x.startswith(('requests','http','urllib')) for x in imports))
        self.assertNotIn('send_control_api',Path(m.__file__).read_text())
        self.assertFalse(any('control_io' in x for x in sys.modules))
    def test_readonly_select_enforced(self):
        class Cursor:
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def execute(self,q,args=()):calls.append(q)
            def fetchone(self):return {'ro':1}
            def fetchall(self):return []
        class Conn:
            def cursor(self):return Cursor()
            def rollback(self):pass
        calls=[];s=m.ReadOnlySource(Conn(),CONFIG);s.begin()
        for q in ['INSERT INTO x VALUES(1)','UPDATE x SET a=1','DELETE FROM x','SELECT 1; DELETE FROM x']:
            with self.assertRaises(ValueError):s.select(q)
        s.page('observations',0);self.assertIn('SET SESSION TRANSACTION READ ONLY',calls);self.assertTrue(all(q.startswith(('SELECT ','SET SESSION TRANSACTION READ ONLY','START TRANSACTION READ ONLY')) for q in calls));self.assertIn('iot_data_idx=%s',calls[-1])
    def test_kernel_no_http_and_exec(self):
        code='''from geas35.collection.receive_only import lock_network
import socket, urllib.request, subprocess
lock_network()
for fn in [lambda: socket.socket(), lambda: urllib.request.urlopen('http://127.0.0.1:9',timeout=1), lambda: subprocess.run(['/bin/true'])]:
 try: fn()
 except Exception: pass
 else: raise AssertionError('isolation failed')
print('ISOLATED')
'''
        p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(Path(m.__file__).parents[2])))
        self.assertEqual(p.returncode,0,p.stderr);self.assertIn('ISOLATED',p.stdout)
    def test_session_lock(self):
        with self.assertRaises(BlockingIOError):m.Store(self.path,CONFIG,{})

if __name__=='__main__':unittest.main()
