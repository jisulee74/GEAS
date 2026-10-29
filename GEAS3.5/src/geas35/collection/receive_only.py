"""Future-F P0. CLI requires explicit authorization; never run for preflight.

Only preopened MySQL socket survives kernel network lockdown. JSONL is source
of truth; checkpoint can lag. Partial writes fail closed (never rewrite history).
"""
from __future__ import annotations
import argparse
import ctypes
import ctypes.util
import datetime as dt
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
from zoneinfo import ZoneInfo

VERSION = 'future_f_receive_only.v1.1'
KST = ZoneInfo('Asia/Seoul')
SENSORS = {'in_temp':'degC','in_temp2':'degC','in_hum':'percent','in_hum2':'percent',
           'out_temp':'degC','out_hum':'percent','out_light':'W/m2'}
STREAMS = {
 'observations': ('data_silla_enc', ['idx','iot_data_idx','reg_date',*SENSORS]),
 'generated_commands': ('optimal_monitoring_copy', ['idx','iot_data_idx','reg_date','pred_ltw','pred_rtw','pred_pc1','pred_pc2','pred_heater','pred_cooler','pred_fan']),
 'sent_commands': ('send_ds_control_api_history', ['idx','n_idx','reg_date','channel','operation','ratio']),
}
FILES = [*STREAMS,'context_events','decision_bindings','collector_events']

def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()

def raw_json(value):
    if isinstance(value, dict): return {k:raw_json(v) for k,v in value.items()}
    if isinstance(value, (dt.datetime,dt.date)): return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    if value is None or isinstance(value,(str,int,float,bool)): return value
    return str(value)

def stamp(): return dt.datetime.now(dt.timezone.utc)

def parse_time(v):
    x=dt.datetime.fromisoformat(str(v)); return x.replace(tzinfo=KST) if x.tzinfo is None else x

def validate_config(c):
    expected={'facility_id':97,'iot_data_idx':97,'crop_info_idx':72,'crop':'cucumber',
              'pilot_id':'pilot_iot97_cycle72','crop_code':'04','transplant_date':'2026-08-14',
              'pilot_mode':'SHADOW_ONLY','phase':'P0','timezone':'Asia/Seoul',
              'schema_version':'future_f_p0.v1','sent_source_binding':'all_history_events_iot97_external_contract',
              'output_relative':'prospective_collection/future_f/pilot_iot97_cycle72'}
    for k,v in expected.items():
        if c.get(k)!=v: raise ValueError('pilot contract mismatch: '+k)
    for k in ('batch_size','initial_replay_ids','poll_seconds','expected_sensor_interval_seconds','receipt_age_warning_seconds'):
        if type(c.get(k)) is not int or not 0<c[k]<=10000: raise ValueError(k)
    if set(c)!=set(expected)|{'batch_size','initial_replay_ids','poll_seconds','expected_sensor_interval_seconds','receipt_age_warning_seconds'}: raise ValueError('unknown config field')
    return c

def lock_network():
    """Linux seccomp default allow; irrevocably deny creation/use of new peers.
    Existing DB send/read is allowed; socket/connect/exec/fd import paths denied.
    Fail closed on unsupported kernels/libraries. No global host modification.
    """
    lib=ctypes.CDLL(ctypes.util.find_library('seccomp') or 'libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=lib.seccomp_init(0x7fff0000)
    if not ctx: raise RuntimeError('seccomp unavailable')
    try:
        for name in ('socket','socketpair','connect','accept','accept4','execve','execveat',
                     'fork','vfork','clone','clone3','ptrace','pidfd_getfd','recvmsg','recvmmsg',
                     'io_uring_setup','io_uring_enter','io_uring_register'):
            nr=lib.seccomp_syscall_resolve_name(name.encode())
            if nr>=0 and lib.seccomp_rule_add(ctx,0x00050000|errno.EPERM,nr,0)!=0:
                raise RuntimeError('seccomp rule failed')
        if lib.seccomp_load(ctx)!=0: raise RuntimeError('seccomp load failed')
    finally: lib.seccomp_release(ctx)
    return 'KERNEL_NEW_NETWORK_AND_EXEC_DENIED'

def db_credentials(path):
    # Parse only five names, never execute dotenv or import old extraction/controller.
    keys={'GEAS_DB_HOST','GEAS_DB_PORT','GEAS_DB_USER','GEAS_DB_PASSWORD','GEAS_DB_NAME'}
    values={}
    with Path(path).open() as f:
        for line in f:
            k,sep,v=line.strip().partition('=')
            if sep and k in keys: values[k]=v.strip().strip('\"\'')
    if set(values)!=keys: raise ValueError('DB credential fields missing')
    return dict(host=values['GEAS_DB_HOST'],port=int(values['GEAS_DB_PORT']),user=values['GEAS_DB_USER'],
                password=values['GEAS_DB_PASSWORD'],database=values['GEAS_DB_NAME'])

class ReadOnlySource:
    def __init__(self, conn, config): self.conn=conn; self.config=config
    def begin(self):
        self.conn.rollback()
        with self.conn.cursor() as cur:
            cur.execute('SET SESSION TRANSACTION READ ONLY')
            cur.execute('START TRANSACTION READ ONLY')
            cur.execute('SELECT @@session.tx_read_only AS ro')
            if cur.fetchone()['ro']!=1: raise RuntimeError('DB readonly not enforced')
    def select(self,q,args=()):
        if not q.startswith('SELECT ') or ';' in q: raise ValueError('SELECT only')
        with self.conn.cursor() as cur: cur.execute(q,args);return cur.fetchall()
    def binding(self):
        c=self.config
        rows=self.select("SELECT idx,subj_cd,trans_crop_date,crop_end_date FROM crop_info WHERE iot_data_idx=%s AND del_yn='N' AND crop_end_date IS NULL AND trans_crop_date IS NOT NULL ORDER BY trans_crop_date DESC,idx DESC",(c['iot_data_idx'],))
        if len(rows)!=1 or int(rows[0]['idx'])!=c['crop_info_idx'] or rows[0]['subj_cd']!=c['crop_code'] or str(rows[0]['trans_crop_date'])[:10]!=c['transplant_date']: raise RuntimeError('pilot cycle changed/ambiguous')
        f=self.select("SELECT idx FROM iot_data_info WHERE idx=%s AND del_yn='N'",(c['facility_id'],))
        if len(f)!=1: raise RuntimeError('facility invalid')
    def where(self,s): return (' AND iot_data_idx=%s',(self.config['iot_data_idx'],)) if s!='sent_commands' else ('',())
    def max_id(self,s):
        w,args=self.where(s); rows=self.select('SELECT MAX(idx) AS n FROM '+STREAMS[s][0]+' WHERE 1=1'+w,args)
        return int(rows[0]['n'] or 0)
    def page(self,s,after,ceiling=None):
        table,cols=STREAMS[s];w,args=self.where(s)
        q='SELECT '+','.join(cols)+' FROM '+table+' WHERE idx>%s'+w
        args=(after,)+args
        if ceiling is not None:q+=' AND idx<=%s';args+=(ceiling,)
        return self.select(q+' ORDER BY idx LIMIT %s',args+(self.config['batch_size'],))

def sensor_qc(raw,receipt,previous,c):
    flags={}; missing=[];nonfinite=[];mismatch=[];invalid=[]
    for k in SENSORS:
        v=raw.get(k)
        if v is None or v=='':missing.append(k);continue
        try:n=float(v)
        except (ValueError,TypeError):mismatch.append(k);continue
        if not math.isfinite(n):nonfinite.append(k)
        elif ('hum' in k and not 0<=n<=100) or ('temp' in k and n< -273.15) or (k=='out_light' and n<0):invalid.append(k)
    flags.update(missing=missing,nonfinite=nonfinite,unit_schema_mismatch=mismatch,physical_invalid=invalid)
    try:
        ts=parse_time(raw['reg_date']);age=(receipt-ts).total_seconds()
        flags['late_arrival']=age>c['receipt_age_warning_seconds']
        flags['stale']=flags['late_arrival']
        flags['stale_definition']='source_timestamp_age_only; not B60 frozen-sensor classification'
        flags['future_timestamp']=age<0
        flags['timestamp_regression']=bool(previous and ts<previous)
        flags['gap']=bool(previous and (ts-previous).total_seconds()>c['expected_sensor_interval_seconds'])
    except (ValueError,TypeError,KeyError):flags.update(timestamp_invalid=True,stale='UNKNOWN',late_arrival='UNKNOWN',timestamp_regression='UNKNOWN',gap='UNKNOWN')
    flags['imputed']=False
    return flags

class Store:
    def __init__(self,path,config,clock,create=False,baseline=None):
        self.lock=None
        try:
            self._initialize(path,config,clock,create,baseline)
        except BaseException:
            self.close()
            raise
    def _initialize(self,path,config,clock,create,baseline):
        self.path=Path(path);self.config=validate_config(config)
        if any(p.is_symlink() for p in (self.path,*self.path.parents)):raise ValueError('symlink storage path')
        if create:self.path.mkdir(parents=True,exist_ok=False,mode=0o700)
        if self.path.is_symlink() or (self.path.stat().st_mode & 0o077):raise ValueError('private nonsymlink session required')
        self.lock=os.fdopen(os.open(self.path/'.lock',os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600),'a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if create:
            payload={'collector_version':VERSION,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     'config':config,'config_hash':digest(config),'session_id':self.path.name,'start_at':stamp().isoformat(),
                     'timezone':config['timezone'],'clock_health':clock,'host':socket.gethostname(),'pid':os.getpid(),
                     'source_tables':[v[0] for v in STREAMS.values()],'no_send':'preopened_DB_only_seccomp',
                     'phase':'P0','objective_status':'OBJECTIVE_TARGET_MISSING','reward_eligible':False,'baseline':baseline or {s:0 for s in STREAMS}}
            self._new('session_manifest.json',encoded({'payload':payload,'hash':digest(payload)})+b'\n')
            for s in FILES:self._new(s+'.jsonl',b'')
        if (self.path/'session_manifest.json').is_symlink():raise ValueError('symlink manifest')
        env=json.loads((self.path/'session_manifest.json').read_text());self.manifest=env['payload']
        if digest(self.manifest)!=env['hash'] or self.manifest['config_hash']!=digest(config):raise ValueError('manifest/config corruption')
        if self.manifest['code_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('code drift: new session required')
        self.seq=0;self.tail='0'*64;self.seen={};self.previous=None
        self.first_record=None;self.start_event=None
        allrows=[]
        for s in FILES:
            p=self.path/(s+'.jsonl')
            if p.is_symlink():raise ValueError('symlink log')
            with p.open('rb') as f:
                for line in f:
                    if not line.endswith(b'\n'):raise ValueError('partial append: quarantine, do not rewrite')
                    entry=json.loads(line)
                    if entry['stream']!=s:raise ValueError('wrong stream')
                    allrows.append(entry)
        for entry in sorted(allrows,key=lambda e:e['collector_sequence']):
            h=entry['entry_hash'];body={k:v for k,v in entry.items() if k!='entry_hash'}
            if entry['collector_sequence']!=self.seq+1 or entry['previous_hash']!=self.tail or digest(body)!=h:raise ValueError('JSONL integrity failure')
            self.seq+=1;self.tail=h
            if entry['stream']=='collector_events' and entry.get('event')=='PILOT_COLLECTION_START':
                if self.start_event is not None or self.first_record is None:raise ValueError('invalid start ordering')
                if entry['pilot_collection_start']!=self.first_record['receipt_at_utc'] or entry['first_record_hash']!=self.first_record['entry_hash']:raise ValueError('start binding mismatch')
                self.start_event=entry
            if entry['stream'] in STREAMS:
                if self.first_record is None:self.first_record=entry
                self.seen[(entry['stream'],entry['source_row_id'])]=entry['source_hash']
                if entry['stream']=='observations':self._advance_time(entry['raw'])
        self.progress={s:{'high':self.manifest['baseline'][s],'scan':self.manifest['baseline'][s]} for s in STREAMS}
        cp=self.path/'checkpoint.json'
        if cp.exists():
            if cp.is_symlink():raise ValueError('symlink checkpoint')
            x=json.loads(cp.read_text());body=x['payload']
            if digest(body)!=x['hash'] or body['manifest_hash']!=digest(self.manifest) or not 0<=body['sequence']<=self.seq:raise ValueError('checkpoint corruption')
            expected='0'*64 if body['sequence']==0 else sorted(allrows,key=lambda e:e['collector_sequence'])[body['sequence']-1]['entry_hash']
            if body['tail']!=expected:raise ValueError('checkpoint tail mismatch')
            self.progress=body['progress']
        if self.first_record is not None and self.start_event is None:
            # Reconfirm durability of recovered bytes before reconstructing metadata.
            fd=os.open(self.path/(self.first_record['stream']+'.jsonl'),os.O_RDONLY|os.O_NOFOLLOW)
            try:os.fsync(fd)
            finally:os.close(fd)
            self._ensure_start()
    @property
    def pilot_collection_started(self):return self.start_event is not None
    @property
    def pilot_collection_start(self):return self.start_event['pilot_collection_start'] if self.start_event else None
    def _ensure_start(self):
        if self.first_record is not None and self.start_event is None:
            r=self.first_record
            self.start_event=self.append('collector_events',{'event':'PILOT_COLLECTION_START',
                'pilot_collection_start':r['receipt_at_utc'],'receipt_at_utc':r['receipt_at_utc'],
                'recorded_at_utc':stamp().isoformat(),'first_record_hash':r['entry_hash'],
                'first_record_sequence':r['collector_sequence'],'first_record_stream':r['stream']})
    def _new(self,name,data):
        fd=os.open(self.path/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    def _advance_time(self,raw):
        try:t=parse_time(raw['reg_date']);self.previous=max(self.previous,t) if self.previous else t
        except (ValueError,TypeError,KeyError):pass
    def append(self,s,body):
        if s not in FILES:raise ValueError('stream')
        entry=dict(body,stream=s,collector_sequence=self.seq+1,session_id=self.manifest['session_id'],previous_hash=self.tail,
                   phase='P0',objective_status='OBJECTIVE_TARGET_MISSING',reward_eligible=False)
        entry['entry_hash']=digest(entry)
        fd=os.open(self.path/(s+'.jsonl'),os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
        with os.fdopen(fd,'ab') as f:f.write(encoded(entry)+b'\n');f.flush();os.fsync(f.fileno())
        self.seq+=1;self.tail=entry['entry_hash'];return entry
    def ingest(self,s,row,receipt=None):
        if s not in STREAMS:raise ValueError('stream')
        if s!='sent_commands' and str(row.get('iot_data_idx'))!=str(self.config['iot_data_idx']):raise ValueError('source mismatch')
        raw=raw_json(row);rid=int(row['idx']);h=digest(raw);key=(s,rid)
        if self.seen.get(key)==h:return False
        receipt=receipt or stamp()
        if receipt.tzinfo is None:raise ValueError('aware receipt required')
        body={'source_table':STREAMS[s][0],'source_row_id':rid,'source_hash':h,'source_timestamp':raw.get('reg_date'),
              'receipt_at_utc':receipt.astimezone(dt.timezone.utc).isoformat(),'receipt_at_kst':receipt.astimezone(KST).isoformat(),
              'raw':raw,'actor':'UNKNOWN','attribution':'UNKNOWN','cross_table_causal_order':'AMBIGUOUS',
              'source_revision':key in self.seen,'qc':{'duplicate':False}}
        if s=='observations':body['qc'].update(sensor_qc(raw,receipt,self.previous,self.config))
        entry=self.append(s,body) # returns only after successful fsync
        if self.first_record is None:self.first_record=entry
        self._ensure_start()
        self.seen[key]=h
        if s=='observations':self._advance_time(raw)
        return True
    def checkpoint(self):
        body={'sequence':self.seq,'tail':self.tail,'progress':self.progress,'manifest_hash':digest(self.manifest)}
        tmp=self.path/'checkpoint.tmp'
        if tmp.exists():tmp.unlink() # mutable checkpoint only, never JSONL
        self._new('checkpoint.tmp',encoded({'payload':body,'hash':digest(body)})+b'\n')
        os.replace(tmp,self.path/'checkpoint.json')
        fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    def close(self):
        if self.lock is not None and not self.lock.closed:
            fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()

def poll(source,store):
    source.begin();source.binding()
    for s in STREAMS:
        p=store.progress[s];count=0;duplicate=0
        # Forward by ID catches newly inserted rows even with old source timestamps.
        # Circular bounded pages revisit entire session ID domain for delayed commits/revisions.
        for after,ceiling,kind in [(p['high'],None,'high'),(p['scan'],p['high'],'scan')]:
            rows=source.page(s,after,ceiling)
            for row in rows:
                duplicate+=not store.ingest(s,row);count+=1
            if rows:p[kind]=int(rows[-1]['idx'])
            elif kind=='scan':p['scan']=store.manifest['baseline'][s]
        store.append('collector_events',{'event':'POLL','source':s,'read_count':count,'duplicate_suppressed':duplicate,'receipt_at_utc':stamp().isoformat()})
    store.checkpoint()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--db-env',required=True)
    parser.add_argument('--authorization',required=True,help='signed/approved collection authorization JSON, not technical readiness')
    parser.add_argument('--session-id',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    # This process must start without inherited descriptors; refuse sockets other than stdio.
    for fd in Path('/proc/self/fd').iterdir():
        try:
            if int(fd.name)>2 and os.readlink(fd).startswith('socket:'):raise RuntimeError('inherited network socket')
        except FileNotFoundError:pass
    os.environ.clear() # never forward API secrets or control configuration
    os.umask(0o077)
    config=validate_config(json.loads(Path(args.config).read_text()))
    auth=json.loads(Path(args.authorization).read_text())
    required=('approved_by','owner_role','approved_at','scope_confirmed','retention_approved','storage_approved','clock_verified_at')
    if auth.get('pilot_id')!=config['pilot_id'] or auth.get('authorize_collection') is not True or not all(auth.get(k) for k in required):raise ValueError('explicit human collection authorization required')
    if parse_time(auth['approved_at'])>stamp() or parse_time(auth['clock_verified_at'])>stamp():raise ValueError('future approval/clock verification')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.session_id):raise ValueError('session id')
    import pymysql
    conn=pymysql.connect(**db_credentials(args.db_env),cursorclass=pymysql.cursors.DictCursor,charset='utf8mb4',autocommit=False,connect_timeout=10,read_timeout=20,write_timeout=20,local_infile=False)
    lock_network()
    source=ReadOnlySource(conn,config);source.begin();source.binding()
    root=Path(__file__).resolve().parents[3]
    path=root/config['output_relative']/args.session_id
    baseline={s:max(0,source.max_id(s)-config['initial_replay_ids']) for s in STREAMS} if not args.resume else None
    # Host NTP must be confirmed before authorization; runtime wall/monotonic checks also logged.
    if auth.get('clock_health')!='SYNCHRONIZED_VERIFIED':raise ValueError('clock health approval required')
    store=Store(path,config,{'status':auth['clock_health'],'verified_at':auth.get('clock_verified_at'),'authorization_sha256':digest(auth)},create=not args.resume,baseline=baseline)
    store.append('collector_events',{'event':'SESSION_RESUME' if args.resume else 'SESSION_INITIALIZED','receipt_at_utc':stamp().isoformat()})
    last_wall=stamp();last_mono=time.monotonic()
    try:
        while True:
            now=stamp();mono=time.monotonic()
            if now<last_wall or abs((now-last_wall).total_seconds()-(mono-last_mono))>1:raise RuntimeError('clock jump: stop')
            last_wall,last_mono=now,mono
            poll(source,store);time.sleep(config['poll_seconds'])
    except Exception as exc:
        store.append('collector_events',{'event':'STOP_ERROR','error_type':type(exc).__name__,'receipt_at_utc':stamp().isoformat()})
        raise SystemExit('collector stopped; inspect protected log and resume after repair') from None
    finally:store.close();conn.close()

if __name__=='__main__': main()
