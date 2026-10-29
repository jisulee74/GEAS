"""Receive-only prospective evaluation sidecar. No controller/network/DB client.

P0 is immutable input. All writes go to a separate SQLite evaluation journal.
Real activation requires an explicit project declaration; a clock override is
allowed only in temporary synthetic storage. No target lookup by outcome/time.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import tempfile
from zoneinfo import ZoneInfo
from .binding import canonical, digest, timestamp, finite, TargetContext, VERSION
from .quality import QualityEvaluator

SEMANTICS='PROJECT_OPERATIONAL_POINT_DISCREPANCY'
IDENTITY={'crop':'cucumber','crop_cycle':72,'source':97}


@dataclass(frozen=True)
class Declaration:
    payload: str

    def __post_init__(self):
        d=json.loads(self.payload)
        required={'target_profile_id','target_version',*IDENTITY,'temperature','rh','effective_from','available_at','declared_by','declaration_basis','profile_hash','claim'}
        if set(d) not in (required,required|{'effective_until'}) or any(d[k]!=v for k,v in IDENTITY.items()):raise ValueError('DECLARATION_SCHEMA')
        if d['claim']!='PROJECT_OPERATIONAL_EVALUATION_TARGET':raise ValueError('UNSUPPORTED_CLAIM')
        for k in ('target_profile_id','target_version','declared_by','declaration_basis'):
            if not isinstance(d[k],str) or not d[k].strip():raise ValueError('DECLARATION_PROVENANCE_REQUIRED')
        for c,u in [('temperature','degC'),('rh','percentage_points')]:
            x=d[c]
            if set(x)!={'active','reference_value','unit','semantics','evidence_source'}:raise ValueError('COMPONENT_SCHEMA')
            if type(x['active']) is not bool or x['unit']!=u or x['semantics']!=SEMANTICS or not isinstance(x['evidence_source'],str) or not x['evidence_source'].strip():raise ValueError('COMPONENT_SEMANTICS')
            if x['active'] and (not finite(x['reference_value']) or (c=='rh' and not 0<=x['reference_value']<=100) or (c=='temperature' and x['reference_value']< -273.15)):raise ValueError('INVALID_REFERENCE')
            if not x['active'] and x['reference_value'] is not None:raise ValueError('INACTIVE_REFERENCE')
        for k in ('available_at','effective_from'):timestamp(d[k])
        if 'effective_until' in d and timestamp(d['effective_until'])<=timestamp(d['effective_from']):raise ValueError('INVALID_WINDOW')
        body={k:v for k,v in d.items() if k!='profile_hash'}
        if digest(body)!=d['profile_hash']:raise ValueError('PROFILE_HASH_MISMATCH')
        object.__setattr__(self,'payload',canonical(d))

    @classmethod
    def create(cls, **body):
        return cls(canonical(dict(body,profile_hash=digest(body))))

    @property
    def data(self):return json.loads(self.payload)


@dataclass(frozen=True)
class OperationalBinding:
    payload: str
    @property
    def data(self):return json.loads(self.payload)
    @property
    def target_hash(self):return digest(self.data)
    @property
    def readiness(self):return 'VALID' if self.data['active'] else 'NOT_ACTIVE_CONTEXT_UNRESOLVED'


def receipt(row,stream,session):
    """Verify collector envelope and raw hash without changing the source row."""
    x=json.loads(canonical(row))
    if x.get('stream')!=stream or x.get('session_id')!=session:raise ValueError('RECEIPT_IDENTITY_MISMATCH')
    if digest({k:v for k,v in x.items() if k!='entry_hash'})!=x.get('entry_hash') or digest(x['raw'])!=x.get('source_hash'):raise ValueError('RECEIPT_HASH_MISMATCH')
    if x.get('source_row_id')!=x['raw'].get('idx'):raise ValueError('RECEIPT_ROW_MISMATCH')
    if stream=='observations' and (x.get('source_table')!='data_silla_enc' or str(x['raw'].get('iot_data_idx'))!='97'):raise ValueError('RECEIPT_SOURCE_MISMATCH')
    if stream=='sent_commands' and x.get('source_table')!='send_ds_control_api_history':raise ValueError('RECEIPT_SOURCE_MISMATCH')
    if x.get('source_timestamp')!=x['raw'].get('reg_date'):raise ValueError('RECEIPT_TIME_MISMATCH')
    t=datetime.fromisoformat(x['source_timestamp'])
    if t.tzinfo is None:t=t.replace(tzinfo=ZoneInfo('Asia/Seoul'))
    t=t.astimezone(timezone.utc)
    at=timestamp(x['receipt_at_utc'])
    if t>at:raise ValueError('FUTURE_SOURCE_TIMESTAMP')
    return x,t,at


class LivePath:
    def __init__(self,directory,contract,session_id,*,synthetic=False,clock=None):
        p=Path(directory).absolute()
        if any(x.is_symlink() for x in (p,*p.parents)) or 'prospective_collection' in p.parts:raise ValueError('P0_STORAGE_FORBIDDEN')
        if synthetic and Path(tempfile.gettempdir()).resolve() not in p.resolve().parents:raise ValueError('SYNTHETIC_TEMP_ONLY')
        if clock is not None and not synthetic:raise ValueError('LIVE_CLOCK_OVERRIDE_FORBIDDEN')
        self.clock=clock or (lambda:datetime.now(timezone.utc).isoformat())
        self.synthetic=synthetic;self.session_id=session_id
        self.quality=QualityEvaluator.from_file(contract)
        p.mkdir(parents=True,exist_ok=True)
        db=p/'evaluation.sqlite'
        if db.is_symlink():raise ValueError('SYMLINK_FORBIDDEN')
        self.db=sqlite3.connect(db)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, key TEXT UNIQUE, at TEXT, payload TEXT, previous_hash TEXT, hash TEXT)')
        for op in ('UPDATE','DELETE'):
            self.db.execute(f"CREATE TRIGGER IF NOT EXISTS no_{op} BEFORE {op} ON events BEGIN SELECT RAISE(ABORT,'IMMUTABLE'); END")
        self.db.commit()
        try:
            prev='0'*64
            for seq,key,at,payload,previous,h in self.db.execute('SELECT * FROM events ORDER BY seq'):
                if previous!=prev or digest([seq,key,at,payload,previous])!=h:raise ValueError('JOURNAL_INTEGRITY_FAILURE')
                prev=h
            meta=self.get('session')
            if meta:
                m=json.loads(meta[2])
                if m['session_id']!=session_id or m['synthetic']!=synthetic:raise ValueError('SESSION_MISMATCH')
                self.started_at=meta[1]
            else:
                self.started_at=self.now();self.append('session',dict(session_id=session_id,synthetic=synthetic),self.started_at)
        except BaseException:self.db.close();raise

    def close(self):self.db.close()
    def now(self):return timestamp(self.clock()).isoformat()
    def get(self,key):return self.db.execute('SELECT seq,at,payload FROM events WHERE key=?',(key,)).fetchone()
    def append(self,key,value,at=None):
        return self.append_many([(key,value)],at)

    def append_many(self,items,at=None):
        at=at or self.now()
        try:
            self.db.execute('BEGIN IMMEDIATE')
            last=self.db.execute('SELECT seq,at,hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            if last and timestamp(at)<timestamp(last[1]):raise ValueError('CLOCK_REGRESSION')
            for key,value in items:
                if self.get(key):raise ValueError('IMMUTABLE_EVENT_CONFLICT')
                body=[last[0]+1 if last else 1,key,at,canonical(value),last[2] if last else '0'*64]
                h=digest(body)
                self.db.execute('INSERT INTO events VALUES (?,?,?,?,?,?)',(*body,h))
                last=(body[0],at,h)
            self.db.commit();return body[0]
        except BaseException:self.db.rollback();raise

    def declare(self,declaration):
        d=declaration.data;now=self.now()
        if not self.synthetic and 'TEST_FIXTURE_ONLY' in declaration.payload:raise ValueError('SYNTHETIC_DECLARATION_FORBIDDEN')
        if timestamp(d['available_at'])>timestamp(now):raise ValueError('TARGET_NOT_AVAILABLE_PRE_ACTION')
        self.append('profile:'+d['target_profile_id']+':'+d['target_version'],d,now)
        return d['profile_hash']

    def profile(self,h):
        for payload, in self.db.execute("SELECT payload FROM events WHERE key LIKE 'profile:%'"):
            d=json.loads(payload)
            if d['profile_hash']==h:return d
        raise ValueError('TARGET_NOT_BOUND')

    def prepare(self,profile_hash,action_id,current_receipt,action_identity_evidence):
        """Reserve an explicitly identified future action; no guessed timestamp ID."""
        now=self.now();d=self.profile(profile_hash)
        if not isinstance(action_id,str) or not action_id or not isinstance(action_identity_evidence,str) or not action_identity_evidence.strip():raise ValueError('ACTION_ATTRIBUTION_UNRESOLVED')
        for k,reason in [('available_at','TARGET_NOT_AVAILABLE_PRE_ACTION'),('effective_from','TARGET_NOT_EFFECTIVE')]:
            if timestamp(d[k])>timestamp(now):raise ValueError(reason)
        if d.get('effective_until') and timestamp(now)>=timestamp(d['effective_until']):raise ValueError('TARGET_EXPIRED')
        row,measured,received=receipt(current_receipt,'observations',self.session_id)
        if not timestamp(self.started_at)<measured<=received<=timestamp(now):raise ValueError('NONPROSPECTIVE_INPUT')
        if self.get('action:'+action_id):raise ValueError('BINDING_AFTER_ACTION')
        bound={}
        for c,name,unit in [('temperature','temperature','°C'),('RH','rh','%RH')]:
            x=d[name]
            bound[c]=dict(objective_version='future_f_operational.v1',target_version=d['target_version'],profile_hash=profile_hash,component=c,active=x['active'],reference_value=x['reference_value'],unit=unit,reference_semantics=SEMANTICS,scope='PROSPECTIVE_OPERATIONAL',decision_time=now,action_time=None,action_id=action_id,component_mask=[k for k,n in [('temperature','temperature'),('RH','rh')] if d[n]['active']],evaluator_version=VERSION)
        data=dict(bindings=bound,input_receipt_hash=row['entry_hash'],input_receipt=row,action_identity_evidence=action_identity_evidence,profile_hash=profile_hash)
        self.append('prepared:'+action_id,data,now)
        # Sample trusted wall clock after the durable binding transaction returns.
        self.append('sealed:'+action_id,dict(binding_commit_completed_at=self.now()))
        return self.context(action_id)

    def context(self,action_id):
        rec=self.get('prepared:'+action_id)
        if not rec or not self.get('sealed:'+action_id):raise ValueError('TARGET_NOT_BOUND')
        b=json.loads(rec[2])
        value={'binding_commit_completed_at':json.loads(self.get('sealed:'+action_id)[2])['binding_commit_completed_at'],'source':97,'crop_cycle':72,'action_id':action_id,'profile_hash':b['profile_hash'],'target_declaration':self.profile(b['profile_hash']),'input_receipt_hash':b['input_receipt_hash'],'observation_channels':{k:b['input_receipt']['raw'].get(v) for k,v in [('T1','in_temp'),('T2','in_temp2'),('RH1','in_hum'),('RH2','in_hum2')]},'bindings':b['bindings'],'binding_hashes':{k:digest(v) for k,v in b['bindings'].items()}}
        return dict(payload=value,context_hash=digest(value))

    def lookup(self,h):
        for payload, in self.db.execute("SELECT payload FROM events WHERE key LIKE 'prepared:%'"):
            for b in json.loads(payload)['bindings'].values():
                obj=OperationalBinding(canonical(b))
                if obj.target_hash==h:return obj
        return None

    def verifies_operational_binding(self,b):
        old=self.lookup(b.target_hash)
        return old is not None and old.payload==b.payload and b.data['scope']=='PROSPECTIVE_OPERATIONAL'

    def null(self,reason,status='UNSCORABLE'):
        out=self.quality.unavailable([{'reason':reason}]);out['evaluation_status']=status
        return out

    def observe_action(self,action_id,sent_receipt,link,policy_context):
        """An explicit, evidenced source-ID association is mandatory, not nearest join."""
        now=self.now();prepared=self.get('prepared:'+action_id)
        if not prepared:return self.null('TARGET_NOT_BOUND')
        row,sent,received=receipt(sent_receipt,'sent_commands',self.session_id)
        ctx=self.context(action_id)
        decl=ctx['payload']['target_declaration']
        if sent<timestamp(decl['effective_from']):return self.null('TARGET_NOT_EFFECTIVE')
        if decl.get('effective_until') and (sent>=timestamp(decl['effective_until']) or timestamp(now)>=timestamp(decl['effective_until'])):return self.null('TARGET_EXPIRED')
        if policy_context!=ctx:return self.null('TARGET_CONTEXT_MISMATCH')
        if link.get('method')!='EXPLICIT_ID' or not link.get('evidence_source') or link.get('action_id')!=action_id or link.get('event_hash')!=row['entry_hash'] or link.get('ambiguous',True):return self.null('ACTION_ATTRIBUTION_UNRESOLVED')
        if not all(b['active'] for b in ctx['payload']['bindings'].values()):return self.null('NOT_ACTIVE_CONTEXT_UNRESOLVED')
        if not timestamp(ctx['payload']['binding_commit_completed_at'])<sent<=received<=timestamp(now):return self.null('BINDING_AFTER_ACTION')
        if self.get('sent_event:'+row['entry_hash']):return self.null('ACTION_ATTRIBUTION_UNRESOLVED')
        items=[('action:'+action_id,dict(sent_at=sent.isoformat(),receipt=row,link=link,context=ctx)),('sent_event:'+row['entry_hash'],dict(action_id=action_id))]
        if not self.get('start'):
            items.append(('start',dict(event='SYNTHETIC_OBJECTIVE_BOUND_EVALUATION_START' if self.synthetic else 'OBJECTIVE_BOUND_EVALUATION_START',action_id=action_id,action_time=sent.isoformat(),binding_commit=prepared[1],context_hash=ctx['context_hash'],synthetic=self.synthetic)))
        self.append_many(items,now)
        return dict(evaluation_status='ACTION_BOUND',context_hash=ctx['context_hash'])

    def evaluate(self,action_id,outcome_receipt,link,policy_context,environment_context,*,accounting_status='COMPLETE',support_termination=None):
        if accounting_status not in ('COMPLETE','TRUNCATED','SYSTEM_INVALID'):raise ValueError('ACCOUNTING_SCHEMA')
        action=self.get('action:'+action_id)
        if not action:result=self.null('ACTION_ATTRIBUTION_UNRESOLVED' if self.get('prepared:'+action_id) else 'TARGET_NOT_BOUND')
        else:
            a=json.loads(action[2]);ctx=self.context(action_id)
            try:
                row,measured,received=receipt(outcome_receipt,'observations',self.session_id)
                decl=ctx['payload']['target_declaration']
                if decl.get('effective_until') and (measured>=timestamp(decl['effective_until']) or timestamp(self.now())>=timestamp(decl['effective_until'])):raise ValueError('TARGET_EXPIRED')
                if not timestamp(a['sent_at'])<measured<=received<=timestamp(self.now()):raise ValueError('OUTCOME_NOT_AFTER_ACTION')
                if link.get('method')!='EXPLICIT_ID' or not link.get('evidence_source') or link.get('action_id')!=action_id or link.get('event_hash')!=row['entry_hash'] or link.get('ambiguous',True) or link.get('intervening_override') is not False:raise ValueError('ACTION_ATTRIBUTION_UNRESOLVED')
                if link.get('profile_hash')!=ctx['payload']['profile_hash']:raise ValueError('TARGET_CONTEXT_MISMATCH')
                if policy_context!=ctx or environment_context!=ctx or a['context']!=ctx:raise ValueError('TARGET_CONTEXT_MISMATCH')
                if accounting_status=='SYSTEM_INVALID':raise ValueError('SYSTEM_INVALID')
                hs=ctx['payload']['binding_hashes'];contexts={k:TargetContext.from_binding(self.lookup(h)) for k,h in hs.items()}
                obs={}
                for ch,raw in [('T1','in_temp'),('T2','in_temp2'),('RH1','in_hum'),('RH2','in_hum2')]:
                    v=row['raw'].get(raw)
                    try:obs[ch]=float(v) if isinstance(v,str) else v
                    except ValueError:obs[ch]=v
                # Action get() for channel evaluator uses source sent time, not late receipt.
                view=_EvaluationView(self,a['sent_at'],action_id)
                result=self.quality.evaluate(view,hs,obs,dict(action_id=action_id,observed_at=measured.isoformat(),binding_hashes=hs),contexts,contexts)
            except (ValueError,KeyError,TypeError) as exc:
                reason=str(exc) if isinstance(exc,ValueError) else 'INVALID_RECEIPT_SCHEMA'
                result=self.null(reason,'SYSTEM_INVALID' if reason=='SYSTEM_INVALID' else 'UNSCORABLE')
        result.update(action_id=action_id,outcome_identity=outcome_receipt.get('entry_hash'),policy_context_hash=policy_context.get('context_hash'),environment_evaluation_context_hash=environment_context.get('context_hash'),accounting_status=accounting_status,support_termination=support_termination,episode_return=None,synthetic=self.synthetic,live_activation=not self.synthetic and bool(self.get('start')))
        self.append('evaluation:'+action_id+':'+str(outcome_receipt.get('entry_hash')),dict(result=result,outcome_receipt=outcome_receipt,attribution=link))
        return result


class _EvaluationView:
    def __init__(self,path,sent_at,action_id):self.path=path;self.sent_at=sent_at;self.action_id=action_id
    def lookup(self,h):return self.path.lookup(h)
    def verifies_operational_binding(self,b):return self.path.verifies_operational_binding(b)
    def get(self,key):
        rec=self.path.get(key)
        if rec and key=='action:'+self.action_id:return rec[0],self.sent_at,rec[2]
        return rec
