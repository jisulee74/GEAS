"""Fixture-only prospective binding and raw channel discrepancies, never reward.

Times are trusted inputs from a synthetic driver, not live clock attestations.
SQLite FULL transactions commit bindings before registered action events. No
live collector, policy, actuator or external DB adapter is connected here.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import sqlite3
import tempfile

VERSION = 'channel_discrepancy.v1'
SCOPE = 'TEST_FIXTURE_ONLY'
CHANNELS = {'temperature': ('T1', 'T2'), 'RH': ('RH1', 'RH2')}
UNITS = {'temperature': '°C', 'RH': '%RH'}
FIELDS = {'objective_version','target_version','component','active','inactive_reason',
          'reference_value','unit','reference_semantics','activation_context',
          'activation_rule_version','source','provenance_class','available_at',
          'effective_from','decision_time','action_id','action_time','component_mask',
          'evaluator_version','scope'}


def canonical(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(v):
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def timestamp(v):
    t = datetime.fromisoformat(v)
    if t.tzinfo is None:
        raise ValueError('AWARE_TIMESTAMP_REQUIRED')
    return t.astimezone(timezone.utc)


def finite(v):
    return type(v) in (int, float) and math.isfinite(v)


@dataclass(frozen=True)
class Binding:
    """Deeply immutable canonical payload; .data returns a detached copy."""
    payload: str

    def __post_init__(self):
        d = json.loads(self.payload)
        if set(d) != FIELDS:
            raise ValueError('BINDING_SCHEMA')
        if d['scope'] != SCOPE or d['provenance_class'] != SCOPE:
            raise ValueError('FIXTURE_ONLY')
        if d['component'] not in CHANNELS or type(d['active']) is not bool:
            raise ValueError('INVALID_COMPONENT')
        mask = d['component_mask']
        if not isinstance(mask, list) or len(set(mask)) != len(mask) or not set(mask) <= set(CHANNELS):
            raise ValueError('INVALID_MASK')
        d['component_mask'] = sorted(mask)
        if d['active'] != (d['component'] in mask):
            raise ValueError('MASK_ACTIVATION_CONFLICT')
        if not isinstance(d['activation_context'], dict):
            raise ValueError('INVALID_CONTEXT')
        if d['active'] and (not d['activation_context'] or d['inactive_reason']):
            raise ValueError('UNRESOLVED_ACTIVE_CONTEXT')
        if not d['active'] and not d['inactive_reason']:
            raise ValueError('INACTIVE_REASON_REQUIRED')
        for k in ('objective_version','target_version','source','activation_rule_version','reference_semantics','action_id','evaluator_version','unit'):
            if not isinstance(d[k], str) or not d[k]:
                raise ValueError('MISSING_PROVENANCE')
        for k in ('available_at','effective_from','decision_time','action_time'):
            d[k] = timestamp(d[k]).isoformat()
        if d['reference_value'] is not None:
            if not finite(d['reference_value']):
                raise ValueError('INVALID_REFERENCE')
            d['reference_value'] = float(d['reference_value'])
        elif d['active']:
            raise ValueError('MISSING_REFERENCE')
        object.__setattr__(self, 'payload', canonical(d))

    @classmethod
    def create(cls, **values):
        return cls(canonical(values))

    @property
    def data(self):
        return json.loads(self.payload)

    @property
    def target_hash(self):
        # Full per-action binding digest, not merely the scalar reference.
        return hashlib.sha256(self.payload.encode()).hexdigest()

    @property
    def readiness(self):
        d = self.data
        if timestamp(d['available_at']) > timestamp(d['decision_time']):
            return 'TARGET_NOT_AVAILABLE_PRE_ACTION'
        if timestamp(d['effective_from']) > timestamp(d['decision_time']):
            return 'TARGET_NOT_EFFECTIVE'
        return 'VALID' if d['active'] else 'NOT_ACTIVE_CONTEXT_UNRESOLVED'


@dataclass(frozen=True)
class TargetContext:
    target_version: str
    target_hash: str
    component_mask: tuple
    evaluator_version: str

    @classmethod
    def from_binding(cls, b):
        d = b.data
        return cls(d['target_version'], b.target_hash, tuple(d['component_mask']), d['evaluator_version'])


class FixtureJournal:
    """Immutable binding/action log, restricted to temporary fixture storage."""
    def __init__(self, directory):
        p = Path(directory).absolute()
        if any(x.is_symlink() for x in (p, *p.parents)) or 'prospective_collection' in p.parts:
            raise ValueError('LIVE_OR_SYMLINK_PATH_FORBIDDEN')
        tmp = Path(tempfile.gettempdir()).resolve()
        if tmp not in p.resolve().parents:
            raise ValueError('TEMP_FIXTURE_STORAGE_REQUIRED')
        p.mkdir(parents=True, exist_ok=True, mode=0o700)
        db = p/'fixture_bindings.sqlite'
        if db.is_symlink():
            raise ValueError('SYMLINK_FORBIDDEN')
        self.db = sqlite3.connect(db, timeout=10)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, event_key TEXT UNIQUE NOT NULL, at TEXT NOT NULL, payload TEXT NOT NULL, previous_hash TEXT NOT NULL, hash TEXT NOT NULL)')
        for operation in ('UPDATE', 'DELETE'):
            self.db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{operation} BEFORE {operation} ON events BEGIN SELECT RAISE(ABORT,'IMMUTABLE'); END")
        self.db.commit()
        try:
            previous = '0'*64
            for seq, kind, key, at, payload, prev, h in self.db.execute('SELECT * FROM events ORDER BY seq'):
                body = [seq, kind, key, at, payload, prev]
                if prev != previous or digest(body) != h:
                    raise ValueError('JOURNAL_INTEGRITY_FAILURE')
                if kind == 'binding' and Binding(payload).payload != payload:
                    raise ValueError('NONCANONICAL_BINDING')
                previous = h
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def get(self, key):
        return self.db.execute('SELECT seq,at,payload FROM events WHERE event_key=?', (key,)).fetchone()

    def _append(self, kind, key, at, payload):
        last = self.db.execute('SELECT seq,at,hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        at = timestamp(at).isoformat()
        if last and timestamp(at) < timestamp(last[1]):
            raise ValueError('TIME_REGRESSION')
        body = [last[0]+1 if last else 1, kind, key, at, payload, last[2] if last else '0'*64]
        self.db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)', (*body, digest(body)))
        return body[0]

    def commit(self, b, committed_at):
        d = b.data; key = 'binding:'+d['action_id']+':'+d['component']
        try:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.get(key)
            if old:
                if old[2] != b.payload:
                    raise ValueError('IMMUTABLE_BINDING_CONFLICT')
                self.db.commit(); return b.target_hash
            if b.readiness in ('TARGET_NOT_AVAILABLE_PRE_ACTION','TARGET_NOT_EFFECTIVE'):
                raise ValueError(b.readiness)
            if self.get('action:'+d['action_id']):
                raise ValueError('BINDING_AFTER_ACTION')
            if not timestamp(d['decision_time']) <= timestamp(committed_at) <= timestamp(d['action_time']):
                raise ValueError('INVALID_COMMIT_ORDER')
            self._append('binding', key, committed_at, b.payload)
            self.db.commit()  # completes before a registered action can proceed
            return b.target_hash
        except BaseException:
            self.db.rollback(); raise

    def action(self, action_id, binding_hashes, at):
        at = timestamp(at).isoformat()
        payload = canonical({'binding_hashes': sorted(binding_hashes)})
        try:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.get('action:'+action_id)
            if old:
                if (old[1],old[2]) != (at,payload):
                    raise ValueError('ACTION_CONFLICT')
                self.db.commit(); return old[0]
            bindings = [Binding(row[0]) for row in self.db.execute("SELECT payload FROM events WHERE kind='binding'")]
            bindings = [b for b in bindings if b.data['action_id']==action_id]
            if not bindings or len(set(binding_hashes)) != len(binding_hashes) or set(binding_hashes) != {b.target_hash for b in bindings}:
                raise ValueError('ACTION_BINDING_MISMATCH')
            if any(b.data['action_time'] != at for b in bindings):
                raise ValueError('ACTION_TIME_MISMATCH')
            masks = {tuple(b.data['component_mask']) for b in bindings}
            if len(masks)!=1 or not set(next(iter(masks))) <= {b.data['component'] for b in bindings}:
                raise ValueError('INCOMPLETE_COMPONENT_BINDING')
            seq = self._append('action', 'action:'+action_id, at, payload)
            self.db.commit(); return seq
        except BaseException:
            self.db.rollback(); raise

    def lookup(self, h):
        for payload, in self.db.execute("SELECT payload FROM events WHERE kind='binding'"):
            b = Binding(payload)
            if b.target_hash==h:
                return b
        return None


def evaluate(journal, binding_hash, component, observations, outcome, policy_context, environment_context):
    """Return per-channel discrepancies. score=None even for VALID results."""
    if component not in CHANNELS:
        raise ValueError('EXCLUDED_COMPONENT')
    b = journal.lookup(binding_hash); d = b.data if b else None
    status = 'VALID'
    if b is None:
        status = 'TARGET_NOT_BOUND'
    elif d['component']!=component or policy_context!=environment_context or policy_context!=TargetContext.from_binding(b) or d['evaluator_version']!=VERSION:
        status = 'TARGET_CONTEXT_MISMATCH'
    else:
        status = b.readiness
        if status=='VALID' and d['unit']!=UNITS[component]:
            status='UNSUPPORTED_UNIT'
        if status=='VALID' and d['reference_semantics']!='SYNTHETIC_POINT_DISCREPANCY':
            # Only the dedicated, persisted operational adapter may supply this
            # semantics. Fixture Binding construction remains fixture-only.
            verifier = getattr(journal, 'verifies_operational_binding', None)
            if d['reference_semantics']!='PROJECT_OPERATIONAL_POINT_DISCREPANCY' or not verifier or not verifier(b):
                status='UNSUPPORTED_REFERENCE_SEMANTICS'
        if status=='VALID':
            action=journal.get('action:'+d['action_id'])
            if not action or outcome.get('ambiguous') or outcome.get('action_id')!=d['action_id'] or outcome.get('binding_hash')!=binding_hash:
                status='AMBIGUOUS_ACTION_LINKAGE'
            else:
                try:
                    t=timestamp(outcome['observed_at']); at=timestamp(action[1])
                    ordered=t>at or (t==at and outcome.get('order_stream')=='fixture_journal' and type(outcome.get('sequence')) is int and outcome['sequence']>action[0])
                    if not ordered: status='OUTCOME_NOT_AFTER_ACTION'
                except (KeyError,ValueError,TypeError):
                    status='OUTCOME_NOT_AFTER_ACTION'
    rows=[]
    for channel in CHANNELS[component]:
        v=observations.get(channel); state=status
        if status=='VALID':
            if v is None: state='MISSING_SENSOR_VALUE'
            elif not finite(v) or (component=='RH' and not 0<=v<=100) or (component=='temperature' and v< -273.15): state='INVALID_SENSOR_VALUE'
        delta=float(v)-d['reference_value'] if state=='VALID' else None
        if delta is not None and not math.isfinite(delta):
            state='INVALID_SENSOR_VALUE'; delta=None
        rows.append(dict(component=component,channel=channel,status=state,reason=None if state=='VALID' else state,
                         observed_value=float(v) if finite(v) else None,reference_value=d['reference_value'] if d else None,
                         unit=UNITS[component],discrepancy_unit='°C' if component=='temperature' else 'percentage_points',
                         raw_difference=delta,absolute_difference=abs(delta) if delta is not None else None,score=None,
                         target_version=d['target_version'] if d else None,target_hash=binding_hash if b else None))
    return rows
