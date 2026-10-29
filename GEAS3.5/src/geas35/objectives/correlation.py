"""Forward correlation prototype: synthetic journals only, no I/O or live adapter.

Mock supervisor attestations model future authoritative producer hooks. They do
not establish provenance for real P0 records. Never call a controller here.
"""
from dataclasses import dataclass
from uuid import uuid4
import json
from .binding import canonical, digest, timestamp
from .live import receipt

VERSION = 'future_f_action_outcome_correlation.v1'
FIELDS = {'channel', 'operation', 'ratio', 'duration', 'n_idx'}


def uid():
    return str(uuid4())


@dataclass(frozen=True)
class Plan:
    payload: str

    @property
    def data(self):
        return json.loads(self.payload)

    @property
    def hash(self):
        return digest(self.data)

    @classmethod
    def prepare(cls, path, profile_hash, input_receipt, actuators, omitted,
                channel_universe, provenance, outcome_sequence, max_gap_seconds):
        if not path.synthetic:
            raise ValueError('MOCK_ONLY')
        if not actuators or any(set(a) != FIELDS for a in actuators):
            raise ValueError('ACTION_SCHEMA')
        channels = [a['channel'] for a in actuators]
        if len(set(channels)) != len(channels) or set(channels) & set(omitted) or set(channels) | set(omitted) != set(channel_universe):
            raise ValueError('OMISSION_SCHEMA')
        if len(set(omitted)) != len(omitted) or len(set(channel_universe)) != len(channel_universe):
            raise ValueError('OMISSION_SCHEMA')
        if set(provenance) != {'controller_name','controller_version','runtime_crop','actual_crop','controller_code_hash'} or any(not v for v in provenance.values()) or provenance['actual_crop'] != 'cucumber':
            raise ValueError('PROVENANCE_SCHEMA')
        if type(outcome_sequence) is not int or outcome_sequence < 1 or type(max_gap_seconds) is not int or max_gap_seconds <= 0:
            raise ValueError('OUTCOME_PLAN_SCHEMA')
        row, _, _ = receipt(input_receipt, 'observations', path.session_id)
        cycle = uid()
        ctx = path.prepare(profile_hash, cycle, row, 'MOCK_ONLY pre-send producer boundary')
        d = dict(version=VERSION, mode='MOCK_ONLY', evaluation_cycle_id=cycle,
                 input_observation_id=row['source_row_id'], input_observation_hash=row['source_hash'],
                 input_receipt_hash=row['entry_hash'], profile_hash=profile_hash,
                 context_hash=ctx['context_hash'], prepared_at=path.now(),
                 action_hash=digest({'actuators':actuators,'omitted':omitted,'universe':channel_universe}),
                 actuators=actuators, omitted=omitted, channel_universe=channel_universe,
                 expected_channel_set=channels, command_ids=[uid() for _ in actuators],
                 outcome_ticket=uid(), outcome_sequence=outcome_sequence,
                 max_gap_seconds=max_gap_seconds, provenance=provenance)
        obj = cls(canonical(d))
        path.append('correlation_plan:'+cycle, dict(plan=d, plan_hash=obj.hash))
        path.append('correlation_sealed:'+cycle, dict(sealed_at=path.now()))
        return obj

    @classmethod
    def reload(cls, path, cycle):
        if not path.synthetic:
            raise ValueError('MOCK_ONLY')
        rec = path.get('correlation_plan:'+cycle)
        if not rec:
            raise ValueError('PLAN_MISSING')
        x = json.loads(rec[2])
        obj = cls(canonical(x['plan']))
        if x['plan_hash'] != obj.hash:
            raise ValueError('PLAN_HASH_MISMATCH')
        return obj


def correlate(path, plan, attempts, completion, outcome, coverage):
    """Validate exact IDs and supervisor interval assertions, never search rows.

    Each attempt includes its exact synthetic sent receipt. outcome contains an
    explicitly fulfilled pre-send ticket and acquisition interval. coverage is
    a synthetic authoritative ALL-send stream attestation, not a DB time query.
    """
    if not path.synthetic:
        raise ValueError('MOCK_ONLY')
    d = plan.data
    result = dict(evaluation_cycle_id=d.get('evaluation_cycle_id'), correlation_status='SYSTEM_INVALID',
                  override_status='OVERRIDE_UNKNOWN', reward=None, sent_history_row_ids=[],
                  input_observation_id=d.get('input_observation_id'), input_observation_hash=d.get('input_observation_hash'),
                  action_hash=d.get('action_hash'), expected_channel_set=d.get('expected_channel_set'),
                  action_started_at=None, action_completed_at=None,
                  outcome_observation_id=None, outcome_observation_hash=None)
    def finish(status, reason):
        result.update(correlation_status=status, reason=reason)
        return result
    try:
        cycle = d['evaluation_cycle_id']
        if Plan.reload(path, cycle).payload != plan.payload:
            raise ValueError('PLAN_CHANGED')
        ctx = path.context(cycle)
        if ctx['context_hash'] != d['context_hash']:
            raise ValueError('BINDING_CHANGED')
        sealed = json.loads(path.get('correlation_sealed:'+cycle)[2])['sealed_at']
        commands = dict(zip(d['command_ids'], d['actuators']))
        seen_attempts, seen_pks, completed_commands = set(), set(), set()
        starts, ends = [], []
        for a in attempts:
            if a['evaluation_cycle_id'] != cycle or a['command_id'] not in commands or a['attempt_id'] in seen_attempts:
                raise ValueError('ATTEMPT_LINEAGE')
            if a['attempt_ordinal'] != sum(x['command_id']==a['command_id'] for x in attempts[:len(seen_attempts)]) + 1:
                raise ValueError('ATTEMPT_ORDER')
            seen_attempts.add(a['attempt_id'])
            payload = commands[a['command_id']]
            if a['payload_hash'] != digest(payload) or a['channel'] != payload['channel']:
                raise ValueError('PAYLOAD_MISMATCH')
            start = timestamp(a['send_started_at']); end = timestamp(a['persistence_confirmed_at'])
            if not timestamp(sealed) < start <= end <= timestamp(path.now()) or (ends and start < ends[-1]):
                raise ValueError('SEND_ORDER_OR_PREBINDING')
            starts.append(start); ends.append(end)
            if a['committed'] is not True or type(a['response_status']) is not int or not 200 <= a['response_status'] < 300 or a['sent_receipt'] is None:
                return finish('ACTION_INCOMPLETE','SEND_OR_COMMIT_MISSING')
            row, logged, received = receipt(a['sent_receipt'],'sent_commands',path.session_id)
            if not start <= logged <= end:
                raise ValueError('SENT_LOG_CLOCK')
            pk = a['committed_history_pk']
            if type(pk) is not int or pk <= 0 or pk != row['source_row_id'] or pk in seen_pks or received < end:
                raise ValueError('COMMITTED_PK_IDENTITY')
            if any(str(row['raw'].get(k)) != str(payload[k]) for k in FIELDS):
                raise ValueError('SENT_PAYLOAD_MISMATCH')
            claim=path.get('correlation_pk:'+str(pk))
            if claim and json.loads(claim[2])['cycle_id'] != cycle:
                raise ValueError('PK_ALREADY_CLAIMED_BY_OTHER_CYCLE')
            seen_pks.add(pk); completed_commands.add(a['command_id'])
            result['sent_history_row_ids'].append(pk)
        if not attempts or completed_commands != set(commands) or not completion:
            return finish('ACTION_INCOMPLETE','EXPECTED_COMMANDS_NOT_COMPLETE')
        if completion['evaluation_cycle_id'] != cycle or completion['attempt_ids'] != [a['attempt_id'] for a in attempts] or completion['all_attempts_closed'] is not True:
            return finish('ACTION_INCOMPLETE','FINAL_SEND_MANIFEST_UNSEALED')
        if [a['command_id'] for a in attempts if a['attempt_ordinal']==1] != d['command_ids']:
            raise ValueError('CHANNEL_ORDER')
        completed = timestamp(completion['completed_at'])
        if not ends[-1] <= completed <= timestamp(path.now()):
            raise ValueError('COMPLETION_CLOCK')
        result.update(action_started_at=starts[0].isoformat(),action_completed_at=completed.isoformat())
        if not outcome:
            return finish('OUTCOME_UNRESOLVED','NO_EXPLICIT_OUTCOME')
        if outcome['evaluation_cycle_id'] != cycle or outcome['ticket'] != d['outcome_ticket']:
            return finish('OUTCOME_UNRESOLVED','OUTCOME_TICKET_MISMATCH')
        row, _, received = receipt(outcome['receipt'],'observations',path.session_id)
        result.update(outcome_observation_id=row['source_row_id'],outcome_observation_hash=row['source_hash'])
        if row['source_row_id'] == d['input_observation_id']:
            raise ValueError('REUSED_INPUT')
        acq_start=timestamp(outcome['acquired_start']);acq_end=timestamp(outcome['acquired_end'])
        if not completed < acq_start <= acq_end <= received <= timestamp(path.now()):
            return finish('OUTCOME_UNRESOLVED','ACQUISITION_NOT_AFTER_COMPLETE_ACTION')
        if outcome['sequence'] != d['outcome_sequence'] or outcome['continuous_sensor_stream'] is not True or (acq_end-completed).total_seconds()>d['max_gap_seconds']:
            return finish('SENSOR_GAP','PLANNED_SENSOR_SEQUENCE_OR_GAP')
        if not coverage or coverage['complete'] is not True:
            return finish('OVERRIDE_UNKNOWN','ALL_SEND_STREAM_INCOMPLETE')
        if coverage['evaluation_cycle_id'] != cycle or not timestamp(coverage['from']) <= starts[0] or not acq_end <= timestamp(coverage['through']) <= timestamp(path.now()):
            return finish('OVERRIDE_UNKNOWN','EXPOSURE_NOT_FULLY_COVERED')
        if not coverage['authority']:
            return finish('OVERRIDE_UNKNOWN','NO_STREAM_AUTHORITY')
        owned=set()
        unknown=False;override=False
        for e in coverage['events']:
            when=timestamp(e['at'])
            if not starts[0] <= when <= acq_end:
                raise ValueError('EVENT_OUTSIDE_ATTESTED_EXPOSURE')
            if e['origin']=='THIS_CYCLE':
                matches=[a for a in attempts if a['attempt_id']==e['attempt_id'] and a['committed_history_pk']==e['pk'] and a['evaluation_cycle_id']==e['cycle_id']]
                if len(matches)!=1 or e['pk'] in owned or timestamp(matches[0]['send_started_at']) != when:
                    raise ValueError('OWN_EVENT_IDENTITY')
                owned.add(e['pk'])
            elif e['origin']=='CONFIRMED_OTHER' and e.get('evidence'):
                override=True
            else:
                unknown=True
        if owned != seen_pks:
            return finish('OVERRIDE_UNKNOWN','OWN_SENDS_MISSING_FROM_STREAM')
        if override:
            result['override_status']='OVERRIDE_DETECTED'
            return finish('OVERRIDE_DETECTED','CONFIRMED_INTERVENING_COMMAND')
        if unknown:
            return finish('OVERRIDE_UNKNOWN','UNCORRELATED_INTERVENING_SEND')
        result['override_status']='NO_OVERRIDE_CONFIRMED'
        return finish('VALID','EXPLICIT_MOCK_CORRELATION')
    except (KeyError,TypeError,ValueError,IndexError) as exc:
        return finish('SYSTEM_INVALID',str(exc))


def evaluate_mock(path, plan, attempts, completion, outcome, coverage,
                  policy_context, environment_context):
    """Aggregate completeness gate before unchanged LivePath/QualityEvaluator.

    First exact sent receipt is a legacy anchor only AFTER all sends validate;
    it never stands in for aggregate completeness or defines outcome selection.
    """
    r=correlate(path,plan,attempts,completion,outcome,coverage)
    if r['correlation_status']!='VALID':
        return dict(correlation=r,quality=path.quality.unavailable([{'reason':r['correlation_status']}]),reward=None)
    d=plan.data;cycle=d['evaluation_cycle_id'];first=attempts[0]['sent_receipt']
    bundle=dict(plan_hash=plan.hash,attempts=attempts,completion=completion,outcome=outcome,coverage=coverage)
    key='correlation_evidence:'+cycle
    old=path.get(key)
    if old and json.loads(old[2]) != bundle:
        r.update(correlation_status='SYSTEM_INVALID',reason='CORRELATION_EVIDENCE_CHANGED')
        return dict(correlation=r,quality=path.quality.unavailable([{'reason':r['reason']}]),reward=None)
    if not old:
        claims=[('correlation_pk:'+str(pk),dict(cycle_id=cycle)) for pk in r['sent_history_row_ids'] if not path.get('correlation_pk:'+str(pk))]
        path.append_many([(key,bundle),*claims])
    if not path.get('action:'+cycle):
        bound=path.observe_action(cycle,first,dict(method='EXPLICIT_ID',evidence_source='MOCK_ONLY exact committed PK',action_id=cycle,event_hash=first['entry_hash'],ambiguous=False),policy_context)
        if bound['evaluation_status']!='ACTION_BOUND':
            return dict(correlation=r,quality=bound,reward=None)
    link=dict(method='EXPLICIT_ID',evidence_source='MOCK_ONLY fulfilled prospective outcome ticket',action_id=cycle,event_hash=outcome['receipt']['entry_hash'],ambiguous=False,intervening_override=False,profile_hash=d['profile_hash'])
    quality=path.evaluate(cycle,outcome['receipt'],link,policy_context,environment_context)
    return dict(correlation=r,quality=quality,reward=quality['reward'])
