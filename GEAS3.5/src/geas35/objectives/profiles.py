"""Frozen interview-derived candidates; activation always explicit and prospective.

No sensor/clock/DAT/reward selector. A clock is used only to reject backdating.
Candidate profiles remain DECLARED_NOT_ACTIVE; an activation is a separate event.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from .binding import canonical,digest,timestamp
from .live import Declaration,SEMANTICS

FROZEN_HASHES = {'cucumber72_interview_a': '430ac240bf8fc7892b8124842a1d9aacaa2b8f4c4ce0b01db5e2f7f342f1b27e', 'cucumber72_interview_b': 'e0ec81f3a446cdda40e933b085adaea63e6766b584b3d8bfb0fa062b9fec81de'}

@dataclass(frozen=True)
class CandidateProfile:
    payload: str
    def __post_init__(self):
        d=json.loads(self.payload)
        h=digest({k:v for k,v in d.items() if k!='profile_hash'})
        if d.get('profile_hash')!=h or FROZEN_HASHES.get(d.get('profile_id'))!=h:
            raise ValueError('FROZEN_PROFILE_INTEGRITY_FAILURE')
        object.__setattr__(self,'payload',canonical(d))
    @property
    def data(self):return json.loads(self.payload)


def load_profiles(path):
    rows=json.loads(Path(path).read_text())['profiles']
    result={}
    for row in rows:
        p=CandidateProfile(canonical(row));key=p.data['profile_id']
        if key in result:raise ValueError('DUPLICATE_PROFILE')
        result[key]=p
    if set(result)!=set(FROZEN_HASHES):raise ValueError('INCOMPLETE_PROFILE_CATALOG')
    return result


def schedule_profile(path,profile,*,activation_id,activation_context,effective_from,declared_by,declaration_basis,effective_until=None):
    """Persist a future operational declaration, not an action/start event.

Caller supplies explicit profile and context assertion, never inferred here.
The actual LivePath.prepare still rejects use before effective_from and seals
before a separately attributed sent action. Both components stay mandatory.
"""
    d=profile.data
    # Revalidate even if a caller hands in a look-alike object.
    CandidateProfile(canonical(d))
    required={'context_label','selection_method','asserted_by','evidence_source'}
    if set(activation_context)!=required or activation_context['context_label']!=d['context_label'] or activation_context['selection_method']!='EXPLICIT_PROJECT_DECLARATION':
        raise ValueError('EXPLICIT_MATCHING_CONTEXT_REQUIRED')
    for k in ('asserted_by','evidence_source'):
        if not isinstance(activation_context[k],str) or not activation_context[k].strip():raise ValueError('CONTEXT_PROVENANCE_REQUIRED')
    if not isinstance(activation_id,str) or not activation_id.strip() or not isinstance(declaration_basis,str) or not declaration_basis.strip():raise ValueError('ACTIVATION_PROVENANCE_REQUIRED')
    now=path.now()
    if timestamp(effective_from)<=timestamp(now):raise ValueError('PROSPECTIVE_EFFECTIVE_TIME_REQUIRED')
    basis=dict(candidate_profile_id=d['profile_id'],candidate_version=d['profile_version'],candidate_hash=d['profile_hash'],activation_context=activation_context,project_declaration_basis=declaration_basis,project_pairing=d['project_pairing'],approximate_provenance={'temperature':True,'rh':True},interview_evidence={c:d[c]['interview_evidence'] for c in ('temperature','rh')})
    window={}
    activation=None
    if effective_until is not None:
        if timestamp(effective_until)<=timestamp(effective_from):raise ValueError('INVALID_WINDOW')
        window={'effective_until':effective_until}
        activation=dict(activation_id=activation_id,profile_id=d['profile_id'],profile_version=d['profile_version'],profile_hash=d['profile_hash'],selection_method='EXPLICIT_PROJECT_DECLARATION',activation_context=d['context_label'],context_source='EXPLICIT_PROJECT_EVALUATION_CONTEXT',declared_by=declared_by,declaration_basis=declaration_basis,effective_from=effective_from,effective_until=effective_until,created_at=now)
        activation['activation_hash']=digest(activation)
        basis['bounded_activation']=activation
    components={c:dict(active=True,reference_value=d[c]['reference_value'],unit=d[c]['unit'],semantics=SEMANTICS,evidence_source=canonical(d[c]['interview_evidence'])) for c in ('temperature','rh')}
    declaration=Declaration.create(target_profile_id=d['profile_id'],target_version=d['profile_version']+':'+activation_id,crop='cucumber',crop_cycle=72,source=97,**components,effective_from=effective_from,available_at=now,declared_by=declared_by,declaration_basis=canonical(basis),claim=d['claim'],**window)
    h=path.declare(declaration)
    return dict(status='SCHEDULED_NOT_EFFECTIVE',candidate_profile_hash=d['profile_hash'],declaration_hash=h,effective_from=effective_from,activation_context=activation_context,activation_record=activation)
