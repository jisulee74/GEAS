"""Frozen environmental quality over authoritative channel evaluations.

No binding creation, calibration, live activation or episode accounting here.
The current binding adapter remains TEST_FIXTURE_ONLY.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from .binding import evaluate, VERSION
from .normalization_prereg import hash_object

SCALE_HASH='6102b15a3887573cb9dbfd1be1ed6f8fa026200e8f59727b6c2f1b42c42b5dce'
SCALAR_HASH='38ef3b5f0c318b7f18256b2ead75d3407b711aa188b5449852c342337cad95c8'
SCALE_VERSION='future_f_H_train_paired_iqr.v1'
SCALAR_VERSION='future_f_quality_scalarization.v1'
S_T=9.100000000000001
S_RH=20.470000000000006
METRICS=('normalized_error_T1','normalized_error_T2','normalized_error_RH1','normalized_error_RH2',
         'fused_temperature_error','fused_rh_error','temperature_quality','rh_quality','quality_reward')


@dataclass(frozen=True)
class QualityEvaluator:
    """Construction fails closed unless the preregistration identity is exact."""
    contract_json: str

    def __post_init__(self):
        x=json.loads(self.contract_json)
        s=x['calibration']['scale_payload'];c=x['scalarization_contract']
        if hash_object(s)!=SCALE_HASH or x['calibration']['scale_sha256']!=SCALE_HASH:
            raise ValueError('FROZEN_SCALE_HASH_MISMATCH')
        if hash_object(c)!=SCALAR_HASH or x['scalarization_sha256']!=SCALAR_HASH:
            raise ValueError('FROZEN_SCALARIZATION_HASH_MISMATCH')
        if s['version']!=SCALE_VERSION or c['version']!=SCALAR_VERSION or s['scales']['temperature']['scale']!=S_T or s['scales']['RH']['scale']!=S_RH:
            raise ValueError('FROZEN_CONSTANT_MISMATCH')

    @classmethod
    def from_file(cls,path):
        return cls(Path(path).read_text())

    def unavailable(self, reasons, rows=(), bindings=None, action_id=None):
        """Preserve upstream rejection reasons; never turn rejected commits into 0."""
        reasons=list(reasons)
        if not reasons:raise ValueError('UNSCORABLE_REASON_REQUIRED')
        return self._envelope(rows, bindings or {}, action_id, reasons)

    @staticmethod
    def _envelope(rows, bindings, action_id, reasons):
        return dict(evaluation_status='UNSCORABLE' if reasons else 'VALID',reward=None,
                    **{m:None for m in METRICS},reasons=reasons,channels=list(rows),
                    scale_version=SCALE_VERSION,scale_hash=SCALE_HASH,scalarization_version=SCALAR_VERSION,
                    scalarization_hash=SCALAR_HASH,evaluator_version='frozen_quality.v1',channel_evaluator_version=VERSION,
                    target_version={k:b.data['target_version'] for k,b in bindings.items()},
                    target_hash={k:b.target_hash for k,b in bindings.items()},action_id=action_id,
                    live_activation=False)

    def evaluate(self,journal,binding_hashes,observations,outcome,policy_contexts,environment_contexts):
        """outcome carries explicit action_id and component->binding_hash mapping.
        Never infer linkage from nearest timestamps or replace absent hashes.
        """
        rows=[];bindings={};reasons=[]
        for component in ('temperature','RH'):
            h=binding_hashes.get(component)
            b=journal.lookup(h)
            if b:bindings[component]=b
            link=dict(outcome)
            link['binding_hash']=outcome.get('binding_hashes',{}).get(component)
            rows.extend(evaluate(journal,h,component,observations,link,
                                 policy_contexts.get(component),environment_contexts.get(component)))
        for row in rows:
            if row['status']!='VALID':
                reasons.append({'component':row['component'],'channel':row['channel'],'reason':row['reason'] or row['status']})
        if len(bindings)==2:
            ds=[b.data for b in bindings.values()]
            # Both core components must have been declared before the same action.
            for field in ('action_id','action_time','decision_time','objective_version','target_version'):
                if ds[0][field]!=ds[1][field]:reasons.append({'reason':'TARGET_CONTEXT_MISMATCH','field':field})
            if any(set(d['component_mask'])!={'temperature','RH'} for d in ds):
                reasons.append({'reason':'REQUIRED_COMPONENT_MASK_MISMATCH'})
        out=self._envelope(rows,bindings,outcome.get('action_id'),reasons)
        if reasons:return out
        expected={'T1':('temperature','°C'),'T2':('temperature','°C'),'RH1':('RH','%RH'),'RH2':('RH','%RH')}
        by={r['channel']:r for r in rows}
        if len(rows)!=4 or set(by)!=set(expected):return self.unavailable([{'reason':'CHANNEL_SCHEMA_MISMATCH'}],rows,bindings,outcome.get('action_id'))
        for ch,(comp,unit) in expected.items():
            r=by[ch];e=r['absolute_difference']
            if r['component']!=comp or r['unit']!=unit or type(e) not in (float,int) or not math.isfinite(e) or e<0:
                return self.unavailable([{'reason':'INVALID_CHANNEL_DISCREPANCY','channel':ch}],rows,bindings,outcome.get('action_id'))
        e={ch:by[ch]['absolute_difference']/(S_T if ch.startswith('T') else S_RH) for ch in expected}
        et=(e['T1']+e['T2'])/2;er=(e['RH1']+e['RH2'])/2
        qt=1/(1+et);qr=1/(1+er);reward=.5*qt+.5*qr
        out.update({f'normalized_error_{ch}':v for ch,v in e.items()})
        out.update(fused_temperature_error=et,fused_rh_error=er,temperature_quality=qt,rh_quality=qr,quality_reward=reward,reward=reward)
        return out
