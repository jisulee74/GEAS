"""Deterministic synthetic non-learning validation; no dataset/P0 reads or writes.
Prints aggregate implementation diagnostics, never agronomic reward results.
"""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
from geas35.objectives.binding import FixtureJournal,TargetContext
from geas35.objectives.quality import QualityEvaluator,S_T,S_RH
from test_future_f_target_binding import make,T1,T2,T3
from test_future_f_quality_prereg import reference_spec


def validate():
    art=ROOT/'experiments/rl_policy_training/artifacts/future_f_quality_scalarization_v1/quality_scalarization.json'
    frozen=json.loads(art.read_text())
    oracle_path=ROOT/'tests/test_future_f_quality_prereg.py'
    assert hashlib.sha256(oracle_path.read_bytes()).hexdigest()==frozen['code_sha256']['tests/test_future_f_quality_prereg.py']
    cases=[(t,t,h,-h) for t in (0.,1.,9.,100.,1e308) for h in (-40.,0.,5.,40.)]
    # Fixed grid is declared before evaluations; no outcome-based sample selection.
    cohort_hash=hashlib.sha256(json.dumps(cases,separators=(',',':')).encode()).hexdigest()
    q=QualityEvaluator.from_file(art);results=[];errors=[]
    with tempfile.TemporaryDirectory(prefix='quality_validation_fixture_') as p:
        j=FixtureJournal(p);bs={};ctx={}
        for c,u,v in [('temperature','°C',17.25),('RH','%RH',47.5)]:
            b=make(component=c,unit=u,reference_value=v,component_mask=['temperature','RH']);j.commit(b,T1);bs[c]=b.target_hash;ctx[c]=TargetContext.from_binding(b)
        j.action('synthetic-A',list(bs.values()),T2)
        outcome={'action_id':'synthetic-A','binding_hashes':bs,'observed_at':T3}
        for a,b,c,d in cases:
            obs={'T1':17.25+a,'T2':17.25+b,'RH1':47.5+c,'RH2':47.5+d}
            out=q.evaluate(j,bs,obs,outcome,ctx,ctx);expected=reference_spec(out['channels'],{'temperature':S_T,'RH':S_RH})
            assert abs(out['reward']-expected)<=1e-15+1e-12*abs(expected)
            json.dumps(out,allow_nan=False);results.append(out);errors.append(abs(out['reward']-expected))
        invalid=[]
        for channel in ('T1','T2','RH1','RH2'):
            obs={'T1':17.25,'T2':17.25,'RH1':47.5,'RH2':47.5};del obs[channel]
            invalid.append(q.evaluate(j,bs,obs,outcome,ctx,ctx))
        invalid.append(q.evaluate(j,{}, {},outcome,{},{}))
        invalid.append(q.evaluate(j,bs,{'T1':17.25,'T2':17.25,'RH1':47.5,'RH2':47.5},dict(outcome,ambiguous=True),ctx,ctx))
        assert all(x['reward'] is None and x['evaluation_status']=='UNSCORABLE' for x in invalid)
        j.close()
    def stats(key):
        v=[r[key] for r in results];return {'min':min(v),'median':statistics.median(v),'max':max(v)}
    return {'cohort':'TEST_FIXTURE_ONLY deterministic 20-case grid + 6 invalid cases','cohort_hash':cohort_hash,
            'count':len(results)+len(invalid),'valid_count':len(results),'unscorable_count':len(invalid),
            'finite_rate_among_valid':1.0,'quality_reward':stats('quality_reward'),
            'temperature_quality':stats('temperature_quality'),'rh_quality':stats('rh_quality'),
            'max_oracle_absolute_error':max(errors),'parity_tolerance':{'rtol':1e-12,'atol':1e-15},
            'oracle_sha256':hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
            'forbidden_dataset_reads':False,'historical_train_observations_used':False,'P0_observations_used':False}

if __name__=='__main__':print(json.dumps(validate(),indent=2,allow_nan=False))
