"""Synthetic mathematical specification only; NOT a runtime scalar evaluator."""
import copy
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
import unittest
from unittest.mock import patch
from geas35.objectives.normalization_prereg import robust_scale, hash_object, load_train, calibrate

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'experiments/rl_policy_training/artifacts/future_f_quality_scalarization_v1/quality_scalarization.json'

def fixture_rows(errors=(0,0,0,0)):
    return [dict(component=k,channel=c,status='VALID',unit=u,absolute_difference=e,target_hash='TEST_FIXTURE_ONLY:'+k)
            for (k,c,u),e in zip([('temperature','T1','°C'),('temperature','T2','°C'),('RH','RH1','%RH'),('RH','RH2','%RH')],errors)]

def reference_spec(rows,scales):
    # Mathematical oracle confined to this test module; never wired to P0.
    expected={('temperature','T1'):'°C',('temperature','T2'):'°C',('RH','RH1'):'%RH',('RH','RH2'):'%RH'}
    if len(rows)!=4 or {(r['component'],r['channel']) for r in rows}!=set(expected):return None
    for r in rows:
        e=r['absolute_difference']
        if r['status']!='VALID' or r['unit']!=expected[(r['component'],r['channel'])] or type(e) not in (float,int) or not math.isfinite(e) or e<0:return None
    if any(len({r['target_hash'] for r in rows if r['component']==k})!=1 for k in scales):return None
    if any(not math.isfinite(s) or s<=0 for s in scales.values()):return None
    with localcontext() as ctx:
        ctx.prec=60
        q=[]
        for k in ('temperature','RH'):
            es=[Decimal(str(r['absolute_difference']))/Decimal(str(scales[k])) for r in rows if r['component']==k]
            q.append(Decimal(1)/(Decimal(1)+(es[0]+es[1])/2))
        return float((q[0]+q[1])/2)

class QualityPreregTests(unittest.TestCase):
    def setUp(self):self.scales={'temperature':2.5,'RH':7.5} # TEST_FIXTURE_ONLY, not frozen real scales
    def test_zero_best(self):self.assertEqual(reference_spec(fixture_rows(),self.scales),1)
    def test_temperature_monotonic(self):
        q=[reference_spec(fixture_rows((x,x,0,0)),self.scales) for x in (0,1,3,20)]
        self.assertEqual(q,sorted(q,reverse=True));self.assertEqual(len(set(q)),4)
    def test_rh_monotonic(self):
        q=[reference_spec(fixture_rows((0,0,x,x)),self.scales) for x in (0,1,3,20)];self.assertEqual(q,sorted(q,reverse=True))
    def test_channel_permutations(self):
        a=reference_spec(fixture_rows((1,8,2,9)),self.scales)
        for e in ((8,1,2,9),(1,8,9,2),(8,1,9,2)):self.assertEqual(a,reference_spec(fixture_rows(e),self.scales))
    def test_scale_reproducibility(self):self.assertEqual(robust_scale([1,2,3,4,5,6,7,8]),robust_scale([8,7,6,5,4,3,2,1]))
    def test_hash_reproducibility(self):self.assertEqual(hash_object({'a':1,'b':2}),hash_object({'b':2,'a':1}))
    def test_leakage_rejected_before_read(self):
        x=json.loads(ART.read_text())['protocol']
        for name in ('validation_registry.parquet','test_registry.parquet'):
            bad=copy.deepcopy(x);bad['paths'][name]=bad['paths'].pop('train_registry.parquet')
            with patch('geas35.objectives.normalization_prereg.pd.read_parquet',side_effect=AssertionError('forbidden read')):
                with self.assertRaisesRegex(ValueError,'TRAIN_ONLY'):load_train(bad)
    def test_missing_T(self):self.assertIsNone(reference_spec(fixture_rows()[1:],self.scales))
    def test_missing_RH(self):self.assertIsNone(reference_spec(fixture_rows()[:-1],self.scales))
    def test_no_renormalization(self):
        rows=fixture_rows();rows[-1].update(status='MISSING_SENSOR_VALUE',absolute_difference=None);self.assertIsNone(reference_spec(rows,self.scales))
    def test_target_mismatch(self):
        rows=fixture_rows();rows[0]['status']='TARGET_CONTEXT_MISMATCH';self.assertIsNone(reference_spec(rows,self.scales))
    def test_unit_mismatch(self):
        rows=fixture_rows();rows[0]['unit']='kelvin';self.assertIsNone(reference_spec(rows,self.scales))
    def test_deterministic(self):
        rows=fixture_rows((2,3,4,5));self.assertEqual(reference_spec(rows,self.scales),reference_spec(list(reversed(rows)),self.scales))
    def test_extreme_and_bounds(self):
        for x in (0,1,1e100,1e308):
            q=reference_spec(fixture_rows((x,x,x,x)),self.scales);self.assertTrue(math.isfinite(q) and 0<=q<=1)
    def test_no_vpd_or_action_terms(self):
        rows=fixture_rows();rows[0]['heat']=1;rows[0]['vent']=100;rows[0]['switching']=1;rows[0]['VPD']=5
        self.assertEqual(reference_spec(rows,self.scales),1)
        rows[2]['component']='VPD';self.assertIsNone(reference_spec(rows,self.scales))
    def test_unit_scaling_invariance(self):
        a=reference_spec(fixture_rows((1,2,3,4)),self.scales)
        self.assertEqual(a,reference_spec(fixture_rows((10,20,.03,.04)),{'temperature':25,'RH':.075}))
    def test_degenerate_no_fallback(self):
        for values in ([1]*8,[1+1e-15*i for i in range(8)],[float('nan')]*8):
            with self.assertRaises(ValueError):robust_scale(values)
    def test_frozen_train_rebuild(self):
        x=json.loads(ART.read_text());self.assertEqual(hash_object(x['protocol']),x['protocol_sha256'])
        c=calibrate(x['protocol']);self.assertEqual(c,x['calibration']);self.assertEqual(hash_object(c['scale_payload']),c['scale_sha256'])
    def test_runtime_evaluator_has_no_scalar(self):
        from geas35.objectives import binding
        self.assertNotIn('reference_spec',Path(binding.__file__).read_text());self.assertIn('score=None',Path(binding.__file__).read_text())

if __name__=='__main__':unittest.main()
