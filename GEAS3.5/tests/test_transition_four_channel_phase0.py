"""Synthetic Phase0 regressions; never accesses Test/P0 or fits a model."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from geas35.models.transition.four_channel_phase0 import *

R=Path(__file__).resolve().parents[1]
P=json.loads((R/'experiments/rl_policy_training/artifacts/transition_4ch_migration_prereg_v1/migration_preregistration.json').read_text())

class FourChannelTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.p=Path(self.tmp.name)
        self.schema={**P['schema'],'model_namespace':MODEL_NAMESPACE}
        self.obs=self.schema['observation_columns']; self.actions=self.schema['action_columns']
        self.row=pd.Series({**dict.fromkeys(self.obs,0.),'crop':'cucumber','series_id':'s','segment_id':'seg','episode_id':'e','reg_date':pd.Timestamp('2026-01-01 23:55'),'source_confidence':'synthetic'})
        self.row[TARGETS]=[20.,27.,80.,60.]
        self.row['obs_outdoor_temp_c']=10.;self.row['obs_current_target_temp_c']=24.
        self.row['obs_derived_heating_degree_minutes']=100.
        self.row['obs_derived_cooling_degree_minutes']=20.
        self.row['obs_derived_ramp_limit_pct']=15.
        # Keep all observations ordered before metadata.
        self.row=self.row[self.obs+[c for c in self.row.index if c not in self.obs]]
        train=self.row.to_frame().T; train['split']='train'
        self.payload=fit_scaler(train,self.obs)
        self.sp=self.p/'schema.json';self.cp=self.p/'scaler.json'
        self.sp.write_bytes(canonical(self.schema));self.payload['schema_sha256']=sha(self.sp)
        self.cp.write_bytes(canonical(self.payload))
        self.action=dict(zip(self.actions,[.4,.2,.1,0.,0.,1.]))
        self.rt=self.make(); self.rt.reset(self.row)
    def tearDown(self):self.tmp.cleanup()
    def make(self,history=None):
        return FourChannelRuntime(self.sp,self.cp,schema_hash=sha(self.sp),scaler_hash=sha(self.cp),prehistory=history)
    def exog(self,row=None):
        e=self.rt.exogenous(self.rt.current if row is None else row)
        e['reg_date']=pd.Timestamp(self.rt.current.reg_date)+pd.Timedelta(minutes=5)
        return e
    def test_schema_dimensions_order_and_co2(self):
        x=self.rt.model_input(self.action)
        self.assertEqual(len(x),62)
        self.assertEqual(list(x.index),self.obs+self.actions)
        self.assertEqual(self.schema['target_order'],TARGETS)
        self.assertFalse(any('co2' in c for c in x.index))
    def test_own_channel_residual_and_alias_not_average(self):
        p=self.rt.next_from_residual([1,2,3,4])
        self.assertEqual(list(p.values()),[21,29,83,64])
        n=self.rt.advance(prediction=p,action=self.action,exogenous=self.exog())
        self.assertEqual(n.obs_indoor_temp_c,21);self.assertEqual(n.obs_indoor_humidity_pct,83)
        self.assertEqual(n['obs_indoor_temp2_c'],29)
        self.assertIn('not_raw_observed',n.representative_alias_provenance)
    def test_invalid_prediction_no_sensor_fallback(self):
        for bad in [np.nan,np.inf,81,-61]:
            p=dict(zip(TARGETS,[bad,20,70,70]))
            with self.assertRaises(ValueError):self.rt.advance(prediction=p,action=self.action,exogenous=self.exog())
        for bad in [-1,101,np.nan]:
            p=dict(zip(TARGETS,[20,21,bad,60]))
            with self.assertRaises(ValueError):self.rt.advance(prediction=p,action=self.action,exogenous=self.exog())
    def test_future_indoor_and_co2_rejected(self):
        for key in ['in_temp','next_obs_indoor_temp1_c','obs_indoor_co2_ppm',TARGETS[0]]:
            e=self.exog();e[key]=999
            with self.assertRaises(ValueError):self.rt.advance(prediction=self.rt.next_from_residual([0]*4),action=self.action,exogenous=e)
    def test_action_invariant_and_carry(self):
        before=self.rt.current.copy(); a=dict(self.action);a['vent_pct']=.9
        x1=self.rt.model_input(self.action);x2=self.rt.model_input(a)
        np.testing.assert_array_equal(x1[self.obs],x2[self.obs]);pd.testing.assert_series_equal(before,self.rt.current)
        n=self.rt.advance(prediction=self.rt.next_from_residual([0]*4),action=a,exogenous=self.exog())
        for key in self.actions:self.assertEqual(n['obs_prev_'+key],a[key])
        self.assertAlmostEqual(n.obs_derived_vent_loss_proxy,9.)
    def test_boundaries(self):
        for key,val in [('reg_date',pd.Timestamp('2026-01-02 00:05')),('crop','melon'),('series_id','other'),('segment_id','other'),('episode_id','other'),('source_confidence','other'),('physical_discontinuity',True)]:
            e=self.exog();e[key]=val
            with self.assertRaises(ValueError):self.rt.advance(prediction=self.rt.next_from_residual([0]*4),action=self.action,exogenous=e)
    def test_train_only_scaler(self):
        for split in ['validation','test','p0']:
            d=self.row.to_frame().T;d['split']=split
            with self.assertRaises(ValueError):fit_scaler(d,self.obs)
        self.assertNotIn('obs_is_daytime',self.payload['columns'])
        self.assertEqual(self.payload['columns'][:4],TARGETS)
    def test_legacy_schema_and_hash_rejected(self):
        with self.assertRaises(ValueError):FourChannelRuntime(self.sp,self.cp,schema_hash='wrong',scaler_hash=sha(self.cp))
        for field,value in [('target_order',['obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm']),('state_dimension',55),('units',['x']*4),('model_namespace','legacy')]:
            s=dict(self.schema);s[field]=value;self.sp.write_bytes(canonical(s))
            with self.assertRaises(ValueError):self.make()
        self.sp.write_bytes(canonical(self.schema))
    def test_reload(self):
        rt=self.make();rt.reset(self.row)
        np.testing.assert_array_equal(rt.model_input(self.action),self.rt.model_input(self.action))
    def test_long_history_midnight_and_horizons(self):
        for step in range(301):
            p=self.rt.next_from_residual([0,0,0,0])
            n=self.rt.advance(prediction=p,action=self.action,exogenous=self.exog())
            self.assertAlmostEqual(n.obs_indoor_temp_c,20)
            if step==0:self.assertAlmostEqual(n.obs_derived_heating_degree_minutes,20.)
            if step+1 in [3,6,12,289,301]:
                self.assertTrue(np.isfinite(n[self.obs].to_numpy(float)).all())
                self.assertEqual(len(self.rt.model_input(self.action)),62)
        self.assertEqual(len(self.rt.updater._history),288)
        self.assertEqual(len(self.rt.updater._sensor_history),12)
    def test_sensor_history_cross_episode_not_gap(self):
        rows=[]
        for i in range(11,0,-1):
            r=self.row.copy();r['reg_date']-=pd.Timedelta(minutes=5*i);r['episode_id']='earlier'
            r['obs_indoor_temp_c']=20;r['obs_indoor_humidity_pct']=95;r=r.drop('source_confidence');rows.append(r)
        rt=self.make(pd.DataFrame(rows));n=rt.reset(self.row)
        self.assertEqual(n.obs_derived_rh90_minutes_1h,55.)
        rows[-1]['reg_date']-=pd.Timedelta(minutes=1)
        rt=self.make(pd.DataFrame(rows));n=rt.reset(self.row)
        self.assertEqual(n.obs_derived_rh90_minutes_1h,0.)
    def test_future_prehistory_never_used(self):
        future=self.row.copy();future['reg_date']+=pd.Timedelta(minutes=5)
        future['obs_indoor_temp_c']=-40.;future['obs_indoor_humidity_pct']=100.
        rt=self.make(pd.DataFrame([future]));n=rt.reset(self.row)
        np.testing.assert_allclose(n[self.obs].to_numpy(float),self.rt.current[self.obs].to_numpy(float))
    def test_raw_admission_flags_are_not_repaired(self):
        raw={'is_resampled_row':0}
        for s,v in zip(RAW,[20.,21.,70.,80.]):
            raw[s+'_raw_value']=v
            for flag in FLAGS:raw[s+'_'+flag]=0
        self.assertTrue(valid_raw(pd.DataFrame([raw]))[0])
        for s in RAW:
            for flag in FLAGS:
                bad=dict(raw);bad[s+'_'+flag]=1
                self.assertFalse(valid_raw(pd.DataFrame([bad]))[0])
        bad=dict(raw);bad['is_resampled_row']=1
        self.assertFalse(valid_raw(pd.DataFrame([bad]))[0])
    def test_prediction_order_and_missing_action_fail(self):
        with self.assertRaises(ValueError):self.rt.model_input({'vent_pct':.1})
        p=dict(reversed(list(self.rt.next_from_residual([0]*4).items())))
        with self.assertRaises(ValueError):self.rt.advance(prediction=p,action=self.action,exogenous=self.exog())

if __name__=='__main__':unittest.main()
