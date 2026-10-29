"""Conditional aggregation of already stored predictions only."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/transition_model_selection/artifacts/stale_snapshot_shadow_b_v1';P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4/strawberry';M=ROOT/'experiments/transition_model_selection/artifacts/command_carry_b_v1/strawberry'
T=['obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm'];inputs={}
def track(p):inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
b=pd.read_parquet(track(P/'validation_physical.parquet'));wm=pd.read_parquet(D/'window_masks.parquet');wm=wm.loc[wm.crop.eq('strawberry')&wm.split.eq('validation')];tm=pd.read_parquet(D/'transition_masks.parquet');tm=tm.loc[tm.crop.eq('strawberry')&tm.split.eq('validation')]
arrays={};one={};meta={}
for model in ('lightgbm','persistence'):
 arrays[model]=np.load(track(M/model/'rollout/predictions.npz'));one[model]=pd.read_parquet(track(M/model/'fit/validation_predictions.parquet'))[T].to_numpy(float);meta[model]=json.loads(track(M/model/'fit/completed.json').read_text())
x=arrays['lightgbm'];y=arrays['persistence'];np.testing.assert_array_equal(x['start_row_id'],y['start_row_id']);np.testing.assert_allclose(x['truth'],y['truth'],equal_nan=True)
scales=np.array([meta['lightgbm']['scales'][t] for t in T]);truth=b[[t.replace('obs_','target_next_',1) for t in T]].to_numpy(float);np.testing.assert_allclose(one['persistence'],b[T].to_numpy(float))
rows=[];scores=[]
conditions=[('unmasked',0,'unmasked')]+[(s,c,mode) for s in ('A','B') for c in (30,60,120,180) for mode in ('causal_forward','confirmed_run_retrospective')]
for sig,cut,mode in conditions:
 if mode=='unmasked':hit=np.zeros(len(b),bool)
 else:
  mt=tm.loc[tm.signature.eq(sig)&tm.cutoff_minutes.eq(cut)&tm['mode'].eq(mode)].set_index('row_id');hit=mt.loc[b.row_id,'masked'].to_numpy(bool)
 for model in arrays:
  e=one[model][~hit]-truth[~hit];rm=np.sqrt(np.mean(e**2,axis=0));components=[float((rm/scales).mean())];co=[float(rm[2]/scales[2])];dr=[];dc=[]
  base={'signature':sig,'cutoff_minutes':cut,'mode':mode,'model':model}
  rows.append({**base,'horizon_minutes':5,'kept':int((~hit).sum()),'masked':int(hit.sum()),'total':len(hit),'CO2_RMSE':rm[2],'all_target_nRMSE':components[0],'masked_CO2_SSE_fraction':float(((one[model]-truth)[hit,2]**2).sum()/((one[model]-truth)[:,2]**2).sum())})
  for h in (3,6,12):
   ix=np.flatnonzero(x['max_horizon']>=h)
   if mode=='unmasked':wh=np.zeros(len(ix),bool)
   else:
    mm=wm.loc[wm.signature.eq(sig)&wm.cutoff_minutes.eq(cut)&wm['mode'].eq(mode)&wm.horizon.eq(h)].set_index('start_row_id');wh=mm.loc[x['start_row_id'][ix],'masked'].to_numpy(bool)
   allerr=arrays[model]['predictions'][ix,:h]-x['truth'][ix,:h];err=allerr[~wh];rm=np.sqrt(np.mean(err**2,axis=(0,1)));tt=np.arange(h)-np.arange(h).mean();drift=(np.abs(err/scales)*tt[None,:,None]).sum(axis=1)/(tt*tt).sum();dr.append(float(drift.mean()));dc.append(float(drift[:,2].mean()));components.append(float((rm/scales).mean()));co.append(float(rm[2]/scales[2]))
   rows.append({**base,'horizon_minutes':h*5,'kept':int((~wh).sum()),'masked':int(wh.sum()),'total':len(wh),'CO2_RMSE':rm[2],'all_target_nRMSE':components[-1],'CO2_endpoint_RMSE':float(np.sqrt(np.mean(err[:,-1,2]**2))),'masked_CO2_SSE_fraction':float((allerr[wh,:,2]**2).sum()/(allerr[:,:,2]**2).sum())})
  score=float(np.dot([.1,.1,.25,.45],components)+.1*np.mean(dr));coscore=float(np.dot([.1,.1,.25,.45],co)+.1*np.mean(dc));scores.append({**base,'all_target_weighted_score':score,'CO2_only_weighted_analog_not_official':coscore})
  if mode=='unmasked':
   original=json.loads(track(M/model/'rollout/completed.json').read_text());np.testing.assert_allclose(score,original['score'],rtol=1e-6,atol=1e-8)
   for r in rows[-3:]:np.testing.assert_allclose(r['CO2_RMSE'],original['rollout'][str(r['horizon_minutes'])+'min']['targets'][T[2]]['rmse'],rtol=1e-6,atol=1e-8)
r=pd.DataFrame(rows);keys=['signature','cutoff_minutes','mode','horizon_minutes'];r=r.merge(r.loc[r.model.eq('persistence'),keys+['CO2_RMSE']].rename(columns={'CO2_RMSE':'Persistence_CO2_RMSE'}),on=keys,validate='many_to_one');r['CO2_skill']=1-r.CO2_RMSE/r.Persistence_CO2_RMSE;r.to_csv(D/'conditional_metrics.csv',index=False);pd.DataFrame(scores).to_csv(D/'conditional_scores.csv',index=False)
for p,h in inputs.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
(D/'metric_input_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n');(D/'metric_verification.json').write_text(json.dumps({'stored_predictions_only':True,'unmasked_score_and_RMSE_exact_tolerance':True,'same_window_ID_alignment':True,'persistence_one_step_equals_current_recorded_state':True,'normalization_denominator_frozen':True,'no_threshold_selection':True},indent=2)+'\n')
print(r.loc[r.model.eq('lightgbm')&r.horizon_minutes.eq(60),['signature','cutoff_minutes','mode','kept','CO2_RMSE','Persistence_CO2_RMSE','CO2_skill']].to_string(index=False))
