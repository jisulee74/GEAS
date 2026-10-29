"""Algebraic carry/action alignment checks; no environment change or rollout fitting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,gzip,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.recorded_light_env_v1 import load_recorded_light_env
from geas35.features.derived import dewpoint_c,_condensation_margin_forecast_10m
from geas35.experiments.rl.step19_5_diagnostics import ART,OUT,read_json,write_json
DEST=ROOT/ART/'step19_5_residual_ood_v1'
results=[]
for crop in ('melon','cucumber'):
 details=[json.loads(l) for l in gzip.open(DEST/f'{crop}_feature_trace.jsonl.gz','rt')]
 for split in ('validation','train'):
  env=load_recorded_light_env(project_root=ROOT,crop=crop,split=split,protocol_manifest_path=ROOT/ART/'step15_v1_4/rl_protocol_manifest.json');p=env.scaler.inverse_transform_frame(env.frame.copy())
  margin=p.obs_indoor_temp_c-dewpoint_c(p.obs_indoor_temp_c,p.obs_indoor_humidity_pct)
  boundary=[c for c in ('crop','series_id','segment_id','episode_id') if c in p]
  forecast=_condensation_margin_forecast_10m(p.reg_date,margin,boundary_frame=p[boundary])
  errors={'degree_carry_restored':[],'vent_teacher_previous_action':[],'vent_recorded_current_action':[],'forecast_with_available_recorded_prehistory':[],'aliases_vs_reset':[]};n=0
  for d in details:
   if d['split']!=split:continue
   index=d['episode_index'];step=d['step'];ep=env.episodes[index];pos=ep['start']+step;r=p.iloc[pos];init=p.iloc[ep['start']];v=d['features'];n+=1
   for c in ('obs_current_vpd_kpa','obs_current_dewpoint_c','obs_current_condensation_margin_c'):errors['aliases_vs_reset'].append(abs(v[c]['rollout']-v[c]['reset']))
   if not step:continue
   same_day=pd.Timestamp(r.reg_date).date()==pd.Timestamp(init.reg_date).date()
   for c in ('obs_derived_heating_degree_minutes','obs_derived_cooling_degree_minutes'):
    errors['degree_carry_restored'].append(abs(v[c]['teacher']+(float(init[c]) if same_day else 0.)-v[c]['recorded']))
   factor=max(float(r.obs_indoor_temp_c-r.obs_outdoor_temp_c),0.)*(1.-min(max(float(r.obs_outdoor_light),0.)/1000.,1.))
   errors['vent_teacher_previous_action'].append(abs(v['obs_derived_vent_loss_proxy']['teacher']-float(env.frame.iloc[pos-1].vent_pct)*factor))
   errors['vent_recorded_current_action'].append(abs(v['obs_derived_vent_loss_proxy']['recorded']-float(env.frame.iloc[pos].vent_pct)*factor))
   errors['forecast_with_available_recorded_prehistory'].append(abs(float(forecast.iloc[pos])-v['obs_derived_condensation_margin_pred_10m']['recorded']))
  result={'crop':crop,'split':split,'rows':n,'checks':{k:{'n':len(v),'max_abs_error':max(v,default=0.),'above_1e_6':sum(x>1e-6 for x in v)} for k,v in errors.items()}};results.append(result);print(result,flush=True)
write_json(DEST/'contract_probe_evidence.json',results)
