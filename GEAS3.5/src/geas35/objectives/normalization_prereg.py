"""Train-only IQR calibration for preregistration; no runtime reward code."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ['series_id','segment_id','episode_id','reg_date']
SENSORS = ['in_temp','in_temp2','in_hum','in_hum2']
FLAGS = ['missing_flag','invalid_flag','imputed_flag','ffill_flag','rule_outlier_flag']

def hash_object(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def robust_scale(values):
    x=np.asarray(values,dtype=np.float64)
    if x.size<8 or not np.isfinite(x).all():raise ValueError('INVALID_CALIBRATION_VALUES')
    lo,hi=np.quantile(x,[.25,.75],method='linear');s=float(hi-lo)
    floor=64*np.finfo(np.float64).eps*float(np.max(np.abs(x)))
    if not np.isfinite(s) or s<=0 or s<=floor:raise ValueError('DEGENERATE_SCALE_NO_FALLBACK')
    return {'q25':float(lo),'q75':float(hi),'scale':s,'numerical_resolution_floor':floor}

def load_train(protocol):
    paths=protocol['paths']
    if set(paths)!={'train_registry.parquet','train_prepared_qc.parquet'}:raise ValueError('TRAIN_ONLY_INPUTS')
    for name,path in paths.items():
        p=Path(path)
        if p.name!=name or p.parent.name!='cucumber' or p.parent.parent.name!='past_only_qc_hlv15_b60_r6_v1':raise ValueError('UNAPPROVED_COHORT')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=protocol['input_sha256'][name]:raise ValueError('INPUT_HASH_MISMATCH')
    reg=pd.read_parquet(paths['train_registry.parquet'],columns=['row_id','crop',*KEYS])
    cols=[*KEYS,'split','is_resampled_row']+[s+'_raw_value' for s in SENSORS]+[s+'_'+f for s in SENSORS for f in FLAGS]
    qc=pd.read_parquet(paths['train_prepared_qc.parquet'],columns=cols)
    if not reg['crop'].eq('cucumber').all() or not qc['split'].eq('train').all():raise ValueError('SPLIT_CROP_LEAKAGE')
    if reg.duplicated(KEYS).any() or qc.duplicated(KEYS).any() or reg['row_id'].duplicated().any():raise ValueError('DUPLICATE_IDENTITY')
    joined=reg.merge(qc,on=KEYS,how='left',validate='one_to_one',indicator=True)
    if not joined['_merge'].eq('both').all():raise ValueError('UNMATCHED_REGISTRY')
    good=joined['is_resampled_row'].eq(0)
    for s in SENSORS:
        v=pd.to_numeric(joined[s+'_raw_value'],errors='coerce')
        joined[s+'_raw_value']=v
        good &= np.isfinite(v)
        good &= v.between(0,100) if 'hum' in s else v.ge(-273.15)
        for f in FLAGS:good &= joined[s+'_'+f].eq(0)
    selected=joined.loc[good].sort_values(KEYS).copy()
    if len(selected)<4:raise ValueError('INSUFFICIENT_CALIBRATION_ROWS')
    return reg,joined,selected

def calibrate(protocol):
    reg,joined,df=load_train(protocol)
    scales={k:robust_scale(df[[s+'_raw_value' for s in channels]].to_numpy().reshape(-1))
            for k,channels in [('temperature',SENSORS[:2]),('RH',SENSORS[2:])]}
    keys=df[['row_id',*KEYS]].astype(str).to_dict('records')
    payload={'version':'future_f_H_train_paired_iqr.v1','method':'linear_quantile_IQR','input_sha256':protocol['input_sha256'],
             'row_registry_hash':hash_object(keys),'selected_count':len(df),'per_variable_pooled_count':len(df)*2,
             'source_time_min':str(df['reg_date'].min()),'source_time_max_cutoff_inclusive':str(df['reg_date'].max()),
             'units':{'temperature':'degC','RH':'percentage_points'},'scales':scales}
    return {'scale_payload':payload,'scale_sha256':hash_object(payload),'registry_count':len(reg),
            'excluded_count':len(joined)-len(df),'no_validation_test_p0_reads':True}
