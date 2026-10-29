"""Read raw send events strictly inside frozen Train/Validation sensor intervals.
No cycle matching is inferred from rounding. Missing linkage remains unknown.
"""
from pathlib import Path
import importlib.util,json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v1';D.mkdir(parents=True,exist_ok=True)
if (D/'source_capture.json').exists():raise FileExistsError('Frozen capture exists; use a new namespace')
s=importlib.util.spec_from_file_location('rawreader',ROOT/'offline_dataset_preparation/scripts/00_extract_raw.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
import pymysql
report={'test_accessed':False,'cycle_mapping_verified':False,'schema':{},'sources':[]}
try:
 with pymysql.connect(**m.db_config_from_env(),connect_timeout=5,read_timeout=20,cursorclass=pymysql.cursors.DictCursor) as conn:
  with conn.cursor() as cur:
   for table in ['optimal_control','send_ds_control_api_history','optimal_monitoring','control_actuation','optimal_monitoring_node']:
    try:
     cur.execute('SHOW COLUMNS FROM '+table);report['schema'][table]=[{'name':x['Field'],'type':x['Type']} for x in cur.fetchall()]
    except pymysql.MySQLError as e:report['schema'][table]={'error_code':e.args[0]}
   for crop in ('melon','cucumber','strawberry'):
    for split in ('train','validation'):
     p=ROOT/f'offline_dataset_preparation/datasets/03_quality_controlled/{crop}/{split}.parquet';f=pd.read_parquet(p,columns=['reg_date']);t=pd.to_datetime(f.reg_date).drop_duplicates().sort_values();t=t[t>pd.Timestamp('2026-01-22 09:05')]
     events=[];intervals=[]
     if len(t):
      groups=t.diff().ne(pd.Timedelta(minutes=5)).cumsum()
      for _,g in t.groupby(groups):
       start=g.min();end=g.max()+pd.Timedelta(minutes=5);intervals.append([str(start),str(end)])
       cur.execute('SELECT idx,n_idx,channel,operation,ratio,duration,reg_date FROM send_ds_control_api_history WHERE reg_date >= %s AND reg_date < %s AND channel IN (1,2,4,5,9,10,13) ORDER BY reg_date,idx',(start.to_pydatetime(),end.to_pydatetime()));events.extend(cur.fetchall())
     raw=pd.DataFrame(events,columns=['idx','n_idx','channel','operation','ratio','duration','reg_date']);out=D/f'{crop}_{split}_raw_api.parquet';raw.to_parquet(out,index=False)
     report['sources'].append({'crop':crop,'split':split,'raw_event_count':len(raw),'allowed_intervals':intervals,'path':str(out.relative_to(ROOT)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'sensor_input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'event_cycle_link':'unknown','device_mapping':'unknown'})
     print(crop,split,len(raw),'unlinked raw events',flush=True)
except Exception as e:
 report['capture_error']={'class':type(e).__name__,'code':e.args[0] if e.args and isinstance(e.args[0],int) else 'details redacted'}
(D/'source_capture.json').write_text(json.dumps(report,indent=2)+'\n')
