"""Read-only live schema and exact allowed Validation timestamp lookup."""
from pathlib import Path
import importlib.util,json
import pandas as pd
import pymysql
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/transition_model_selection/artifacts/strawberry_co2_source_audit_b_v1'
spec=importlib.util.spec_from_file_location('extract',ROOT/'offline_dataset_preparation/scripts/00_extract_raw.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
status={'read_only':True,'Test_accessed':False}
try:
 cfg=mod.db_config_from_env();conn=pymysql.connect(**cfg,charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor,connect_timeout=8,read_timeout=20,write_timeout=8)
 with conn.cursor() as cur:
  cur.execute('SET TRANSACTION READ ONLY');cur.execute('START TRANSACTION')
  cur.execute('SELECT COLUMN_NAME, DATA_TYPE, COLUMN_COMMENT FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION',('data_silla_enc',));schema=cur.fetchall();(D/'db_data_silla_enc_schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2))
  q=pd.read_parquet(ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4/strawberry/validation_prepared_qc.parquet',columns=['reg_date']);times=q.loc[q.reg_date.ge('2025-11-10')&q.reg_date.lt('2025-11-12'),'reg_date']
  rows=[]
  for start in range(0,len(times),100):
   ts=times.iloc[start:start+100].dt.to_pydatetime().tolist()
   cur.execute('SELECT idx,iot_data_idx,reg_date,in_co2,in_co2_2,in_temp,in_temp2,in_hum,in_hum2,out_light,out_windsp,cont_skyl_vol,cont_skyr_vol FROM data_silla_enc WHERE iot_data_idx=%s AND reg_date IN ('+','.join(['%s']*len(ts))+') ORDER BY reg_date',[97]+ts)
   rows.extend(cur.fetchall())
  pd.DataFrame(rows).to_csv(D/'db_worst_episode_raw_rows.csv.gz',index=False)
  status.update({'success':True,'rows':len(rows),'allowed_timestamps':len(times),'timestamp_columns':[r for r in schema if any(x in r['DATA_TYPE'] for x in ('date','time'))]})
 conn.rollback();conn.close()
except Exception as e:
 status.update({'success':False,'error_type':type(e).__name__,'note':'No credentials or server address recorded; live DB unavailable if unsuccessful.'})
(D/'db_read_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2,default=str));print(json.dumps(status,ensure_ascii=False,default=str))
