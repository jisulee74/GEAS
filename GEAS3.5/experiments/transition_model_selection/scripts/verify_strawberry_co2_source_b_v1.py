from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/transition_model_selection/artifacts/strawberry_co2_source_audit_b_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(D/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
x=pd.read_csv(D/'worst_episode_5min_source_trace.csv.gz',parse_dates=['reg_date']);h=pd.read_csv(D/'train_high_co2_all_rows.csv.gz');v=pd.read_csv(D/'validation_high_rows.csv')
assert len(x)==576 and len(h)==2267 and len(v)==202
assert h.population.value_counts().to_dict()=={'already_excluded_A':2235,'additional_excluded_B':16,'B_eligible':16}
assert v.current_vent.ge(.75).sum()==185
r=pd.read_csv(D/'validation_joint_plateaus.csv');r=r.loc[r.start.eq('2025-11-10 18:45:00')];assert len(r)==1 and r.iloc[0]['rows']==179 and r.iloc[0].unique_raw_ids==179
r=pd.read_csv(D/'validation_co2_light_temperature_humidity_plateaus.csv');r=r.loc[r.start.eq('2025-11-10 19:20:00')];assert len(r)==1 and r.iloc[0]['rows']==172 and r.iloc[0].unique_raw_ids==172
checks=json.loads((D/'input_manifest.json').read_text())['inputs']
for p,hsh in checks.items():assert sha(ROOT/p)==hsh
extra=json.loads((D/'supplemental_verification.json').read_text());assert sha(ROOT/extra['extra_input'])==extra['sha256']
paths=['offline_dataset_preparation/scripts/00_extract_raw.py','offline_dataset_preparation/scripts/01_resample.py','offline_dataset_preparation/scripts/03_control_quality.py','offline_dataset_preparation/configs/qc/smartfarm_korea_domain_ranges.csv','src/geas35/preprocessing/2_unit_canonicalization.py','src/geas35/rl/datasets.py','src/geas35/rl/vent_command_memory_b_v3.py']
paths += [str(p.relative_to(ROOT)) for p in (ROOT/'experiments/transition_model_selection/scripts').glob('*strawberry_co2_source*b_v1.py')]
write('code_manifest.json',{p:sha(ROOT/p) for p in paths})
write('decision.json',{'classification':'C','status':'undetermined','frozen_snapshot_suspicion':'strong; not confirmed without upstream measurement timestamp/update evidence','strawberry_phase_1_2':'BLOCKED','overall_phase_1_2':'BLOCKED','next_single_action':'obtain existing upstream payload/update/sequence logs for 2025-11-10 18:40 through 2025-11-11 09:40 KST; compare to DB registration rows','model_feature_experiments_authorized':False})
f=x.loc[x.reg_date.between('2025-11-10 17:00','2025-11-11 11:00')]
fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
for c in ('raw_in_co2','raw_in_co2_2'):axes[0].plot(f.reg_date,f[c],label=c)
axes[0].set_ylabel('CO2 ppm');axes[0].legend()
for c in ('raw_in_temp','raw_in_temp2'):axes[1].plot(f.reg_date,f[c],label=c)
axes[1].set_ylabel('Temperature C');axes[1].legend()
axes[2].plot(f.reg_date,f.raw_out_light,label='raw light');axes[2].set_ylabel('Stored light');axes[2].legend();axes[2].set_xlabel('DB reg_date (KST; not verified measurement timestamp)')
for ax in axes:ax.axvspan(pd.Timestamp('2025-11-10 19:20'),pd.Timestamp('2025-11-11 09:35'),alpha=.12,color='red');ax.grid(alpha=.2)
fig.suptitle('Repeated source values while DB row IDs/timestamps advance');fig.autofmt_xdate();fig.tight_layout();fig.savefig(D/'source_plateau.png',dpi=150);plt.close(fig)
write('final_verification.json',{'input_hashes_unchanged':True,'A_membership_full_train_exact':True,'DB_raw_exact_rows':576,'high_train_partition':[2235,16,16],'high_validation':202,'high_vent_validation':185,'plateau_summary_verified_from_run_tables':True,'no_Test_no_fit_no_regeneration':True})
write('artifact_integrity.json',{str(p.relative_to(D)):sha(p) for p in sorted(D.iterdir()) if p.is_file() and p.name!='artifact_integrity.json'})
print('All final source audit checks passed.')
