"""Summarize endpoint contrasts; these are diagnostic substitutions, not policy runs."""
from pathlib import Path
import json,gzip
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];ART=ROOT/'experiments/rl_policy_training/artifacts';OUT=ART/'step19_5_residual_ood_v1'
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(l) for l in gzip.open(p,'rt')]
rows=[];summaries=[]
for crop in ('melon','cucumber'):
 endpoints=read(OUT/f'{crop}_endpoints.json');summary=read(OUT/f'{crop}_summary.json');all_detail=lines(OUT/f'{crop}_feature_trace.jsonl.gz')
 for split in ('validation','train'):
  terms=[e for e in endpoints if e['is_terminal'] and e['split']==split]
  summaries.append({'crop':crop,'split':split,'terminals':len(terms),'variants_below_threshold':{k:sum(e['scores'][k]<=e['threshold'] for e in terms) for k in terms[0]['scores']},'max_steps':max(s['steps'] for s in summary if s['split']==split),'teacher_nonzero_episode_counts':{k:sum(s['teacher_max_abs_error'][k]>1e-6 for s in summary if s['split']==split) for k in summary[0]['teacher_max_abs_error']}})
  registry=read(ART/'step19_5_diagnostics_v1/input_manifest.json')['registries'][crop][split]['indices']
  for e in terms:
   idx=e['episode_index'];ordinal=registry.index(idx);step=e['step']
   old=ART/'step19_5_diagnostics_v1/jobs'/f'{crop}__{split}__recorded-one-step__representative_0__42__524288__0/episodes/{ordinal:04d}.jsonl.gz'
   teacher=lines(old);trace=lines(ART/f'step19_5_light_contract_v1/{crop}_{split}/episodes/{ordinal:04d}.jsonl.gz')
   d=[v for v in all_detail if v['split']==split and v['episode_index']==idx and v['step']==step][0]
   distance=e['distance']['squared_distance_components'];top=sorted(distance,key=distance.get,reverse=True)[:5]
   rows.append({**e,'top_distance_components':{k:distance[k] for k in top},'feature_values':d['features'],'alias_formula_error':d['alias_formula_error'],'recorded_one_step_error_to_endpoint':teacher[step-1]['one_step_error_physical'] if step else None,'prior_actions_modified_count':sum(bool(r.get('recorded_action_modified')) for r in trace[:step]),'prior_fallback_count':sum(bool(r['info'].get('safety_fallback_applied')) for r in trace[:step])})
(OUT/'endpoint_interpretation.json').write_text(json.dumps(rows,indent=2)+'\n')
(OUT/'aggregate_evidence.json').write_text(json.dumps(summaries,indent=2)+'\n')
print(json.dumps(summaries,indent=2))
