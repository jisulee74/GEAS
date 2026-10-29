#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
PROJECT_ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT_ROOT/"src"))
from geas35.models.transition.official_v1_4 import run_official_combination

CONFIGS=[PROJECT_ROOT/f"experiments/transition_model_selection/configs/{crop}_v1_4_{flags}_flags.yaml" for crop in ("strawberry","melon","cucumber") for flags in ("with","without")]
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--config"); p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44]); args=p.parse_args(argv)
 configs=[Path(args.config).resolve()] if args.config else CONFIGS
 results=[]
 for config in configs:
  print(f"[v1.4] starting {config.name}",flush=True)
  try: result=run_official_combination(config,seeds=args.seeds)
  except Exception as exc: result={"status":"orchestration_error","error_type":type(exc).__name__,"message":str(exc),"test_opened":False}
  results.append({"config":str(config),**result}); print(f"[v1.4] {config.name}: {result['status']}",flush=True)
 manifest=PROJECT_ROOT/"experiments/transition_model_selection/artifacts/step12_v1_4_root_cause/official_rerun/combined_status.json"
 manifest.parent.mkdir(parents=True,exist_ok=True); manifest.write_text(json.dumps(results,indent=2,default=str))
 return 0 if all(r['status'] in ('completed','no_eligible_model','no_eligible_seed') for r in results) else 1
if __name__=="__main__": raise SystemExit(main())
