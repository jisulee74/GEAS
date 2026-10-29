#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from geas35.experiments.rl.step15_v1_4 import run_step15_v1_4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    parser.add_argument("--config")
    args = parser.parse_args()
    print(run_step15_v1_4(project_root=ROOT, output_root=args.output_root,
                         config_path=args.config))
