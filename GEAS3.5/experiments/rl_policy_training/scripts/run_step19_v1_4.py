#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geas35.experiments.rl.step19_v1_4 import run_step19_trainability_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GEAS RL Step 19 v1.4 trainability pilot.")
    parser.add_argument("--output-root")
    parser.add_argument("--config")
    args = parser.parse_args()
    result = run_step19_trainability_pilot(
        project_root=PROJECT_ROOT, output_root=args.output_root, config_path=args.config,
    )
    print(f"Step 19 v1.4 integrity manifest: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
