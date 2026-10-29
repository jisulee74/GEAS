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

from geas35.experiments.rl.step18_v1_4 import run_step18_hybrid_ppo_freeze


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze and validate GEAS RL Step 18 v1.4.")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    result = run_step18_hybrid_ppo_freeze(project_root=PROJECT_ROOT, output_root=args.output_root)
    print(f"Step 18 v1.4 integrity manifest: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
