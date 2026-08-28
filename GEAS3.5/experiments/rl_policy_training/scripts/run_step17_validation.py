#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.rl.step17_validation import run_step17_environment_validation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run GEAS RL Step 17 validation.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    result = run_step17_environment_validation(
        project_root=args.project_root, output_root=args.output_root
    )
    print(f"Step 17 integrity manifest: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
