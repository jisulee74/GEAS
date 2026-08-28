#!/usr/bin/env python3
"""Run the official Step 12 all-crop real-data benchmark."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.transition.real_data_benchmark import run_step12_real_data_benchmark

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GEAS Transition Step 12.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--handoff")
    parser.add_argument("--output-root")
    parser.add_argument("--config", action="append", dest="configs")
    args = parser.parse_args(argv)
    result = run_step12_real_data_benchmark(
        project_root=args.project_root, config_paths=args.configs,
        handoff_path=args.handoff, output_root=args.output_root,
    )
    print(f"Step 12 manifest: {result.manifest_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
