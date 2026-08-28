#!/usr/bin/env python3
"""Run the official Step 13 reproducibility and integrity review."""
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

from geas35.experiments.transition.reproducibility_review import run_step13_reproducibility_review


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GEAS Transition Step 13 review.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--step12-manifest")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    path = run_step13_reproducibility_review(
        project_root=args.project_root,
        step12_manifest_path=args.step12_manifest,
        output_root=args.output_root,
    )
    print(f"Step 13 manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
