#!/usr/bin/env python3
"""Validate an external researcher decision and publish Step 14 references."""
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

from geas35.experiments.transition.deployment_handoff import create_step14_deployment_handoff


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GEAS Transition Step 14 handoff.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--researcher-decision", required=True)
    parser.add_argument("--step13-manifest")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    manifest, handoff = create_step14_deployment_handoff(
        project_root=args.project_root,
        researcher_decision_path=args.researcher_decision,
        step13_manifest_path=args.step13_manifest,
        output_root=args.output_root,
    )
    print(f"Step 14 manifest: {manifest}")
    print(f"Explicit candidate handoff: {handoff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
