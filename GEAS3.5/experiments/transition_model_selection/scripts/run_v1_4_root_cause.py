#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from geas35.models.transition.root_cause_v1_4 import audit_melon_inputs, run_cucumber_controlled_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description="GEAS3.5 transition rollout root-cause diagnostics v1.4")
    parser.add_argument("--mode", choices=("cucumber-controls", "melon-audit", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--conditions", nargs="+", choices=("R", "A", "B", "C", "D"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    if args.mode in ("melon-audit", "all"):
        audit_melon_inputs(PROJECT_ROOT, output_root=args.output_root)
    if args.mode in ("cucumber-controls", "all"):
        result = run_cucumber_controlled_comparison(
            PROJECT_ROOT, seeds=args.seeds, condition_codes=args.conditions,
            output_root=args.output_root, resume=not args.no_resume,
        )
        print(result["cause_assessment"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
