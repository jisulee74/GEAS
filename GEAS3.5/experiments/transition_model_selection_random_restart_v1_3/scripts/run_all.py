"""Run the six v1.3 experiments sequentially to avoid oversubscription."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime_src"))

from geas35.experiments.transition.random_restart import build_cross_experiment_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only and args.aggregate_only:
        parser.error("--preflight-only and --aggregate-only are mutually exclusive")
    if args.aggregate_only:
        build_cross_experiment_report(ROOT / "artifacts")
        return 0
    configs = sorted((ROOT / "configs").glob("*_v1_3.yaml"))
    if len(configs) != 6:
        raise RuntimeError(f"Expected six v1.3 configs, found {len(configs)}")
    for config in configs:
        command = [sys.executable, str(ROOT / "scripts" / "run.py"), "--config", str(config)]
        if args.preflight_only:
            command.append("--preflight-only")
        subprocess.run(command, check=True, cwd=ROOT)
    if not args.preflight_only:
        build_cross_experiment_report(ROOT / "artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
