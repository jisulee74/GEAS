"""Run one isolated v1.3 transition random-restart experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SRC = ROOT / "runtime_src"
sys.path.insert(0, str(RUNTIME_SRC))

from geas35.experiments.transition.random_restart import run_random_restart_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run_random_restart_experiment(args.config, preflight_only=args.preflight_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
