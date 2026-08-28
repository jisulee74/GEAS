#!/usr/bin/env python3
"""Prepare, smoke-test, execute, or finalize GEAS Step 20 HPO."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from geas35.experiments.rl.step20_protocol import prepare_step20_protocol
from geas35.experiments.rl.step20_runner import finalize_official_hpo, run_crop_hpo

def _indices(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GEAS Step 20 Hybrid PPO HPO")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--official", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--crop", choices=("strawberry", "melon", "cucumber"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--transition-n-jobs", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-trials", type=int, default=1)
    parser.add_argument("--smoke-rung-steps", type=int, nargs=2, default=(16, 24))
    parser.add_argument("--validation-episodes", type=_indices, default=())
    args = parser.parse_args(argv)
    if args.prepare:
        print(prepare_step20_protocol(project_root=args.project_root)); return 0
    if args.finalize:
        print(finalize_official_hpo(project_root=args.project_root)); return 0
    if args.crop is None:
        parser.error("--official and --smoke require --crop")
    path = run_crop_hpo(
        project_root=args.project_root, crop=args.crop, device=args.device,
        resume=args.resume, official=args.official,
        trials_override=None if args.official else args.smoke_trials,
        rung_budgets_override=None if args.official else args.smoke_rung_steps,
        validation_episode_indices=args.validation_episodes,
        transition_n_jobs=args.transition_n_jobs,
    )
    print(path); return 0

if __name__ == "__main__":
    raise SystemExit(main())
