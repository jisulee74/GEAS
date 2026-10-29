#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner import add_workers, finalize, prepare, run_crop


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-crop", action="store_true")
    mode.add_argument("--add-workers", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--crop", choices=("strawberry", "melon", "cucumber"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--transition-n-jobs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--idle-timeout-seconds", type=float, default=3600)
    args = parser.parse_args()

    if args.prepare:
        print(prepare(ROOT))
    elif args.finalize:
        print(finalize(ROOT))
    elif args.add_workers:
        if not args.crop:
            parser.error("--add-workers requires --crop")
        print(add_workers(ROOT, args.crop, args.workers, args.device,
                          args.transition_n_jobs, args.idle_timeout_seconds))
    else:
        if not args.crop:
            parser.error("--run-crop requires --crop")
        print(run_crop(ROOT, args.crop, args.device, args.transition_n_jobs, args.workers))


if __name__ == "__main__":
    main()
