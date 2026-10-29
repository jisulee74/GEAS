#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

PROJECT_PYTHON = Path("/NHNHOME/WORKSPACE/26mafra001_A/BASE/theimc/jslee/envs/jslee_py311/bin/python")
if PROJECT_PYTHON.exists() and Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
    os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from geas35.experiments.rl.step19_5_runner_v2_2 import (
    audit_crop, extend_crop, finalize_audit, finalize_v2_2, prepare_v2_2,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--audit-crop", action="store_true")
    mode.add_argument("--finalize-audit", action="store_true")
    mode.add_argument("--extend-crop", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--crop", choices=("strawberry", "melon", "cucumber"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--transition-n-jobs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="Independent extension jobs to run concurrently")
    args = parser.parse_args()
    if args.prepare:
        path = prepare_v2_2(ROOT)
    elif args.finalize_audit:
        path = finalize_audit(ROOT)
    elif args.finalize:
        path = finalize_v2_2(ROOT)
    else:
        if not args.crop:
            parser.error("--audit-crop and --extend-crop require --crop")
        if args.audit_crop:
            path = audit_crop(ROOT, args.crop, device=args.device,
                              transition_n_jobs=args.transition_n_jobs)
        else:
            path = extend_crop(ROOT, args.crop, device=args.device,
                               transition_n_jobs=args.transition_n_jobs, workers=args.workers)
    print(path)


if __name__ == "__main__":
    main()
