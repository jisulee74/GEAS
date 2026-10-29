#!/usr/bin/env python3
"""Run the Step 13 review against frozen v1.4 official artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.transition.reproducibility_review_v1_4 import (
    run_step13_v1_4_reproducibility_review,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    path = run_step13_v1_4_reproducibility_review(
        project_root=PROJECT_ROOT,
        official_root=args.official_root,
        output_root=args.output_root,
    )
    print(path)


if __name__ == "__main__":
    main()
