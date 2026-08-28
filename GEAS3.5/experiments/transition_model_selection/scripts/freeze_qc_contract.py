"""Validate and freeze the Step 11.1 QC input contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    src = args.project_root.resolve() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from geas35.experiments.transition.qc_contract_freeze import freeze_existing_qc_contract

    output = freeze_existing_qc_contract(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        artifact_root=args.artifact_root,
        output_path=args.output,
    )
    print(f"Step 11.1 freeze manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
