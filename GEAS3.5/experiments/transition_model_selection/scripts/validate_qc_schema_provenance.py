"""Validate the Step 11.2 QC Train/Validation schema and provenance."""

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
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    src = args.project_root.resolve() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from geas35.experiments.transition.qc_schema_validation import validate_existing_qc_schema_and_provenance

    output = validate_existing_qc_schema_and_provenance(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        freeze_manifest_path=args.freeze_manifest,
        output_path=args.output,
    )
    print(f"Step 11.2 validation manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
