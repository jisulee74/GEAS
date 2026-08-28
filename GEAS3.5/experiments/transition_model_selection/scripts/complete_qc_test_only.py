"""Generate only the missing Step 11.3 quality-controlled Test splits."""

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
    parser.add_argument("--step11-2-manifest", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args(argv)
    src = args.project_root.resolve() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from geas35.experiments.transition.qc_test_completion import complete_missing_qc_test_only

    output = complete_missing_qc_test_only(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        step11_2_manifest_path=args.step11_2_manifest,
        output_manifest_path=args.output_manifest,
    )
    print(f"Step 11.3 completion manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
