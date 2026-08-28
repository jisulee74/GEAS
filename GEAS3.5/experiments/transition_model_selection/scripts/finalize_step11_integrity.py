"""Run the Step 11.6 all-crop integrity gate and publish Step 12 handoff."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    root = _project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from geas35.experiments.transition.integrity_handoff import finalize_step11_integrity

    parser = argparse.ArgumentParser(description="Finalize GEAS Transition Step 11.6")
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--rl-manifest", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    manifest, handoff = finalize_step11_integrity(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        rl_manifest_path=args.rl_manifest,
        output_manifest_path=args.output_manifest,
        handoff_path=args.handoff,
        run_id=args.run_id,
    )
    print(f"Step 11.6 integrity manifest: {manifest}")
    print(f"Step 12 handoff: {handoff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
