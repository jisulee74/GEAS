"""Prepare RL-ready transition datasets from quality-controlled splits."""

from __future__ import annotations

import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_src_on_path() -> None:
    src_path = _project_root() / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()
    from geas35.rl.datasets import main as rl_dataset_main

    return rl_dataset_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
