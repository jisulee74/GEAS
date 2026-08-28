"""Run the transition candidate-evaluation experiment from this folder."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_src_on_path() -> None:
    src_path = _project_root() / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()
    args = [] if argv is None else list(argv)
    if "--config" not in args:
        args = ["--config", str(_experiment_root() / "configs" / "default.yaml"), *args]
    os.chdir(_experiment_root())

    from geas35.experiments.transition.cli import main as transition_main

    return transition_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
