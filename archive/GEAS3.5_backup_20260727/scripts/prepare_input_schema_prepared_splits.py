"""CLI wrapper for the offline input-schema preparation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.offline.prepare_input_schema_prepared_splits import main


if __name__ == "__main__":
    raise SystemExit(main())
