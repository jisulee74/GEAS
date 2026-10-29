#!/usr/bin/env python3
"""Run the Step 17 v1.4 environment validation suite."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geas35.experiments.rl.step17_v1_4 import run_step17_environment_validation


if __name__ == "__main__":
    print(run_step17_environment_validation(project_root=PROJECT_ROOT))
