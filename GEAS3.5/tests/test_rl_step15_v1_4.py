from __future__ import annotations
from pathlib import Path
import pytest
from geas35.experiments.rl.step15_v1_4 import Step15V14Error, authorize_rl_test_access

def test_rl_test_requires_frozen_protocol_even_with_authorization(tmp_path: Path):
    with pytest.raises((Step15V14Error, FileNotFoundError)):
        authorize_rl_test_access(tmp_path / "missing.json", tmp_path / "auth.json",
                                 crop="strawberry", final_evaluation=True)
