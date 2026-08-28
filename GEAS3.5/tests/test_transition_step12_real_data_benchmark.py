from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.transition.real_data_benchmark import (
    OFFICIAL_CANDIDATES,
    OFFICIAL_CROPS,
    OFFICIAL_TARGETS,
    Step12BenchmarkError,
    _validate_configs,
    _verify_handoff,
)



def test_step12_official_configs_are_bound_to_step11_immutable_datasets() -> None:
    handoff = (
        PROJECT_ROOT
        / "offline_dataset_preparation/datasets/5_rl_dataset/step12_handoff.json"
    )
    immutable_path, immutable_hash, immutable = _verify_handoff(PROJECT_ROOT, handoff)
    configs = [
        PROJECT_ROOT / f"experiments/transition_model_selection/configs/{crop}.yaml"
        for crop in OFFICIAL_CROPS
    ]
    loaded = _validate_configs(
        PROJECT_ROOT,
        configs,
        immutable_path=immutable_path,
        immutable_hash=immutable_hash,
        immutable=immutable,
    )

    assert tuple(loaded) == OFFICIAL_CROPS
    assert OFFICIAL_TARGETS == (
        "obs_indoor_temp_c",
        "obs_indoor_humidity_pct",
        "obs_indoor_co2_ppm",
    )
    for crop, config in loaded.items():
        assert config.experiment_config.crop == crop
        assert tuple(spec.model_name for spec in config.experiment_config.models) == OFFICIAL_CANDIDATES
        assert config.experiment_config.hpo_config.budget == 20
        assert config.experiment_config.rollout_horizon_steps == (3, 6, 12)


def test_step12_rejects_tampered_handoff_manifest_hash(tmp_path: Path) -> None:
    immutable = tmp_path / "immutable.json"
    immutable.write_text("{}", encoding="utf-8")
    wrong_hash = "0" * 64
    handoff = tmp_path / "handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "status": "ready",
                "integrity_gate": "passed",
                "immutable_rl_manifest": {
                    "path": str(immutable),
                    "sha256": wrong_hash,
                },
            }
        ),
        encoding="utf-8",
    )

    assert hashlib.sha256(immutable.read_bytes()).hexdigest() != wrong_hash
    with pytest.raises(Step12BenchmarkError, match="hash"):
        _verify_handoff(tmp_path, handoff)
