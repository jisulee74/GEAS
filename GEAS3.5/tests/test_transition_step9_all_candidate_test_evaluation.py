from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.transition.cli import run_from_config
from geas35.models.transition import (
    CANDIDATE_TEST_METRICS_CSV,
    CANDIDATE_TEST_METRICS_JSON,
    TEST_METRICS_FILENAME,
    TRANSITION_SUMMARY_MD,
)
from synthetic_experiment_fixtures import (
    SYNTHETIC_CROP,
    transition_rl_ready_frame,
    write_identity_transition_scaler,
)


def test_step9_writes_all_candidate_test_metrics_without_changing_validation_ranking(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "transition_rl_dataset" / SYNTHETIC_CROP
    output_dir = tmp_path / "transition_artifacts"
    dataset_dir.mkdir(parents=True)
    write_identity_transition_scaler(dataset_dir / "observation_scaler.json")
    split_paths = {}
    for split, offset in (("train", 0.0), ("validation", 0.25), ("test", 0.5)):
        path = dataset_dir / f"{split}.parquet"
        transition_rl_ready_frame(offset=offset).to_parquet(path, index=False)
        split_paths[split] = path

    config_path = tmp_path / "transition_step9.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                f"  crop: {SYNTHETIC_CROP}",
                f"  train: '{split_paths['train'].as_posix()}'",
                f"  validation: '{split_paths['validation'].as_posix()}'",
                f"  test: '{split_paths['test'].as_posix()}'",
                "experiment:",
                f"  output_dir: '{output_dir.as_posix()}'",
                "  random_seed: 11",
                "models:",
                "  candidates:",
                "    - name: persistence",
                "      params: {}",
                "    - name: linear_regression",
                "      params: {}",
                "hpo:",
                "  enabled: true",
                "  budget: 1",
                "  random_seed: 42",
                "rollout:",
                "  enabled: true",
                "  horizon_steps: [3, 6, 12]",
                "  step_minutes: 5",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.experiment_result.test_output_paths == {}
    assert not (output_dir / CANDIDATE_TEST_METRICS_CSV).exists()
    assert not (output_dir / CANDIDATE_TEST_METRICS_JSON).exists()
    for candidate_name in ("persistence", "linear_regression"):
        assert not (output_dir / SYNTHETIC_CROP / candidate_name / TEST_METRICS_FILENAME).exists()
    summary_md = (output_dir / TRANSITION_SUMMARY_MD).read_text(encoding="utf-8")
    assert "Test Evaluation" not in summary_md
    assert not (output_dir / "all_candidate_test_report.json").exists()
