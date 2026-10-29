from pathlib import Path
import pytest
from geas35.rl.model_driven_env import load_model_driven_env_from_step15
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"experiments/rl_policy_training/artifacts/step15_v1_4/rl_protocol_manifest.json"
def test_v1_4_environment_loads_selected_model_and_schema():
    env=load_model_driven_env_from_step15(project_root=ROOT,crop="melon",split="train",protocol_manifest_path=PROTOCOL)
    assert env.transition_bundle.model_name=="xgboost" and env.observation_shape==(55,) and env.action_shape==(6,)
def test_v1_4_rl_test_remains_locked():
    with pytest.raises(ValueError,match="locked"):
        load_model_driven_env_from_step15(project_root=ROOT,crop="melon",split="test",protocol_manifest_path=PROTOCOL)
