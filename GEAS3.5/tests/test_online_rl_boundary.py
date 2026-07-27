from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.models.transition import (
    LoggedActionProvider,
    PolicyActionProvider,
    load_selected_transition_model,
    predict_next_observation_reward,
)
from geas35.realtime import sanitize_for_controller


def test_policy_action_provider_is_unsupported_boundary() -> None:
    assert "action" in PolicyActionProvider.__abstractmethods__
    with pytest.raises(TypeError):
        PolicyActionProvider()


def test_transition_reward_prediction_requires_explicit_action() -> None:
    signature = inspect.signature(predict_next_observation_reward)

    assert signature.parameters["action"].default is inspect.Signature.empty


def test_future_online_boundary_exports_existing_reusable_helpers() -> None:
    assert inspect.isclass(LoggedActionProvider)
    assert callable(load_selected_transition_model)
    assert callable(sanitize_for_controller)
