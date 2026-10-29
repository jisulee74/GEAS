"""Public contracts for causal replay, shadow, and opt-in live inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Protocol

import pandas as pd


class InferenceMode(str, Enum):
    REPLAY = "replay"
    SHADOW = "shadow"
    LIVE = "live"


@dataclass(frozen=True)
class ObservationEvent:
    timestamp: pd.Timestamp
    values: Mapping[str, Any]
    received_at: pd.Timestamp | None = None
    control_log: pd.DataFrame | None = None
    segment_id: str | None = None
    crop_cycle_id: str | None = None


@dataclass(frozen=True)
class QualityResult:
    frame: pd.DataFrame
    unresolved_columns: tuple[str, ...] = ()
    blocked: bool = False
    reason: str | None = None
    latency_ms: float = 0.0


@dataclass
class InferenceState:
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    previous_action: dict[str, float] | None = None
    previous_previous_action: dict[str, float] | None = None
    last_timestamp: pd.Timestamp | None = None
    last_segment_id: str | None = None
    last_crop_cycle_id: str | None = None
    fallback_streak: int = 0
    result_by_timestamp: dict[str, "InferenceResult"] = field(default_factory=dict)

    def clone(self) -> "InferenceState":
        return InferenceState(
            history=self.history.copy(deep=True),
            previous_action=None if self.previous_action is None else dict(self.previous_action),
            previous_previous_action=(
                None if self.previous_previous_action is None
                else dict(self.previous_previous_action)
            ),
            last_timestamp=self.last_timestamp,
            last_segment_id=self.last_segment_id,
            last_crop_cycle_id=self.last_crop_cycle_id,
            fallback_streak=self.fallback_streak,
            result_by_timestamp=dict(self.result_by_timestamp),
        )


@dataclass(frozen=True)
class InferenceResult:
    timestamp: pd.Timestamp
    mode: InferenceMode
    status: str
    freshness_seconds: float | None
    unresolved_columns: tuple[str, ...]
    policy_action: Mapping[str, float] | None
    executed_action: Mapping[str, float] | None
    support_projected: bool
    state_ood_level: str
    joint_ood_level: str
    fallback_applied: bool
    command_emitted: bool
    failure_stage: str | None
    reason: str | None
    latencies_ms: Mapping[str, float]
    artifact_hashes: Mapping[str, str]
    idempotent_replay: bool = False


class ObservationSource(Protocol):
    def __iter__(self): ...


class PolicyActionProvider(Protocol):
    def action(self, observation: pd.Series, feasible: Any) -> dict[str, float]: ...


class QualityProcessor(Protocol):
    def process(
        self, frame: pd.DataFrame, control_log: pd.DataFrame | None = None
    ) -> QualityResult: ...


class ControlSink(Protocol):
    def emit(self, result: InferenceResult) -> bool: ...


def utc_now() -> datetime:
    return datetime.utcnow()


__all__ = [
    "ControlSink", "InferenceMode", "InferenceResult", "InferenceState",
    "ObservationEvent", "ObservationSource", "PolicyActionProvider",
    "QualityProcessor", "QualityResult",
]
