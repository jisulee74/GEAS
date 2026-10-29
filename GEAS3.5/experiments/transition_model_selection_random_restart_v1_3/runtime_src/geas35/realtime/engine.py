"""Atomic single-timestep causal inference engine."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Mapping

import pandas as pd

from geas35.realtime.contracts import InferenceMode, InferenceResult, InferenceState, ObservationEvent
from geas35.realtime.policy import (
    apply_action_safety, build_feasible_action_spec, state_support_score, support_level,
)
from geas35.rl import MDP_V1_ACTION_COLUMNS, MdpV1Config, mdp_v1_observation_columns


class OnlineInferenceEngine:
    def __init__(
        self, *, mode: InferenceMode | str, preprocessor, scaler, policy, support, sink,
        artifact_hashes: Mapping[str, str] | None = None,
        safe_fallback: Mapping[str, float] | None = None,
        mdp_config: MdpV1Config | None = None, stale_after_seconds: float = 600.0,
        max_gap_minutes: float = 10.0, live_enabled: bool = False,
    ):
        self.mode = InferenceMode(mode)
        if self.mode is InferenceMode.LIVE and not live_enabled:
            raise PermissionError("Live inference requires explicit live_enabled=True")
        self.preprocessor = preprocessor; self.scaler = scaler; self.policy = policy
        self.support = support; self.sink = sink
        self.artifact_hashes = dict(artifact_hashes or {})
        self.safe_fallback = None if safe_fallback is None else self._validated_action(safe_fallback)
        self.mdp_config = mdp_config or MdpV1Config()
        self.stale_after_seconds = float(stale_after_seconds)
        self.max_gap = pd.Timedelta(minutes=max_gap_minutes)
        self.state = InferenceState()

    @staticmethod
    def _validated_action(action: Mapping[str, float]) -> dict[str, float]:
        missing = [column for column in MDP_V1_ACTION_COLUMNS if column not in action]
        if missing: raise ValueError("Action is missing columns: " + ", ".join(missing))
        values = {column: float(action[column]) for column in MDP_V1_ACTION_COLUMNS}
        if any(not 0.0 <= values[column] <= 1.0 for column in MDP_V1_ACTION_COLUMNS[:3]):
            raise ValueError("Continuous fallback action must be in [0, 1]")
        if any(values[column] not in {0.0, 1.0} for column in MDP_V1_ACTION_COLUMNS[3:]):
            raise ValueError("Binary fallback action must be zero or one")
        if values["heat_run"] == 1.0 and values["cool_run"] == 1.0:
            raise ValueError("Fallback cannot enable heating and cooling together")
        return values

    @staticmethod
    def _inject_previous_action(row: dict, action: Mapping[str, float] | None) -> None:
        if action is None: return
        mapping = {
            "cont_skyl_vol": "vent_pct", "cont_skyr_vol": "vent_pct",
            "cont_cur_vol": "shade_curtain_pct",
            "cont_kwcur_vol": "thermal_curtain_pct",
            "cont_heater_run": "heat_run", "cont_cooler_run": "cool_run",
            "cont_fan_run": "fan_run",
        }
        for target, source in mapping.items():
            if target not in row or pd.isna(row[target]): row[target] = float(action[source])

    def _fallback(self, state: InferenceState) -> dict[str, float] | None:
        if state.fallback_streak == 0 and state.previous_action is not None:
            return dict(state.previous_action)
        return None if self.safe_fallback is None else dict(self.safe_fallback)

    def _blocked(self, event, state, *, stage, reason, freshness=None, unresolved=(), quality_ms=0.0):
        action = self._fallback(state); state.fallback_streak += 1
        result = InferenceResult(
            timestamp=event.timestamp, mode=self.mode, status="blocked",
            freshness_seconds=freshness, unresolved_columns=tuple(unresolved),
            policy_action=None, executed_action=action, support_projected=False,
            state_ood_level="unknown", joint_ood_level="unknown",
            fallback_applied=action is not None, command_emitted=False,
            failure_stage=stage, reason=reason,
            latencies_ms={"quality": quality_ms, "total": 0.0},
            artifact_hashes=self.artifact_hashes,
        )
        if action is not None:
            sink_started = perf_counter()
            try:
                emitted = bool(self.sink.emit(result))
            except Exception:
                emitted = False
            result = replace(
                result, command_emitted=emitted,
                latencies_ms={**result.latencies_ms, "sink": (perf_counter() - sink_started) * 1000.0},
            )
        # A rejected observation never enters causal history, but its decision is
        # cached so the same timestamp cannot emit a fallback twice.
        self.state.fallback_streak = state.fallback_streak
        self.state.result_by_timestamp[event.timestamp.isoformat()] = result
        return result

    def step(self, event: ObservationEvent) -> InferenceResult:
        started = perf_counter(); timestamp = pd.Timestamp(event.timestamp)
        event = replace(event, timestamp=timestamp); key = timestamp.isoformat()
        cached = self.state.result_by_timestamp.get(key)
        if cached is not None:
            return replace(cached, idempotent_replay=True, command_emitted=False)
        candidate = self.state.clone()
        if candidate.last_timestamp is not None and timestamp < candidate.last_timestamp:
            return self._blocked(event, candidate, stage="timestamp", reason="timestamp_regression")
        freshness = None
        if event.received_at is not None:
            freshness = max(0.0, (pd.Timestamp(event.received_at) - timestamp).total_seconds())
            if freshness > self.stale_after_seconds:
                return self._blocked(event, candidate, stage="timestamp", reason="stale_observation", freshness=freshness)
        boundary = (
            candidate.last_timestamp is not None and timestamp - candidate.last_timestamp > self.max_gap
        ) or (
            event.segment_id is not None and candidate.last_segment_id is not None
            and event.segment_id != candidate.last_segment_id
        ) or (
            event.crop_cycle_id is not None and candidate.last_crop_cycle_id is not None
            and event.crop_cycle_id != candidate.last_crop_cycle_id
        )
        if boundary:
            candidate.history = pd.DataFrame(); candidate.previous_action = None
            candidate.previous_previous_action = None; candidate.fallback_streak = 0
        row = dict(event.values); self._inject_previous_action(row, candidate.previous_action)
        row["reg_date"] = timestamp
        candidate.history = pd.concat([candidate.history, pd.DataFrame([row])], ignore_index=True).tail(self.preprocessor.history_rows)
        try:
            quality, physical, _ = self.preprocessor.process(candidate.history, event.control_log)
        except Exception as exc:
            return self._blocked(event, candidate, stage="quality", reason=f"quality_error:{type(exc).__name__}", freshness=freshness)
        if quality.blocked or physical is None:
            return self._blocked(
                event, candidate, stage="quality", reason=quality.reason or "quality_blocked",
                freshness=freshness, unresolved=quality.unresolved_columns, quality_ms=quality.latency_ms,
            )
        scaled = self.scaler.transform_frame(physical.to_frame().T).iloc[0]
        policy_columns = mdp_v1_observation_columns(physical.to_frame().T)
        policy_observation = scaled.loc[policy_columns]
        state_frame = scaled.to_frame().T
        state_level = support_level(self.support, state_support_score(self.support, state_frame), kind="state")
        if state_level == "severe":
            return self._blocked(event, candidate, stage="support", reason="severe_state_ood", freshness=freshness, quality_ms=quality.latency_ms)
        feasible = build_feasible_action_spec(
            self.support, scaled, physical, config=self.mdp_config, fallback_action=self.safe_fallback,
        )
        policy_started = perf_counter()
        try:
            policy_action = self._validated_action(self.policy.action(policy_observation, feasible))
            safe = apply_action_safety(policy_action, feasible)
        except Exception as exc:
            return self._blocked(event, candidate, stage="policy", reason=f"policy_error:{type(exc).__name__}", freshness=freshness, quality_ms=quality.latency_ms)
        policy_ms = (perf_counter() - policy_started) * 1000.0
        joint_score = float(self.support.scores(state_frame, pd.DataFrame([safe.action]))["joint"][0])
        joint_level = support_level(self.support, joint_score, kind="joint")
        fallback = joint_level == "severe"
        executed = self._fallback(candidate) if fallback else safe.action
        result = InferenceResult(
            timestamp=timestamp, mode=self.mode, status="ok" if executed is not None else "blocked",
            freshness_seconds=freshness, unresolved_columns=(), policy_action=policy_action,
            executed_action=executed, support_projected=safe.projected,
            state_ood_level=state_level, joint_ood_level=joint_level,
            fallback_applied=fallback, command_emitted=False,
            failure_stage=("support" if executed is None else None),
            reason=("severe_joint_ood" if fallback else safe.reason),
            latencies_ms={"quality": quality.latency_ms, "policy": policy_ms, "total": 0.0},
            artifact_hashes=self.artifact_hashes,
        )
        sink_started = perf_counter()
        try:
            emitted = executed is not None and bool(self.sink.emit(result))
        except Exception:
            emitted = False
        result = replace(result, command_emitted=emitted, latencies_ms={
            **result.latencies_ms, "sink": (perf_counter() - sink_started) * 1000.0,
            "total": (perf_counter() - started) * 1000.0,
        })
        if executed is not None and not emitted:
            failed = replace(result, status="blocked", failure_stage="sink", reason="control_sink_failed")
            self.state.result_by_timestamp[key] = failed
            return failed
        candidate.previous_previous_action = candidate.previous_action
        candidate.previous_action = None if executed is None else dict(executed)
        candidate.last_timestamp = timestamp; candidate.last_segment_id = event.segment_id
        candidate.last_crop_cycle_id = event.crop_cycle_id
        candidate.fallback_streak = 1 if fallback else 0
        candidate.result_by_timestamp[key] = result; self.state = candidate
        return result


__all__ = ["OnlineInferenceEngine"]
