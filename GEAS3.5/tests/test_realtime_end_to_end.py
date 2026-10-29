from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geas35.realtime import (
    InferenceMode, ObservationEvent, OnlineInferenceEngine,
    RecordOnlyControlSink, ReplayObservationSource, load_deployment_bundle,
    summarize_inference_results,
)
from geas35.realtime.adapters import HttpControlSink
from geas35.realtime.artifacts import BUNDLE_SCHEMA_VERSION
from geas35.realtime.contracts import QualityResult
from geas35.realtime.policy import apply_action_safety
from geas35.rl import MDP_V1_ACTION_COLUMNS, MDP_V1_OBSERVATION_COLUMNS
from geas35.rl.hybrid_ppo import FeasibleActionSpec


class IdentityScaler:
    def transform_frame(self, frame): return frame.copy()


class FakePreprocessor:
    history_rows = 288
    def __init__(self, blocked=False): self.blocked = blocked; self.history_lengths = []
    def process(self, history, control_log=None):
        del control_log
        self.history_lengths.append(len(history))
        if self.blocked:
            return QualityResult(history, ("obs_indoor_temp_c",), True, "quality_blocked"), None, None
        values = {column: 0.0 for column in MDP_V1_OBSERVATION_COLUMNS}
        values.update({
            "obs_indoor_temp_c": 22.0, "obs_indoor_humidity_pct": 70.0,
            "obs_indoor_co2_ppm": 500.0, "obs_outdoor_wind_speed": 1.0,
            "obs_rain_flag": 0.0,
        })
        current = pd.Series(values)
        return QualityResult(history), current, current.to_frame().T


class FakeNearest:
    def kneighbors(self, values, return_distance=True):
        distances = np.zeros((len(values), 1), dtype=float)
        indices = np.zeros((len(values), 1), dtype=int)
        return (distances, indices) if return_distance else indices


class FakeSupport:
    def __init__(self, severe=False):
        self.config = {
            "state_columns": list(MDP_V1_OBSERVATION_COLUMNS),
            "thresholds": {"state_warning": 1.0, "state_severe": -1.0 if severe else 2.0,
                           "joint_warning": 1.0, "joint_severe": 2.0},
        }
        self.state_weights = np.ones(len(MDP_V1_OBSERVATION_COLUMNS))
        self._state_nn = FakeNearest()
    def local_action_support(self, state):
        return [{"continuous": {column: [0.0, 1.0] for column in MDP_V1_ACTION_COLUMNS[:3]},
                 "binary": {column: [0, 1] for column in MDP_V1_ACTION_COLUMNS[3:]}}]
    def scores(self, state, action): return {"state": np.array([0.0]), "joint": np.array([0.0])}


class FakePolicy:
    def action(self, observation, feasible):
        del observation, feasible
        return dict(zip(MDP_V1_ACTION_COLUMNS, (0.4, 0.3, 0.2, 1.0, 0.0, 1.0)))


def make_engine(*, mode="replay", blocked=False, support=None, live_enabled=False):
    sink = RecordOnlyControlSink()
    engine = OnlineInferenceEngine(
        mode=mode, preprocessor=FakePreprocessor(blocked), scaler=IdentityScaler(),
        policy=FakePolicy(), support=support or FakeSupport(), sink=sink,
        safe_fallback=dict.fromkeys(MDP_V1_ACTION_COLUMNS, 0.0), live_enabled=live_enabled,
    )
    return engine, sink


def event(ts="2026-06-01 12:00:00", received=None):
    timestamp = pd.Timestamp(ts)
    return ObservationEvent(timestamp, {"value": 1.0}, received_at=pd.Timestamp(received or timestamp))


def test_replay_and_shadow_share_the_same_core_action():
    replay, _ = make_engine(mode="replay"); shadow, _ = make_engine(mode="shadow")
    assert replay.step(event()).executed_action == shadow.step(event()).executed_action


def test_duplicate_timestamp_is_idempotent_and_not_emitted_twice():
    engine, sink = make_engine(); first = engine.step(event()); second = engine.step(event())
    assert first.command_emitted and second.idempotent_replay and not second.command_emitted
    assert len(sink.records) == 1


def test_stale_and_quality_blocked_fail_closed_without_policy_action():
    stale, stale_sink = make_engine()
    result = stale.step(event(received="2026-06-01 12:20:00"))
    assert result.status == "blocked" and result.reason == "stale_observation"
    assert result.policy_action is None and len(stale_sink.records) == 1
    blocked, _ = make_engine(blocked=True)
    result = blocked.step(event())
    assert result.failure_stage == "quality" and result.unresolved_columns


def test_timestamp_regression_and_severe_state_ood_are_blocked():
    engine, _ = make_engine(); engine.step(event("2026-06-01 12:05:00"))
    assert engine.step(event("2026-06-01 12:00:00")).reason == "timestamp_regression"
    severe, _ = make_engine(support=FakeSupport(severe=True))
    assert severe.step(event()).reason == "severe_state_ood"


def test_live_requires_explicit_gate():
    with pytest.raises(PermissionError): make_engine(mode="live")
    engine, _ = make_engine(mode="live", live_enabled=True)
    assert engine.mode is InferenceMode.LIVE


def test_http_sink_requires_both_live_gates():
    with pytest.raises(PermissionError):
        HttpControlSink(endpoint="http://localhost", auth_endpoint=None, credentials=None,
                        device_mapping={"vent_pct": [{"device_id": "1"}]},
                        live_enabled=True, confirm_live_control=False)


def test_http_sink_builds_all_payloads_without_retry(monkeypatch):
    requests = []
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
    class Opener:
        def open(self, outgoing, timeout):
            requests.append((json.loads(outgoing.data), timeout)); return Response()
    monkeypatch.setattr("geas35.realtime.adapters.request.build_opener", lambda *args: Opener())
    mapping = {column: [{"device_id": str(index), "scale": 100 if index < 3 else 1}]
               for index, column in enumerate(MDP_V1_ACTION_COLUMNS)}
    sink = HttpControlSink(endpoint="http://controller", auth_endpoint=None, credentials=None,
                           device_mapping=mapping, live_enabled=True,
                           confirm_live_control=True)
    engine, _ = make_engine(); result = engine.step(event())
    assert sink.emit(result) and len(requests) == 6
    assert not sink.emit(result) and len(requests) == 6


def test_heat_and_cool_conflict_is_projected_to_both_off():
    feasible = FeasibleActionSpec.unconstrained()
    action = dict(zip(MDP_V1_ACTION_COLUMNS, (0.5, 0.5, 0.5, 1, 1, 0)))
    result = apply_action_safety(action, feasible)
    assert result.projected and result.reason == "heat_cool_conflict"
    assert result.action["heat_run"] == result.action["cool_run"] == 0.0


def test_replay_source_sorts_and_preserves_boundaries():
    frame = pd.DataFrame({"reg_date": ["2026-01-02", "2026-01-01"],
                          "segment_id": ["b", "a"], "crop_cycle_id": [2, 1]})
    events = list(ReplayObservationSource(frame))
    assert events[0].timestamp < events[1].timestamp and events[0].segment_id == "a"


def test_deployment_bundle_checks_hashes_and_optional_policy(tmp_path: Path):
    roles = ["quality_model", "observation_scaler", "state_action_support", "transition_model"]
    artifacts = {}
    for role in roles:
        path = tmp_path / f"{role}.bin"; path.write_bytes(role.encode())
        artifacts[role] = {"path": path.name, "sha256": hashlib.sha256(role.encode()).hexdigest()}
    manifest = tmp_path / "bundle.json"
    manifest.write_text(json.dumps({"schema_version": BUNDLE_SCHEMA_VERSION, "crop": "cucumber", "artifacts": artifacts}))
    bundle = load_deployment_bundle(manifest, require_policy=False)
    assert bundle.crop == "cucumber"
    artifacts["quality_model"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps({"schema_version": BUNDLE_SCHEMA_VERSION, "crop": "cucumber", "artifacts": artifacts}))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_deployment_bundle(manifest, require_policy=False)


def test_summary_reports_latency_and_fail_closed_rates():
    engine, _ = make_engine(); results = [engine.step(event())]
    summary = summarize_inference_results(results)
    assert summary["steps"] == 1 and summary["valid_step_rate"] == 1.0
    assert "total" in summary["latency"]


def test_long_gap_resets_causal_history():
    processor = FakePreprocessor(); sink = RecordOnlyControlSink()
    engine = OnlineInferenceEngine(
        mode="replay", preprocessor=processor, scaler=IdentityScaler(), policy=FakePolicy(),
        support=FakeSupport(), sink=sink,
        safe_fallback=dict.fromkeys(MDP_V1_ACTION_COLUMNS, 0.0), max_gap_minutes=10,
    )
    engine.step(event("2026-06-01 12:00:00")); engine.step(event("2026-06-01 12:05:00"))
    engine.step(event("2026-06-01 12:30:00"))
    assert processor.history_lengths == [1, 2, 1]


def test_sink_failure_is_cached_and_never_retried_for_same_timestamp():
    class FailingSink:
        def __init__(self): self.calls = 0
        def emit(self, result): self.calls += 1; raise TimeoutError
    sink = FailingSink()
    engine = OnlineInferenceEngine(
        mode="replay", preprocessor=FakePreprocessor(), scaler=IdentityScaler(),
        policy=FakePolicy(), support=FakeSupport(), sink=sink,
        safe_fallback=dict.fromkeys(MDP_V1_ACTION_COLUMNS, 0.0),
    )
    first = engine.step(event()); second = engine.step(event())
    assert first.failure_stage == "sink" and second.idempotent_replay and sink.calls == 1


def test_invalid_fallback_is_rejected():
    sink = RecordOnlyControlSink()
    bad = dict.fromkeys(MDP_V1_ACTION_COLUMNS, 0.0); bad["heat_run"] = bad["cool_run"] = 1.0
    with pytest.raises(ValueError, match="heating and cooling"):
        OnlineInferenceEngine(mode="replay", preprocessor=FakePreprocessor(), scaler=IdentityScaler(),
                              policy=FakePolicy(), support=FakeSupport(), sink=sink, safe_fallback=bad)
