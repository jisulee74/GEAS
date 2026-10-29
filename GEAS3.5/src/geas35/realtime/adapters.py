"""Replay, database, record-only, and explicitly enabled HTTP adapters."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping
from urllib import request

import pandas as pd

from geas35.realtime.contracts import InferenceResult, ObservationEvent
from geas35.rl import MDP_V1_ACTION_COLUMNS


def _json_value(value: Any):
    if isinstance(value, pd.Timestamp): return value.isoformat()
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        try: return value.value
        except Exception: pass
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


class ReplayObservationSource:
    def __init__(self, frame: pd.DataFrame, *, time_column="reg_date"):
        if time_column not in frame.columns: raise KeyError(f"Missing timestamp column: {time_column}")
        self.frame = frame.copy(); self.time_column = time_column
        self.frame[time_column] = pd.to_datetime(self.frame[time_column], errors="coerce")
        self.frame = self.frame.dropna(subset=[time_column]).sort_values(time_column)

    def __iter__(self):
        for _, row in self.frame.iterrows():
            values = row.to_dict(); timestamp = pd.Timestamp(values.pop(self.time_column))
            yield ObservationEvent(
                timestamp=timestamp, values=values, received_at=timestamp,
                segment_id=(None if pd.isna(row.get("segment_id")) else str(row.get("segment_id"))),
                crop_cycle_id=(None if pd.isna(row.get("crop_cycle_id")) else str(row.get("crop_cycle_id"))),
            )


class DatabaseObservationSource:
    """Polling source with a caller-owned DB-API connection factory."""
    def __init__(
        self, connection_factory: Callable[[], Any], *, table: str, farm_column: str,
        farm_id: int, columns: Iterable[str], timestamp_column="reg_date",
        primary_key="idx", poll_seconds=5.0, retry_seconds=5.0,
    ):
        if not table.replace("_", "").isalnum(): raise ValueError("Unsafe table name")
        for name in [farm_column, timestamp_column, primary_key, *columns]:
            if not str(name).replace("_", "").isalnum(): raise ValueError("Unsafe column name")
        self.connection_factory = connection_factory; self.table = table
        self.farm_column = farm_column; self.farm_id = int(farm_id)
        self.columns = tuple(dict.fromkeys([timestamp_column, *columns]))
        self.timestamp_column = timestamp_column; self.primary_key = primary_key
        self.poll_seconds = float(poll_seconds); self.retry_seconds = float(retry_seconds)
        self.last_timestamp: pd.Timestamp | None = None; self.last_primary_key = -1

    def fetch_new(self, *, limit=1000) -> list[ObservationEvent]:
        if self.last_timestamp is None:
            where = f"`{self.farm_column}`=%s"; params = (self.farm_id, int(limit))
        else:
            where = (
                f"`{self.farm_column}`=%s AND (`{self.timestamp_column}`>%s OR "
                f"(`{self.timestamp_column}`=%s AND `{self.primary_key}`>%s))"
            )
            params = (self.farm_id, self.last_timestamp.to_pydatetime(), self.last_timestamp.to_pydatetime(), self.last_primary_key, int(limit))
        selected = ",".join(f"`{column}`" for column in (*self.columns, self.primary_key))
        sql = (
            f"SELECT {selected} FROM `{self.table}` WHERE {where} "
            f"ORDER BY `{self.timestamp_column}`,`{self.primary_key}` LIMIT %s"
        )
        connection = self.connection_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, params); names = [item[0] for item in cursor.description]
                rows = [dict(zip(names, row)) for row in cursor.fetchall()]
        finally:
            connection.close()
        events = []
        for row in rows:
            timestamp = pd.Timestamp(row.pop(self.timestamp_column)); pk = int(row.pop(self.primary_key))
            events.append(ObservationEvent(timestamp=timestamp, received_at=pd.Timestamp.now(), values=row))
            self.last_timestamp = timestamp; self.last_primary_key = pk
        return events

    def __iter__(self):
        while True:
            try:
                rows = self.fetch_new()
                if rows:
                    yield from rows
                else:
                    time.sleep(self.poll_seconds)
            except Exception:
                time.sleep(self.retry_seconds)


class RecordOnlyControlSink:
    def __init__(self, output_path: str | Path | None = None):
        self.output_path = None if output_path is None else Path(output_path)
        self.records: list[InferenceResult] = []; self._timestamps: set[str] = set()

    def emit(self, result: InferenceResult) -> bool:
        key = result.timestamp.isoformat()
        if key in self._timestamps: return False
        self._timestamps.add(key)
        recorded = result.__class__(**{**asdict(result), "command_emitted": True})
        self.records.append(recorded)
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(recorded), default=_json_value, sort_keys=True) + "\n")
        return True


class HttpControlSink:
    """No-retry device sender; construction requires explicit live confirmation."""
    def __init__(
        self, *, endpoint: str, auth_endpoint: str | None, credentials: Mapping[str, str] | None,
        device_mapping: Mapping[str, Iterable[Mapping[str, Any]]],
        live_enabled: bool, confirm_live_control: bool, timeout_seconds=5.0,
    ):
        if not (live_enabled and confirm_live_control):
            raise PermissionError("HTTP control requires both live gates")
        if not endpoint or not device_mapping: raise ValueError("Endpoint and device mapping are required")
        missing_actions = sorted(set(MDP_V1_ACTION_COLUMNS).difference(device_mapping))
        if missing_actions:
            raise ValueError("Device mapping is incomplete: " + ", ".join(missing_actions))
        self.endpoint = endpoint; self.auth_endpoint = auth_endpoint
        self.credentials = None if credentials is None else dict(credentials)
        self.device_mapping = {key: tuple(value) for key, value in device_mapping.items()}
        self.timeout_seconds = float(timeout_seconds); self._timestamps: set[str] = set()

    def _payloads(self, action: Mapping[str, float]):
        payloads = []
        for action_name, mappings in self.device_mapping.items():
            if action_name not in action: raise KeyError(f"Unmapped action: {action_name}")
            for mapping in mappings:
                payload = dict(mapping); scale = float(payload.pop("scale", 1.0))
                payload["target"] = str(round(float(action[action_name]) * scale))
                payloads.append(payload)
        return payloads

    def emit(self, result: InferenceResult) -> bool:
        if result.executed_action is None: return False
        key = result.timestamp.isoformat()
        if key in self._timestamps: return False
        opener = request.build_opener(request.HTTPCookieProcessor())
        headers = {"Content-Type": "application/json"}
        if self.auth_endpoint is not None:
            auth = request.Request(self.auth_endpoint, data=json.dumps(self.credentials or {}).encode(), headers=headers)
            with opener.open(auth, timeout=self.timeout_seconds): pass
        for payload in self._payloads(result.executed_action):
            outgoing = request.Request(self.endpoint, data=json.dumps(payload).encode(), headers=headers)
            with opener.open(outgoing, timeout=self.timeout_seconds) as response:
                if response.status < 200 or response.status >= 300: return False
        self._timestamps.add(key)
        return True


def run_inference_source(engine, source) -> list[InferenceResult]:
    return [engine.step(event) for event in source]


__all__ = [
    "DatabaseObservationSource", "HttpControlSink", "RecordOnlyControlSink",
    "ReplayObservationSource", "run_inference_source",
]
