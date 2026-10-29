"""Deterministic run summaries for replay and shadow evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from geas35.realtime.contracts import InferenceResult


def summarize_inference_results(results: Iterable[InferenceResult]) -> dict[str, object]:
    rows = list(results); count = len(rows)
    latency_keys = sorted({key for row in rows for key in row.latencies_ms})
    latency = {}
    for key in latency_keys:
        values = np.asarray([row.latencies_ms[key] for row in rows if key in row.latencies_ms], dtype=float)
        finite = values[np.isfinite(values)]
        latency[key] = ({"p50_ms": float(np.quantile(finite, .5)),
                         "p95_ms": float(np.quantile(finite, .95)),
                         "max_ms": float(np.max(finite))} if finite.size else {})
    rate = lambda predicate: 0.0 if count == 0 else float(sum(predicate(row) for row in rows) / count)
    return {
        "steps": count,
        "valid_step_rate": rate(lambda row: row.status == "ok"),
        "quality_block_rate": rate(lambda row: row.failure_stage == "quality"),
        "fallback_rate": rate(lambda row: row.fallback_applied),
        "support_projection_rate": rate(lambda row: row.support_projected),
        "command_emission_rate": rate(lambda row: row.command_emitted),
        "latency": latency,
    }


def write_run_manifest(
    path: str | Path, *, results: Iterable[InferenceResult], mode: str,
    artifact_hashes: Mapping[str, str], config_sha256: str,
    code_version: str | None = None,
) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "geas35.realtime.run_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": mode, "code_version": code_version,
        "config_sha256": config_sha256,
        "artifact_hashes": dict(artifact_hashes),
        "summary": summarize_inference_results(results),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target); return target


__all__ = ["summarize_inference_results", "write_run_manifest"]
