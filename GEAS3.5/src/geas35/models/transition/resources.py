"""Resource benchmarking helpers for transition candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile

import numpy as np

from geas35.models.transition.base import BaseTransitionModel, TransitionDataset


DEFAULT_INFERENCE_BENCHMARK_REPEATS = 5
DEFAULT_INFERENCE_BENCHMARK_ROWS = 8


def benchmark_transition_resources(
    model: BaseTransitionModel,
    dataset: TransitionDataset,
    *,
    artifact_dir: Path,
    model_path: Path,
    training_time_seconds: float,
    hpo_total_time_seconds: float,
    inference_repeats: int = DEFAULT_INFERENCE_BENCHMARK_REPEATS,
    inference_rows: int = DEFAULT_INFERENCE_BENCHMARK_ROWS,
) -> dict[str, Any]:
    """Measure Step 6 resource metrics for one saved transition candidate."""

    serialized_model_size_bytes = _file_size(model_path)
    artifact_dir_size_bytes = _directory_size(artifact_dir)
    inference = _subprocess_inference_benchmark(
        model_path=model_path,
        dataset=dataset,
        repeats=inference_repeats,
        rows=inference_rows,
    )
    return {
        "stage": "transition_candidate_resource_benchmark",
        "protocol": {
            "inference_benchmark": "subprocess_cold_load_predict",
            "inference_repeats": int(inference_repeats),
            "inference_rows": int(min(inference_rows, len(dataset.x.index))),
            "time_unit": "seconds",
            "latency_unit": "milliseconds",
            "memory_unit": "bytes",
            "resource_metrics_used_in_validation_score": False,
            "test_used_for_resource_benchmark": False,
        },
        "training_time_seconds": _finite_or_none(training_time_seconds),
        "hpo_total_time_seconds": _finite_or_none(hpo_total_time_seconds),
        "inference_latency_median_ms": inference.get("latency_median_ms"),
        "inference_latency_p95_ms": inference.get("latency_p95_ms"),
        "peak_cpu_rss_bytes": inference.get("peak_cpu_rss_bytes"),
        "peak_gpu_allocated_bytes": None,
        "serialized_model_size_bytes": int(serialized_model_size_bytes),
        "artifact_dir_size_bytes": int(artifact_dir_size_bytes),
        "benchmark_status": inference.get("status", "unknown"),
        "benchmark_reason": inference.get("reason"),
        "device": _device_payload(),
        "library_versions": _library_versions(),
    }


def _subprocess_inference_benchmark(
    *,
    model_path: Path,
    dataset: TransitionDataset,
    repeats: int,
    rows: int,
) -> dict[str, Any]:
    if repeats <= 0:
        return {"status": "skipped", "reason": "inference_repeats_not_positive"}
    if dataset.x.empty:
        return {"status": "skipped", "reason": "dataset_empty"}
    sample = dataset.x.loc[:, list(dataset.input_columns)].head(max(1, rows)).copy()
    with tempfile.TemporaryDirectory(prefix="geas_transition_resource_") as tmp:
        sample_path = Path(tmp) / "sample.pkl"
        sample_path.write_bytes(pickle.dumps(sample))
        code = _subprocess_code()
        env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parents[3])
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            src_root
            if not existing_pythonpath
            else os.pathsep.join([src_root, existing_pythonpath])
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(model_path),
                str(sample_path),
                str(int(repeats)),
            ],
            cwd=str(Path(__file__).resolve().parents[4]),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "reason": completed.stderr.strip()[:1000] or "subprocess_failed",
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {"status": "failed", "reason": f"invalid_subprocess_json: {exc}"}
    payload["status"] = "completed"
    payload.setdefault("reason", None)
    return payload


def _subprocess_code() -> str:
    return r"""
import json
import pickle
import statistics
import sys
import time
import tracemalloc

model_path, sample_path, repeats_text = sys.argv[1:4]
repeats = int(repeats_text)
with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(sample_path, "rb") as f:
    sample = pickle.load(f)
tracemalloc.start()
model.predict(sample)
latencies = []
for _ in range(repeats):
    start = time.perf_counter()
    model.predict(sample)
    latencies.append((time.perf_counter() - start) * 1000.0)
_, peak = tracemalloc.get_traced_memory()
rss = None
try:
    import psutil
    rss = int(psutil.Process().memory_info().rss)
except Exception:
    rss = int(peak)
latencies_sorted = sorted(latencies)
if len(latencies_sorted) == 1:
    p95 = latencies_sorted[0]
else:
    p95 = statistics.quantiles(latencies_sorted, n=20, method="inclusive")[18]
print(json.dumps({
    "latency_median_ms": float(statistics.median(latencies_sorted)),
    "latency_p95_ms": float(p95),
    "peak_cpu_rss_bytes": rss,
}))
"""


def _device_payload() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpu_available": _torch_gpu_available(),
        "gpu_name": _torch_gpu_name(),
    }


def _library_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "numpy": np.__version__,
    }
    for module_name in ("pandas", "sklearn", "lightgbm", "xgboost", "psutil"):
        try:
            module = __import__(module_name)
        except Exception:
            versions[module_name] = None
        else:
            versions[module_name] = getattr(module, "__version__", "unknown")
    return versions


def _torch_gpu_available() -> bool | None:
    try:
        import torch
    except Exception:
        return None
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return None


def _torch_gpu_name() -> str | None:
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        return None
    return None


def _file_size(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            total += int(item.stat().st_size)
    return total


def _finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


__all__ = [
    "DEFAULT_INFERENCE_BENCHMARK_REPEATS",
    "DEFAULT_INFERENCE_BENCHMARK_ROWS",
    "benchmark_transition_resources",
]
