from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


COMMON_THETA_RANGES: Dict[str, tuple[float, float]] = {
    "UA": (50.0, 2.0e4),
    "C": (1.0e5, 1.0e9),
    "eta": (0.0, 2.0),
    "k_heat": (0.0, 2.0e5),
    "a0": (0.0, 5.0),
    "a1": (0.0, 100.0),
    "a2": (0.0, 20.0),
    "a3": (0.0, 20.0),
}


VALIDATION_RATE_THRESHOLDS: Dict[str, float] = {
    "temp_viol_rate": 0.20,
    "cond_viol_rate": 0.20,
    "rh_viol_rate": 0.25,
    "vpd_viol_rate": 0.25,
    "hv_ineff_rate": 0.35,
}


ROBUST_RATE_THRESHOLDS: Dict[str, float] = {
    "q90.temp_viol_rate": 0.25,
    "q90.cond_viol_rate": 0.25,
    "cvar90.temp_viol_rate": 0.35,
    "cvar90.cond_viol_rate": 0.35,
    "cvar90.hv_ineff_rate": 0.45,
}


FIT_THRESHOLDS: Dict[str, float] = {
    "loss": 1.0e6,
    "value": 1.0e6,
}


def _resolve_geas_root() -> Path:
    return Path(os.getenv("GEAS_PROJECT_ROOT", "/home/ljs/jslee/GEAS ver3.0"))


def _load_store_api():
    geas_root = _resolve_geas_root()
    if str(geas_root) not in sys.path:
        sys.path.insert(0, str(geas_root))
    from physics_store import PhysicsStore, save_backend_params
    return PhysicsStore, save_backend_params


def _store_path(path: Optional[str | Path] = None) -> Path:
    if path is not None:
        return Path(path)
    return _resolve_geas_root() / "physics_store" / "physics_params_store.json"


def _is_finite_number(x: Any) -> bool:
    try:
        x = float(x)
    except Exception:
        return False
    return x == x and x not in (float("inf"), float("-inf"))


def _lookup_nested_metric(metadata: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = metadata
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def evaluate_theta_candidate(
    theta: Mapping[str, Any],
    *,
    data_case: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    reasons: list[str] = []
    metrics: Dict[str, Any] = {}

    for key, (lo, hi) in COMMON_THETA_RANGES.items():
        val = theta.get(key)
        metrics[key] = val
        if val is None or not _is_finite_number(val):
            reasons.append(f"missing_or_nonfinite:{key}")
            continue
        fval = float(val)
        if not (lo <= fval <= hi):
            reasons.append(f"out_of_range:{key}")

    if float(theta.get("a1", 0.0)) + float(theta.get("a2", 0.0)) <= 0.0:
        reasons.append("nonpositive_ventilation_gain")

    case = str(data_case).lower()
    n_obs = metadata.get("n_obs")
    if n_obs is not None:
        metrics["n_obs"] = int(n_obs)

    for metric_name, threshold in VALIDATION_RATE_THRESHOLDS.items():
        val = metadata.get(metric_name)
        if val is None:
            continue
        metrics[metric_name] = val
        if (not _is_finite_number(val)) or float(val) > float(threshold):
            reasons.append(f"threshold_exceeded:{metric_name}")

    for dotted_key, threshold in ROBUST_RATE_THRESHOLDS.items():
        val = _lookup_nested_metric(metadata, dotted_key)
        if val is None:
            continue
        metrics[dotted_key] = val
        if (not _is_finite_number(val)) or float(val) > float(threshold):
            reasons.append(f"threshold_exceeded:{dotted_key}")

    if case == "sufficient":
        if n_obs is not None and int(n_obs) < 72:
            reasons.append("insufficient_observations_for_sufficient_case")
        for metric_name in ("value", "loss"):
            if metric_name in metadata:
                val = metadata.get(metric_name)
                metrics[metric_name] = val
                if not _is_finite_number(val):
                    reasons.append(f"nonfinite_{metric_name}")
                elif float(val) > FIT_THRESHOLDS[metric_name]:
                    reasons.append(f"threshold_exceeded:{metric_name}")
        conv = metadata.get("convergence", metadata.get("conv"))
        if conv is not None:
            metrics["convergence"] = conv
            try:
                if int(conv) < 0:
                    reasons.append("invalid_convergence")
            except Exception:
                reasons.append("invalid_convergence")

    approved = len(reasons) == 0
    return {
        "approved": approved,
        "data_case": case,
        "reasons": reasons,
        "metrics": metrics,
    }


def approve_and_save_theta_case(
    theta: Mapping[str, Any],
    *,
    farm_sn: Optional[int] = None,
    stage_name: Optional[str] = None,
    data_case: str,
    model_name: str = "default",
    store_path: Optional[str | Path] = None,
    source: str = "modeling",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    approval = evaluate_theta_candidate(theta, data_case=data_case, metadata=metadata)
    out: Dict[str, Any] = {"approval": approval}
    if not approval["approved"]:
        return out

    record = save_theta_case(
        theta,
        farm_sn=farm_sn,
        stage_name=stage_name,
        data_case=data_case,
        model_name=model_name,
        store_path=store_path,
        source=source,
        metadata={**metadata, "approval": approval},
    )
    out["record"] = record
    return out


def approve_and_save_identification_result(
    result: Mapping[str, Any],
    *,
    farm_sn: Optional[int] = None,
    stage_name: Optional[str] = None,
    data_case: str = "sufficient",
    model_name: str = "default",
    store_path: Optional[str | Path] = None,
    source: str = "modeling_identification",
    metadata: Optional[Dict[str, Any]] = None,
    ach_defaults: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    par_est = result.get("par_est")
    if par_est is None:
        raise ValueError("result['par_est'] is required for approve_and_save_identification_result")

    vals: Sequence[float] = list(par_est)
    if len(vals) < 5:
        raise ValueError("par_est must contain at least 5 values: C, k_heat, eta, UA, Kvent")

    ach_defaults = dict(ach_defaults or {})
    mapped = {
        "C": float(vals[0]),
        "k_heat": float(vals[1]),
        "eta": float(vals[2]),
        "UA": float(vals[3]),
        "a0": float(ach_defaults.get("a0", 0.05)),
        "a1": float(vals[4]),
        "a2": float(ach_defaults.get("a2", 0.10)),
        "a3": float(ach_defaults.get("a3", 0.00)),
    }

    extra_meta = {
        "convergence": result.get("convergence"),
        "value": result.get("value"),
        "raw_par_est": list(vals),
    }
    if metadata:
        extra_meta.update(metadata)

    return approve_and_save_theta_case(
        mapped,
        farm_sn=farm_sn,
        stage_name=stage_name,
        data_case=data_case,
        model_name=model_name,
        store_path=store_path,
        source=source,
        metadata=extra_meta,
    )


def save_theta_case(
    theta: Mapping[str, Any],
    *,
    farm_sn: Optional[int] = None,
    stage_name: Optional[str] = None,
    data_case: str,
    model_name: str = "default",
    store_path: Optional[str | Path] = None,
    source: str = "modeling",
    metadata: Optional[Dict[str, Any]] = None,
):
    PhysicsStore, save_backend_params = _load_store_api()
    store = PhysicsStore(_store_path(store_path))
    return save_backend_params(
        store,
        backend_result=dict(theta),
        farm_sn=farm_sn,
        stage_name=stage_name,
        data_case=data_case,
        model_name=model_name,
        source=source,
        metadata=metadata,
    )


def save_identification_result(
    result: Mapping[str, Any],
    *,
    farm_sn: Optional[int] = None,
    stage_name: Optional[str] = None,
    data_case: str = "sufficient",
    model_name: str = "default",
    store_path: Optional[str | Path] = None,
    source: str = "modeling_identification",
    metadata: Optional[Dict[str, Any]] = None,
    ach_defaults: Optional[Mapping[str, float]] = None,
):
    approved = approve_and_save_identification_result(
        result,
        farm_sn=farm_sn,
        stage_name=stage_name,
        data_case=data_case,
        model_name=model_name,
        store_path=store_path,
        source=source,
        metadata=metadata,
        ach_defaults=ach_defaults,
    )
    if "record" not in approved:
        raise ValueError(f"identification result was not approved for storage: {approved['approval']['reasons']}")
    return approved["record"]
