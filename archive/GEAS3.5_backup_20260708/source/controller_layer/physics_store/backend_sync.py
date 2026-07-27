from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .repository import PhysicsRecord, PhysicsStore


ALIASES = {
    "UA": ("UA", "ua"),
    "C": ("C", "c", "capacitance"),
    "eta": ("eta", "g_solar", "solar_gain", "solar_efficiency"),
    "a0": ("a0",),
    "a1": ("a1",),
    "a2": ("a2",),
    "a3": ("a3",),
    "k_heat": ("k_heat", "Q_heat", "heater_power"),
    "k_cool": ("k_cool", "Q_cool", "cooler_power"),
    "rho_cp": ("rho_cp",),
    "A": ("A", "area_m2", "greenhouse_area_m2"),
    "V": ("V", "volume_m3", "greenhouse_volume_m3"),
    "dt_sec": ("dt_sec", "sample_time_sec", "control_interval_sec"),
    "k_shade": ("k_shade", "shade_factor"),
    "k_thermal": ("k_thermal", "thermal_factor"),
}


def normalize_backend_params(raw: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    if not isinstance(raw, Mapping):
        return normalized

    ach_coef = raw.get("ACH_coef")
    if isinstance(ach_coef, Mapping):
        for key in ("a0", "a1", "a2", "a3"):
            value = ach_coef.get(key)
            if value is not None:
                normalized[key] = value

    for target, aliases in ALIASES.items():
        for alias in aliases:
            if alias in raw and raw[alias] is not None:
                normalized[target] = raw[alias]
                break
    return normalized


def save_backend_params(
    store: PhysicsStore,
    *,
    backend_result: Mapping[str, Any],
    farm_sn: Optional[int] = None,
    stage_name: Optional[str] = None,
    data_case: Optional[str] = None,
    model_name: Optional[str] = None,
    valid_from: Any = None,
    valid_to: Any = None,
    source: str = 'backend',
    metadata: Optional[Dict[str, Any]] = None,
) -> PhysicsRecord:
    params = normalize_backend_params(backend_result)
    metadata = dict(metadata or {})
    if data_case is not None:
        metadata.setdefault('data_case', data_case)
    return store.save_record(
        farm_sn=farm_sn,
        stage_name=stage_name,
        model_name=model_name,
        valid_from=valid_from,
        valid_to=valid_to,
        params=params,
        source=source,
        metadata=metadata,
    )
