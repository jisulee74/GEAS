"""Second-stage local HPO proposal generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def build_local_hpo_params(
    top_params: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int = 43,
    forbidden: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], ...]:
    """Generate unique valid neighbors around the three best broad configurations."""

    if not top_params or count <= 0:
        return ()
    rng = np.random.default_rng(seed)
    seen = {_signature(item) for item in forbidden}
    output = []
    attempts = 0
    while len(output) < count and attempts < count * 200:
        attempts += 1
        parent = dict(top_params[attempts % min(3, len(top_params))])
        candidate = _perturb_mapping(parent, rng)
        candidate["random_state"] = seed
        signature = _signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        output.append(candidate)
    if len(output) != count:
        raise ValueError(f"Could not construct {count} unique local HPO trials.")
    return tuple(output)


def _perturb_mapping(values: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    out = {}
    for key, value in values.items():
        if key == "random_state":
            out[key] = 43
        elif key in {"n_jobs", "verbosity"}:
            out[key] = value
        elif isinstance(value, Mapping):
            out[key] = _perturb_mapping(value, rng)
        elif key == "hidden_layer_sizes" and isinstance(value, (list, tuple)):
            factor = float(rng.choice([0.75, 1.0, 1.25]))
            out[key] = tuple(max(8, int(round(int(width) * factor / 8) * 8)) for width in value)
        elif isinstance(value, bool) or value is None or isinstance(value, str):
            out[key] = value
        elif isinstance(value, int):
            step = int(rng.choice([-1, 0, 1]))
            if key in {"n_estimators", "max_iter"}:
                out[key] = max(10, int(round(value * (1.0 + 0.25 * step))))
            else:
                out[key] = max(1, value + step)
        elif isinstance(value, float):
            factor = float(rng.choice([0.5, 0.75, 1.0, 1.25, 2.0]))
            updated = value * factor if value != 0.0 else float(rng.choice([0.0, 0.001, 0.01]))
            if key in {"subsample", "colsample_bytree", "max_features"}:
                updated = min(1.0, max(0.1, updated))
            out[key] = float(updated)
        else:
            out[key] = value
    return out


def _signature(values: Mapping[str, Any]) -> str:
    return repr(_freeze(values))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


__all__ = ["build_local_hpo_params"]
