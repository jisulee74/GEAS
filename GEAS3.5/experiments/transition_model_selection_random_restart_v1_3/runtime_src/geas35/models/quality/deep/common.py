"""Shared utilities for AI quality models.

The deep quality models use feature-level masked reconstruction: a lookback
window provides temporal context, and only selected observation cells at the
current timestep are masked and reconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from geas35.models.quality.base import validate_observation_columns


QUALITY_FLAG_SUFFIXES = (
    "_missing_flag",
    "_rule_outlier_flag",
    "_ai_outlier_flag",
    "_invalid_flag",
    "_ffill_flag",
)


class TorchUnavailableError(ImportError):
    """Raised when a deep quality model requires PyTorch but it is unavailable."""


@dataclass(frozen=True)
class SlidingWindow:
    """Row-position based lookback window whose last row is the current timestep."""

    positions: tuple[int, ...]

    @property
    def start_position(self) -> int:
        return self.positions[0]

    @property
    def end_position(self) -> int:
        return self.positions[-1]

    @property
    def current_position(self) -> int:
        return self.positions[-1]


@dataclass(frozen=True)
class FeatureMaskingConfig:
    """Configuration for current-timestep feature-level synthetic masking."""

    mask_fraction: float = 0.1
    random_state: int | None = 0
    min_unmasked_features_per_row: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.mask_fraction <= 1.0:
            raise ValueError("mask_fraction must be in the interval (0, 1].")
        if self.min_unmasked_features_per_row < 0:
            raise ValueError("min_unmasked_features_per_row must be >= 0.")


def require_torch():
    """Import PyTorch lazily and raise a clear optional-dependency error."""

    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise TorchUnavailableError(
            "PyTorch is required for deep quality models. Install torch before "
            "using ModernTCN, TimesNet, or PatchTST quality models."
        ) from exc
    return torch


def build_valid_observation_mask(
    df: pd.DataFrame,
    observation_columns: Iterable[str],
) -> pd.DataFrame:
    """Return normal, observed numeric cells eligible for masked reconstruction."""

    columns = validate_observation_columns(observation_columns)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        preview = ", ".join(missing[:5])
        if len(missing) > 5:
            preview = f"{preview}, ..."
        raise ValueError(f"Frame is missing observation columns: {preview}")

    mask = pd.DataFrame(True, index=df.index, columns=list(columns))
    for col in columns:
        mask[col] &= pd.to_numeric(df[col], errors="coerce").notna()
        for suffix in QUALITY_FLAG_SUFFIXES:
            flag_col = f"{col}{suffix}"
            if flag_col in df.columns:
                mask[col] &= ~(
                    pd.to_numeric(df[flag_col], errors="coerce")
                    .fillna(0)
                    .astype(bool)
                )
    return mask.astype(bool)


def build_sliding_windows(
    df: pd.DataFrame,
    *,
    lookback: int,
    time_column: str = "reg_date",
    group_columns: Iterable[str] = ("crop", "series_id", "segment_id", "episode_id"),
    expected_frequency: str | pd.Timedelta | None = None,
) -> list[SlidingWindow]:
    """Build row-position windows without crossing groups or time gaps.

    ``lookback`` is the number of rows in each window, including the current
    timestep. The reconstruction target for later models is always the last
    row of each window.
    """

    if lookback < 1:
        raise ValueError("lookback must be >= 1.")
    if df.empty:
        return []

    grouped_positions = _grouped_positions(df, group_columns)
    windows: list[SlidingWindow] = []
    for positions in grouped_positions:
        for contiguous in _split_contiguous_positions(
            df,
            positions,
            time_column=time_column,
            expected_frequency=expected_frequency,
        ):
            if len(contiguous) < lookback:
                continue
            for start in range(0, len(contiguous) - lookback + 1):
                window_positions = tuple(int(pos) for pos in contiguous[start : start + lookback])
                windows.append(SlidingWindow(positions=window_positions))
    return windows


def make_current_timestep_feature_mask(
    df: pd.DataFrame,
    observation_columns: Iterable[str],
    *,
    windows: Sequence[SlidingWindow],
    valid_mask: pd.DataFrame | None = None,
    config: FeatureMaskingConfig | None = None,
) -> pd.DataFrame:
    """Mask feature-level observation cells only at each window's current row."""

    columns = validate_observation_columns(observation_columns)
    cfg = config or FeatureMaskingConfig()
    valid = (
        build_valid_observation_mask(df, columns)
        if valid_mask is None
        else valid_mask.loc[:, list(columns)].astype(bool)
    )
    mask = pd.DataFrame(False, index=df.index, columns=list(columns))
    if not windows:
        return mask

    rng = np.random.default_rng(cfg.random_state)
    values = mask.to_numpy(dtype=bool).copy()
    valid_values = valid.to_numpy(dtype=bool)
    n_columns = len(columns)

    for window in windows:
        current = window.current_position
        candidates = np.flatnonzero(valid_values[current])
        if len(candidates) == 0:
            continue
        max_mask = max(0, n_columns - cfg.min_unmasked_features_per_row)
        n_mask = max(1, int(round(len(candidates) * cfg.mask_fraction)))
        if max_mask > 0:
            n_mask = min(n_mask, max_mask)
        else:
            n_mask = min(n_mask, len(candidates))
        if n_mask <= 0:
            continue
        selected = rng.choice(candidates, size=n_mask, replace=False)
        values[current, selected] = True

    return pd.DataFrame(values, index=df.index, columns=list(columns))


def apply_feature_mask(
    df: pd.DataFrame,
    feature_mask: pd.DataFrame,
    observation_columns: Iterable[str],
    *,
    fill_value: object = np.nan,
) -> pd.DataFrame:
    """Return a copy with selected observation cells replaced by ``fill_value``."""

    columns = validate_observation_columns(observation_columns)
    result = df.copy()
    for col in columns:
        if col not in result.columns or col not in feature_mask.columns:
            continue
        result.loc[feature_mask[col].astype(bool), col] = fill_value
    return result


def current_timestep_loss_mask(
    feature_mask: pd.DataFrame,
    valid_mask: pd.DataFrame,
    observation_columns: Iterable[str],
) -> pd.DataFrame:
    """Return loss mask for valid current-timestep cells selected for masking."""

    columns = validate_observation_columns(observation_columns)
    return (
        feature_mask.loc[:, list(columns)].astype(bool)
        & valid_mask.loc[:, list(columns)].astype(bool)
    )


def score_iqr_scale(
    scores: pd.DataFrame | pd.Series,
    *,
    epsilon: float = 1e-6,
) -> pd.Series | float:
    """Return robust IQR scale for confidence calibration."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if isinstance(scores, pd.Series):
        numeric = pd.to_numeric(scores, errors="coerce")
        q75 = numeric.quantile(0.75)
        q25 = numeric.quantile(0.25)
        scale = float(q75 - q25)
        return max(scale, epsilon) if np.isfinite(scale) else epsilon

    numeric_df = scores.apply(pd.to_numeric, errors="coerce")
    scale = numeric_df.quantile(0.75) - numeric_df.quantile(0.25)
    scale = scale.astype(float).where(np.isfinite(scale), epsilon)
    return scale.clip(lower=epsilon)


def confidence_from_scores(
    scores: pd.DataFrame | pd.Series,
    thresholds: float | pd.Series | dict[str, float],
    scales: float | pd.Series | dict[str, float],
    *,
    epsilon: float = 1e-6,
) -> pd.DataFrame | pd.Series:
    """Convert anomaly-score margins to invalid-decision confidence.

    confidence = sigmoid((score - threshold) / max(scale, epsilon))
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if isinstance(scores, pd.Series):
        threshold = _resolve_named_value(thresholds, scores.name)
        scale = max(_resolve_named_value(scales, scores.name), epsilon)
        return _sigmoid((pd.to_numeric(scores, errors="coerce") - threshold) / scale)

    numeric_scores = scores.apply(pd.to_numeric, errors="coerce")
    result = pd.DataFrame(index=scores.index, columns=scores.columns, dtype="float64")
    for col in scores.columns:
        threshold = _resolve_named_value(thresholds, str(col))
        scale = max(_resolve_named_value(scales, str(col)), epsilon)
        result[col] = _sigmoid((numeric_scores[col] - threshold) / scale)
    return result


def _grouped_positions(
    df: pd.DataFrame,
    group_columns: Iterable[str],
) -> list[np.ndarray]:
    existing_group_columns = [col for col in group_columns if col in df.columns]
    if not existing_group_columns:
        return [np.arange(len(df), dtype=int)]

    temp = df.loc[:, existing_group_columns].copy()
    temp["_row_position"] = np.arange(len(df), dtype=int)
    grouped = temp.groupby(existing_group_columns, sort=False, dropna=False)
    return [group["_row_position"].to_numpy(dtype=int) for _, group in grouped]


def _split_contiguous_positions(
    df: pd.DataFrame,
    positions: np.ndarray,
    *,
    time_column: str,
    expected_frequency: str | pd.Timedelta | None,
) -> list[np.ndarray]:
    if expected_frequency is None or time_column not in df.columns or len(positions) <= 1:
        return [positions]

    expected_delta = pd.Timedelta(expected_frequency)
    times = pd.to_datetime(df.iloc[positions][time_column], errors="coerce")
    segments: list[list[int]] = [[int(positions[0])]]
    previous = times.iloc[0]
    for pos, current in zip(positions[1:], times.iloc[1:]):
        if pd.isna(previous) or pd.isna(current) or current - previous != expected_delta:
            segments.append([int(pos)])
        else:
            segments[-1].append(int(pos))
        previous = current
    return [np.asarray(segment, dtype=int) for segment in segments]


def _resolve_named_value(
    value: float | pd.Series | dict[str, float],
    column: str | None,
) -> float:
    if isinstance(value, pd.Series):
        if column is not None and column in value.index:
            resolved = value.loc[column]
        elif "default" in value.index:
            resolved = value.loc["default"]
        else:
            resolved = value.iloc[0]
    elif isinstance(value, dict):
        if column is not None and column in value:
            resolved = value[column]
        elif "default" in value:
            resolved = value["default"]
        else:
            resolved = next(iter(value.values()))
    else:
        resolved = value

    numeric = pd.to_numeric(resolved, errors="coerce")
    if pd.isna(numeric):
        raise ValueError(f"Could not resolve numeric value for column {column!r}.")
    return float(numeric)


def _sigmoid(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    clipped = numeric.clip(lower=-60.0, upper=60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


__all__ = [
    "FeatureMaskingConfig",
    "SlidingWindow",
    "TorchUnavailableError",
    "apply_feature_mask",
    "build_sliding_windows",
    "build_valid_observation_mask",
    "confidence_from_scores",
    "current_timestep_loss_mask",
    "make_current_timestep_feature_mask",
    "require_torch",
    "score_iqr_scale",
]
