"""Causal current-and-past-only preprocessing for online inference."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd

from geas35.models.quality import load_quality_model_application
from geas35.preprocessing import (
    add_growth_stage_columns_from_crop_cycles,
    canonicalize_feature_units,
    prepare_geas_input_schema,
    prepare_missing_outliers_handled_features,
)
from geas35.realtime.contracts import QualityResult
from geas35.rl import MDP_V1_OBSERVATION_COLUMNS, MdpV1Config, prepare_mdp_v1_frame


def load_observation_scaler(path: str | Path):
    from geas35.rl import MdpV1ObservationScaler
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("fit_split") != "train":
        raise ValueError("Online inference requires a frozen Train observation scaler.")
    return MdpV1ObservationScaler(
        columns=tuple(payload["columns"]), means=payload["mean"], scales=payload["scale"]
    )


def _add_representative_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the canonical indoor observations without hiding provenance."""
    output = frame.copy()
    specifications = (
        ("in_temp_representative", "in_temp", "in_temp2"),
        ("in_hum_representative", "in_hum", "in_hum2"),
        ("in_co2_representative", "in_co2", "in_co2_2"),
    )
    for target, primary, secondary in specifications:
        primary_values = pd.to_numeric(output.get(primary), errors="coerce")
        secondary_values = pd.to_numeric(output.get(secondary), errors="coerce")
        primary_invalid = primary_values.isna()
        secondary_invalid = secondary_values.isna()
        for suffix in ("_missing_flag", "_rule_outlier_flag", "_ai_outlier_flag"):
            if f"{primary}{suffix}" in output:
                primary_invalid |= pd.to_numeric(output[f"{primary}{suffix}"], errors="coerce").fillna(0).astype(bool)
            if f"{secondary}{suffix}" in output:
                secondary_invalid |= pd.to_numeric(output[f"{secondary}{suffix}"], errors="coerce").fillna(0).astype(bool)
        values = primary_values.mask(primary_invalid)
        values = values.where(values.notna(), secondary_values.mask(secondary_invalid))
        imputed_flag = f"{primary}_imputed_flag"
        if imputed_flag in output:
            use_imputed = values.isna() & pd.to_numeric(output[imputed_flag], errors="coerce").fillna(0).astype(bool)
            values.loc[use_imputed] = primary_values.loc[use_imputed]
        output[target] = values
    return output


class ArtifactQualityProcessor:
    """Apply one frozen quality-model application without fitting online."""

    def __init__(self, artifact_path: str | Path, *, required_columns: Iterable[str] = ()):
        application = load_quality_model_application(artifact_path)
        self.model = application.model
        self.thresholds = application.thresholds
        self.observation_columns = tuple(application.observation_columns or ())
        self.imputation_confidence_threshold = application.imputation_confidence_threshold
        self.required_columns = tuple(required_columns)

    def process(self, frame: pd.DataFrame, control_log: pd.DataFrame | None = None) -> QualityResult:
        started = perf_counter()
        output = prepare_missing_outliers_handled_features(
            frame, df_control_log=control_log, quality_model=self.model,
            thresholds=self.thresholds, observation_columns=self.observation_columns,
            imputation_columns=self.observation_columns, keep_extra_columns=True,
            imputation_confidence_threshold=(
                0.5 if self.imputation_confidence_threshold is None
                else self.imputation_confidence_threshold
            ),
        )
        output = _add_representative_observations(output)
        unresolved = tuple(
            column for column in self.required_columns
            if column not in output or pd.isna(output[column].iloc[-1])
        )
        return QualityResult(
            frame=output, unresolved_columns=unresolved, blocked=bool(unresolved),
            reason=("required_quality_values_unresolved" if unresolved else None),
            latency_ms=(perf_counter() - started) * 1000.0,
        )


class PassthroughQualityProcessor:
    """Test seam that preserves input while enforcing required values."""

    def __init__(self, required_columns: Iterable[str] = ()):
        self.required_columns = tuple(required_columns)

    def process(self, frame: pd.DataFrame, control_log: pd.DataFrame | None = None) -> QualityResult:
        del control_log
        unresolved = tuple(
            column for column in self.required_columns
            if column not in frame or pd.isna(frame[column].iloc[-1])
        )
        return QualityResult(
            frame=frame.copy(), unresolved_columns=unresolved, blocked=bool(unresolved),
            reason=("required_quality_values_unresolved" if unresolved else None),
        )


class CausalPreprocessor:
    def __init__(
        self, quality_processor, *, crop_cycles: pd.DataFrame | str | Path | None = None,
        history_rows: int = 288, mdp_config: MdpV1Config | None = None,
    ):
        if history_rows <= 0:
            raise ValueError("history_rows must be positive")
        self.quality_processor = quality_processor
        self.crop_cycles = crop_cycles
        self.history_rows = history_rows
        self.mdp_config = mdp_config or MdpV1Config()

    def process(self, history: pd.DataFrame, control_log: pd.DataFrame | None = None):
        frame = prepare_geas_input_schema(history, keep_extra_columns=True)
        frame = canonicalize_feature_units(frame)
        if self.crop_cycles is not None:
            frame = add_growth_stage_columns_from_crop_cycles(frame, self.crop_cycles)
        quality = self.quality_processor.process(frame, control_log)
        if quality.blocked:
            return quality, None, None
        # Only current and past rows are supplied. shift(-1) transition targets are
        # ignored; the final row observation is therefore causal.
        mdp = prepare_mdp_v1_frame(quality.frame, config=self.mdp_config)
        current = mdp.iloc[-1]
        columns = [column for column in MDP_V1_OBSERVATION_COLUMNS if column in current.index]
        unresolved = tuple(column for column in columns if pd.isna(current[column]))
        if unresolved:
            return QualityResult(
                frame=quality.frame, unresolved_columns=unresolved, blocked=True,
                reason="policy_observation_unresolved", latency_ms=quality.latency_ms,
            ), None, None
        return quality, current, mdp


__all__ = [
    "ArtifactQualityProcessor", "CausalPreprocessor", "PassthroughQualityProcessor",
    "load_observation_scaler",
]
