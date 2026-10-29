"""Explicit feature contracts for transition-model ablation experiments."""
from __future__ import annotations
from enum import Enum
import hashlib
import json
from typing import Iterable
import pandas as pd
from geas35.rl import MDP_V1_OBSERVATION_COLUMNS, MdpV1ObservationScaler

class MdpObservationSchemaVersion(str, Enum):
    DEDUPLICATED_V2 = "geas35.mdp.observation.deduplicated.v2"

class TransitionFeatureVariant(str, Enum):
    DEDUPLICATED_WITH_QUALITY_FLAGS = "deduplicated_with_quality_flags"
    DEDUPLICATED_WITHOUT_QUALITY_FLAGS = "deduplicated_without_quality_flags"

class TransitionDiagnosticFeatureSchema(str, Enum):
    LEGACY_V1_1 = "legacy_v1_1"
    DEDUPLICATED_WITHOUT_CONDENSATION_PREDICTION = "deduplicated_without_condensation_prediction"
    DEDUPLICATED_WITH_CONDENSATION_PREDICTION = "deduplicated_with_condensation_prediction"

QUALITY_FLAG_COLUMNS = ("obs_quality_missing_imputed_flag", "obs_quality_outlier_flag")
LEGACY_CONDENSATION_PREDICTION_COLUMN = "dTcond_pred_10m"
CONDENSATION_PREDICTION_COLUMN = "obs_derived_condensation_margin_pred_10m"
EXPECTED_BASE_OBSERVATION_COUNT = 55
LEGACY_DUPLICATE_COLUMNS = (
    "obs_derived_vpd_kpa", "obs_derived_dewpoint_c",
    "obs_derived_condensation_margin_c",
)

def observation_columns_for_variant(variant: TransitionFeatureVariant | str) -> tuple[str, ...]:
    selected = TransitionFeatureVariant(variant)
    base = tuple(MDP_V1_OBSERVATION_COLUMNS)
    if len(base) != EXPECTED_BASE_OBSERVATION_COUNT:
        raise ValueError(f"Deduplicated MDP schema must contain {EXPECTED_BASE_OBSERVATION_COUNT} base observations, found {len(base)}.")
    return (*base, *QUALITY_FLAG_COLUMNS) if selected is TransitionFeatureVariant.DEDUPLICATED_WITH_QUALITY_FLAGS else base

def prepare_frame_for_feature_variant(frame: pd.DataFrame, variant: TransitionFeatureVariant | str) -> pd.DataFrame:
    output = frame.copy()
    if CONDENSATION_PREDICTION_COLUMN not in output:
        if LEGACY_CONDENSATION_PREDICTION_COLUMN not in output:
            raise ValueError(f"Missing causal condensation feature {CONDENSATION_PREDICTION_COLUMN!r}.")
        output[CONDENSATION_PREDICTION_COLUMN] = output[LEGACY_CONDENSATION_PREDICTION_COLUMN]
    missing = [column for column in observation_columns_for_variant(variant) if column not in output]
    if missing:
        raise ValueError(f"Feature variant is missing observations: {missing}")
    return output

def feature_variant_contract(variant: TransitionFeatureVariant | str, *, action_columns: Iterable[str]) -> dict[str, object]:
    selected = TransitionFeatureVariant(variant)
    observations = observation_columns_for_variant(selected)
    actions = tuple(action_columns)
    payload: dict[str, object] = {
        "mdp_observation_schema_version": MdpObservationSchemaVersion.DEDUPLICATED_V2.value,
        "feature_variant": selected.value,
        "quality_flags_included": selected is TransitionFeatureVariant.DEDUPLICATED_WITH_QUALITY_FLAGS,
        "observation_columns": list(observations), "action_columns": list(actions),
        "observation_count": len(observations), "input_dimension": len(observations) + len(actions),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["schema_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload

def project_frozen_observation_scaler(
    source: MdpV1ObservationScaler, observation_columns: Iterable[str]
) -> MdpV1ObservationScaler:
    """Project the original physical-unit Train scaler onto a feature schema.

    RL-ready parquet observations are already standardized. Re-fitting a scaler
    on those values destroys the physical-unit residual adapter contract.
    """
    requested = tuple(observation_columns)
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in requested:
        source_column = (
            LEGACY_CONDENSATION_PREDICTION_COLUMN
            if column == CONDENSATION_PREDICTION_COLUMN else column
        )
        if source_column in source.means and source_column in source.scales:
            means[column] = float(source.means[source_column])
            scales[column] = float(source.scales[source_column])
    required = {"obs_indoor_temp_c", "obs_indoor_humidity_pct", "obs_indoor_co2_ppm"}
    missing = required.difference(means)
    if missing:
        raise ValueError(f"Frozen scaler is missing transition targets: {sorted(missing)}")
    return MdpV1ObservationScaler(columns=tuple(means), means=means, scales=scales)

def observation_columns_for_diagnostic_schema(schema: TransitionDiagnosticFeatureSchema | str) -> tuple[str, ...]:
    selected = TransitionDiagnosticFeatureSchema(schema)
    base = tuple(MDP_V1_OBSERVATION_COLUMNS)
    without_prediction = tuple(c for c in base if c != CONDENSATION_PREDICTION_COLUMN)
    if selected is TransitionDiagnosticFeatureSchema.LEGACY_V1_1:
        return (*without_prediction, *LEGACY_DUPLICATE_COLUMNS, *QUALITY_FLAG_COLUMNS)
    if selected is TransitionDiagnosticFeatureSchema.DEDUPLICATED_WITHOUT_CONDENSATION_PREDICTION:
        return (*without_prediction, *QUALITY_FLAG_COLUMNS)
    return (*base, *QUALITY_FLAG_COLUMNS)

def diagnostic_feature_contract(schema: TransitionDiagnosticFeatureSchema | str, *, action_columns: Iterable[str]) -> dict[str, object]:
    selected = TransitionDiagnosticFeatureSchema(schema)
    observations = observation_columns_for_diagnostic_schema(selected)
    actions = tuple(action_columns)
    payload: dict[str, object] = {
        "diagnostic_schema": selected.value, "quality_flags_included": True,
        "observation_columns": list(observations), "action_columns": list(actions),
        "observation_count": len(observations),
        "input_dimension": len(observations) + len(actions),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["schema_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload

__all__ = ["CONDENSATION_PREDICTION_COLUMN", "LEGACY_DUPLICATE_COLUMNS", "MdpObservationSchemaVersion", "QUALITY_FLAG_COLUMNS", "TransitionDiagnosticFeatureSchema", "TransitionFeatureVariant", "diagnostic_feature_contract", "feature_variant_contract", "observation_columns_for_diagnostic_schema", "observation_columns_for_variant", "prepare_frame_for_feature_variant", "project_frozen_observation_scaler"]
