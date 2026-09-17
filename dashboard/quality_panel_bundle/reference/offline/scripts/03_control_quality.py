"""Apply GEAS quality control to offline split datasets.

This script is the offline dataset preparation entry point for the shared
preprocessing modules:

1. input schema preparation;
2. unit canonicalization;
3. combined missing/outlier handling.

By default, this entry point applies the already selected crop-specific model
and validation-calibrated per-variable thresholds to train/validation. Model
selection, HPO, and threshold calibration remain outside this script.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dataset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets"


DEFAULT_INPUT_DIR_NAME = "02_split"
DEFAULT_OUTPUT_DIR_NAME = "03_quality_controlled"
DEFAULT_SPLITS = ("train", "validation")
DEFAULT_GROWTH_STAGE_SPLITS = ("train", "validation")
SUPPORTED_GROWTH_STAGE_SPLITS = (*DEFAULT_GROWTH_STAGE_SPLITS, "test")
DEFAULT_CROP_CYCLE_MANIFEST_NAME = "crop_cycle_manifest.csv"
DEFAULT_SELECTED_MODELS = {
    "strawberry": "modern_tcn",
    "melon": "modern_tcn",
    "cucumber": "patch_tst",
}
PROVISIONAL_SHORT_GAP_MAX_ROWS = 3
QUALITY_UNCONFIRMED_FLAG_COLUMN = "quality_unconfirmed_flag"

CONTINUOUS_EXTERNAL_COLUMNS = (
    "out_temp", "out_hum", "out_windsp", "out_rainfall", "out_airpress",
)
CIRCULAR_EXTERNAL_COLUMNS = ("out_winddirec",)
SOLAR_EXTERNAL_COLUMNS = ("out_light", "out_light_sum")
STATE_PERSISTENCE_COLUMNS = ("out_rain", "etc_plc_norm")

RULE_BASED_OUTLIER_COLUMNS = (
    "in_temp", "in_temp2", "in_hum", "in_hum2", "in_co2", "in_co2_2",
    "out_temp", "out_winddirec", "out_windsp", "out_rain", "out_light",
    "out_light_sum", "in_medium_hum1", "in_medium_hum2",
    "in_medium_temp1", "in_medium_temp2",
)
AI_BASED_OUTLIER_COLUMNS = (
    "in_medium_temp1", "in_temp", "in_temp2", "in_hum", "in_hum2",
    "in_medium_hum1", "in_co2", "in_co2_2",
)
# Sensor2 values are diagnosed but kept untouched for audit/fallback use.
AI_IMPUTATION_COLUMNS = AI_BASED_OUTLIER_COLUMNS
REPRESENTATIVE_SENSOR_SPECS = (
    ("in_temp_representative", "in_temp", "in_temp2", True),
    ("in_hum_representative", "in_hum", "in_hum2", True),
    ("in_co2_representative", "in_co2", "in_co2_2", True),
    # Substrate sensor2 is retained for audit only because of long inactive periods.
    ("in_medium_temp_representative", "in_medium_temp1", None, True),
    ("in_medium_hum_representative", "in_medium_hum1", None, True),
    ("in_medium_ec_representative", "in_medium_ec1", None, False),
)


def _usable_sensor_value(df: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    raw_column = f"{column}_raw_value"
    values = pd.to_numeric(
        df[raw_column] if raw_column in df.columns else df[column],
        errors="coerce",
    )
    invalid = values.isna()
    for suffix in ("_missing_flag", "_rule_outlier_flag", "_ai_outlier_flag"):
        flag_column = f"{column}{suffix}"
        if flag_column in df.columns:
            invalid |= (
                pd.to_numeric(df[flag_column], errors="coerce")
                .fillna(0)
                .astype(bool)
            )
    return values, ~invalid


def _add_representative_sensor_values(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Create one MDP-ready value per duplicated physical sensor quantity."""

    df = df_raw.copy()
    for output_column, primary, secondary, allow_ai_fallback in REPRESENTATIVE_SENSOR_SPECS:
        primary_values, primary_usable = _usable_sensor_value(df, primary)
        representative = pd.Series(float("nan"), index=df.index, dtype="Float64")
        source = pd.Series("unavailable", index=df.index, dtype="string")

        representative.loc[primary_usable] = primary_values.loc[primary_usable]
        source.loc[primary_usable] = "sensor1"

        if secondary is not None:
            secondary_values, secondary_usable = _usable_sensor_value(df, secondary)
            use_secondary = ~primary_usable & secondary_usable
            representative.loc[use_secondary] = secondary_values.loc[use_secondary]
            source.loc[use_secondary] = "sensor2"
        else:
            use_secondary = pd.Series(False, index=df.index)

        imputed_flag = f"{primary}_imputed_flag"
        if allow_ai_fallback and imputed_flag in df.columns:
            use_ai = (
                ~primary_usable
                & ~use_secondary
                & pd.to_numeric(df[imputed_flag], errors="coerce")
                .fillna(0)
                .astype(bool)
                & pd.to_numeric(df[primary], errors="coerce").notna()
            )
            representative.loc[use_ai] = pd.to_numeric(
                df.loc[use_ai, primary], errors="coerce"
            )
            source.loc[use_ai] = "ai_imputed_sensor1"

        df[output_column] = representative
        df[f"{output_column}_source"] = source
    return df


@dataclass(frozen=True)
class QualityControlledSplitSummary:
    crop: str
    split: str
    input_path: str
    output_path: str
    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    missing_cells: int
    restored_action_cells: int
    rule_outlier_rows: int
    rule_outlier_cells: int
    ai_outlier_rows: int
    invalid_rows: int
    invalid_cells: int
    imputed_rows: int
    imputed_cells: int
    provisional_imputed_cells: int
    unconfirmed_rows: int
    unconfirmed_cells: int


@dataclass(frozen=True)
class QualityControlApplication:
    quality_model: object | None
    thresholds: object | None
    observation_columns: tuple[str, ...] | None
    imputation_confidence_threshold: float | None
    artifact_path: Path | None
    raw_payload: dict[str, object] | None


def _quality_artifact_root() -> Path:
    return (
        _project_root()
        / "experiments"
        / "quality_control_model_selection"
        / "artifacts"
    )


def _selected_model_artifact(crop: str) -> Path:
    if crop not in DEFAULT_SELECTED_MODELS:
        raise KeyError(f"No selected quality model is configured for crop: {crop}")
    return (
        _quality_artifact_root()
        / crop
        / DEFAULT_SELECTED_MODELS[crop]
        / "quality_model_application.json"
    )


def _ensure_src_on_path() -> None:
    src_path = _project_root() / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _rebuild_loaded_torch_network(model: object) -> object:
    """Rebind a pickled local Torch network while preserving learned weights."""

    stored_network = getattr(model, "model_", None)
    observation_columns = tuple(getattr(model, "observation_columns_", ()) or ())
    build_network = getattr(model, "_build_network", None)
    if stored_network is None or not observation_columns or not callable(build_network):
        return model

    from geas35.models.quality.deep import require_torch

    torch = require_torch()
    rebuilt = build_network(torch, torch.nn, n_features=len(observation_columns))
    rebuilt.load_state_dict(stored_network.state_dict(), strict=True)
    try:
        device = next(stored_network.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    rebuilt.to(device)
    rebuilt.eval()
    model.model_ = rebuilt
    return model


def _load_quality_control_application(
    artifact_path: Path | None,
) -> QualityControlApplication:
    if artifact_path is None:
        return QualityControlApplication(
            quality_model=None,
            thresholds=None,
            observation_columns=None,
            imputation_confidence_threshold=None,
            artifact_path=None,
            raw_payload=None,
        )

    from geas35.models.quality import load_quality_model_application

    application = load_quality_model_application(artifact_path)
    quality_model = _rebuild_loaded_torch_network(application.model)
    return QualityControlApplication(
        quality_model=quality_model,
        thresholds=application.thresholds,
        observation_columns=application.observation_columns,
        imputation_confidence_threshold=application.imputation_confidence_threshold,
        artifact_path=application.artifact_path,
        raw_payload=application.raw_payload,
    )


def _contiguous_true_runs(mask: pd.Series) -> list[tuple[int, int]]:
    values = mask.fillna(False).to_numpy(dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for position, active in enumerate(values):
        if active and start is None:
            start = position
        elif not active and start is not None:
            runs.append((start, position))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def _provisional_group_positions(df: pd.DataFrame) -> list[list[int]]:
    for column in ("episode_id", "segment_id", "source_segment_id"):
        if column in df.columns:
            return [list(positions) for positions in df.groupby(column, sort=False, dropna=False).indices.values()]
    return [list(range(len(df)))]


def _short_linear_interpolation(
    values: pd.Series,
    timestamps: pd.Series,
    invalid: pd.Series,
    group_positions: list[list[int]],
    *,
    circular_degrees: bool = False,
) -> tuple[pd.Series, pd.Series]:
    result = pd.to_numeric(values, errors="coerce").astype(float).mask(invalid)
    filled = pd.Series(False, index=values.index)
    for positions in group_positions:
        if not positions:
            continue
        group_values = result.iloc[positions].reset_index(drop=True)
        group_times = timestamps.iloc[positions].reset_index(drop=True)
        group_invalid = invalid.iloc[positions].reset_index(drop=True)
        for start, end in _contiguous_true_runs(group_invalid):
            run_length = end - start
            if run_length > PROVISIONAL_SHORT_GAP_MAX_ROWS or start == 0 or end >= len(positions):
                continue
            left_value = group_values.iloc[start - 1]
            right_value = group_values.iloc[end]
            left_time = group_times.iloc[start - 1]
            right_time = group_times.iloc[end]
            if pd.isna(left_value) or pd.isna(right_value) or pd.isna(left_time) or pd.isna(right_time):
                continue
            duration = (right_time - left_time).total_seconds()
            if duration <= 0:
                continue
            for local_position in range(start, end):
                current_time = group_times.iloc[local_position]
                if pd.isna(current_time):
                    continue
                fraction = (current_time - left_time).total_seconds() / duration
                if not 0.0 <= fraction <= 1.0:
                    continue
                if circular_degrees:
                    left_angle = float(left_value) % 360.0
                    right_angle = float(right_value) % 360.0
                    difference = ((right_angle - left_angle + 180.0) % 360.0) - 180.0
                    interpolated = (left_angle + fraction * difference) % 360.0
                else:
                    interpolated = float(left_value) + fraction * (
                        float(right_value) - float(left_value)
                    )
                absolute_position = positions[local_position]
                result.iloc[absolute_position] = interpolated
                filled.iloc[absolute_position] = True
    return result, filled


def _short_state_persistence(
    values: pd.Series,
    invalid: pd.Series,
    group_positions: list[list[int]],
) -> tuple[pd.Series, pd.Series]:
    result = pd.to_numeric(values, errors="coerce").astype(float).mask(invalid)
    filled = pd.Series(False, index=values.index)
    for positions in group_positions:
        if not positions:
            continue
        group_values = result.iloc[positions].reset_index(drop=True)
        group_invalid = invalid.iloc[positions].reset_index(drop=True)
        for start, end in _contiguous_true_runs(group_invalid):
            if end - start > PROVISIONAL_SHORT_GAP_MAX_ROWS or start == 0:
                continue
            previous = group_values.iloc[start - 1]
            if pd.isna(previous):
                continue
            for local_position in range(start, end):
                absolute_position = positions[local_position]
                result.iloc[absolute_position] = float(previous)
                filled.iloc[absolute_position] = True
    return result, filled


def _invalid_non_ai_mask(df: pd.DataFrame, column: str) -> pd.Series:
    invalid = pd.to_numeric(df[column], errors="coerce").isna()
    for suffix in ("_missing_flag", "_rule_outlier_flag"):
        flag_column = f"{column}{suffix}"
        if flag_column in df.columns:
            invalid |= pd.to_numeric(df[flag_column], errors="coerce").fillna(0).astype(bool)
    return invalid


def _apply_provisional_non_ai_imputation(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the provisional non-AI fallback policy without hiding provenance.

    Short means at most three consecutive 5-minute rows. Long gaps remain NaN
    and receive an unconfirmed flag because external-weather substitution and
    the detailed fallback contract are intentionally deferred.
    """

    from geas35.preprocessing import QUALITY_IMPUTED_FLAG_COLUMN, STATE_COLUMNS

    df = df_raw.copy()
    timestamps = pd.to_datetime(df.get("reg_date"), errors="coerce")
    group_positions = _provisional_group_positions(df)
    provisional_filled_columns: list[str] = []
    unconfirmed_columns: list[str] = []

    for column in STATE_COLUMNS:
        if column in AI_IMPUTATION_COLUMNS or column not in df.columns:
            continue
        invalid = _invalid_non_ai_mask(df, column)
        if not bool(invalid.any()):
            continue

        if column in CONTINUOUS_EXTERNAL_COLUMNS:
            values, filled = _short_linear_interpolation(
                df[column], timestamps, invalid, group_positions
            )
            method_name = "short_time_linear_interpolation"
            unresolved_method = "external_weather_required_or_unconfirmed"
        elif column in CIRCULAR_EXTERNAL_COLUMNS:
            values, filled = _short_linear_interpolation(
                df[column], timestamps, invalid, group_positions, circular_degrees=True
            )
            method_name = "short_circular_time_interpolation"
            unresolved_method = "circular_weather_unconfirmed"
        elif column in SOLAR_EXTERNAL_COLUMNS:
            values, filled = _short_linear_interpolation(
                df[column], timestamps, invalid, group_positions
            )
            values.loc[filled] = values.loc[filled].clip(lower=0.0)
            method_name = (
                "short_solar_accumulation_interpolation"
                if column == "out_light_sum"
                else "short_solar_cycle_time_interpolation"
            )
            unresolved_method = "solar_weather_unconfirmed"
        elif column in STATE_PERSISTENCE_COLUMNS:
            values, filled = _short_state_persistence(
                df[column], invalid, group_positions
            )
            method_name = "short_previous_state_persistence"
            unresolved_method = "abnormal_or_unconfirmed_state"
        else:
            # Internal audit/fallback sensors outside the eight AI variables are
            # preserved with their existing missing/rule flags. Their detailed
            # fallback policy is intentionally not invented at this stage.
            continue

        unresolved = invalid & ~filled
        df[column] = values
        imputed_flag_column = f"{column}_imputed_flag"
        existing_imputed = (
            pd.to_numeric(df[imputed_flag_column], errors="coerce").fillna(0).astype(bool)
            if imputed_flag_column in df.columns
            else pd.Series(False, index=df.index)
        )
        df[imputed_flag_column] = (existing_imputed | filled).astype(int)
        df[f"{column}_unconfirmed_flag"] = unresolved.astype(int)
        method = pd.Series("not_applicable", index=df.index, dtype="string")
        method.loc[filled] = method_name
        method.loc[unresolved] = unresolved_method
        df[f"{column}_imputation_method"] = method
        provisional_filled_columns.append(imputed_flag_column)
        unconfirmed_columns.append(f"{column}_unconfirmed_flag")

    provisional_row_mask = pd.Series(False, index=df.index)
    for column in provisional_filled_columns:
        provisional_row_mask |= pd.to_numeric(df[column], errors="coerce").fillna(0).astype(bool)
    existing_quality_imputed = (
        pd.to_numeric(df[QUALITY_IMPUTED_FLAG_COLUMN], errors="coerce").fillna(0).astype(bool)
        if QUALITY_IMPUTED_FLAG_COLUMN in df.columns
        else pd.Series(False, index=df.index)
    )
    df[QUALITY_IMPUTED_FLAG_COLUMN] = (
        existing_quality_imputed | provisional_row_mask
    ).astype(int)

    unconfirmed_row_mask = pd.Series(False, index=df.index)
    for column in unconfirmed_columns:
        unconfirmed_row_mask |= pd.to_numeric(df[column], errors="coerce").fillna(0).astype(bool)
    df[QUALITY_UNCONFIRMED_FLAG_COLUMN] = unconfirmed_row_mask.astype(int)
    return df


def _prepare_quality_controlled_frame(
    df_raw: pd.DataFrame,
    *,
    domain_csv_path: Path,
    add_flags: bool,
    imputation_confidence_threshold: float,
    application: QualityControlApplication,
    df_control_log: pd.DataFrame | None,
    crop_cycles: pd.DataFrame | None,
    add_growth_stage: bool,
) -> pd.DataFrame:
    from geas35.preprocessing import (
        add_growth_stage_columns_from_crop_cycles,
        canonicalize_feature_units,
        prepare_geas_input_schema,
        prepare_missing_outliers_handled_features,
    )

    df = prepare_geas_input_schema(df_raw, keep_extra_columns=True)
    df = canonicalize_feature_units(df)
    if add_growth_stage:
        if crop_cycles is None:
            raise ValueError("crop_cycles are required for growth-stage enrichment.")
        df = add_growth_stage_columns_from_crop_cycles(
            df,
            crop_cycles,
            include_stage_name=True,
        )
    df = prepare_missing_outliers_handled_features(
        df,
        df_control_log=df_control_log,
        quality_model=application.quality_model,
        thresholds=application.thresholds,
        domain_csv_path=domain_csv_path,
        observation_columns=AI_BASED_OUTLIER_COLUMNS,
        imputation_columns=AI_IMPUTATION_COLUMNS,
        rule_columns=RULE_BASED_OUTLIER_COLUMNS,
        keep_extra_columns=True,
        add_flags=add_flags,
        imputation_confidence_threshold=(
            application.imputation_confidence_threshold
            if application.imputation_confidence_threshold is not None
            else imputation_confidence_threshold
        ),
    )
    df = _apply_provisional_non_ai_imputation(df)
    return _add_representative_sensor_values(df)


def _prepare_split_file(
    input_path: Path,
    output_path: Path,
    *,
    crop: str,
    split: str,
    domain_csv_path: Path,
    add_flags: bool,
    imputation_confidence_threshold: float,
    application: QualityControlApplication,
    df_control_log: pd.DataFrame | None,
    crop_cycles: pd.DataFrame | None,
    growth_stage_splits: frozenset[str],
) -> QualityControlledSplitSummary:
    from geas35.preprocessing import (
        ACTION_RESTORED_FLAG_SUFFIX,
        DEFAULT_RULE_AGGREGATE_FLAG,
        MISSING_FLAG_SUFFIX,
        QUALITY_AI_OUTLIER_FLAG_COLUMN,
        QUALITY_IMPUTED_FLAG_COLUMN,
        QUALITY_INVALID_FLAG_COLUMN,
        STATE_COLUMNS,
    )
    from geas35.io_utils import flag_cell_count, flag_sum

    df_input = pd.read_parquet(input_path)
    df_output = _prepare_quality_controlled_frame(
        df_input,
        domain_csv_path=domain_csv_path,
        add_flags=add_flags,
        imputation_confidence_threshold=imputation_confidence_threshold,
        application=application,
        df_control_log=df_control_log,
        crop_cycles=crop_cycles,
        add_growth_stage=split in growth_stage_splits,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_parquet(output_path, index=False)

    return QualityControlledSplitSummary(
        crop=crop,
        split=split,
        input_path=str(input_path),
        output_path=str(output_path),
        input_rows=len(df_input),
        output_rows=len(df_output),
        input_columns=len(df_input.columns),
        output_columns=len(df_output.columns),
        missing_cells=flag_cell_count(df_output, MISSING_FLAG_SUFFIX),
        restored_action_cells=flag_cell_count(
            df_output,
            ACTION_RESTORED_FLAG_SUFFIX,
        ),
        rule_outlier_rows=flag_sum(df_output, DEFAULT_RULE_AGGREGATE_FLAG),
        rule_outlier_cells=flag_cell_count(
            df_output,
            "_rule_outlier_flag",
            exclude={DEFAULT_RULE_AGGREGATE_FLAG},
        ),
        ai_outlier_rows=flag_sum(df_output, QUALITY_AI_OUTLIER_FLAG_COLUMN),
        invalid_rows=flag_sum(df_output, QUALITY_INVALID_FLAG_COLUMN),
        invalid_cells=flag_cell_count(
            df_output,
            "_invalid_flag",
            exclude={QUALITY_INVALID_FLAG_COLUMN},
        ),
        imputed_rows=flag_sum(df_output, QUALITY_IMPUTED_FLAG_COLUMN),
        imputed_cells=flag_cell_count(
            df_output,
            "_imputed_flag",
            exclude={QUALITY_IMPUTED_FLAG_COLUMN},
        ),
        provisional_imputed_cells=sum(
            flag_sum(df_output, f"{column}_imputed_flag")
            for column in STATE_COLUMNS
            if column not in AI_IMPUTATION_COLUMNS
        ),
        unconfirmed_rows=flag_sum(df_output, QUALITY_UNCONFIRMED_FLAG_COLUMN),
        unconfirmed_cells=flag_cell_count(
            df_output,
            "_unconfirmed_flag",
            exclude={QUALITY_UNCONFIRMED_FLAG_COLUMN},
        ),
    )


def _load_crop_control_log(
    control_log_dir: Path | None,
    crop: str,
) -> tuple[pd.DataFrame | None, Path | None]:
    """Load an optional crop control_actuation snapshot for action restoration."""

    if control_log_dir is None:
        return None, None
    root = Path(control_log_dir)
    candidates = (
        root / f"{crop}.parquet",
        root / crop / "control_actuation.parquet",
    )
    for path in candidates:
        if path.is_file():
            return pd.read_parquet(path), path
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Control-log parquet not found; expected one of: {expected}")


def prepare_quality_controlled_splits(
    *,
    dataset_root: Path = _dataset_root(),
    input_dir_name: str = DEFAULT_INPUT_DIR_NAME,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    crops: Iterable[str] | None = None,
    splits: Iterable[str] = DEFAULT_SPLITS,
    domain_csv_path: Path | None = None,
    add_flags: bool = True,
    imputation_confidence_threshold: float | None = None,
    quality_model_artifact: Path | None = None,
    control_log_dir: Path | None = None,
    crop_cycle_manifest_path: Path | None = None,
    growth_stage_splits: Iterable[str] = DEFAULT_GROWTH_STAGE_SPLITS,
) -> list[QualityControlledSplitSummary]:
    """Run complete QC with each crop's manually selected quality model."""

    from geas35.preprocessing import (
        DEFAULT_DOMAIN_RANGE_PATH,
        DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
    )
    from geas35.io_utils import existing_crops, write_json

    dataset_root = Path(dataset_root)
    input_root = dataset_root / input_dir_name
    output_root = dataset_root / output_dir_name
    domain_csv_path = DEFAULT_DOMAIN_RANGE_PATH if domain_csv_path is None else domain_csv_path
    imputation_confidence_threshold = (
        DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD
        if imputation_confidence_threshold is None
        else imputation_confidence_threshold
    )

    crop_names = list(crops) if crops is not None else existing_crops(input_root)
    split_names = list(splits)
    if quality_model_artifact is not None and len(crop_names) != 1:
        raise ValueError(
            "--quality-model-artifact overrides one crop only; also pass exactly one --crops value."
        )
    growth_stage_split_names = frozenset(growth_stage_splits)
    unsupported_growth_splits = growth_stage_split_names.difference(
        SUPPORTED_GROWTH_STAGE_SPLITS
    )
    if unsupported_growth_splits:
        names = ", ".join(sorted(unsupported_growth_splits))
        raise ValueError(f"Unsupported growth-stage splits: {names}")

    crop_cycles: pd.DataFrame | None = None
    effective_crop_cycle_manifest_path: Path | None = None
    if growth_stage_split_names.intersection(split_names):
        effective_crop_cycle_manifest_path = (
            dataset_root / "00_raw" / DEFAULT_CROP_CYCLE_MANIFEST_NAME
            if crop_cycle_manifest_path is None
            else Path(crop_cycle_manifest_path)
        )
        if not effective_crop_cycle_manifest_path.exists():
            raise FileNotFoundError(
                "Crop-cycle manifest not found: "
                f"{effective_crop_cycle_manifest_path}"
            )
        crop_cycles = pd.read_csv(
            effective_crop_cycle_manifest_path,
            dtype={"subj_cd": "string", "kind_cd": "string"},
        )

    summaries: list[QualityControlledSplitSummary] = []
    crop_applications: dict[str, dict[str, object]] = {}

    for crop in crop_names:
        artifact_path = (
            Path(quality_model_artifact)
            if quality_model_artifact is not None
            else _selected_model_artifact(crop)
        )
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Selected quality-model artifact not found: {artifact_path}")
        application = _load_quality_control_application(artifact_path)
        model_name = str(
            getattr(
                application.quality_model,
                "model_name",
                application.quality_model.__class__.__name__,
            )
        )
        selected_model_name = DEFAULT_SELECTED_MODELS.get(crop)
        if quality_model_artifact is None and model_name != selected_model_name:
            raise ValueError(
                f"{crop}: expected selected model {selected_model_name}, found {model_name}"
            )
        artifact_columns = tuple(application.observation_columns or ())
        if artifact_columns != AI_BASED_OUTLIER_COLUMNS:
            raise ValueError(
                f"{crop}: selected model was not trained for the required 8 AI columns. "
                f"Expected {AI_BASED_OUTLIER_COLUMNS}, found {artifact_columns}. "
                "Rerun quality_control_model_selection with the crop YAML."
            )
        if not isinstance(application.thresholds, dict) or set(application.thresholds) != set(AI_BASED_OUTLIER_COLUMNS):
            raise ValueError(
                f"{crop}: selected model must contain one calibrated threshold per AI column."
            )

        df_control_log, effective_control_log_path = _load_crop_control_log(
            control_log_dir,
            crop,
        )
        crop_applications[crop] = {
            "quality_model": model_name,
            "selected_quality_model_artifact": application.artifact_path,
            "thresholds": application.thresholds,
            "ai_based_outlier_columns": AI_BASED_OUTLIER_COLUMNS,
            "ai_imputation_columns": AI_IMPUTATION_COLUMNS,
            "representative_sensor_specs": REPRESENTATIVE_SENSOR_SPECS,
            "rule_based_outlier_columns": RULE_BASED_OUTLIER_COLUMNS,
            "best_hyperparameters": (
                application.raw_payload.get("metadata", {}).get("hpo_best_candidate")
                if application.raw_payload is not None
                else None
            ),
            "imputation_confidence_threshold": (
                application.imputation_confidence_threshold
                if application.imputation_confidence_threshold is not None
                else imputation_confidence_threshold
            ),
            "control_log_path": effective_control_log_path,
            "control_log_policy": (
                "same_timestamp_missing_action_restoration"
                if effective_control_log_path is not None
                else "use_action_values_already_present_in_02_split"
            ),
        }
        print(
            f"[quality_control] {crop}: model={model_name}, "
            f"threshold={application.thresholds}, artifact={artifact_path}"
        )

        for split in split_names:
            input_path = input_root / crop / f"{split}.parquet"
            if not input_path.exists():
                raise FileNotFoundError(f"Split parquet not found: {input_path}")
            output_path = output_root / crop / f"{split}.parquet"
            summaries.append(
                _prepare_split_file(
                    input_path,
                    output_path,
                    crop=crop,
                    split=split,
                    domain_csv_path=domain_csv_path,
                    add_flags=add_flags,
                    imputation_confidence_threshold=imputation_confidence_threshold,
                    application=application,
                    df_control_log=df_control_log,
                    crop_cycles=crop_cycles,
                    growth_stage_splits=growth_stage_split_names,
                )
            )

    write_json(
        output_root / "quality_controlled_manifest.json",
        {
            "stage": "quality_controlled",
            "input_root": input_root,
            "output_root": output_root,
            "processed_splits": split_names,
            "test_processed": "test" in split_names,
            "input_schema_stage": "geas35.preprocessing.1_input_schema_preparation",
            "unit_canonicalization_stage": "geas35.preprocessing.2_unit_canonicalization",
            "growth_stage_stage": "geas35.preprocessing.growth_stage_features",
            "growth_stage_splits": sorted(growth_stage_split_names),
            "growth_stage_test_policy": "deferred_to_online_inference_preprocessing",
            "crop_cycle_manifest_path": effective_crop_cycle_manifest_path,
            "missing_outlier_stage": "geas35.preprocessing.3_missing_outliers_handling",
            "selected_model_policy": "crop_specific_manually_selected_model",
            "selected_models": DEFAULT_SELECTED_MODELS,
            "rule_based_outlier_columns": RULE_BASED_OUTLIER_COLUMNS,
            "ai_based_outlier_columns": AI_BASED_OUTLIER_COLUMNS,
            "ai_imputation_columns": AI_IMPUTATION_COLUMNS,
            "representative_sensor_specs": REPRESENTATIVE_SENSOR_SPECS,
            "representative_value_policy": "sensor1_then_sensor2_then_ai_imputed_sensor1",
            "substrate_sensor2_policy": "audit_flags_only_excluded_from_representative_and_ai_imputation",
            "threshold_policy": "crop_model_and_variable_specific_validation_calibration",
            "provisional_non_ai_imputation_policy": {
                "short_gap_max_rows": PROVISIONAL_SHORT_GAP_MAX_ROWS,
                "sampling_interval_minutes": 5,
                "continuous_external": "short_time_linear_interpolation; long_external_weather_required_or_unconfirmed",
                "wind_direction": "short_circular_interpolation; long_unconfirmed",
                "solar": "short_solar_aware_interpolation; long_unconfirmed",
                "rain_and_system_state": "short_previous_state_persistence; long_abnormal_or_unconfirmed",
                "other_non_ai_sensors": "retain_original_values_and_existing_audit_flags; policy_pending",
                "status": "provisional_subject_to_future_refinement",
            },
            "crop_applications": crop_applications,
            "future_control_log_contract": {
                "source_table": "control_actuation",
                "time_column": "reg_date",
                "farm_column": "farm_sn",
                "matching": "same_timestamp_only",
                "known_gap": "GEAS3.0 control_actuation has no pred_co2 column",
            },
            "domain_csv_path": domain_csv_path,
            "add_flags": add_flags,
            "splits": summaries,
        },
        convert=True,
    )
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    _ensure_src_on_path()
    from geas35.preprocessing import (
        DEFAULT_DOMAIN_RANGE_PATH,
        DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
    )

    parser = argparse.ArgumentParser(
        description="Build 03_quality_controlled datasets from 02_split parquet files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_dataset_root(),
        help="Root of the offline dataset directory.",
    )
    parser.add_argument(
        "--input-dir-name",
        default=DEFAULT_INPUT_DIR_NAME,
        help="Input directory name under dataset root.",
    )
    parser.add_argument(
        "--output-dir-name",
        default=DEFAULT_OUTPUT_DIR_NAME,
        help="Output directory name under dataset root.",
    )
    parser.add_argument(
        "--domain-csv-path",
        type=Path,
        default=DEFAULT_DOMAIN_RANGE_PATH,
        help="Domain range CSV used for rule-based outlier flags.",
    )
    parser.add_argument(
        "--crops",
        nargs="+",
        default=None,
        help="Crop folders to process. Defaults to all crop folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names to process. Defaults to train validation.",
    )
    parser.add_argument(
        "--no-flags",
        action="store_true",
        help="Do not add missing/restored/quality flag columns.",
    )
    parser.add_argument(
        "--imputation-confidence-threshold",
        type=float,
        default=DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
        help="Confidence gate reserved for selected AI quality models.",
    )
    parser.add_argument(
        "--crop-cycle-manifest-path",
        type=Path,
        default=None,
        help=(
            "Crop-cycle snapshot used for growth-stage features. Defaults to "
            "<dataset-root>/00_raw/crop_cycle_manifest.csv."
        ),
    )
    parser.add_argument(
        "--growth-stage-splits",
        nargs="+",
        choices=list(SUPPORTED_GROWTH_STAGE_SPLITS),
        default=list(DEFAULT_GROWTH_STAGE_SPLITS),
        help=(
            "Splits enriched with crop-cycle and growth-stage columns. "
            "Defaults to train validation; Step 11.3 may explicitly request test."
        ),
    )
    parser.add_argument(
        "--quality-model-artifact",
        type=Path,
        default=None,
        help=(
            "Optional single-crop artifact override. By default each crop uses "
            "its configured selected-model quality_model_application.json."
        ),
    )
    parser.add_argument(
        "--control-log-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing <crop>.parquet or "
            "<crop>/control_actuation.parquet. Missing action values are restored "
            "only on exact reg_date matches."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()
    args = build_arg_parser().parse_args(argv)
    summaries = prepare_quality_controlled_splits(
        dataset_root=args.dataset_root,
        input_dir_name=args.input_dir_name,
        output_dir_name=args.output_dir_name,
        crops=args.crops,
        splits=args.splits,
        domain_csv_path=args.domain_csv_path,
        add_flags=not args.no_flags,
        imputation_confidence_threshold=args.imputation_confidence_threshold,
        quality_model_artifact=args.quality_model_artifact,
        control_log_dir=args.control_log_dir,
        crop_cycle_manifest_path=args.crop_cycle_manifest_path,
        growth_stage_splits=args.growth_stage_splits,
    )

    for summary in summaries:
        print(
            f"[quality_controlled] {summary.crop}/{summary.split}: "
            f"{summary.input_rows} -> {summary.output_rows} rows, "
            f"{summary.invalid_rows} invalid rows, "
            f"{summary.imputed_cells} cells imputed, wrote {summary.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
