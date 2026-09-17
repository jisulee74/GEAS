"""Growth-stage feature columns for offline and streaming datasets."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

import pandas as pd

from geas35.core.growth_stage import (
    DEFAULT_GROWTH_STAGE_RULE_PATH,
    GrowthStageRule,
    load_growth_stage_rules,
    normalize_crop,
)


GROWTH_STAGE_ORDER_COLUMN = "growth_stage_order"
GROWTH_STAGE_DAT_COLUMN = "growth_stage_dat"
GROWTH_STAGE_NAME_COLUMN = "growth_stage_name"
DEFAULT_TRANSPLANT_DATE_COLUMNS = ("trans_crop_date", "transplant_date")

CROP_CYCLE_ID_COLUMN = "crop_cycle_id"
TRANSPLANT_DATE_COLUMN = "trans_crop_date"
EFFECTIVE_CROP_END_DATE_COLUMN = "effective_crop_end_date"
GROWTH_STAGE_UNMATCHED_FLAG_COLUMN = "growth_stage_unmatched_flag"


def growth_stage_rules_sha256(
    path: Path | str = DEFAULT_GROWTH_STAGE_RULE_PATH,
) -> str:
    """Return a stable hash for the growth-stage rule table used by a dataset."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_rule_index(
    rules: Iterable[GrowthStageRule],
) -> dict[str, list[GrowthStageRule]]:
    by_crop: dict[str, list[GrowthStageRule]] = {}
    for rule in rules:
        by_crop.setdefault(rule.crop, []).append(rule)
    for crop_rules in by_crop.values():
        crop_rules.sort(key=lambda item: item.dat_start)
    return by_crop


def _date_series_from_mapping(
    df: pd.DataFrame,
    *,
    series_col: str,
    transplant_date_by_series: Mapping[object, object],
) -> pd.Series:
    normalized = {str(key): value for key, value in transplant_date_by_series.items()}
    mapped = df[series_col].map(lambda value: normalized.get(str(value)))
    return pd.to_datetime(mapped, errors="coerce")


def _infer_transplant_dates_from_time(
    df: pd.DataFrame,
    *,
    parsed_time: pd.Series,
    series_col: str,
) -> pd.Series:
    if series_col in df.columns:
        inferred = parsed_time.groupby(df[series_col], dropna=False).transform("min")
    else:
        inferred = pd.Series(parsed_time.min(), index=df.index)
    return pd.to_datetime(inferred, errors="coerce")


def _resolve_transplant_dates(
    df: pd.DataFrame,
    *,
    parsed_time: pd.Series,
    transplant_date: object | None,
    transplant_date_by_series: Mapping[object, object] | None,
    series_col: str,
    transplant_date_columns: Iterable[str],
) -> tuple[pd.Series, str]:
    if transplant_date is not None:
        resolved = pd.Series(pd.to_datetime(transplant_date, errors="coerce"), index=df.index)
        return resolved, "argument:transplant_date"

    for column in transplant_date_columns:
        if column in df.columns:
            return pd.to_datetime(df[column], errors="coerce"), f"column:{column}"

    if transplant_date_by_series and series_col in df.columns:
        return (
            _date_series_from_mapping(
                df,
                series_col=series_col,
                transplant_date_by_series=transplant_date_by_series,
            ),
            f"mapping:{series_col}",
        )

    return (
        _infer_transplant_dates_from_time(df, parsed_time=parsed_time, series_col=series_col),
        "inferred:min_reg_date",
    )


def add_growth_stage_columns(
    df: pd.DataFrame,
    *,
    crop: str | None = None,
    crop_col: str = "crop",
    time_col: str = "reg_date",
    transplant_date: object | None = None,
    transplant_date_by_series: Mapping[object, object] | None = None,
    series_col: str = "series_id",
    transplant_date_columns: Iterable[str] = DEFAULT_TRANSPLANT_DATE_COLUMNS,
    rules: Iterable[GrowthStageRule] | None = None,
    include_stage_name: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Add numeric DAT and growth-stage-order columns to a dataframe.

    ``growth_stage_name`` is intentionally opt-in so parquet files can keep the
    compact numeric contract while rule-table metadata carries the labels.
    """

    if time_col not in df.columns:
        raise KeyError(f"Missing timestamp column: {time_col}")

    result = df.copy()
    parsed_time = pd.to_datetime(result[time_col], errors="coerce")
    transplant_dates, _ = _resolve_transplant_dates(
        result,
        parsed_time=parsed_time,
        transplant_date=transplant_date,
        transplant_date_by_series=transplant_date_by_series,
        series_col=series_col,
        transplant_date_columns=transplant_date_columns,
    )

    dat_values = (
        parsed_time.dt.normalize() - transplant_dates.dt.normalize()
    ).dt.days + 1
    dat_values = dat_values.where(dat_values >= 1)
    result[GROWTH_STAGE_DAT_COLUMN] = dat_values.astype("Int64")

    if crop is not None:
        crop_values = pd.Series(normalize_crop(crop), index=result.index)
    elif crop_col in result.columns:
        crop_values = result[crop_col].map(normalize_crop)
    else:
        if strict:
            raise KeyError(f"Missing crop column: {crop_col}")
        crop_values = pd.Series(pd.NA, index=result.index, dtype="object")

    rule_index = _normalized_rule_index(rules or load_growth_stage_rules())
    stage_order = pd.Series(pd.NA, index=result.index, dtype="Int64")
    stage_name = pd.Series(pd.NA, index=result.index, dtype="object")

    for crop_key, crop_rules in rule_index.items():
        crop_mask = crop_values == crop_key
        if not bool(crop_mask.any()):
            continue
        crop_dat = result.loc[crop_mask, GROWTH_STAGE_DAT_COLUMN]
        for rule in crop_rules:
            rule_mask = crop_dat >= rule.dat_start
            if rule.dat_end is not None:
                rule_mask &= crop_dat <= rule.dat_end
            matched_index = crop_dat.index[rule_mask.fillna(False)]
            stage_order.loc[matched_index] = rule.stage_order
            if include_stage_name:
                stage_name.loc[matched_index] = rule.stage_name

    if strict:
        missing_dat = result[GROWTH_STAGE_DAT_COLUMN].isna()
        missing_stage = stage_order.isna()
        if bool(missing_dat.any()):
            raise ValueError(f"{int(missing_dat.sum())} rows have invalid growth-stage DAT.")
        if bool(missing_stage.any()):
            raise ValueError(
                f"{int(missing_stage.sum())} rows do not match a growth-stage rule."
            )

    result[GROWTH_STAGE_ORDER_COLUMN] = stage_order
    if include_stage_name:
        result[GROWTH_STAGE_NAME_COLUMN] = stage_name
    return result


def add_growth_stage_columns_from_crop_cycles(
    df: pd.DataFrame,
    crop_cycles: pd.DataFrame | str | Path,
    *,
    crop_col: str = "crop",
    time_col: str = "reg_date",
    iot_data_idx_col: str = "iot_data_idx",
    rules: Iterable[GrowthStageRule] | None = None,
    include_stage_name: bool = True,
) -> pd.DataFrame:
    """Match rows to crop cycles and add deterministic growth-stage features.

    Missing row-level greenhouse identifiers are filled only when the snapshot
    contains exactly one greenhouse. Rows outside every effective crop interval
    remain unmatched; their DAT and growth-stage values stay missing.
    """

    if time_col not in df.columns:
        raise KeyError(f"Missing timestamp column: {time_col}")

    cycles = (
        pd.read_csv(crop_cycles, dtype={"subj_cd": "string", "kind_cd": "string"})
        if isinstance(crop_cycles, (str, Path))
        else crop_cycles.copy()
    )
    required = {
        "crop_cycle_id",
        "iot_data_idx",
        "crop",
        "transplant_date",
        "effective_crop_end_date",
    }
    missing = sorted(required.difference(cycles.columns))
    if missing:
        raise KeyError(f"Crop-cycle manifest is missing columns: {', '.join(missing)}")

    result = df.copy()
    parsed_time = pd.to_datetime(result[time_col], errors="coerce")
    row_crop = (
        result[crop_col].map(normalize_crop)
        if crop_col in result.columns
        else pd.Series(pd.NA, index=result.index, dtype="object")
    )
    row_iot = (
        pd.to_numeric(result[iot_data_idx_col], errors="coerce")
        if iot_data_idx_col in result.columns
        else pd.Series(float("nan"), index=result.index)
    )

    cycle_iot = pd.to_numeric(cycles["iot_data_idx"], errors="coerce")
    unique_iot = cycle_iot.dropna().unique()
    if len(unique_iot) == 1:
        row_iot = row_iot.fillna(float(unique_iot[0]))

    result[CROP_CYCLE_ID_COLUMN] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result[TRANSPLANT_DATE_COLUMN] = pd.NaT
    result[EFFECTIVE_CROP_END_DATE_COLUMN] = pd.NaT

    cycles = cycles.copy()
    cycles["iot_data_idx"] = cycle_iot
    cycles["crop"] = cycles["crop"].map(normalize_crop)
    cycles["transplant_date"] = pd.to_datetime(cycles["transplant_date"], errors="coerce")
    cycles["effective_crop_end_date"] = pd.to_datetime(
        cycles["effective_crop_end_date"], errors="coerce"
    )
    cycles = cycles.sort_values(["iot_data_idx", "transplant_date", "crop_cycle_id"])

    for cycle in cycles.itertuples(index=False):
        start = cycle.transplant_date
        if pd.isna(start):
            continue
        mask = (
            parsed_time.ge(start)
            & row_iot.eq(float(cycle.iot_data_idx))
            & row_crop.eq(cycle.crop)
        )
        end = cycle.effective_crop_end_date
        if pd.notna(end):
            mask &= parsed_time.le(end)
        matched = result.index[mask.fillna(False)]
        result.loc[matched, CROP_CYCLE_ID_COLUMN] = int(cycle.crop_cycle_id)
        result.loc[matched, TRANSPLANT_DATE_COLUMN] = start
        result.loc[matched, EFFECTIVE_CROP_END_DATE_COLUMN] = end

    result = add_growth_stage_columns(
        result,
        crop_col=crop_col,
        time_col=time_col,
        transplant_date_columns=(TRANSPLANT_DATE_COLUMN,),
        rules=rules,
        include_stage_name=include_stage_name,
        strict=False,
    )
    result[GROWTH_STAGE_UNMATCHED_FLAG_COLUMN] = (
        result[CROP_CYCLE_ID_COLUMN].isna().astype(int)
    )
    return result


__all__ = [
    "CROP_CYCLE_ID_COLUMN",
    "EFFECTIVE_CROP_END_DATE_COLUMN",
    "GROWTH_STAGE_UNMATCHED_FLAG_COLUMN",
    "TRANSPLANT_DATE_COLUMN",
    "add_growth_stage_columns_from_crop_cycles",
    "DEFAULT_TRANSPLANT_DATE_COLUMNS",
    "GROWTH_STAGE_DAT_COLUMN",
    "GROWTH_STAGE_NAME_COLUMN",
    "GROWTH_STAGE_ORDER_COLUMN",
    "add_growth_stage_columns",
    "growth_stage_rules_sha256",
]
