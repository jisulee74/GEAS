from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "offline_dataset_preparation" / "datasets"
RAW_ROOT = DATASET_ROOT / "00_raw"
OUTPUT_ROOT = DATASET_ROOT / "01_resampled"
RAW_MANIFEST_PATH = RAW_ROOT / "series_manifest.csv"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"
SEGMENTS_PATH = OUTPUT_ROOT / "segments.csv"
NOTES_PATH = OUTPUT_ROOT / "preprocessing_notes.json"
SUMMARY_PATH = OUTPUT_ROOT / "resampling_summary.csv"

GRID_FREQ = "5min"
EXPECTED_ROWS_PER_DAY = 288
MAX_SEGMENT_GAP_MIN = 10.0
MIN_SEGMENT_ROWS = EXPECTED_ROWS_PER_DAY
SOURCE_INTERVAL_MINUTES = 10.0

EXCLUDED_FROM_RESAMPLED_COLUMNS = (
    "etc_blackout",
    "etc_plc_abnorm",
    "cont_co2_run",
)

FFILL_COLUMNS = (
    "in_medium_temp1", "in_medium_temp2", "in_temp", "in_temp2",
    "in_water_hot", "in_water_cold", "in_hum", "in_hum2",
    "in_medium_hum1", "in_medium_hum2", "in_co2", "in_co2_2",
    "in_medium_ec1", "in_medium_ec2", "etc_plc_norm", "out_temp", "out_hum", "out_winddirec", "out_windsp",
    "out_light", "out_rainfall", "out_rain", "out_airpress",
    "cont_skyl_vol", "cont_skyr_vol", "cont_cur_vol", "cont_kwcur_vol",
    "cont_pump1_run", "cont_pump2_run", "cont_heater_run",
    "cont_cooler_run", "cont_3way1_vol", "cont_3way2_vol", "cont_fan_run",
)
CUMULATIVE_LIGHT_COLUMN = "out_light_sum"
LIGHT_COLUMN = "out_light"
LIGHT_SUM_ESTIMATED_FLAG = "out_light_sum_estimated_flag"
LIGHT_SUM_EXTRAPOLATED_FLAG = "out_light_sum_extrapolated_flag"
LIGHT_SUM_FFILL_FLAG = "out_light_sum_ffill_flag"
LIGHT_SUM_METHOD_COLUMN = "out_light_sum_estimation_method"
LIGHT_SUM_METHOD_NONE = "none"
LIGHT_SUM_METHOD_INTEGRATION = "solar_integration"
LIGHT_SUM_METHOD_EXTRAPOLATION = "causal_linear_extrapolation"
LIGHT_SUM_METHOD_FFILL = "ffill"
SECONDS_PER_STEP = 5 * 60
SQUARE_METRES_PER_SQUARE_CENTIMETRE = 10_000
FFILL_FLAG_SUFFIX = "_ffill_flag"

METADATA_COLUMNS = {
    "series_id",
    "segment_id",
    "segment_index",
    "crop",
    "geas_version",
    "source_segment_id",
    "source_segment_index",
    "source_file_path",
    "is_resampled_row",
    "is_causal_ffill_row",
    "source_interval_minutes",
    "dt_min",
}


def read_manifest_paths() -> list[Path]:
    manifest = pd.read_csv(RAW_MANIFEST_PATH)
    paths: list[Path] = []
    for value in manifest["file_path"]:
        for part in str(value).split(";"):
            part = part.strip().replace("\\", "/")
            if part:
                paths.append(PROJECT_ROOT / part)
    return list(dict.fromkeys(paths))


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def split_on_large_gaps(df: pd.DataFrame) -> pd.Series:
    gap_min = df["reg_date"].diff().dt.total_seconds().div(60)
    return gap_min.gt(MAX_SEGMENT_GAP_MIN).fillna(False).cumsum()


def _causal_ffill_inserted_midpoints(
    out: pd.DataFrame,
    source_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    """Fill the midpoint of an exact 10-minute source gap using past data only."""

    result = out.copy()
    inserted = ~result["reg_date"].isin(source_dates)
    previous_source = result["reg_date"] - pd.Timedelta(minutes=5)
    next_source = result["reg_date"] + pd.Timedelta(minutes=5)
    ten_minute_midpoint = (
        inserted
        & previous_source.isin(source_dates)
        & next_source.isin(source_dates)
    )

    result["source_interval_minutes"] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    result.loc[ten_minute_midpoint, "source_interval_minutes"] = SOURCE_INTERVAL_MINUTES
    any_filled = pd.Series(False, index=result.index)
    for column in FFILL_COLUMNS:
        flag_column = f"{column}{FFILL_FLAG_SUFFIX}"
        result[flag_column] = 0
        if column not in result.columns:
            continue
        previous_value = result[column].shift(1)
        fill_mask = ten_minute_midpoint & result[column].isna() & previous_value.notna()
        result.loc[fill_mask, column] = previous_value.loc[fill_mask]
        result.loc[fill_mask, flag_column] = 1
        any_filled |= fill_mask
    result["is_causal_ffill_row"] = any_filled

    result[LIGHT_SUM_ESTIMATED_FLAG] = 0
    result[LIGHT_SUM_EXTRAPOLATED_FLAG] = 0
    result[LIGHT_SUM_FFILL_FLAG] = 0
    result[LIGHT_SUM_METHOD_COLUMN] = pd.Series(
        LIGHT_SUM_METHOD_NONE, index=result.index, dtype="string"
    )
    result["is_light_sum_estimated_row"] = False
    if CUMULATIVE_LIGHT_COLUMN not in result.columns:
        return result

    cumulative = pd.to_numeric(result[CUMULATIVE_LIGHT_COLUMN], errors="coerce").astype(float)
    light = (
        pd.to_numeric(result[LIGHT_COLUMN], errors="coerce").astype(float)
        if LIGHT_COLUMN in result.columns
        else pd.Series(float("nan"), index=result.index)
    )
    actual = result["reg_date"].isin(source_dates)
    previous_cumulative = cumulative.shift(1)
    previous_light = light.shift(1)
    eligible = ten_minute_midpoint & cumulative.isna() & previous_cumulative.notna()

    integration_mask = eligible & previous_light.notna()
    integration_increment = previous_light.clip(lower=0.0) * (
        SECONDS_PER_STEP / SQUARE_METRES_PER_SQUARE_CENTIMETRE
    )
    cumulative.loc[integration_mask] = (
        previous_cumulative.loc[integration_mask]
        + integration_increment.loc[integration_mask]
    )
    result.loc[integration_mask, LIGHT_SUM_METHOD_COLUMN] = LIGHT_SUM_METHOD_INTEGRATION

    source_history = pd.DataFrame(
        {
            "reg_date": result.loc[actual, "reg_date"].to_numpy(),
            "cumulative": cumulative.loc[actual].to_numpy(),
        }
    )
    source_history["prior_date"] = source_history["reg_date"].shift(1)
    source_history["prior_cumulative"] = source_history["cumulative"].shift(1)
    history_by_date = source_history.set_index("reg_date")
    previous_actual_date = previous_source.map(history_by_date["prior_date"])
    previous_actual_cumulative = previous_source.map(history_by_date["prior_cumulative"])
    elapsed_minutes = (
        previous_source - pd.to_datetime(previous_actual_date)
    ).dt.total_seconds().div(60.0)
    same_day = (
        previous_source.dt.normalize()
        == pd.to_datetime(previous_actual_date).dt.normalize()
    )
    increase = previous_cumulative - pd.to_numeric(
        previous_actual_cumulative, errors="coerce"
    )
    extrapolation_mask = (
        eligible
        & ~integration_mask
        & previous_actual_cumulative.notna()
        & elapsed_minutes.gt(0)
        & same_day.fillna(False)
        & increase.ge(0)
    )
    extrapolated_increment = increase.clip(lower=0.0) * (5.0 / elapsed_minutes)
    cumulative.loc[extrapolation_mask] = (
        previous_cumulative.loc[extrapolation_mask]
        + extrapolated_increment.loc[extrapolation_mask]
    )
    result.loc[extrapolation_mask, LIGHT_SUM_METHOD_COLUMN] = (
        LIGHT_SUM_METHOD_EXTRAPOLATION
    )
    result.loc[extrapolation_mask, LIGHT_SUM_EXTRAPOLATED_FLAG] = 1

    fallback_mask = eligible & cumulative.isna()
    cumulative.loc[fallback_mask] = previous_cumulative.loc[fallback_mask]
    result.loc[fallback_mask, LIGHT_SUM_METHOD_COLUMN] = LIGHT_SUM_METHOD_FFILL
    result.loc[fallback_mask, LIGHT_SUM_FFILL_FLAG] = 1

    estimated = integration_mask | extrapolation_mask | fallback_mask
    # Within a day, an estimated cumulative value must not decrease.
    cumulative.loc[estimated] = cumulative.loc[estimated].clip(
        lower=previous_cumulative.loc[estimated]
    )
    result[CUMULATIVE_LIGHT_COLUMN] = cumulative
    result.loc[estimated, LIGHT_SUM_ESTIMATED_FLAG] = 1
    result.loc[estimated, "is_light_sum_estimated_row"] = True
    return result


def resample_part(part: pd.DataFrame) -> pd.DataFrame | None:
    part = part.sort_values("reg_date").drop_duplicates("reg_date", keep="last").copy()
    start = part["reg_date"].min()
    end = part["reg_date"].max()
    grid = pd.date_range(start=start, end=end, freq=GRID_FREQ)
    if len(grid) < MIN_SEGMENT_ROWS:
        return None

    source_dates = set(part["reg_date"])
    indexed = part.set_index("reg_date").reindex(grid)
    indexed.index.name = "reg_date"

    out = indexed.reset_index()
    out["is_resampled_row"] = ~out["reg_date"].isin(source_dates)
    out = _causal_ffill_inserted_midpoints(out, source_dates)
    out = out.drop(columns=list(EXCLUDED_FROM_RESAMPLED_COLUMNS), errors="ignore")
    out["dt_min"] = out["reg_date"].diff().dt.total_seconds().div(60)
    return out


def collect_resampled_parts() -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for path in read_manifest_paths():
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
        df = df.dropna(
            subset=["reg_date", "source_segment_id", "series_id"]
        ).sort_values("reg_date")

        for source_segment_id, segment_df in df.groupby(
            "source_segment_id", sort=False
        ):
            segment_df = segment_df.sort_values("reg_date").copy()
            segment_df["_gap_part"] = split_on_large_gaps(segment_df)
            for _, part in segment_df.groupby("_gap_part", sort=False):
                part = part.drop(columns=["_gap_part"])
                resampled = resample_part(part)
                if resampled is None:
                    continue
                first = part.iloc[0]
                parts.append(
                    {
                        "df": resampled,
                        "series_id": str(first["series_id"]),
                        "crop": str(first["crop"]),
                        "geas_version": str(first["geas_version"]),
                        "source_segment_id": str(source_segment_id),
                        "source_segment_index": int(first["source_segment_index"]),
                        "source_file_path": relative(path),
                        "actual_start": resampled["reg_date"].min(),
                        "actual_end": resampled["reg_date"].max(),
                    }
                )
    return parts


def assign_output_segments(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    for part in sorted(parts, key=lambda p: (p["series_id"], p["actual_start"])):
        sid = part["series_id"]
        counters[sid] = counters.get(sid, 0) + 1
        segment_index = counters[sid]
        segment_id = f"iot97_{sid}_seg{segment_index:02d}"
        df = part["df"].copy()
        df["series_id"] = sid
        df["segment_id"] = segment_id
        df["segment_index"] = segment_index
        df["crop"] = part["crop"]
        df["geas_version"] = part["geas_version"]
        df["source_segment_id"] = part["source_segment_id"]
        df["source_segment_index"] = part["source_segment_index"]
        df["source_file_path"] = part["source_file_path"]
        part["df"] = df
        part["segment_id"] = segment_id
        part["segment_index"] = segment_index
    return parts


def output_path_for(series_id: str, crop: str) -> Path:
    return OUTPUT_ROOT / crop / f"{series_id}.parquet"


def write_outputs(parts: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    by_file: dict[Path, list[pd.DataFrame]] = {}
    for part in parts:
        out_path = output_path_for(part["series_id"], part["crop"])
        by_file.setdefault(out_path, []).append(part["df"])

    for out_path, frames in by_file.items():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = pd.concat(frames, ignore_index=True).sort_values(["reg_date", "segment_id"])
        data.to_parquet(out_path, index=False)
        print(f"{len(data):,} rows -> {out_path}")

    segment_rows = []
    for part in parts:
        out_path = output_path_for(part["series_id"], part["crop"])
        segment_rows.append(
            {
                "segment_id": part["segment_id"],
                "series_id": part["series_id"],
                "crop": part["crop"],
                "geas_version": part["geas_version"],
                "segment_index": part["segment_index"],
                "actual_start": part["df"]["reg_date"].min(),
                "actual_end": part["df"]["reg_date"].max(),
                "row_count": len(part["df"]),
                "n_inserted_rows": int(part["df"]["is_resampled_row"].sum()),
                "n_causal_ffill_rows": int(part["df"]["is_causal_ffill_row"].sum()),
                "n_causal_ffill_cells": int(
                    part["df"][[f"{c}{FFILL_FLAG_SUFFIX}" for c in FFILL_COLUMNS]].sum().sum()
                ),
                "n_light_sum_estimated_rows": int(
                    part["df"][LIGHT_SUM_ESTIMATED_FLAG].sum()
                ),
                "n_light_sum_solar_integration": int(
                    part["df"][LIGHT_SUM_METHOD_COLUMN].eq(LIGHT_SUM_METHOD_INTEGRATION).sum()
                ),
                "n_light_sum_causal_extrapolation": int(
                    part["df"][LIGHT_SUM_METHOD_COLUMN].eq(LIGHT_SUM_METHOD_EXTRAPOLATION).sum()
                ),
                "n_light_sum_ffill": int(
                    part["df"][LIGHT_SUM_METHOD_COLUMN].eq(LIGHT_SUM_METHOD_FFILL).sum()
                ),
                "source_segment_id": part["source_segment_id"],
                "source_segment_index": part["source_segment_index"],
                "source_file_path": part["source_file_path"],
                "file_path": relative(out_path),
            }
        )
    segments = pd.DataFrame(segment_rows).sort_values("actual_start")
    segments.to_csv(SEGMENTS_PATH, index=False, encoding="utf-8-sig")

    manifest = (
        segments.groupby(["series_id", "crop", "geas_version"], sort=False)
        .agg(
            actual_start=("actual_start", "min"),
            actual_end=("actual_end", "max"),
            row_count=("row_count", "sum"),
            n_segments=("segment_id", "nunique"),
            n_inserted_rows=("n_inserted_rows", "sum"),
            n_causal_ffill_rows=("n_causal_ffill_rows", "sum"),
            n_causal_ffill_cells=("n_causal_ffill_cells", "sum"),
            n_light_sum_estimated_rows=("n_light_sum_estimated_rows", "sum"),
            n_light_sum_solar_integration=("n_light_sum_solar_integration", "sum"),
            n_light_sum_causal_extrapolation=("n_light_sum_causal_extrapolation", "sum"),
            n_light_sum_ffill=("n_light_sum_ffill", "sum"),
            file_paths=("file_path", lambda s: ";".join(dict.fromkeys(map(str, s)))),
        )
        .reset_index()
    )
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")

    crop_display = {"strawberry": "딸기", "melon": "멜론", "cucumber": "오이"}
    crop_order = {"strawberry": 0, "melon": 1, "cucumber": 2}
    summary = manifest.copy()
    summary["_crop_order"] = summary["crop"].map(crop_order).fillna(len(crop_order))
    summary = summary.sort_values(["_crop_order", "geas_version"])
    summary["crop"] = summary["crop"].map(crop_display).fillna(summary["crop"])
    summary = summary.rename(
        columns={
            "crop": "작물",
            "geas_version": "GEAS 버전",
            "row_count": "전체 행",
            "n_inserted_rows": "삽입 행",
            "n_causal_ffill_rows": "ffill 행",
            "n_causal_ffill_cells": "ffill 셀",
            "n_light_sum_estimated_rows": "누적일사량 추정 행",
            "n_light_sum_solar_integration": "일사량 적분",
            "n_light_sum_causal_extrapolation": "선형 외삽",
            "n_light_sum_ffill": "최종 ffill",
        }
    )
    summary = summary[[
        "작물", "GEAS 버전", "전체 행", "삽입 행", "ffill 행", "ffill 셀",
        "누적일사량 추정 행", "일사량 적분", "선형 외삽", "최종 ffill",
    ]]
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    notes = {
        "source": relative(RAW_ROOT),
        "output": relative(OUTPUT_ROOT),
        "grid_freq": GRID_FREQ,
        "max_segment_gap_min": MAX_SEGMENT_GAP_MIN,
        "gap_rule": "dt <= 10 minutes is aligned onto 5-minute grid; dt > 10 starts a new segment candidate",
        "minimum_segment_rows": MIN_SEGMENT_ROWS,
        "minimum_segment_duration": "at least one 24h day on 5-minute grid",
        "inserted_row_policy": (
            "only the midpoint between adjacent 10-minute source rows is eligible "
            "for causal forward-fill"
        ),
        "excluded_from_resampled": {
            "columns": list(EXCLUDED_FROM_RESAMPLED_COLUMNS),
            "reason": "constant-zero or unavailable across the database; retained only in 00_raw",
        },
        "causal_ffill": {
            "method": "ffill from the immediately preceding real observation",
            "eligible_columns": list(FFILL_COLUMNS),
            "eligible_column_count": len(FFILL_COLUMNS),
            "eligible_source_interval_minutes": SOURCE_INTERVAL_MINUTES,
            "large_gap_fill": False,
            "ordinary_missing_value_fill": False,
            "future_information_used": False,
            "flags": {
                "row": "is_causal_ffill_row",
                "cell_suffix": FFILL_FLAG_SUFFIX,
                "source_interval": "source_interval_minutes",
            },
        },
        "cumulative_light_estimation": {
            "column": CUMULATIVE_LIGHT_COLUMN,
            "priority": [
                LIGHT_SUM_METHOD_INTEGRATION,
                LIGHT_SUM_METHOD_EXTRAPOLATION,
                LIGHT_SUM_METHOD_FFILL,
            ],
            "solar_integration_formula": (
                "previous_actual_out_light_w_m2 * 300_seconds / 10000"
            ),
            "extrapolation_source": "two most recent unfilled actual cumulative values",
            "night_rule": "negative solar energy increment is clipped to zero",
            "daily_reset_rule": "previous-day growth rate is not reused",
            "monotonic_rule": "estimate cannot be below previous cumulative value within the day",
            "future_information_used": False,
            "flags": {
                "cell": LIGHT_SUM_ESTIMATED_FLAG,
                "extrapolated_cell": LIGHT_SUM_EXTRAPOLATED_FLAG,
                "ffill_cell": LIGHT_SUM_FFILL_FLAG,
                "method": LIGHT_SUM_METHOD_COLUMN,
                "row": "is_light_sum_estimated_row",
            },
        },
        "action_fill": "causal ffill under zero-order-hold control assumption",
        "raw_is_not_modified": True,
    }
    NOTES_PATH.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parts = assign_output_segments(collect_resampled_parts())
    if not parts:
        raise RuntimeError("No resampled segments were produced.")
    write_outputs(parts)
    print(f"manifest={MANIFEST_PATH}")
    print(f"segments={SEGMENTS_PATH}")
    print(f"notes={NOTES_PATH}")
    print(f"summary={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
