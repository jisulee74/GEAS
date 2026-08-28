from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "offline_dataset_preparation" / "datasets"
RAW_ROOT = DATASET_ROOT / "00_raw"
SERIES_MANIFEST_PATH = RAW_ROOT / "series_manifest.csv"
CROP_CYCLE_MANIFEST_PATH = RAW_ROOT / "crop_cycle_manifest.csv"
SEGMENTS_PATH = RAW_ROOT / "segments.csv"
ENV_PATH = PROJECT_ROOT / ".env"
REQUIRED_DB_ENV = (
    "GEAS_DB_HOST",
    "GEAS_DB_PORT",
    "GEAS_DB_USER",
    "GEAS_DB_PASSWORD",
    "GEAS_DB_NAME",
)

IOT_DATA_IDX = 97
CONTROL_API_HISTORY_CUTOFF = pd.Timestamp("2026-01-22 09:05:00")

ACTION_API_CHANNEL_MAP = {
    1: "cont_skyl_vol",
    2: "cont_skyr_vol",
    4: "cont_cur_vol",
    5: "cont_kwcur_vol",
    6: "cont_co2_run",
    7: "cont_pump1_run",
    8: "cont_pump2_run",
    9: "cont_heater_run",
    10: "cont_cooler_run",
    11: "cont_3way1_vol",
    12: "cont_3way2_vol",
    13: "cont_fan_run",
}
ACTION_COLUMNS = list(ACTION_API_CHANNEL_MAP.values())

CROP_SLUG = {
    "딸기": "strawberry",
    "멜론": "melon",
    "오이": "cucumber",
}

CROP_DISPLAY_NAME = {
    "strawberry": "딸기",
    "melon": "멜론",
    "cucumber": "오이",
}

CONTROL_RULE = (
    "reg_date <= 2026-01-22 09:05:00 uses data_silla_enc cont*; "
    "reg_date > cutoff uses send_ds_control_api_history channel/ratio rounded to 5min"
)

SEGMENT_PLAN = [
    ("strawberry", "2.0", "GEAS2.0", "2024-03-05 10:50:00", "2024-11-19 15:50:00"),
    ("strawberry", "2.0", "GEAS2.0", "2024-11-27 15:25:00", "2025-01-30 20:10:00"),
    ("strawberry", "2.0", "GEAS2.0", "2025-02-03 09:20:00", "2025-04-04 14:00:00"),
    ("strawberry", "2.0", "GEAS2.0", "2025-04-07 13:55:00", "2025-05-07 23:55:00"),
    ("melon", "2.0", "GEAS2.0", "2025-05-08 00:00:00", "2025-06-25 14:20:00"),
    ("melon", "2.0", "GEAS2.0", "2025-06-27 17:20:00", "2025-08-22 14:25:00"),
    ("melon", "2.0", "GEAS2.0", "2025-08-26 15:50:00", "2025-09-09 13:55:00"),
    ("melon", "2.5", "GEAS2.5", "2025-09-09 14:00:00", "2025-10-16 23:55:00"),
    ("strawberry", "2.5", "GEAS2.5", "2025-10-17 00:00:00", "2025-12-01 08:25:00"),
    ("strawberry", "3.0", "GEAS3.0", "2025-12-01 08:30:00", "2025-12-16 09:35:00"),
    ("strawberry", "3.0", "GEAS3.0", "2025-12-19 15:25:00", "2026-01-22 09:05:00"),
    ("strawberry", "2.5", "GEAS2.5", "2026-01-22 09:10:00", "2026-02-12 10:40:00"),
    ("cucumber", "2.5", "GEAS2.5", "2026-03-05 00:00:00", "2099-12-31 23:59:59"),
]


def series_id(crop: str, geas_version: str) -> str:
    return f"{crop}_geas{str(geas_version).replace('.', '_')}"


def segment_plan() -> pd.DataFrame:
    rows = []
    counters: dict[str, int] = {}
    for crop, version, control_mode, start, end in SEGMENT_PLAN:
        sid = series_id(crop, version)
        counters[sid] = counters.get(sid, 0) + 1
        rows.append(
            {
                "series_id": sid,
                "source_segment_id": f"iot97_{sid}_source_seg{counters[sid]:02d}",
                "source_segment_index": counters[sid],
                "crop": crop,
                "geas_version": version,
                "control_mode": control_mode,
                "planned_start": pd.Timestamp(start),
                "planned_end": pd.Timestamp(end),
            }
        )
    return pd.DataFrame(rows)


def load_project_env(path: Path = ENV_PATH) -> None:
    """Load GEAS3.5/.env without overriding already exported variables."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, _strip_env_quotes(value.strip()))


def _strip_env_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def db_config_from_env(path: Path = ENV_PATH) -> dict[str, object]:
    load_project_env(path)
    missing = [name for name in REQUIRED_DB_ENV if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing required DB environment variables: " + ", ".join(missing)
        )
    try:
        port = int(os.environ["GEAS_DB_PORT"])
    except ValueError as exc:
        raise RuntimeError("GEAS_DB_PORT must be an integer.") from exc
    return {
        "host": os.environ["GEAS_DB_HOST"],
        "port": port,
        "user": os.environ["GEAS_DB_USER"],
        "password": os.environ["GEAS_DB_PASSWORD"],
        "database": os.environ["GEAS_DB_NAME"],
    }


def connect():
    import pymysql

    config = db_config_from_env()
    return pymysql.connect(
        host=str(config["host"]),
        port=int(config["port"]),
        user=str(config["user"]),
        password=str(config["password"]),
        database=str(config["database"]),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def slugify(value: Any) -> str:
    text = CROP_SLUG.get(str(value).strip(), str(value).strip())
    text = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", text)
    return text.strip("_").lower()


def find_period_csv() -> Path:
    candidates = [
        path
        for root in (PROJECT_ROOT, WORKSPACE_ROOT)
        for path in root.glob("*.csv")
        if "GEAS" in path.name and "260707" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError("Could not find crop/GEAS period CSV in project root.")
    return candidates[0]


def parse_periods() -> pd.DataFrame:
    periods = pd.read_csv(find_period_csv(), encoding="utf-8-sig")
    period_col, crop_col, version_col = periods.columns[:3]
    periods = periods.dropna(subset=[period_col, crop_col, version_col]).copy()
    split = periods[period_col].astype(str).str.split("~", n=1, expand=True)
    periods["period_start"] = pd.to_datetime(split[0].str.strip(), errors="coerce")
    periods["period_end"] = pd.to_datetime(split[1].str.strip(), errors="coerce")
    periods = periods.dropna(subset=["period_start", "period_end"]).reset_index(drop=True)
    periods["period_id"] = periods.index + 1
    periods = periods.rename(columns={crop_col: "crop", version_col: "geas_version"})
    periods["crop"] = periods["crop"].astype(str).str.strip().map(lambda v: CROP_SLUG.get(v, v))
    periods["geas_version"] = periods["geas_version"].astype(str).str.strip()
    return periods[["period_id", "period_start", "period_end", "crop", "geas_version"]]


def fetch_state(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    query = """
        SELECT *
        FROM data_silla_enc
        WHERE iot_data_idx = %s
          AND reg_date IS NOT NULL
          AND reg_date >= %s
          AND reg_date <= %s
        ORDER BY reg_date
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (IOT_DATA_IDX, start.to_pydatetime(), end.to_pydatetime()))
            return pd.DataFrame(cur.fetchall())


def fetch_action_history(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    query = """
        SELECT idx, n_idx, channel, operation, ratio, reg_date
        FROM send_ds_control_api_history
        WHERE reg_date IS NOT NULL
          AND reg_date >= %s
          AND reg_date <= %s
        ORDER BY reg_date, idx
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (start.to_pydatetime(), end.to_pydatetime()))
            return pd.DataFrame(cur.fetchall())


def fetch_crop_cycles(iot_data_idx: int = IOT_DATA_IDX) -> pd.DataFrame:
    """Read cultivation-cycle metadata from crop_info."""
    query = """
        SELECT
            idx AS crop_cycle_id,
            iot_data_idx,
            crop_nm AS crop_name,
            subj_cd,
            kind_cd,
            trans_crop_date AS transplant_date,
            crop_end_date AS db_crop_end_date
        FROM crop_info
        WHERE iot_data_idx = %s
          AND COALESCE(del_yn, 'N') = 'N'
          AND trans_crop_date IS NOT NULL
        ORDER BY trans_crop_date, idx
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (iot_data_idx,))
            return pd.DataFrame(cur.fetchall())


def prepare_crop_cycle_manifest(crop_cycles: pd.DataFrame) -> pd.DataFrame:
    """Add non-overlapping, five-minute-grid effective end dates."""
    columns = [
        "crop_cycle_id",
        "iot_data_idx",
        "crop",
        "crop_name",
        "subj_cd",
        "kind_cd",
        "transplant_date",
        "db_crop_end_date",
        "effective_crop_end_date",
        "end_adjusted",
    ]
    if crop_cycles.empty:
        return pd.DataFrame(columns=columns)

    out = crop_cycles.copy()
    out["subj_cd"] = out["subj_cd"].map(
        lambda value: None if pd.isna(value) else str(value).zfill(2)
    )
    out["kind_cd"] = out["kind_cd"].map(
        lambda value: None if pd.isna(value) else str(value).zfill(4)
    )
    out["crop"] = out["subj_cd"].map(
        {"01": "strawberry", "04": "cucumber", "05": "melon"}
    )
    out["transplant_date"] = pd.to_datetime(out["transplant_date"], errors="coerce")
    out["db_crop_end_date"] = pd.to_datetime(out["db_crop_end_date"], errors="coerce")
    out = out.sort_values(
        ["iot_data_idx", "transplant_date", "crop_cycle_id"]
    ).reset_index(drop=True)

    next_transplant = out.groupby("iot_data_idx")["transplant_date"].shift(-1)
    overlaps_next = (
        next_transplant.notna()
        & out["db_crop_end_date"].notna()
        & out["db_crop_end_date"].ge(next_transplant)
    )
    out["effective_crop_end_date"] = out["db_crop_end_date"]
    out.loc[overlaps_next, "effective_crop_end_date"] = (
        next_transplant.loc[overlaps_next] - pd.Timedelta(minutes=5)
    )
    out["end_adjusted"] = overlaps_next.astype(int)
    return out[columns]


def is_missing_text(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def build_action_wide(action_history: pd.DataFrame) -> pd.DataFrame:
    if action_history.empty:
        return pd.DataFrame(columns=["reg_date", *ACTION_COLUMNS])

    log = action_history.copy()
    log["api_reg_date"] = pd.to_datetime(log["reg_date"], errors="coerce")
    log["reg_date"] = log["api_reg_date"].dt.round("5min")
    log["channel"] = pd.to_numeric(log["channel"], errors="coerce").astype("Int64")
    log["ratio"] = pd.to_numeric(log["ratio"], errors="coerce")
    log["action_col"] = log["channel"].map(ACTION_API_CHANNEL_MAP)
    valid = (
        log["reg_date"].notna()
        & log["api_reg_date"].notna()
        & log["action_col"].notna()
        & ~is_missing_text(log["idx"])
        & ~is_missing_text(log["n_idx"])
        & ~is_missing_text(log["operation"])
        & log["ratio"].notna()
    )
    log = log.loc[valid].copy()
    if log.empty:
        return pd.DataFrame(columns=["reg_date", *ACTION_COLUMNS])

    log["alignment_seconds"] = (log["api_reg_date"] - log["reg_date"]).abs().dt.total_seconds()
    log = (
        log.sort_values(
            ["reg_date", "action_col", "alignment_seconds", "api_reg_date", "idx"],
            ascending=[True, True, True, False, False],
        )
        .drop_duplicates(["reg_date", "action_col"], keep="first")
    )
    wide = log.pivot(index="reg_date", columns="action_col", values="ratio").reset_index()
    wide = wide.rename_axis(columns=None)
    for col in ACTION_COLUMNS:
        if col not in wide.columns:
            wide[col] = pd.NA
    return wide[["reg_date", *ACTION_COLUMNS]]


def apply_action_source_rule(state: pd.DataFrame, action_wide: pd.DataFrame) -> pd.DataFrame:
    state = state.copy()
    state["reg_date"] = pd.to_datetime(state["reg_date"], errors="coerce")
    for col in ACTION_COLUMNS:
        if col in state.columns:
            state[f"{col}_data_silla_raw"] = state[col]
        else:
            state[f"{col}_data_silla_raw"] = pd.NA
            state[col] = pd.NA

    if action_wide.empty:
        merged = state
        for col in ACTION_COLUMNS:
            merged[f"{col}_api_history"] = pd.NA
    else:
        api = action_wide.rename(columns={col: f"{col}_api_history" for col in ACTION_COLUMNS})
        merged = state.merge(api, on="reg_date", how="left")

    pre_mask = merged["reg_date"].notna() & (merged["reg_date"] <= CONTROL_API_HISTORY_CUTOFF)
    post_mask = merged["reg_date"].notna() & (merged["reg_date"] > CONTROL_API_HISTORY_CUTOFF)
    for col in ACTION_COLUMNS:
        merged[col] = pd.NA
        merged.loc[pre_mask, col] = pd.to_numeric(
            merged.loc[pre_mask, f"{col}_data_silla_raw"], errors="coerce"
        )
        merged.loc[post_mask, col] = pd.to_numeric(
            merged.loc[post_mask, f"{col}_api_history"], errors="coerce"
        )

    helper_cols = [c for c in merged.columns if c.endswith("_data_silla_raw") or c.endswith("_api_history")]
    return merged.drop(columns=helper_cols)


def output_path_for_series(crop: str, geas_version: str) -> Path:
    crop_slug = slugify(crop)
    return RAW_ROOT / crop_slug / f"{series_id(crop_slug, geas_version)}.parquet"


def assign_metadata(df: pd.DataFrame, crop: str, geas_version: str, plan: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    reg_date = pd.to_datetime(df["reg_date"], errors="coerce")
    sid = series_id(crop, geas_version)
    df["series_id"] = sid
    df["source_segment_id"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["source_segment_index"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df["crop"] = crop
    df["geas_version"] = geas_version
    for _, segment in plan[(plan["crop"] == crop) & (plan["geas_version"] == geas_version)].iterrows():
        mask = reg_date.ge(segment["planned_start"]) & reg_date.le(segment["planned_end"])
        df.loc[mask, "source_segment_id"] = segment["source_segment_id"]
        df.loc[mask, "source_segment_index"] = int(segment["source_segment_index"])
    if df["source_segment_id"].isna().any():
        missing_dates = reg_date[df["source_segment_id"].isna()]
        raise RuntimeError(f"{sid} has rows outside segment plan: {missing_dates.min()} to {missing_dates.max()}")
    return df


def write_manifests(
    file_rows: list[dict[str, Any]],
    plan: pd.DataFrame,
    crop_cycles: pd.DataFrame,
) -> None:
    file_df = pd.DataFrame(file_rows)
    segment_rows = []
    for _, file_row in file_df.iterrows():
        df = pd.read_parquet(file_row["file_path"])
        grouped = (
            df.groupby("source_segment_id", sort=False)
            .agg(
                series_id=("series_id", "first"),
                crop=("crop", "first"),
                geas_version=("geas_version", "first"),
                source_segment_index=("source_segment_index", "first"),
                actual_start=("reg_date", "min"),
                actual_end=("reg_date", "max"),
                row_count=("reg_date", "size"),
            )
            .reset_index()
        )
        grouped["file_path"] = file_row["relative_path"]
        segment_rows.append(grouped)

    segments = pd.concat(segment_rows, ignore_index=True)
    planned = plan[["source_segment_id", "control_mode", "planned_start", "planned_end"]]
    segments = segments.merge(planned, on="source_segment_id", how="left")
    segments = segments[
        [
            "source_segment_id",
            "series_id",
            "crop",
            "geas_version",
            "control_mode",
            "source_segment_index",
            "planned_start",
            "planned_end",
            "actual_start",
            "actual_end",
            "row_count",
            "file_path",
        ]
    ].sort_values("actual_start")
    segments.to_csv(SEGMENTS_PATH, index=False, encoding="utf-8-sig")

    series_manifest = (
        segments.groupby(["series_id", "crop", "geas_version"], sort=False)
        .agg(
            actual_start=("actual_start", "min"),
            actual_end=("actual_end", "max"),
            row_count=("row_count", "sum"),
            n_source_segments=("source_segment_id", "nunique"),
            file_paths=("file_path", lambda s: ";".join(dict.fromkeys(map(str, s)))),
        )
        .reset_index()
    )
    series_manifest = series_manifest.sort_values("actual_start").reset_index(drop=True)
    series_manifest.insert(0, "series_number", series_manifest.index + 1)
    series_manifest["crop_name"] = series_manifest["crop"].map(CROP_DISPLAY_NAME)
    series_manifest = series_manifest.rename(
        columns={
            "actual_start": "data_start",
            "actual_end": "data_end",
            "file_paths": "file_path",
        }
    )
    series_manifest = series_manifest[
        [
            "series_number",
            "series_id",
            "data_start",
            "data_end",
            "crop",
            "crop_name",
            "geas_version",
            "row_count",
            "n_source_segments",
            "file_path",
        ]
    ]
    series_manifest.to_csv(
        SERIES_MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )
    prepare_crop_cycle_manifest(crop_cycles).to_csv(
        CROP_CYCLE_MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build 00_raw datasets from the configured GEAS DB."
    )
    parser.add_argument(
        "--check-db-config",
        action="store_true",
        help="Load GEAS3.5/.env and validate required DB variable names without connecting.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.check_db_config:
        db_config_from_env()
        print("DB config loaded: " + ", ".join(REQUIRED_DB_ENV))
        return 0

    db_config_from_env()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    periods = parse_periods()
    plan = segment_plan()
    action_start = max(periods["period_start"].min(), CONTROL_API_HISTORY_CUTOFF) - pd.Timedelta(minutes=10)
    action_end = periods["period_end"].max() + pd.Timedelta(minutes=10)
    action_wide = build_action_wide(fetch_action_history(action_start, action_end))

    series_frames: dict[str, list[pd.DataFrame]] = {}
    series_paths: dict[str, Path] = {}
    for _, period in periods.iterrows():
        crop = str(period["crop"])
        geas_version = str(period["geas_version"])
        state = fetch_state(period["period_start"], period["period_end"])

        if state.empty:
            dataset = pd.DataFrame()
        else:
            dataset = apply_action_source_rule(state, action_wide)
            dataset = assign_metadata(dataset, crop, geas_version, plan)
            dataset["source_period_id"] = int(period["period_id"])
            dataset["source_period_start"] = period["period_start"]
            dataset["source_period_end"] = period["period_end"]

        sid = series_id(crop, geas_version)
        if not dataset.empty:
            series_frames.setdefault(sid, []).append(dataset)
            series_paths[sid] = output_path_for_series(crop, geas_version)
        print(f"fetched period {int(period['period_id']):02d} {crop} GEAS {geas_version}: {len(dataset):,} rows")

    file_rows: list[dict[str, Any]] = []
    for sid, frames in series_frames.items():
        out_path = series_paths[sid]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = pd.concat(frames, ignore_index=True).sort_values("reg_date")
        dataset.to_parquet(out_path, index=False)
        file_rows.append(
            {
                "file_path": str(out_path),
                "relative_path": str(out_path.relative_to(PROJECT_ROOT)),
                "row_count": len(dataset),
            }
        )
        print(f"series {sid}: {len(dataset):,} rows -> {out_path}")

    write_manifests(file_rows, plan, fetch_crop_cycles())
    print(f"series_manifest={SERIES_MANIFEST_PATH}")
    print(f"crop_cycle_manifest={CROP_CYCLE_MANIFEST_PATH}")
    print(f"segments={SEGMENTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
