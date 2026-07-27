from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "offline_dataset_preparation" / "datasets"
RAW_ROOT = DATASET_ROOT / "00_raw"
OUTPUT_ROOT = DATASET_ROOT / "01_resampled"
RAW_MANIFEST_PATH = RAW_ROOT / "manifest.csv"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"
SEGMENTS_PATH = OUTPUT_ROOT / "segments.csv"
NOTES_PATH = OUTPUT_ROOT / "preprocessing_notes.json"

GRID_FREQ = "5min"
EXPECTED_ROWS_PER_DAY = 288
MAX_SEGMENT_GAP_MIN = 10.0
MIN_SEGMENT_ROWS = EXPECTED_ROWS_PER_DAY

METADATA_COLUMNS = {
    "series_id",
    "segment_id",
    "segment_index",
    "crop",
    "geas_version",
    "source_segment_id",
    "source_file_path",
    "is_resampled_row",
    "dt_min",
}


def read_manifest_paths() -> list[Path]:
    manifest = pd.read_csv(RAW_MANIFEST_PATH)
    paths: list[Path] = []
    for value in manifest["file_paths"]:
        for part in str(value).split(";"):
            part = part.strip()
            if part:
                paths.append(PROJECT_ROOT / part)
    return list(dict.fromkeys(paths))


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def split_on_large_gaps(df: pd.DataFrame) -> pd.Series:
    gap_min = df["reg_date"].diff().dt.total_seconds().div(60)
    return gap_min.gt(MAX_SEGMENT_GAP_MIN).fillna(False).cumsum()


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
    out["dt_min"] = out["reg_date"].diff().dt.total_seconds().div(60)
    return out


def collect_resampled_parts() -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for path in read_manifest_paths():
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
        df = df.dropna(subset=["reg_date", "segment_id", "series_id"]).sort_values("reg_date")

        for source_segment_id, segment_df in df.groupby("segment_id", sort=False):
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
                "source_segment_id": part["source_segment_id"],
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
            file_paths=("file_path", lambda s: ";".join(dict.fromkeys(map(str, s)))),
        )
        .reset_index()
    )
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")

    notes = {
        "source": relative(RAW_ROOT),
        "output": relative(OUTPUT_ROOT),
        "grid_freq": GRID_FREQ,
        "max_segment_gap_min": MAX_SEGMENT_GAP_MIN,
        "gap_rule": "dt <= 10 minutes is aligned onto 5-minute grid; dt > 10 starts a new segment candidate",
        "minimum_segment_rows": MIN_SEGMENT_ROWS,
        "minimum_segment_duration": "at least one 24h day on 5-minute grid",
        "inserted_row_policy": "values introduced by 5-minute grid insertion are left as NaN",
        "sensor_fill": "none at this stage; missing sensor values remain NaN",
        "action_fill": "none at this stage; missing action values remain NaN",
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


if __name__ == "__main__":
    main()
