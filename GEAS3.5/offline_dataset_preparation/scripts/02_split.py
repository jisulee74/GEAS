from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "offline_dataset_preparation" / "datasets"
SOURCE_ROOT = DATASET_ROOT / "01_resampled"
OUTPUT_ROOT = DATASET_ROOT / "02_split"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "manifest.csv"
SPLIT_MANIFEST_PATH = OUTPUT_ROOT / "split_manifest.csv"
CROP_CYCLE_MANIFEST_PATH = DATASET_ROOT / "00_raw" / "crop_cycle_manifest.csv"
CROP_CYCLE_FILTER_SUMMARY_PATH = OUTPUT_ROOT / "crop_cycle_filter_summary.csv"

EXPECTED_ROWS_PER_DAY = 288
MIN_EPISODE_COVERAGE = 0.90
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def read_source_paths() -> list[Path]:
    manifest = pd.read_csv(SOURCE_MANIFEST_PATH)
    paths: list[Path] = []
    for value in manifest["file_paths"]:
        for part in str(value).split(";"):
            part = part.strip()
            if part:
                paths.append(PROJECT_ROOT / part)
    return list(dict.fromkeys(paths))


def load_resampled() -> pd.DataFrame:
    frames = []
    for path in read_source_paths():
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df["source_file_path"] = relative(path)
        frames.append(df)
    if not frames:
        raise RuntimeError("No resampled parquet data found. Run 01_resample.py first.")
    data = pd.concat(frames, ignore_index=True)
    data["reg_date"] = pd.to_datetime(data["reg_date"], errors="coerce")
    return data.dropna(subset=["reg_date", "crop", "segment_id"]).sort_values("reg_date")



def filter_to_effective_crop_cycles(
    data: pd.DataFrame,
    crop_cycles: pd.DataFrame | Path = CROP_CYCLE_MANIFEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only rows inside a declared transplant-to-effective-end interval.

    A missing row-level greenhouse id is filled only when the crop-cycle
    snapshot contains exactly one greenhouse, matching the growth-stage
    enrichment contract. Overlapping cycle matches fail closed.
    """

    cycles = (
        pd.read_csv(crop_cycles)
        if isinstance(crop_cycles, Path)
        else crop_cycles.copy()
    )
    required = {
        "crop_cycle_id", "iot_data_idx", "crop", "transplant_date",
        "effective_crop_end_date",
    }
    missing = sorted(required.difference(cycles.columns))
    if missing:
        raise KeyError(
            "Crop-cycle manifest is missing columns: " + ", ".join(missing)
        )

    frame = data.copy()
    frame["reg_date"] = pd.to_datetime(frame["reg_date"], errors="coerce")
    cycles["transplant_date"] = pd.to_datetime(
        cycles["transplant_date"], errors="coerce"
    )
    cycles["effective_crop_end_date"] = pd.to_datetime(
        cycles["effective_crop_end_date"], errors="coerce"
    )
    cycle_iot = pd.to_numeric(cycles["iot_data_idx"], errors="coerce")
    row_iot = pd.to_numeric(frame.get("iot_data_idx"), errors="coerce")
    unique_iot = cycle_iot.dropna().unique()
    if len(unique_iot) == 1:
        row_iot = row_iot.fillna(float(unique_iot[0]))

    match_count = pd.Series(0, index=frame.index, dtype="int64")
    matched_cycle = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    matched_transplant = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    matched_end = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    for cycle in cycles.itertuples(index=False):
        start = cycle.transplant_date
        if pd.isna(start) or pd.isna(cycle.iot_data_idx):
            continue
        mask = (
            row_iot.eq(float(cycle.iot_data_idx))
            & frame["crop"].astype(str).eq(str(cycle.crop))
            & frame["reg_date"].ge(start)
        )
        end = cycle.effective_crop_end_date
        if pd.notna(end):
            mask &= frame["reg_date"].le(end)
        mask = mask.fillna(False)
        match_count.loc[mask] += 1
        matched_cycle.loc[mask] = int(cycle.crop_cycle_id)
        matched_transplant.loc[mask] = start
        matched_end.loc[mask] = end

    overlaps = match_count.gt(1)
    if bool(overlaps.any()):
        raise ValueError(
            f"{int(overlaps.sum())} rows match overlapping effective crop cycles"
        )
    keep = match_count.eq(1)
    filtered = frame.loc[keep].copy()
    filtered["crop_cycle_id"] = matched_cycle.loc[keep]
    filtered["trans_crop_date"] = matched_transplant.loc[keep]
    filtered["effective_crop_end_date"] = matched_end.loc[keep]

    audit_rows: list[dict[str, Any]] = []
    for crop, crop_frame in frame.groupby("crop", sort=True):
        crop_keep = keep.loc[crop_frame.index]
        excluded = crop_frame.loc[~crop_keep]
        audit_rows.append(
            {
                "crop": crop,
                "input_rows": len(crop_frame),
                "matched_cycle_rows": int(crop_keep.sum()),
                "excluded_outside_cycle_rows": int((~crop_keep).sum()),
                "excluded_start": (
                    excluded["reg_date"].min() if not excluded.empty else pd.NaT
                ),
                "excluded_end": (
                    excluded["reg_date"].max() if not excluded.empty else pd.NaT
                ),
                "matched_crop_cycles": int(
                    matched_cycle.loc[crop_frame.index].dropna().nunique()
                ),
                "policy": (
                    "transplant_date <= reg_date <= effective_crop_end_date"
                ),
            }
        )
    return filtered.sort_values("reg_date"), pd.DataFrame(audit_rows)

def build_episode_table(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["episode_date"] = df["reg_date"].dt.date.astype(str)
    rows: list[dict[str, Any]] = []
    for (crop, segment_id, episode_date), group in df.groupby(
        ["crop", "segment_id", "episode_date"], sort=True
    ):
        start = group["reg_date"].min()
        end = group["reg_date"].max()
        n_rows = len(group)
        coverage = n_rows / EXPECTED_ROWS_PER_DAY
        rows.append(
            {
                "crop": crop,
                "segment_id": segment_id,
                "series_id": group["series_id"].iloc[0],
                "geas_version": group["geas_version"].iloc[0],
                "episode_date": episode_date,
                "episode_id": f"{segment_id}_{episode_date.replace('-', '')}",
                "episode_start": start,
                "episode_end": end,
                "n_rows": n_rows,
                "coverage": coverage,
                "is_valid_episode": coverage >= MIN_EPISODE_COVERAGE,
            }
        )
    return pd.DataFrame(rows).sort_values(["crop", "episode_start"])


def assign_splits(episodes: pd.DataFrame) -> pd.DataFrame:
    out = episodes.copy()
    out["split"] = pd.NA
    for crop, idx in out[out["is_valid_episode"]].groupby("crop").groups.items():
        crop_episodes = out.loc[list(idx)].sort_values("episode_start")
        n = len(crop_episodes)
        if n == 0:
            continue
        n_train = int(n * TRAIN_RATIO)
        n_validation = int(n * VALIDATION_RATIO)
        if n >= 3:
            n_train = max(1, n_train)
            n_validation = max(1, n_validation)
            if n_train + n_validation >= n:
                n_validation = max(0, n - n_train - 1)
        train_ids = crop_episodes.iloc[:n_train].index
        validation_ids = crop_episodes.iloc[n_train : n_train + n_validation].index
        test_ids = crop_episodes.iloc[n_train + n_validation :].index
        out.loc[train_ids, "split"] = "train"
        out.loc[validation_ids, "split"] = "validation"
        out.loc[test_ids, "split"] = "test"
    return out


def write_split_parquets(data: pd.DataFrame, episodes: pd.DataFrame) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    episode_split = episodes[episodes["split"].notna()][["episode_id", "split"]]
    data = data.copy()
    data["episode_date"] = data["reg_date"].dt.date.astype(str)
    data["episode_id"] = data["segment_id"].astype(str) + "_" + data["episode_date"].str.replace("-", "")
    data = data.merge(episode_split, on="episode_id", how="inner")

    manifest_rows = []
    for crop, crop_df in data.groupby("crop", sort=True):
        crop_dir = OUTPUT_ROOT / crop
        crop_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ["train", "validation", "test"]:
            split_df = crop_df[crop_df["split"] == split_name].sort_values("reg_date")
            out_path = crop_dir / f"{split_name}.parquet"
            split_df.to_parquet(out_path, index=False)
            split_episodes = episodes[(episodes["crop"] == crop) & (episodes["split"] == split_name)]
            manifest_rows.append(
                {
                    "crop": crop,
                    "split": split_name,
                    "n_rows": len(split_df),
                    "n_episodes": int(split_episodes["episode_id"].nunique()),
                    "actual_start": split_df["reg_date"].min() if not split_df.empty else pd.NaT,
                    "actual_end": split_df["reg_date"].max() if not split_df.empty else pd.NaT,
                    "file_path": relative(out_path),
                }
            )
            print(f"{crop} {split_name}: {len(split_df):,} rows -> {out_path}")

        episodes[episodes["crop"] == crop].to_csv(
            crop_dir / "episode_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(manifest_rows).to_csv(SPLIT_MANIFEST_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    data = load_resampled()
    data, crop_cycle_audit = filter_to_effective_crop_cycles(data)
    episodes = assign_splits(build_episode_table(data))
    write_split_parquets(data, episodes)
    crop_cycle_audit.to_csv(
        CROP_CYCLE_FILTER_SUMMARY_PATH, index=False, encoding="utf-8-sig"
    )
    print(f"crop_cycle_filter_summary={CROP_CYCLE_FILTER_SUMMARY_PATH}")
    print(f"split_manifest={SPLIT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
