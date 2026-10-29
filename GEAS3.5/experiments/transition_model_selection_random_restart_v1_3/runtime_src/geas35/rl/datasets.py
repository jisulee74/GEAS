"""Build the Step 11.5 RL-ready datasets from verified QC outputs."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from geas35.io_utils import existing_crops, write_json
from geas35.preprocessing import TIME_COLUMN
from geas35.rl.mdp_v1 import (
    MDP_V1_ACTION_COLUMNS,
    MDP_V1_ROLLOUT_ID_COLUMN,
    MDP_V1_STEP_MINUTES,
    MDP_V1_TRANSITION_OBSERVATION_COLUMNS,
    MDP_V1_TRANSITION_TARGET_COLUMNS,
    MDP_V1_VALID_TRANSITION_COLUMN,
    MdpV1Config,
    MdpV1ObservationScaler,
    compute_mdp_v1_reward,
    fit_mdp_v1_observation_scaler,
    logged_mdp_v1_action,
    mdp_v1_observation_columns,
    prepare_mdp_v1_frame,
)

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "offline_dataset_preparation" / "datasets"
DEFAULT_INPUT_DIR_NAME = "03_quality_controlled"
DEFAULT_OUTPUT_DIR_NAME = "5_rl_dataset"
DEFAULT_SPLITS = ("train", "validation", "test")

# The seven logged columns are the sole sources of the official six-dimensional
# action. Unrelated pump and three-way-valve logs must not exclude an RL row.
MDP_V1_REQUIRED_SOURCE_ACTION_COLUMNS = (
    "cont_skyl_vol",
    "cont_skyr_vol",
    "cont_cur_vol",
    "cont_kwcur_vol",
    "cont_heater_run",
    "cont_cooler_run",
    "cont_fan_run",
)

RL_REWARD_COLUMN = "reward"
RL_DONE_COLUMN = "done"
RL_VALID_TRANSITION_COLUMN = "rl_valid_transition"
RL_EXCLUSION_REASON_COLUMN = "rl_exclusion_reason"
NEXT_OBSERVATION_PREFIX = "next_"
REWARD_TERM_PREFIX = "reward_term_"
EXCLUSION_REASON_ORDER = (
    "action_nan",
    "time_discontinuity",
    "episode_or_series_boundary",
    "required_state_missing",
    "target_missing",
)


@dataclass(frozen=True)
class RlDatasetSplitSummary:
    crop: str
    split: str
    input_path: str
    output_path: str
    input_rows: int
    mdp_rows: int
    output_rows: int
    excluded_rows: int
    exclusion_reason_counts: dict[str, int]
    exclusion_reason_flag_counts: dict[str, int]
    observation_columns: list[str]
    action_columns: list[str]
    transition_target_columns: list[str]
    required_source_action_columns: list[str]

    @property
    def excluded_action_nan_rows(self) -> int:
        return int(self.exclusion_reason_flag_counts.get("action_nan", 0))

    @property
    def excluded_invalid_transition_rows(self) -> int:
        return self.excluded_rows - int(self.exclusion_reason_counts.get("action_nan", 0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_action_nan_mask(frame: pd.DataFrame, required_action_columns: Sequence[str]) -> pd.Series:
    missing = [column for column in required_action_columns if column not in frame.columns]
    if missing:
        raise ValueError("RL dataset source frame is missing action columns: " + ", ".join(missing))
    values = frame.loc[:, list(required_action_columns)].apply(pd.to_numeric, errors="coerce")
    return ~np.isfinite(values.to_numpy(dtype=float)).all(axis=1)


def _finite_mask(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError("RL dataset frame is missing required columns: " + ", ".join(missing))
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(values.to_numpy(dtype=float)).all(axis=1), index=frame.index)


def _transition_exclusion_masks(
    frame: pd.DataFrame,
    observation_columns: Sequence[str],
    required_action_columns: Sequence[str],
) -> dict[str, pd.Series]:
    timestamps = pd.to_datetime(frame[TIME_COLUMN], errors="coerce")
    next_timestamps = timestamps.shift(-1)
    has_next = next_timestamps.notna()
    step_minutes = next_timestamps.sub(timestamps).dt.total_seconds().div(60.0)
    time_discontinuity = has_next & ~step_minutes.eq(MDP_V1_STEP_MINUTES)

    boundary = has_next & ~timestamps.dt.date.eq(next_timestamps.dt.date).fillna(False)
    for column in ("crop", "series_id", "segment_id", "episode_id"):
        if column in frame.columns:
            current = frame[column].astype("string")
            boundary |= has_next & ~current.eq(current.shift(-1)).fillna(False)

    return {
        "action_nan": pd.Series(
            _source_action_nan_mask(frame, required_action_columns), index=frame.index
        ).astype(bool),
        "time_discontinuity": time_discontinuity.astype(bool),
        "episode_or_series_boundary": boundary.astype(bool),
        "required_state_missing": (~_finite_mask(frame, observation_columns)).astype(bool),
        "target_missing": (~_finite_mask(frame, MDP_V1_TRANSITION_TARGET_COLUMNS)).astype(bool),
    }


def _exclusive_reasons(masks: Mapping[str, pd.Series], index: pd.Index) -> pd.Series:
    reasons = pd.Series("", index=index, dtype="string")
    for reason in EXCLUSION_REASON_ORDER:
        select = reasons.eq("") & masks[reason]
        reasons.loc[select] = reason
    return reasons


def _logged_action_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [logged_mdp_v1_action(row) for _, row in frame.iterrows()]
    return pd.DataFrame(rows, index=frame.index, columns=MDP_V1_ACTION_COLUMNS)


def _reward_columns(
    frame: pd.DataFrame,
    valid_mask: pd.Series,
    action_frame: pd.DataFrame,
    *,
    config: MdpV1Config | None = None,
) -> pd.DataFrame:
    rewards = pd.Series(np.nan, index=frame.index, dtype="float64")
    term_values: dict[str, pd.Series] = {}
    rollout_ids = frame[MDP_V1_ROLLOUT_ID_COLUMN]
    for position, index in enumerate(frame.index):
        if not bool(valid_mask.loc[index]) or position >= len(frame) - 1:
            continue
        same_previous = position > 0 and rollout_ids.iloc[position - 1] == rollout_ids.iloc[position]
        prev_position = position - 1 if same_previous else position
        same_previous_previous = (
            prev_position > 0
            and rollout_ids.iloc[prev_position - 1] == rollout_ids.iloc[position]
        )
        prev_prev_position = prev_position - 1 if same_previous_previous else prev_position
        reward, terms = compute_mdp_v1_reward(
            frame.iloc[position + 1],
            action_frame.iloc[position].to_dict(),
            prev_action=action_frame.iloc[prev_position].to_dict(),
            prev_prev_action=action_frame.iloc[prev_prev_position].to_dict(),
            config=config,
        )
        rewards.loc[index] = reward
        for key, value in terms.items():
            column = f"{REWARD_TERM_PREFIX}{key}"
            term_values.setdefault(
                column, pd.Series(np.nan, index=frame.index, dtype="float64")
            ).loc[index] = value
    return pd.concat(
        [pd.DataFrame({RL_REWARD_COLUMN: rewards}), pd.DataFrame(term_values)], axis=1
    )


def _next_observation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(MDP_V1_TRANSITION_OBSERVATION_COLUMNS)
    return frame.loc[:, columns].shift(-1).rename(
        columns={column: f"{NEXT_OBSERVATION_PREFIX}{column}" for column in columns}
    )


def _metadata_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        TIME_COLUMN, "crop", "series_id", "segment_id", "episode_id",
        MDP_V1_ROLLOUT_ID_COLUMN, MDP_V1_VALID_TRANSITION_COLUMN,
    ]
    return [column for column in candidates if column in frame.columns]


def prepare_rl_dataset_frame(
    df_raw: pd.DataFrame,
    *,
    config: MdpV1Config | None = None,
    required_source_action_columns: Sequence[str] = MDP_V1_REQUIRED_SOURCE_ACTION_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create unscaled valid transitions and auditable row-exclusion counts."""
    if df_raw is None or df_raw.empty:
        summary = {
            "input_rows": 0 if df_raw is None else len(df_raw), "mdp_rows": 0,
            "output_rows": 0, "excluded_rows": 0,
            "exclusion_reason_counts": {reason: 0 for reason in EXCLUSION_REASON_ORDER},
            "exclusion_reason_flag_counts": {reason: 0 for reason in EXCLUSION_REASON_ORDER},
            "observation_columns": [], "action_columns": list(MDP_V1_ACTION_COLUMNS),
            "transition_target_columns": list(MDP_V1_TRANSITION_TARGET_COLUMNS),
            "required_source_action_columns": list(required_source_action_columns),
        }
        return pd.DataFrame(), summary

    frame = prepare_mdp_v1_frame(df_raw, config=config)
    observation_columns = mdp_v1_observation_columns(frame)
    masks = _transition_exclusion_masks(frame, observation_columns, required_source_action_columns)
    reasons = _exclusive_reasons(masks, frame.index)
    valid_mask = reasons.eq("")

    # Keep the canonical MDP validity computation and the explicit reason audit in lockstep.
    mdp_valid = pd.to_numeric(
        frame[MDP_V1_VALID_TRANSITION_COLUMN], errors="coerce"
    ).fillna(0).astype(bool)
    expected_valid = mdp_valid & ~masks["action_nan"]
    mismatch = int(
        np.count_nonzero(
            valid_mask.to_numpy(dtype=bool) != expected_valid.to_numpy(dtype=bool)
        )
    )
    if mismatch:
        raise RuntimeError(f"Explicit exclusion provenance disagrees with MDP validity for {mismatch} rows")

    action_frame = _logged_action_frame(frame)
    rewards = _reward_columns(frame, valid_mask, action_frame, config=config)
    next_valid = valid_mask.shift(-1).fillna(False).astype(bool)
    assembled = pd.concat(
        [
            frame.loc[:, _metadata_columns(frame)],
            frame.loc[:, observation_columns],
            action_frame,
            frame.loc[:, list(MDP_V1_TRANSITION_TARGET_COLUMNS)],
            _next_observation_frame(frame),
            rewards,
            pd.DataFrame(
                {
                    RL_DONE_COLUMN: (~next_valid).astype(int),
                    RL_VALID_TRANSITION_COLUMN: valid_mask.astype(int),
                    RL_EXCLUSION_REASON_COLUMN: reasons,
                }, index=frame.index,
            ),
        ], axis=1,
    )
    output = assembled.loc[valid_mask].reset_index(drop=True)
    ordered = [
        *[column for column in _metadata_columns(frame) if column in output.columns],
        *observation_columns,
        *MDP_V1_ACTION_COLUMNS,
        *MDP_V1_TRANSITION_TARGET_COLUMNS,
        *[f"{NEXT_OBSERVATION_PREFIX}{column}" for column in MDP_V1_TRANSITION_OBSERVATION_COLUMNS],
        *[column for column in output.columns if column.startswith(REWARD_TERM_PREFIX)],
        RL_REWARD_COLUMN, RL_DONE_COLUMN, RL_VALID_TRANSITION_COLUMN,
    ]
    output = output.loc[:, ordered]
    exclusive_counts = {
        reason: int(reasons.eq(reason).sum()) for reason in EXCLUSION_REASON_ORDER
    }
    flag_counts = {reason: int(mask.sum()) for reason, mask in masks.items()}
    summary = {
        "input_rows": len(df_raw), "mdp_rows": len(frame), "output_rows": len(output),
        "excluded_rows": int((~valid_mask).sum()),
        "exclusion_reason_counts": exclusive_counts,
        "exclusion_reason_flag_counts": flag_counts,
        "observation_columns": list(observation_columns),
        "action_columns": list(MDP_V1_ACTION_COLUMNS),
        "transition_target_columns": list(MDP_V1_TRANSITION_TARGET_COLUMNS),
        "required_source_action_columns": list(required_source_action_columns),
    }
    return output, summary


def _summary_record(crop: str, split: str, input_path: Path, output_path: Path, summary: Mapping[str, object]) -> RlDatasetSplitSummary:
    return RlDatasetSplitSummary(
        crop=crop, split=split, input_path=str(input_path), output_path=str(output_path),
        input_rows=int(summary["input_rows"]), mdp_rows=int(summary["mdp_rows"]),
        output_rows=int(summary["output_rows"]), excluded_rows=int(summary["excluded_rows"]),
        exclusion_reason_counts=dict(summary["exclusion_reason_counts"]),
        exclusion_reason_flag_counts=dict(summary["exclusion_reason_flag_counts"]),
        observation_columns=list(summary["observation_columns"]),
        action_columns=list(summary["action_columns"]),
        transition_target_columns=list(summary["transition_target_columns"]),
        required_source_action_columns=list(summary["required_source_action_columns"]),
    )


def prepare_rl_dataset_split_file(
    input_path: Path,
    output_path: Path,
    *,
    crop: str,
    split: str,
    config: MdpV1Config | None = None,
    required_source_action_columns: Sequence[str] = MDP_V1_REQUIRED_SOURCE_ACTION_COLUMNS,
    scaler: MdpV1ObservationScaler | None = None,
) -> RlDatasetSplitSummary:
    """Prepare one split; callers must pass the crop's Train-fitted scaler."""
    output, summary = prepare_rl_dataset_frame(
        pd.read_parquet(input_path), config=config,
        required_source_action_columns=required_source_action_columns,
    )
    if scaler is not None:
        output = scaler.transform_frame(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return _summary_record(crop, split, input_path, output_path, summary)


def _scaler_payload(
    scaler: MdpV1ObservationScaler,
    *,
    crop: str,
    train_path: Path,
    train_valid_rows: int,
) -> dict[str, object]:
    return {
        "schema_version": "geas35.rl_observation_scaler.step11.5.v1",
        "crop": crop,
        "fit_split": "train",
        "fit_valid_transition_rows": train_valid_rows,
        "columns": list(scaler.columns),
        "mean": {column: float(scaler.means[column]) for column in scaler.columns},
        "scale": {column: float(scaler.scales[column]) for column in scaler.columns},
        "raw_train": {"path": str(train_path), "sha256": _sha256(train_path)},
    }


def prepare_rl_dataset_splits(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    input_dir_name: str = DEFAULT_INPUT_DIR_NAME,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    crops: Iterable[str] | None = None,
    splits: Iterable[str] = DEFAULT_SPLITS,
    config: MdpV1Config | None = None,
    required_source_action_columns: Sequence[str] = MDP_V1_REQUIRED_SOURCE_ACTION_COLUMNS,
) -> list[RlDatasetSplitSummary]:
    """Build all Step 11.5 outputs with one Train-only scaler per crop."""
    dataset_root = Path(dataset_root)
    input_root = dataset_root / input_dir_name
    output_root = dataset_root / output_dir_name
    crop_names = list(crops) if crops is not None else existing_crops(input_root)
    split_names = list(splits)
    if set(split_names) != set(DEFAULT_SPLITS):
        raise ValueError("Step 11.5 requires exactly train, validation, and test splits")

    summaries: list[RlDatasetSplitSummary] = []
    scaler_artifacts: dict[str, str] = {}
    reference_schema: list[str] | None = None
    for crop in crop_names:
        crop_output_root = output_root / crop
        train_path = input_root / crop / "train.parquet"
        if not train_path.is_file():
            raise FileNotFoundError(f"Verified QC Train parquet not found: {train_path}")
        train_output, train_summary = prepare_rl_dataset_frame(
            pd.read_parquet(train_path), config=config,
            required_source_action_columns=required_source_action_columns,
        )
        if train_output.empty:
            raise ValueError(f"Step 11.5 produced an empty RL dataset: {crop}/train")
        scaler = fit_mdp_v1_observation_scaler(
            train_output, observation_columns=train_summary["observation_columns"]
        )
        scaler_path = crop_output_root / "observation_scaler.json"
        write_json(
            scaler_path,
            _scaler_payload(
                scaler, crop=crop, train_path=train_path,
                train_valid_rows=int(train_summary["output_rows"]),
            ),
        )
        scaler_artifacts[crop] = str(scaler_path)

        for split in split_names:
            input_path = input_root / crop / f"{split}.parquet"
            if not input_path.is_file():
                raise FileNotFoundError(f"Verified QC parquet not found: {input_path}")
            if split == "train":
                raw_output, summary = train_output, train_summary
            else:
                raw_output, summary = prepare_rl_dataset_frame(
                    pd.read_parquet(input_path), config=config,
                    required_source_action_columns=required_source_action_columns,
                )
            if raw_output.empty:
                raise ValueError(f"Step 11.5 produced an empty RL dataset: {crop}/{split}")
            if list(summary["transition_target_columns"]) != list(MDP_V1_TRANSITION_TARGET_COLUMNS):
                raise ValueError(f"Non-canonical three-target schema: {crop}/{split}")
            output = scaler.transform_frame(raw_output)
            if not np.isfinite(output.loc[:, list(scaler.columns)].to_numpy(dtype=float)).all():
                raise ValueError(f"Scaled State contains non-finite values: {crop}/{split}")
            if reference_schema is None:
                reference_schema = list(output.columns)
            elif list(output.columns) != reference_schema:
                raise ValueError(f"RL dataset schema differs across crop/split: {crop}/{split}")
            output_path = crop_output_root / f"{split}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output.to_parquet(output_path, index=False)
            summaries.append(_summary_record(crop, split, input_path, output_path, summary))

    manifest = {
        "schema_version": "geas35.rl_dataset.step11.5.v1",
        "stage": "11.5_rl_ready_dataset_scaling_and_row_exclusion_provenance",
        "status": "passed",
        "input_root": str(input_root), "output_root": str(output_root),
        "policy": {
            "observation_source": "verified_quality_controlled",
            "action_source": "logged_sources_for_official_six_actions",
            "action_nan_policy": "exclude_transition_start",
            "ai_action_imputation": False,
            "scaler_fit_scope": "crop_train_valid_transitions_only",
            "validation_test_scaler": "reuse_crop_train_scaler",
        },
        "required_source_action_columns": list(required_source_action_columns),
        "transition_target_columns": list(MDP_V1_TRANSITION_TARGET_COLUMNS),
        "output_schema": reference_schema or [],
        "scaler_artifacts": scaler_artifacts,
        "splits": [asdict(summary) for summary in summaries],
    }
    write_json(output_root / "rl_dataset_manifest.json", manifest)
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Step 11.5 RL transition datasets")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--input-dir-name", default=DEFAULT_INPUT_DIR_NAME)
    parser.add_argument("--output-dir-name", default=DEFAULT_OUTPUT_DIR_NAME)
    parser.add_argument("--crops", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summaries = prepare_rl_dataset_splits(
        dataset_root=args.dataset_root, input_dir_name=args.input_dir_name,
        output_dir_name=args.output_dir_name, crops=args.crops, splits=args.splits,
    )
    for summary in summaries:
        print(
            f"[rl_dataset] {summary.crop}/{summary.split}: "
            f"{summary.input_rows} input -> {summary.output_rows} transitions; "
            f"excluded={summary.exclusion_reason_counts}; wrote {summary.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
