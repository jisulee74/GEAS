"""Prepare RL-ready transition datasets from clean quality-model outputs.

Step 9 starts after the AI Quality Model stage. Inputs are expected under
``4_preprocessed/3_missing_outliers_handled`` and outputs are written to
``5_rl_dataset``. Rows with unresolved action NaNs are excluded only here; this
preserves preprocessing data integrity while keeping the RL dataset action-safe.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from geas35.offline._common import existing_crops, write_json
from geas35.preprocessing import ACTION_COLUMNS, TIME_COLUMN
from geas35.rl import (
    MDP_V1_ACTION_COLUMNS,
    MDP_V1_ROLLOUT_ID_COLUMN,
    MDP_V1_TRANSITION_TARGET_COLUMNS,
    MDP_V1_VALID_TRANSITION_COLUMN,
    MdpV1Config,
    compute_mdp_v1_reward,
    logged_mdp_v1_action,
    mdp_v1_observation_columns,
    prepare_mdp_v1_frame,
)


DEFAULT_DATASET_ROOT = Path("../datasets/iot97_historical")
DEFAULT_INPUT_DIR_NAME = "4_preprocessed/3_missing_outliers_handled"
DEFAULT_OUTPUT_DIR_NAME = "5_rl_dataset"
DEFAULT_SPLITS = ("train", "validation", "test")

RL_REWARD_COLUMN = "reward"
RL_DONE_COLUMN = "done"
RL_VALID_TRANSITION_COLUMN = "rl_valid_transition"
RL_EXCLUSION_REASON_COLUMN = "rl_exclusion_reason"
NEXT_OBSERVATION_PREFIX = "next_"
REWARD_TERM_PREFIX = "reward_term_"


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
    excluded_action_nan_rows: int
    excluded_invalid_transition_rows: int
    observation_columns: list[str]
    action_columns: list[str]
    transition_target_columns: list[str]
    required_source_action_columns: list[str]


def _source_action_nan_mask(
    frame: pd.DataFrame,
    required_action_columns: Sequence[str],
) -> pd.Series:
    missing = [col for col in required_action_columns if col not in frame.columns]
    if missing:
        preview = ", ".join(missing[:5])
        if len(missing) > 5:
            preview = f"{preview}, ..."
        raise ValueError(f"RL dataset source frame is missing action columns: {preview}")

    action_values = frame.loc[:, list(required_action_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return action_values.isna().any(axis=1)


def _logged_action_frame(frame: pd.DataFrame) -> pd.DataFrame:
    action_rows = [logged_mdp_v1_action(row) for _, row in frame.iterrows()]
    return pd.DataFrame(action_rows, index=frame.index, columns=MDP_V1_ACTION_COLUMNS)


def _reward_columns(
    frame: pd.DataFrame,
    valid_mask: pd.Series,
    action_frame: pd.DataFrame,
    *,
    config: MdpV1Config | None = None,
) -> pd.DataFrame:
    rewards = pd.Series(np.nan, index=frame.index, dtype="float64")
    term_values: dict[str, pd.Series] = {}

    for position, index in enumerate(frame.index):
        if not bool(valid_mask.loc[index]) or position >= len(frame) - 1:
            continue

        current_action = action_frame.loc[index].to_dict()
        prev_index = frame.index[position - 1] if position > 0 else index
        prev_prev_index = frame.index[position - 2] if position > 1 else prev_index
        prev_action = action_frame.loc[prev_index].to_dict()
        prev_prev_action = action_frame.loc[prev_prev_index].to_dict()
        reward, terms = compute_mdp_v1_reward(
            frame.iloc[position + 1],
            current_action,
            prev_action=prev_action,
            prev_prev_action=prev_prev_action,
            config=config,
        )
        rewards.loc[index] = reward
        for key, value in terms.items():
            col = f"{REWARD_TERM_PREFIX}{key}"
            if col not in term_values:
                term_values[col] = pd.Series(np.nan, index=frame.index, dtype="float64")
            term_values[col].loc[index] = value

    return pd.concat(
        [pd.DataFrame({RL_REWARD_COLUMN: rewards}), pd.DataFrame(term_values)],
        axis=1,
    )


def _next_observation_frame(
    frame: pd.DataFrame,
    observation_columns: Sequence[str],
) -> pd.DataFrame:
    return frame.loc[:, list(observation_columns)].shift(-1).rename(
        columns={col: f"{NEXT_OBSERVATION_PREFIX}{col}" for col in observation_columns}
    )


def _metadata_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        TIME_COLUMN,
        "crop",
        "series_id",
        "segment_id",
        "episode_id",
        MDP_V1_ROLLOUT_ID_COLUMN,
        MDP_V1_VALID_TRANSITION_COLUMN,
    ]
    return [col for col in candidates if col in frame.columns]


def prepare_rl_dataset_frame(
    df_raw: pd.DataFrame,
    *,
    config: MdpV1Config | None = None,
    required_source_action_columns: Sequence[str] = tuple(ACTION_COLUMNS),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return an RL-ready transition table and filter summary.

    ``required_source_action_columns`` defaults to all GEAS action columns. This
    matches the preprocessing policy that unresolved action NaNs are preserved
    until RL dataset creation, where affected transition starts are excluded.
    """

    if df_raw is None or df_raw.empty:
        empty_summary = {
            "input_rows": 0 if df_raw is None else len(df_raw),
            "mdp_rows": 0,
            "output_rows": 0,
            "excluded_rows": 0,
            "excluded_action_nan_rows": 0,
            "excluded_invalid_transition_rows": 0,
            "observation_columns": [],
            "action_columns": list(MDP_V1_ACTION_COLUMNS),
            "transition_target_columns": list(MDP_V1_TRANSITION_TARGET_COLUMNS),
            "required_source_action_columns": list(required_source_action_columns),
        }
        return pd.DataFrame(), empty_summary

    frame = prepare_mdp_v1_frame(df_raw, config=config)
    if frame.empty:
        empty_summary = {
            "input_rows": len(df_raw),
            "mdp_rows": 0,
            "output_rows": 0,
            "excluded_rows": len(df_raw),
            "excluded_action_nan_rows": 0,
            "excluded_invalid_transition_rows": 0,
            "observation_columns": [],
            "action_columns": list(MDP_V1_ACTION_COLUMNS),
            "transition_target_columns": list(MDP_V1_TRANSITION_TARGET_COLUMNS),
            "required_source_action_columns": list(required_source_action_columns),
        }
        return pd.DataFrame(), empty_summary

    action_nan = _source_action_nan_mask(frame, required_source_action_columns)
    mdp_valid = (
        pd.to_numeric(frame[MDP_V1_VALID_TRANSITION_COLUMN], errors="coerce")
        .fillna(0)
        .astype(bool)
    )
    valid_mask = mdp_valid & ~action_nan

    observation_columns = mdp_v1_observation_columns(frame)
    action_frame = _logged_action_frame(frame)
    next_observations = _next_observation_frame(frame, observation_columns)
    rewards = _reward_columns(frame, valid_mask, action_frame, config=config)
    next_valid_dataset_start = valid_mask.shift(-1).fillna(False).astype(bool)

    assembled = pd.concat(
        [
            frame.loc[:, _metadata_columns(frame)],
            frame.loc[:, observation_columns],
            action_frame,
            frame.loc[:, list(MDP_V1_TRANSITION_TARGET_COLUMNS)],
            next_observations,
            rewards,
            pd.DataFrame(
                {
                    RL_DONE_COLUMN: (~next_valid_dataset_start).astype(int),
                    RL_VALID_TRANSITION_COLUMN: valid_mask.astype(int),
                    RL_EXCLUSION_REASON_COLUMN: np.where(
                        action_nan,
                        "action_nan",
                        np.where(mdp_valid, "", "invalid_transition"),
                    ),
                },
                index=frame.index,
            ),
        ],
        axis=1,
    )

    output = assembled.loc[valid_mask].reset_index(drop=True)
    ordered_cols = [
        *[col for col in _metadata_columns(frame) if col in output.columns],
        *[col for col in observation_columns if col in output.columns],
        *[col for col in MDP_V1_ACTION_COLUMNS if col in output.columns],
        *[
            col
            for col in MDP_V1_TRANSITION_TARGET_COLUMNS
            if col in output.columns
        ],
        *[
            f"{NEXT_OBSERVATION_PREFIX}{col}"
            for col in observation_columns
            if f"{NEXT_OBSERVATION_PREFIX}{col}" in output.columns
        ],
        *[col for col in output.columns if col.startswith(REWARD_TERM_PREFIX)],
        RL_REWARD_COLUMN,
        RL_DONE_COLUMN,
        RL_VALID_TRANSITION_COLUMN,
    ]
    output = output.loc[:, [col for col in ordered_cols if col in output.columns]]

    summary = {
        "input_rows": len(df_raw),
        "mdp_rows": len(frame),
        "output_rows": len(output),
        "excluded_rows": int((~valid_mask).sum()),
        "excluded_action_nan_rows": int(action_nan.sum()),
        "excluded_invalid_transition_rows": int((~mdp_valid).sum()),
        "observation_columns": list(observation_columns),
        "action_columns": list(MDP_V1_ACTION_COLUMNS),
        "transition_target_columns": list(MDP_V1_TRANSITION_TARGET_COLUMNS),
        "required_source_action_columns": list(required_source_action_columns),
    }
    return output, summary


def prepare_rl_dataset_split_file(
    input_path: Path,
    output_path: Path,
    *,
    crop: str,
    split: str,
    config: MdpV1Config | None = None,
    required_source_action_columns: Sequence[str] = tuple(ACTION_COLUMNS),
) -> RlDatasetSplitSummary:
    """Prepare one clean split parquet and write an RL transition parquet."""

    df_input = pd.read_parquet(input_path)
    df_output, summary = prepare_rl_dataset_frame(
        df_input,
        config=config,
        required_source_action_columns=required_source_action_columns,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_parquet(output_path, index=False)

    return RlDatasetSplitSummary(
        crop=crop,
        split=split,
        input_path=str(input_path),
        output_path=str(output_path),
        input_rows=int(summary["input_rows"]),
        mdp_rows=int(summary["mdp_rows"]),
        output_rows=int(summary["output_rows"]),
        excluded_rows=int(summary["excluded_rows"]),
        excluded_action_nan_rows=int(summary["excluded_action_nan_rows"]),
        excluded_invalid_transition_rows=int(summary["excluded_invalid_transition_rows"]),
        observation_columns=list(summary["observation_columns"]),
        action_columns=list(summary["action_columns"]),
        transition_target_columns=list(summary["transition_target_columns"]),
        required_source_action_columns=list(summary["required_source_action_columns"]),
    )


def prepare_rl_dataset_splits(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    input_dir_name: str = DEFAULT_INPUT_DIR_NAME,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    crops: Iterable[str] | None = None,
    splits: Iterable[str] = DEFAULT_SPLITS,
    config: MdpV1Config | None = None,
    required_source_action_columns: Sequence[str] = tuple(ACTION_COLUMNS),
) -> list[RlDatasetSplitSummary]:
    """Prepare RL-ready transition parquet files for selected splits."""

    dataset_root = Path(dataset_root)
    input_root = dataset_root / input_dir_name
    output_root = dataset_root / output_dir_name
    crop_names = list(crops) if crops is not None else existing_crops(input_root)
    split_names = list(splits)

    summaries: list[RlDatasetSplitSummary] = []
    for crop in crop_names:
        for split in split_names:
            input_path = input_root / crop / f"{split}.parquet"
            if not input_path.exists():
                raise FileNotFoundError(f"Clean quality parquet not found: {input_path}")

            output_path = output_root / crop / f"{split}.parquet"
            summaries.append(
                prepare_rl_dataset_split_file(
                    input_path,
                    output_path,
                    crop=crop,
                    split=split,
                    config=config,
                    required_source_action_columns=required_source_action_columns,
                )
            )

    manifest_path = output_root / "rl_dataset_manifest.json"
    manifest = {
        "stage": "rl_dataset",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "policy": {
            "observation_source": "quality_model_clean_observations",
            "action_source": "original_or_action_log_restored_actions_only",
            "action_nan_policy": "exclude_transition_start",
            "ai_action_imputation": False,
        },
        "required_source_action_columns": list(required_source_action_columns),
        "splits": [asdict(summary) for summary in summaries],
    }
    write_json(manifest_path, manifest)
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Step 9 RL transition datasets from clean GEAS datasets."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root of the iot97_historical dataset directory.",
    )
    parser.add_argument(
        "--input-dir-name",
        default=DEFAULT_INPUT_DIR_NAME,
        help="Clean quality-model input directory under dataset root.",
    )
    parser.add_argument(
        "--output-dir-name",
        default=DEFAULT_OUTPUT_DIR_NAME,
        help="RL dataset output directory under dataset root.",
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
        help="Split names to process. Defaults to train validation test.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summaries = prepare_rl_dataset_splits(
        dataset_root=args.dataset_root,
        input_dir_name=args.input_dir_name,
        output_dir_name=args.output_dir_name,
        crops=args.crops,
        splits=args.splits,
    )

    for summary in summaries:
        print(
            f"[rl_dataset] {summary.crop}/{summary.split}: "
            f"{summary.input_rows} input rows -> {summary.output_rows} transitions, "
            f"{summary.excluded_action_nan_rows} action-NaN rows excluded, "
            f"wrote {summary.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
