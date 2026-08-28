"""Dataset builders for GEAS transition models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from geas35.models.transition.base import TransitionDataset
from geas35.models.transition.features import (
    TransitionFeatureSchema,
    TransitionTargetPolicy,
    resolve_transition_feature_schema,
)
from geas35.rl import (
    MDP_V1_ACTION_COLUMNS,
    MDP_V1_BINARY_ACTION_COLUMNS,
    MDP_V1_CONTINUOUS_ACTION_COLUMNS,
    MDP_V1_ROLLOUT_ID_COLUMN,
    MDP_V1_VALID_TRANSITION_COLUMN,
)

RL_VALID_TRANSITION_COLUMN = "rl_valid_transition"
RL_DONE_COLUMN = "done"

DEFAULT_METADATA_COLUMNS = (
    "reg_date",
    "timestamp",
    "crop",
    "greenhouse",
    "greenhouse_id",
    "series_id",
    "segment_id",
    "episode_id",
    MDP_V1_ROLLOUT_ID_COLUMN,
    MDP_V1_VALID_TRANSITION_COLUMN,
    RL_VALID_TRANSITION_COLUMN,
    RL_DONE_COLUMN,
)


@dataclass(frozen=True)
class TransitionDatasetBuildSummary:
    """Summary of an RL-frame to TransitionDataset conversion."""

    input_rows: int
    output_rows: int
    excluded_invalid_transition_rows: int
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    next_target_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    missing_dynamic_target_columns: tuple[str, ...]


@dataclass(frozen=True)
class TransitionDatasetBuildResult:
    """Dataset builder output with schema and conversion summary."""

    dataset: TransitionDataset
    feature_schema: TransitionFeatureSchema
    summary: TransitionDatasetBuildSummary


@dataclass(frozen=True)
class TransitionDatasetBuilder:
    """Build transition-model datasets from RL-ready parquet frames."""

    target_policy: TransitionTargetPolicy = TransitionTargetPolicy()
    valid_transition_column: str = RL_VALID_TRANSITION_COLUMN
    metadata_columns: tuple[str, ...] = DEFAULT_METADATA_COLUMNS

    def from_rl_dataset_path(
        self,
        path: Path | str,
        *,
        observation_columns: Sequence[str] | None = None,
        action_columns: Sequence[str] | None = None,
    ) -> TransitionDatasetBuildResult:
        """Read an RL parquet file and build a TransitionDataset."""

        return self.from_rl_dataset_frame(
            pd.read_parquet(path),
            observation_columns=observation_columns,
            action_columns=action_columns,
        )

    def from_rl_dataset_frame(
        self,
        frame: pd.DataFrame,
        *,
        observation_columns: Sequence[str] | None = None,
        action_columns: Sequence[str] | None = None,
    ) -> TransitionDatasetBuildResult:
        """Build a TransitionDataset from an RL-ready transition frame."""

        if frame is None or frame.empty:
            raise ValueError("Transition dataset source frame must not be empty.")

        schema = resolve_transition_feature_schema(
            frame,
            observation_columns=observation_columns,
            action_columns=action_columns,
            policy=self.target_policy,
        )
        if not schema.dynamic_target_columns:
            raise ValueError("No dynamic transition target columns were resolved.")

        valid_mask = self._valid_transition_mask(frame)
        valid_frame = frame.loc[valid_mask].reset_index(drop=True)
        _validate_official_action_contract(valid_frame)
        excluded_invalid = int((~valid_mask).sum())
        if valid_frame.empty:
            raise ValueError("No valid transition rows are available.")

        x = valid_frame.loc[:, list(schema.input_columns)].copy()
        y = valid_frame.loc[:, list(schema.next_target_columns)].copy()
        y.columns = list(schema.dynamic_target_columns)

        metadata_cols = tuple(
            column for column in self.metadata_columns if column in valid_frame.columns
        )
        metadata = valid_frame.loc[:, list(metadata_cols)].copy()
        dataset = TransitionDataset(
            x=x,
            y=y,
            metadata=metadata,
            input_columns=schema.input_columns,
            target_columns=schema.dynamic_target_columns,
            observation_columns=tuple(
                column for column in schema.input_columns if column.startswith("obs_")
            ),
            action_columns=tuple(
                column for column in schema.input_columns if not column.startswith("obs_")
            ),
        )
        summary = TransitionDatasetBuildSummary(
            input_rows=int(len(frame.index)),
            output_rows=int(len(dataset.x.index)),
            excluded_invalid_transition_rows=excluded_invalid,
            input_columns=dataset.input_columns,
            target_columns=dataset.target_columns,
            next_target_columns=schema.next_target_columns,
            metadata_columns=metadata_cols,
            missing_dynamic_target_columns=schema.missing_dynamic_target_columns,
        )
        return TransitionDatasetBuildResult(
            dataset=dataset,
            feature_schema=schema,
            summary=summary,
        )

    def _valid_transition_mask(self, frame: pd.DataFrame) -> pd.Series:
        if self.valid_transition_column not in frame.columns:
            return pd.Series(True, index=frame.index, dtype=bool)
        return (
            pd.to_numeric(frame[self.valid_transition_column], errors="coerce")
            .fillna(0)
            .astype(bool)
        )


def _validate_official_action_contract(frame: pd.DataFrame) -> None:
    missing = [column for column in MDP_V1_ACTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Official MDP action contract is incomplete: {missing}")
    values = frame.loc[:, MDP_V1_ACTION_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not values.notna().all().all():
        raise ValueError("Official MDP actions must be finite and must not be imputed by the model layer.")
    for column in MDP_V1_CONTINUOUS_ACTION_COLUMNS:
        if not values[column].between(0.0, 1.0).all():
            raise ValueError(f"Continuous action {column} must be in [0, 1].")
    for column in MDP_V1_BINARY_ACTION_COLUMNS:
        if not values[column].isin((0.0, 1.0)).all():
            raise ValueError(f"Binary action {column} must be in {{0, 1}}.")


def build_transition_dataset_from_rl_frame(
    frame: pd.DataFrame,
    *,
    observation_columns: Sequence[str] | None = None,
    action_columns: Sequence[str] | None = None,
    target_policy: TransitionTargetPolicy | None = None,
) -> TransitionDatasetBuildResult:
    """Build a transition dataset from an RL-ready frame using default policy."""

    builder = TransitionDatasetBuilder(
        target_policy=target_policy or TransitionTargetPolicy()
    )
    return builder.from_rl_dataset_frame(
        frame,
        observation_columns=observation_columns,
        action_columns=action_columns,
    )


def build_transition_dataset_from_rl_path(
    path: Path | str,
    *,
    observation_columns: Sequence[str] | None = None,
    action_columns: Sequence[str] | None = None,
    target_policy: TransitionTargetPolicy | None = None,
) -> TransitionDatasetBuildResult:
    """Build a transition dataset from an RL-ready parquet file."""

    builder = TransitionDatasetBuilder(
        target_policy=target_policy or TransitionTargetPolicy()
    )
    return builder.from_rl_dataset_path(
        path,
        observation_columns=observation_columns,
        action_columns=action_columns,
    )


__all__ = [
    "DEFAULT_METADATA_COLUMNS",
    "RL_DONE_COLUMN",
    "RL_VALID_TRANSITION_COLUMN",
    "TransitionDatasetBuildResult",
    "TransitionDatasetBuildSummary",
    "TransitionDatasetBuilder",
    "build_transition_dataset_from_rl_frame",
    "build_transition_dataset_from_rl_path",
]
