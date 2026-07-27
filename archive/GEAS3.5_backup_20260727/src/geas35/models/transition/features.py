"""Feature schema helpers for GEAS transition models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from geas35.features import MDP_DERIVED_CONTINUOUS_COLUMNS
from geas35.rl import (
    MDP_V1_ACTION_COLUMNS,
    MDP_V1_DAYLIGHT_CONDITION_ONEHOT_COLUMNS,
    MDP_V1_GROWTH_STAGE_ONEHOT_COLUMNS,
    MDP_V1_ROLLOUT_ID_COLUMN,
    MDP_V1_SOLAR_PERIOD_ONEHOT_COLUMNS,
    MDP_V1_VALID_TRANSITION_COLUMN,
)

NEXT_OBSERVATION_PREFIX = "next_"

MDP_V1_DYNAMIC_TARGET_BASE_COLUMNS = (
    "obs_indoor_temp_c",
    "obs_indoor_humidity_pct",
    "obs_outdoor_temp_c",
    "obs_outdoor_humidity_pct",
    "obs_outdoor_light",
    "obs_outdoor_wind_speed",
    "obs_rain_flag",
    "obs_current_target_temp_min_c",
    "obs_current_target_temp_max_c",
    "obs_current_target_temp_c",
    "obs_growth_stage_dat",
    "obs_current_vpd_kpa",
    "obs_current_dewpoint_c",
    "obs_current_condensation_margin_c",
    *MDP_DERIVED_CONTINUOUS_COLUMNS,
)

DETERMINISTIC_OBSERVATION_COLUMNS = (
    "obs_hour_sin",
    "obs_hour_cos",
    "obs_is_daytime",
    *MDP_V1_GROWTH_STAGE_ONEHOT_COLUMNS,
    *MDP_V1_SOLAR_PERIOD_ONEHOT_COLUMNS,
    *MDP_V1_DAYLIGHT_CONDITION_ONEHOT_COLUMNS,
)

POSTPROCESSED_OBSERVATION_COLUMNS = (
    "obs_prev_vent_pct",
    "obs_prev_shade_curtain_pct",
    "obs_prev_thermal_curtain_pct",
    "obs_prev_heat_run",
    "obs_prev_cool_run",
    "obs_prev_fan_run",
)

STATIC_METADATA_COLUMNS = (
    "crop",
    "greenhouse",
    "greenhouse_id",
    "series_id",
    "segment_id",
    "episode_id",
    "reg_date",
    "timestamp",
    "time",
    "date",
)

MANAGEMENT_COLUMNS = (
    "done",
    "reward",
    "rl_valid_transition",
    "rl_exclusion_reason",
    MDP_V1_VALID_TRANSITION_COLUMN,
    MDP_V1_ROLLOUT_ID_COLUMN,
)


@dataclass(frozen=True)
class TransitionFeatureSchema:
    """Resolved feature schema for transition model training or inference."""

    input_columns: tuple[str, ...]
    dynamic_target_columns: tuple[str, ...]
    next_target_columns: tuple[str, ...]
    deterministic_observation_columns: tuple[str, ...]
    excluded_observation_columns: tuple[str, ...]
    postprocessed_columns: tuple[str, ...]
    missing_dynamic_target_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class TransitionTargetPolicy:
    """Semantic allowlist and denylist policy for next-observation targets."""

    dynamic_target_allowlist: tuple[str, ...] = MDP_V1_DYNAMIC_TARGET_BASE_COLUMNS
    next_prefix: str = NEXT_OBSERVATION_PREFIX

    def __post_init__(self) -> None:
        invalid = [
            column
            for column in self.dynamic_target_allowlist
            if is_denied_observation_column(column)
        ]
        if invalid:
            preview = ", ".join(invalid[:5])
            if len(invalid) > 5:
                preview = f"{preview}, ..."
            raise ValueError(f"Dynamic target allowlist contains denied columns: {preview}")

    def next_column(self, observation_column: str) -> str:
        return f"{self.next_prefix}{observation_column}"

    def resolve(
        self,
        frame: pd.DataFrame,
        *,
        observation_columns: Iterable[str] | None = None,
        action_columns: Iterable[str] | None = None,
    ) -> TransitionFeatureSchema:
        if frame is None:
            frame = pd.DataFrame()

        observed = tuple(observation_columns or infer_observation_columns(frame))
        actions = tuple(action_columns or infer_action_columns(frame))
        frame_columns = set(frame.columns)
        observed_set = set(observed)

        dynamic_targets: list[str] = []
        next_targets: list[str] = []
        missing_targets: list[str] = []
        for column in self.dynamic_target_allowlist:
            if column not in observed_set:
                continue
            next_column = self.next_column(column)
            if next_column in frame_columns:
                dynamic_targets.append(column)
                next_targets.append(next_column)
            else:
                missing_targets.append(column)

        deterministic = tuple(
            column for column in observed if column in DETERMINISTIC_OBSERVATION_COLUMNS
        )
        postprocessed = tuple(
            column for column in observed if column in POSTPROCESSED_OBSERVATION_COLUMNS
        )
        excluded = tuple(
            column
            for column in observed
            if column not in dynamic_targets
            and (
                column in deterministic
                or column in postprocessed
                or is_denied_observation_column(column)
                or column not in self.dynamic_target_allowlist
            )
        )
        input_columns = tuple(
            column for column in (*observed, *actions) if column in frame_columns
        )
        return TransitionFeatureSchema(
            input_columns=input_columns,
            dynamic_target_columns=tuple(dynamic_targets),
            next_target_columns=tuple(next_targets),
            deterministic_observation_columns=deterministic,
            excluded_observation_columns=excluded,
            postprocessed_columns=postprocessed,
            missing_dynamic_target_columns=tuple(missing_targets),
        )


def infer_observation_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Infer current observation columns from an RL transition frame."""

    return tuple(
        column
        for column in frame.columns
        if column.startswith("obs_") and not column.startswith(NEXT_OBSERVATION_PREFIX)
    )


def infer_action_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Infer MDP v1 action columns present in a frame."""

    return tuple(column for column in MDP_V1_ACTION_COLUMNS if column in frame.columns)


def is_denied_observation_column(column: str) -> bool:
    """Return whether a column must not be learned as a dynamic target."""

    return (
        column in STATIC_METADATA_COLUMNS
        or column in MANAGEMENT_COLUMNS
        or column in DETERMINISTIC_OBSERVATION_COLUMNS
        or column in POSTPROCESSED_OBSERVATION_COLUMNS
        or column.startswith(NEXT_OBSERVATION_PREFIX)
        or column.startswith("target_")
        or column.startswith("reward")
        or column.startswith("obs_quality_")
        or column.endswith("_missing_flag")
        or column.endswith("_outlier_flag")
        or column.endswith("_imputed_flag")
        or column.endswith("_valid_transition")
    )


def resolve_transition_feature_schema(
    frame: pd.DataFrame,
    *,
    observation_columns: Sequence[str] | None = None,
    action_columns: Sequence[str] | None = None,
    policy: TransitionTargetPolicy | None = None,
) -> TransitionFeatureSchema:
    """Resolve transition input and dynamic next-observation target columns."""

    return (policy or TransitionTargetPolicy()).resolve(
        frame,
        observation_columns=observation_columns,
        action_columns=action_columns,
    )


__all__ = [
    "DETERMINISTIC_OBSERVATION_COLUMNS",
    "MANAGEMENT_COLUMNS",
    "MDP_V1_DYNAMIC_TARGET_BASE_COLUMNS",
    "NEXT_OBSERVATION_PREFIX",
    "POSTPROCESSED_OBSERVATION_COLUMNS",
    "STATIC_METADATA_COLUMNS",
    "TransitionFeatureSchema",
    "TransitionTargetPolicy",
    "infer_action_columns",
    "infer_observation_columns",
    "is_denied_observation_column",
    "resolve_transition_feature_schema",
]
