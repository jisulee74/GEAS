"""Reinforcement-learning environment helpers for GEAS3.5."""

from geas35.rl.mdp_v1 import (
    MDP_V1_ACTION_COLUMNS,
    MDP_V1_AGGREGATE_QUALITY_FLAG_COLUMNS,
    MDP_V1_BINARY_ACTION_COLUMNS,
    MDP_V1_CONTINUOUS_ACTION_COLUMNS,
    MDP_V1_DAYLIGHT_CONDITION_ONEHOT_COLUMNS,
    MDP_V1_GROWTH_STAGE_ONEHOT_COLUMNS,
    MDP_V1_MAX_GROWTH_STAGE_ORDER,
    MDP_V1_OBSERVATION_COLUMNS,
    MDP_V1_QUALITY_FLAG_FALLBACKS,
    MDP_V1_QUALITY_FLAG_PREFIX,
    MDP_V1_REWARD_TERM_KEYS,
    MDP_V1_ROLLOUT_ID_COLUMN,
    MDP_V1_SOLAR_PERIOD_ONEHOT_COLUMNS,
    MDP_V1_STEP_MINUTES,
    MDP_V1_TRANSITION_METRIC_KEYS,
    MDP_V1_TRANSITION_OBSERVATION_COLUMNS,
    MDP_V1_TRANSITION_TARGET_COLUMNS,
    MDP_V1_VALID_TRANSITION_COLUMN,
    MDP_V1_ZSCORE_OBSERVATION_COLUMNS,
    MdpV1Config,
    MdpV1Env,
    MdpV1ObservationScaler,
    MdpV1RewardNormalizer,
    compute_mdp_v1_reward,
    evaluate_mdp_v1_transition_predictions,
    fit_mdp_v1_observation_scaler,
    fit_mdp_v1_reward_normalizer,
    logged_mdp_v1_action,
    normalize_mdp_v1_action,
    project_mdp_v1_action_constraints,
    mdp_v1_observation_columns,
    prepare_mdp_v1_frame,
)
from geas35.rl.datasets import (
    NEXT_OBSERVATION_PREFIX,
    REWARD_TERM_PREFIX,
    RL_DONE_COLUMN,
    RL_EXCLUSION_REASON_COLUMN,
    RL_REWARD_COLUMN,
    RL_VALID_TRANSITION_COLUMN,
    RlDatasetSplitSummary,
    prepare_rl_dataset_frame,
    prepare_rl_dataset_split_file,
    prepare_rl_dataset_splits,
)


from geas35.rl.support_v1 import (
    SUPPORT_SCHEMA_VERSION,
    StateActionSupportModel,
    SupportBuildConfig,
    build_support_artifact,
    conformal_quantile,
)


_STEP16_EXPORTS = {
    "GeasModelDrivenEnv",
    "ModelDrivenEnvConfig",
    "STEP16_ENV_VERSION",
    "load_model_driven_env_from_step15",
}

_STEP18_EXPORTS = {
    "FeasibleActionSpec", "HybridActorCritic", "HybridPPOConfig", "HybridPPOTrainer",
    "RolloutBuffer", "STEP18_PPO_VERSION", "SupportAwareActionAdapter", "collect_rollout",
    "compute_gae", "load_ppo_checkpoint", "save_ppo_checkpoint", "seed_everything",
}


def __getattr__(name: str):
    """Load Step 16 lazily so transition feature imports remain acyclic."""
    if name in _STEP16_EXPORTS:
        from geas35.rl import model_driven_env

        return getattr(model_driven_env, name)
    if name in _STEP18_EXPORTS:
        from geas35.rl import hybrid_ppo

        return getattr(hybrid_ppo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FeasibleActionSpec", "HybridActorCritic", "HybridPPOConfig", "HybridPPOTrainer",
    "RolloutBuffer", "STEP18_PPO_VERSION", "SupportAwareActionAdapter", "collect_rollout",
    "compute_gae", "load_ppo_checkpoint", "save_ppo_checkpoint", "seed_everything",
    "GeasModelDrivenEnv",
    "ModelDrivenEnvConfig",
    "STEP16_ENV_VERSION",
    "load_model_driven_env_from_step15",
    "SUPPORT_SCHEMA_VERSION",
    "StateActionSupportModel",
    "SupportBuildConfig",
    "build_support_artifact",
    "conformal_quantile",
    "MDP_V1_ACTION_COLUMNS",
    "MDP_V1_AGGREGATE_QUALITY_FLAG_COLUMNS",
    "MDP_V1_BINARY_ACTION_COLUMNS",
    "MDP_V1_CONTINUOUS_ACTION_COLUMNS",
    "MDP_V1_DAYLIGHT_CONDITION_ONEHOT_COLUMNS",
    "MDP_V1_GROWTH_STAGE_ONEHOT_COLUMNS",
    "MDP_V1_MAX_GROWTH_STAGE_ORDER",
    "MDP_V1_OBSERVATION_COLUMNS",
    "MDP_V1_QUALITY_FLAG_FALLBACKS",
    "MDP_V1_QUALITY_FLAG_PREFIX",
    "MDP_V1_REWARD_TERM_KEYS",
    "MDP_V1_ROLLOUT_ID_COLUMN",
    "MDP_V1_SOLAR_PERIOD_ONEHOT_COLUMNS",
    "MDP_V1_STEP_MINUTES",
    "MDP_V1_TRANSITION_METRIC_KEYS",
    "MDP_V1_TRANSITION_OBSERVATION_COLUMNS",
    "MDP_V1_TRANSITION_TARGET_COLUMNS",
    "MDP_V1_VALID_TRANSITION_COLUMN",
    "MDP_V1_ZSCORE_OBSERVATION_COLUMNS",
    "MdpV1Config",
    "MdpV1Env",
    "MdpV1ObservationScaler",
    "MdpV1RewardNormalizer",
    "NEXT_OBSERVATION_PREFIX",
    "REWARD_TERM_PREFIX",
    "RL_DONE_COLUMN",
    "RL_EXCLUSION_REASON_COLUMN",
    "RL_REWARD_COLUMN",
    "RL_VALID_TRANSITION_COLUMN",
    "RlDatasetSplitSummary",
    "compute_mdp_v1_reward",
    "evaluate_mdp_v1_transition_predictions",
    "fit_mdp_v1_observation_scaler",
    "fit_mdp_v1_reward_normalizer",
    "logged_mdp_v1_action",
    "normalize_mdp_v1_action",
    "project_mdp_v1_action_constraints",
    "mdp_v1_observation_columns",
    "prepare_mdp_v1_frame",
    "prepare_rl_dataset_frame",
    "prepare_rl_dataset_split_file",
    "prepare_rl_dataset_splits",
]
