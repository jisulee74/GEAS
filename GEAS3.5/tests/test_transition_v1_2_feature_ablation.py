from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from geas35.features import add_mdp_derived_features
from geas35.models.transition.feature_variants import (
    TransitionFeatureVariant, feature_variant_contract,
    observation_columns_for_variant, prepare_frame_for_feature_variant,
)
from geas35.models.transition.quality_flag_ablation import paired_bootstrap_difference
from geas35.rl import MDP_V1_ACTION_COLUMNS, MDP_V1_OBSERVATION_COLUMNS

REMOVED = {"obs_derived_vpd_kpa", "obs_derived_dewpoint_c", "obs_derived_condensation_margin_c"}

def test_deduplicated_variant_dimensions_and_columns():
    assert len(MDP_V1_OBSERVATION_COLUMNS) == 55
    assert not REMOVED.intersection(MDP_V1_OBSERVATION_COLUMNS)
    with_flags=observation_columns_for_variant(TransitionFeatureVariant.DEDUPLICATED_WITH_QUALITY_FLAGS)
    without=observation_columns_for_variant(TransitionFeatureVariant.DEDUPLICATED_WITHOUT_QUALITY_FLAGS)
    assert len(with_flags)==57 and len(without)==55
    assert set(with_flags)-set(without)=={"obs_quality_missing_imputed_flag","obs_quality_outlier_flag"}
    assert feature_variant_contract("deduplicated_with_quality_flags",action_columns=MDP_V1_ACTION_COLUMNS)["input_dimension"]==63
    assert feature_variant_contract("deduplicated_without_quality_flags",action_columns=MDP_V1_ACTION_COLUMNS)["input_dimension"]==61

def test_legacy_condensation_name_is_promoted_without_changing_values():
    frame=pd.DataFrame({c:[0.0] for c in observation_columns_for_variant("deduplicated_without_quality_flags") if c!="obs_derived_condensation_margin_pred_10m"})
    frame["dTcond_pred_10m"]=[1.25]
    result=prepare_frame_for_feature_variant(frame,"deduplicated_without_quality_flags")
    assert result["obs_derived_condensation_margin_pred_10m"].iloc[0]==pytest.approx(1.25)

def test_condensation_builder_is_causal_at_episode_boundary():
    frame=pd.DataFrame({"reg_date":pd.date_range("2026-01-01",periods=4,freq="5min"),"episode_id":["a","a","b","b"],"obs_indoor_temp_c":[20,20,30,30],"obs_indoor_humidity_pct":[80,80,50,50],"obs_outdoor_temp_c":[10]*4,"obs_outdoor_light":[0]*4,"obs_current_target_temp_c":[20]*4})
    out=add_mdp_derived_features(frame)
    assert np.isfinite(out["obs_derived_condensation_margin_pred_10m"]).all()

def test_paired_bootstrap_effect_classification_inputs():
    mean,lo,hi=paired_bootstrap_difference(np.array([1.,1.,1.,1.]),np.array([2.,2.,2.,2.]),samples=200,seed=42)
    assert mean==-1.0 and hi<0 and lo<0
