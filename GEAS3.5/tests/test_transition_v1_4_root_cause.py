from __future__ import annotations
import numpy as np
import pandas as pd
from geas35.models.transition.action_response import ActionResponseReport
from geas35.models.transition.evaluator import TransitionOneStepEvaluationReport
from geas35.models.transition.feature_variants import (
    TransitionDiagnosticFeatureSchema, diagnostic_feature_contract,
    observation_columns_for_diagnostic_schema, project_frozen_observation_scaler,
)
from geas35.models.transition.root_cause_v1_4 import assess_controlled_cause
from geas35.models.transition.selector import (
    TransitionModelRankingCandidate, ValidationRolloutWeightedScoreStrategy,
    rank_transition_models,
)
from geas35.models.transition.trainer import _hpo_trial_eligibility
from geas35.rl import MDP_V1_ACTION_COLUMNS, MdpV1ObservationScaler


def test_diagnostic_schema_dimensions_are_explicit():
    assert len(observation_columns_for_diagnostic_schema("legacy_v1_1")) == 59
    assert len(observation_columns_for_diagnostic_schema("deduplicated_without_condensation_prediction")) == 56
    assert len(observation_columns_for_diagnostic_schema("deduplicated_with_condensation_prediction")) == 57
    contract = diagnostic_feature_contract("deduplicated_with_condensation_prediction", action_columns=MDP_V1_ACTION_COLUMNS)
    assert contract["input_dimension"] == 63


def test_projected_scaler_keeps_physical_units_and_renames_condensation_feature():
    source = MdpV1ObservationScaler(
        columns=("obs_indoor_temp_c", "obs_indoor_humidity_pct", "obs_indoor_co2_ppm", "dTcond_pred_10m"),
        means={"obs_indoor_temp_c":20., "obs_indoor_humidity_pct":60., "obs_indoor_co2_ppm":500., "dTcond_pred_10m":3.},
        scales={"obs_indoor_temp_c":5., "obs_indoor_humidity_pct":10., "obs_indoor_co2_ppm":100., "dTcond_pred_10m":2.},
    )
    projected = project_frozen_observation_scaler(source, (
        "obs_indoor_temp_c", "obs_indoor_humidity_pct", "obs_indoor_co2_ppm",
        "obs_derived_condensation_margin_pred_10m",
    ))
    scaled = pd.DataFrame({"obs_indoor_temp_c":[1.], "obs_indoor_humidity_pct":[0.], "obs_indoor_co2_ppm":[-1.], "obs_derived_condensation_margin_pred_10m":[2.]})
    physical = projected.inverse_transform_frame(scaled)
    assert physical.iloc[0].to_dict() == {
        "obs_indoor_temp_c":25., "obs_indoor_humidity_pct":60.,
        "obs_indoor_co2_ppm":400., "obs_derived_condensation_margin_pred_10m":7.,
    }


def test_hpo_trial_requires_finite_rollout_safety_and_action_response():
    one = TransitionOneStepEvaluationReport(1, ("y",), {"y":{"rmse":1.}}, {"mean_rmse":1.}, {})
    action = ActionResponseReport(actions={}, eligible=True, fully_action_insensitive=False)
    rollout = {h:{"safety_gate_passed":1., "physical_violation_rate":0., "nan_inf_count":0., "trajectory_nrmse":1.} for h in ("15min","30min","60min")}
    eligible, reasons = _hpo_trial_eligibility(one, rollout, action, 1.0)
    assert eligible and not reasons
    rollout["60min"]["physical_violation_rate"] = 0.1
    eligible, reasons = _hpo_trial_eligibility(one, rollout, action, 1.0)
    assert not eligible and "60min:physical_violation" in reasons


def test_validation_ranking_uses_recorded_rollout_objective():
    candidates = [
        TransitionModelRankingCandidate("a", metadata={"validation_rollout_weighted_score":2.}),
        TransitionModelRankingCandidate("b", metadata={"validation_rollout_weighted_score":1.}),
    ]
    result = rank_transition_models(candidates, strategy=ValidationRolloutWeightedScoreStrategy())
    assert [x.model_name for x in result.candidate_scores] == ["b", "a"]


def test_controlled_cause_rules():
    rows=[]
    for code, stable in {"R":True,"A":True,"B":False,"C":True,"D":False}.items():
        rows.extend({"condition":code,"stability":"stable" if stable else "explosive_error"} for _ in range(3))
    assert assess_controlled_cause(rows)["cause"] == "condensation_prediction_recursive_feedback"
