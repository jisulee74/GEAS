from geas35.experiments.transition.deployment_handoff_v1_4 import SELECTED_CANDIDATES

def test_step14_choices_are_explicit_and_without_flags():
    assert SELECTED_CANDIDATES=={
      "strawberry":("deduplicated_without_quality_flags","lightgbm"),
      "melon":("deduplicated_without_quality_flags","xgboost"),
      "cucumber":("deduplicated_without_quality_flags","lightgbm")}
