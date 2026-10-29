"""Opt-in recorded-weather feature contract; legacy environments stay unchanged.

Only action-independent solar features are copied at the *current* exogenous
provider timestamp. This is not a forecast/deployment or physical-sensor repair.
"""
from __future__ import annotations

import numpy as np

from geas35.models.transition.causal_state import CausalStateUpdater
from geas35.models.transition.rollout import RecordedWeatherProvider
from geas35.rl.model_driven_env import load_model_driven_env_from_step15

VERSION = 'geas35.rl.recorded_light_contract.v1'
SOLAR_FEATURES = (
    'obs_derived_dli_mol_m2', 'obs_derived_light_sum',
    'obs_derived_clear_sky_sum', 'obs_derived_light_sum_ratio',
    'obs_derived_light_eta_minutes',
)


class RecordedLightStateUpdater(CausalStateUpdater):
    """Preserve frozen recorded-feature semantics, including genuine recorded 0."""

    def advance(self, *, current_scaled, recorded_next_scaled, prediction_physical, action):
        recorded = self._inverse(recorded_next_scaled)
        missing = [name for name in SOLAR_FEATURES if name not in recorded or not np.isfinite(float(recorded[name]))]
        if missing:
            raise ValueError(f'Recorded solar contract requires finite features: {missing}')
        if set(prediction_physical).intersection(SOLAR_FEATURES):
            raise ValueError('Transition targets must not override exogenous solar features')
        super().advance(current_scaled=current_scaled, recorded_next_scaled=recorded_next_scaled,
                        prediction_physical=prediction_physical, action=action)
        # Restore all five as a unit so ratio/ETA do not retain a recomputed definition.
        # Update the same history row used by current_physical and subsequent steps.
        for name in SOLAR_FEATURES:
            self._history[-1][name] = float(recorded[name])
        return self.scaler.transform_frame(self.current_physical.to_frame().T).iloc[0]


def load_recorded_light_env(*, project_root, crop, split, protocol_manifest_path):
    if split not in ('train', 'validation'):
        raise ValueError('Recorded-light verification permits Train/Validation only; Test locked')
    env = load_model_driven_env_from_step15(project_root=project_root, crop=crop, split=split,
                                          protocol_manifest_path=protocol_manifest_path)
    if type(env.exogenous_provider) is not RecordedWeatherProvider:
        raise ValueError('Recorded-light contract requires RecordedWeatherProvider')
    env.state_updater = RecordedLightStateUpdater(env.scaler)
    env.diagnostic_environment_version = VERSION
    return env
