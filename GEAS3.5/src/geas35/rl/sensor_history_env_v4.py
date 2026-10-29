"""B Phase 0 candidate: continuous sensor windows, isolated command history."""
import numpy as np
import pandas as pd
from geas35.rl.causal_contract_env_v2 import CausalContractStateUpdater
from geas35.features.derived import add_mdp_derived_features

VERSION = 'geas35.rl.sensor_history_b.v4_candidate'
SENSOR_FEATURES = ('obs_derived_low_vpd_minutes_1h', 'obs_derived_rh90_minutes_1h',
                   'obs_derived_ach_min_day', 'obs_derived_ach_min_night')
IDENTITY = ('crop', 'series_id', 'segment_id')
SOURCE_KEYS = ('sensor_source', 'source_table', 'source_confidence')
BREAK_FLAGS = ('sensor_discontinuity', 'physical_discontinuity')
CONTRACT = {'sensor_window': '(t-60min,t], 5min continuous sensor history; episode boundary allowed',
 'sensor_boundaries': 'crop/series/segment/source changes, explicit discontinuity, gaps or duplicate timestamp stop history',
 'command_history': 'unchanged v3 B: same episode e_(t-1), e_(t-2); sensor buffer never supplies commands',
 'other_features': 'unchanged causal v2 solar/alias/degree/condensation contracts',
 'admission': 'Phase 0 candidate; no model/PPO fit or official environment promotion'}


def continuous(previous, current):
    if pd.Timestamp(current.reg_date)-pd.Timestamp(previous.reg_date) != pd.Timedelta(minutes=5):
        return False
    for key in IDENTITY:
        if key not in previous or key not in current or pd.isna(previous[key]) or pd.isna(current[key]) or str(previous[key]) != str(current[key]):
            return False
    for key in SOURCE_KEYS:
        if key in previous or key in current:
            if key not in previous or key not in current or pd.isna(previous[key]) or pd.isna(current[key]) or str(previous[key]) != str(current[key]):
                return False
    for key in BREAK_FLAGS:
        if key in current and (pd.isna(current[key]) or bool(current[key])):
            return False
    return True


class SensorHistoryStateUpdater(CausalContractStateUpdater):
    """Only four sensor-window features use cross-episode prehistory.

    The parent history (condensation, degrees and action carry) is untouched.
    After reset, this buffer receives predicted states only, never future records.
    """
    def reset(self, initial_scaled):
        super().reset(initial_scaled)
        initial = self.current_physical
        t = pd.Timestamp(initial.reg_date)
        selected = self.prehistory
        rows = [initial.copy()]
        if not selected.empty:
            selected = selected.loc[(selected.reg_date < t) & (selected.reg_date > t-pd.Timedelta(minutes=60))].sort_values('reg_date')
            duplicates = set(selected.loc[selected.reg_date.duplicated(keep=False), 'reg_date'])
            for _, row in selected.iloc[::-1].iterrows():
                if row.reg_date in duplicates or not continuous(row, rows[-1]):
                    break
                rows.append(row.copy())
        self._sensor_history = list(reversed(rows))
        self._update_sensor_features()

    def _update_sensor_features(self):
        # Reuse the dataset formula, including its exact defaults and ACH thresholds.
        recomputed = add_mdp_derived_features(pd.DataFrame(self._sensor_history)).iloc[-1]
        for name in SENSOR_FEATURES:
            self._history[-1][name] = float(recomputed[name])

    def advance(self, *, current_scaled, recorded_next_scaled, prediction_physical, action):
        super().advance(current_scaled=current_scaled, recorded_next_scaled=recorded_next_scaled,
                        prediction_physical=prediction_physical, action=action)
        row = self.current_physical
        # Source metadata may be absent from the parent's exogenous allowlist.
        recorded = self._inverse(recorded_next_scaled)
        for key in (*SOURCE_KEYS, *BREAK_FLAGS):
            if key in recorded:
                row[key] = recorded[key]
        if not continuous(self._sensor_history[-1], row):
            self._sensor_history = []
        cutoff = pd.Timestamp(row.reg_date)-pd.Timedelta(minutes=60)
        self._sensor_history = [r for r in self._sensor_history if pd.Timestamp(r.reg_date) > cutoff] + [row]
        self._update_sensor_features()
        return self.scaler.transform_frame(self.current_physical.to_frame().T).iloc[0]
