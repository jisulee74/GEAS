"""Frozen four-channel Phase0 contract. No learner, controller, or data loader.

Only caller-supplied legal exogenous values can enter advance(). Sensor-history
uses existing formulas, with physical T1/RH1 aliases excluded from model inputs.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from geas35.rl.mdp_v1 import MdpV1ObservationScaler, MDP_V1_ZSCORE_OBSERVATION_COLUMNS
from geas35.rl.sensor_history_env_v4 import SensorHistoryStateUpdater, SOURCE_KEYS, BREAK_FLAGS
from geas35.rl.recorded_light_env_v1 import SOLAR_FEATURES
from geas35.models.transition.causal_state import CausalStateUpdater

VERSION = 'geas35.transition.four_channel.phase0.v1'
TARGETS = ['obs_indoor_temp1_c', 'obs_indoor_temp2_c', 'obs_indoor_humidity1_pct', 'obs_indoor_humidity2_pct']
RAW = ['in_temp', 'in_temp2', 'in_hum', 'in_hum2']
FLAGS = ['missing_flag', 'invalid_flag', 'imputed_flag', 'ffill_flag', 'rule_outlier_flag']
KEYS = ['crop', 'series_id', 'segment_id', 'episode_id', 'reg_date']
ALIASES = {'obs_indoor_temp_c': TARGETS[0], 'obs_indoor_humidity_pct': TARGETS[2]}
UNITS = ['degC', 'degC', 'percent_RH', 'percent_RH']
MODEL_NAMESPACE = 'command_carry_hlv15_b60_r6_4ch_v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()

def valid_raw(frame):
    valid = frame.is_resampled_row.eq(0).to_numpy(copy=True)
    for s in RAW:
        v = frame[s + '_raw_value'].to_numpy(float)
        valid &= np.isfinite(v) & ((v >= -273.15) if 'temp' in s else ((v >= 0) & (v <= 100)))
        for flag in FLAGS:
            valid &= frame[s + '_' + flag].eq(0).to_numpy()
    return valid

def physical_gate(values):
    x = np.asarray(values, dtype=float)
    if x.shape[-1] != 4 or not np.isfinite(x).all():
        raise ValueError('Four finite channels required')
    if ((x[..., :2] < -60) | (x[..., :2] > 80)).any() or ((x[..., 2:] < 0) | (x[..., 2:] > 100)).any():
        raise ValueError('Four-channel physical violation; no substitution or clipping')

def fit_scaler(train, obs):
    if 'split' not in train or not train.split.eq('train').all() or train.empty:
        raise ValueError('Only nonempty eligible Train may fit scaler')
    columns = TARGETS + [c for c in MDP_V1_ZSCORE_OBSERVATION_COLUMNS if c in obs and c not in TARGETS]
    values = train[columns].astype(float)
    if not np.isfinite(values).all().all():
        raise ValueError('Nonfinite Train scaler input')
    means = values.mean().to_dict()
    scales = values.std(ddof=0).to_dict()
    scales = {c: (v if v > 0 else 1.0) for c, v in scales.items()}
    return {'columns': columns, 'means': means, 'scales': scales, 'fit_split': 'train'}

def scaler_from(payload):
    return MdpV1ObservationScaler(tuple(payload['columns']), payload['means'], payload['scales'])

class FourChannelRuntime:
    """Schema-bound physical recurrence; no model fitting/loading or reward.

    advance receives *only* legal contemporaneous exogenous fields, not a full
    future record. Diagnostic teacher forcing is performed by the test caller.
    """
    def __init__(self, schema_path, scaler_path, *, schema_hash, scaler_hash, prehistory=None):
        if sha(schema_path) != schema_hash or sha(scaler_path) != scaler_hash:
            raise ValueError('Schema/scaler hash mismatch')
        self.schema = json.loads(Path(schema_path).read_text())
        s = self.schema
        self.obs, self.actions = s['observation_columns'], s['action_columns']
        if (len(self.obs) != 56 or len(set(self.obs)) != 56 or len(self.actions) != 6
            or s['input_order'] != self.obs + self.actions or s['target_order'] != TARGETS
            or self.obs[:4] != TARGETS or s['units'] != UNITS or s['input_dimension'] != 62
            or s['state_dimension'] != 56 or any('co2' in c.lower() for c in s['input_order'])
            or s['model_namespace'] != MODEL_NAMESPACE):
            raise ValueError('Not a bound 56/6/4 contract; legacy bundle rejected')
        payload = json.loads(Path(scaler_path).read_text())
        expected = TARGETS + [c for c in MDP_V1_ZSCORE_OBSERVATION_COLUMNS if c in self.obs]
        if payload['columns'] != expected or payload['schema_sha256'] != schema_hash or payload['fit_split'] != 'train':
            raise ValueError('Scaler identity/schema mismatch')
        self.scaler = scaler_from(payload)
        self.schema_hash, self.scaler_hash = schema_hash, scaler_hash
        # History contains only the two representative historical aliases needed
        # by the frozen formulas; no raw CO2 column is retained or synthesized.
        history = pd.DataFrame() if prehistory is None else prehistory.copy()
        keep = [c for c in history if c in self.obs or c in ALIASES or self.legal_exogenous(c)]
        # source_confidence describes command evidence, not the sensor stream.
        # Keep it outside the sensor buffer; real sensor_source/source_table
        # boundaries remain enforced by the frozen updater.
        keep = [c for c in keep if c != 'source_confidence']
        self.updater = SensorHistoryStateUpdater(self.scaler, history[keep])
        self.current = None

    @staticmethod
    def legal_exogenous(column):
        return CausalStateUpdater._recorded_column_allowed(column) or column in (*SOLAR_FEATURES, *SOURCE_KEYS, *BREAK_FLAGS)

    def exogenous(self, row):
        out = row[[c for c in row.index if self.legal_exogenous(c)]].copy()
        if 'source_confidence' not in out:
            out['source_confidence'] = self.command_source_confidence
        return out

    def _aliases(self, row):
        result = row.copy()
        physical_gate(result[TARGETS].to_numpy(float))
        for alias, source in ALIASES.items():
            result[alias] = float(result[source])
        result['representative_alias_provenance'] = 'derived_from_four_channel_state_not_raw_observed'
        return result

    def reset(self, physical):
        self.validate_state(physical)
        self.command_source_confidence = physical.get('source_confidence')
        if not isinstance(self.command_source_confidence, str) or not self.command_source_confidence:
            raise ValueError('Explicit command provenance required')
        row = physical[[c for c in physical.index if c != 'source_confidence' and (c in self.obs or self.legal_exogenous(c))]].copy()
        row = self._aliases(row)
        self.updater.reset(self.scaler.transform_frame(row.to_frame().T).iloc[0])
        self.current = self.updater.current_physical
        return self.current.copy()

    def validate_state(self, state):
        if list(c for c in state.index if c in self.obs) != self.obs:
            raise ValueError('State schema/order mismatch')
        if not np.isfinite(state[self.obs].to_numpy(float)).all():
            raise ValueError('Nonfinite state')
        physical_gate(state[TARGETS].to_numpy(float))

    def validate_action(self, action):
        if list(action) != self.actions or not np.isfinite(list(action.values())).all():
            raise ValueError('Explicit ordered six-action contract required')
        if any(not 0 <= v <= 1 for v in action.values()) or any(action[a] not in (0, 1) for a in self.actions[3:]):
            raise ValueError('Invalid normalized action')

    def model_input(self, action):
        self.validate_action(action)
        self.validate_state(self.current)
        row = self.scaler.transform_frame(self.current.to_frame().T)[self.obs].iloc[0]
        return pd.Series([*row.to_numpy(float), *action.values()], index=self.schema['input_order'])

    def next_from_residual(self, delta):
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (4,) or not np.isfinite(delta).all():
            raise ValueError('Four finite physical residuals required')
        values = self.current[TARGETS].to_numpy(float) + delta
        physical_gate(values)
        return dict(zip(TARGETS, values))

    def advance(self, *, prediction, action, exogenous):
        if list(prediction) != TARGETS:
            raise ValueError('Target order mismatch')
        physical_gate(list(prediction.values()))
        self.validate_action(action)
        if any(not self.legal_exogenous(c) for c in exogenous.index):
            raise ValueError('Future endogenous/unknown field prohibited')
        if pd.Timestamp(exogenous.reg_date) - pd.Timestamp(self.current.reg_date) != pd.Timedelta(minutes=5):
            raise ValueError('Exact +5min required; reset after gap')
        if exogenous.get('source_confidence') != self.command_source_confidence:
            raise ValueError('Command source crossing requires reset')
        exogenous = exogenous.drop('source_confidence')
        for key in ('crop', 'series_id', 'segment_id', 'episode_id', *SOURCE_KEYS):
            if key in self.current and (key not in exogenous or str(exogenous[key]) != str(self.current[key])):
                raise ValueError('Boundary crossing requires explicit reset')
        if any(bool(exogenous.get(c, False)) for c in BREAK_FLAGS):
            raise ValueError('Physical discontinuity requires reset')
        pred = dict(prediction)
        pred.update({a: prediction[s] for a, s in ALIASES.items()})
        self.updater.advance(
            current_scaled=self.scaler.transform_frame(self.current.to_frame().T).iloc[0],
            recorded_next_scaled=self.scaler.transform_frame(exogenous.to_frame().T).iloc[0],
            prediction_physical=pred, action=action)
        self.current = self.updater.current_physical
        self.validate_state(self.current)
        return self.current.copy()
