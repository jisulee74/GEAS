"""Profile-A offline simulator adapter. No trainer, optimizer, or live interface."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import copy
import hashlib
import json

import numpy as np
import pandas as pd

from geas35.models.transition.four_channel_phase0 import TARGETS
from geas35.rl.bound_support_four_channel_v1 import load_bound_support
from geas35.rl.mdp_v1 import MdpV1Config, project_mdp_v1_action_constraints
from geas35.objectives.binding import TargetContext, VERSION
from geas35.objectives.quality import QualityEvaluator, SCALE_HASH, SCALAR_HASH

SEMANTIC_HASH = '98a4e738bb2ba9341c7a5c90d0c66a02b592b92714b9d6f414d67207a2b45116'
COHORT_HASH = 'ce8cbbc3b5b14b086b18ba7bf6d326975fe395e6e4a8de394f7e072d74cf7a5c'
PROFILE_HASH = '430ac240bf8fc7892b8124842a1d9aacaa2b8f4c4ce0b01db5e2f7f342f1b27e'
ACT = ['vent_pct', 'shade_curtain_pct', 'thermal_curtain_pct', 'heat_run', 'cool_run', 'fan_run']
CFG = MdpV1Config()
CATEGORIES = {'COMPLETE', 'TRUNCATED', 'FAILURE', 'SYSTEM_INVALID', 'UNSCORABLE'}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locked_json(path, key, expected):
    value = json.loads(Path(path).read_text())
    payload = dict(value)
    if payload.pop(key) != expected or digest(payload) != expected:
        raise ValueError('LOCK_MISMATCH')
    return value


def masked_gae(rewards, values, next_values, bootstrap_mask, trace_mask, identities):
    """Fixed baseline GAE; final values must already be identity-checked."""
    arrays = [np.asarray(x, dtype=float) for x in (rewards, values, next_values, bootstrap_mask, trace_mask)]
    r, v, nv, b, c = arrays
    if any(x.ndim != 1 or x.shape != r.shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError('INVALID_GAE_ARRAYS')
    if len(identities) != len(r) or not np.isin(b, [0, 1]).all() or not np.isin(c, [0, 1]).all():
        raise ValueError('INVALID_MASKS')
    for i in range(len(r)):
        if c[i] and (i + 1 == len(r) or identities[i] != identities[i + 1]):
            raise ValueError('TRACE_CROSSES_EPISODE_OR_BUFFER')
    a = np.zeros_like(r)
    acc = 0.
    for i in range(len(r) - 1, -1, -1):
        delta = r[i] + .99 * b[i] * nv[i] - v[i]
        acc = delta + .99 * .95 * c[i] * acc
        a[i] = acc
    return a, a + v


class EpisodeStaging:
    """All fragments remain provisional until terminal episode validity is known."""
    def __init__(self, identity):
        self.identity = identity
        self.rows = []
        self.category = None
        self.events = []
        self.closed = False

    def append(self, row):
        if self.closed or row['identity'] != self.identity:
            raise ValueError('EPISODE_IDENTITY')
        if row['reward'] is None or not np.isfinite(row['reward']):
            raise ValueError('INVALID_REWARD_NOT_ADMITTED')
        if row['next_identity'] != self.identity:
            raise ValueError('RESET_OBSERVATION_BOOTSTRAP')
        if self.rows and row['step'] != self.rows[-1]['step'] + 1:
            raise ValueError('STEP_DISCONTINUITY')
        self.rows.append(copy.deepcopy(row))

    def truncate(self, final_observation, identity):
        if self.closed or not self.rows or identity != self.identity:
            raise ValueError('INVALID_TRUNCATION')
        if not np.array_equal(final_observation, self.rows[-1]['next_observation']):
            raise ValueError('RESET_OBSERVATION_BOOTSTRAP')
        self.rows[-1].update(bootstrap_mask=1, trace_mask=0)
        self.events.append('TRUNCATED')

    def finalize(self, category):
        if category not in CATEGORIES or category == 'TRUNCATED' or self.closed:
            raise ValueError('INVALID_FINALIZATION')
        self.category = category
        self.closed = True
        self.events.append(category)
        if category in {'SYSTEM_INVALID', 'UNSCORABLE'}:
            self.rows.clear()
        elif self.rows:
            self.rows[-1].update(bootstrap_mask=0, trace_mask=0)

    def arrays(self):
        if not self.closed:
            raise ValueError('PROVISIONAL_EPISODE')
        if self.category in {'SYSTEM_INVALID', 'UNSCORABLE'} or not self.rows:
            return None
        rows = self.rows
        a, returns = masked_gae(*[[r[k] for r in rows] for k in
            ('reward', 'value', 'next_value', 'bootstrap_mask', 'trace_mask')], [r['identity'] for r in rows])
        return {'actions': np.array([r['policy_action'] for r in rows]),
                'log_probs': np.array([r['policy_log_prob'] for r in rows]),
                'observations': np.array([r['observation'] for r in rows]),
                'advantages': a, 'returns': returns,
                'bootstrap_mask': np.array([r['bootstrap_mask'] for r in rows]),
                'trace_mask': np.array([r['trace_mask'] for r in rows])}

    @property
    def numeric_return(self):
        if not self.closed or not self.rows or self.category in {'SYSTEM_INVALID', 'UNSCORABLE'}:
            return None
        return float(sum(.99 ** i * r['reward'] for i, r in enumerate(self.rows)))


@dataclass(frozen=True)
class SimulatorBinding:
    payload: str

    @property
    def data(self):
        return json.loads(self.payload)

    @property
    def target_hash(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()

    @property
    def readiness(self):
        d = self.data
        return 'VALID' if d['available_at'] <= d['decision_time'] and d['effective_from'] <= d['decision_time'] else 'TARGET_NOT_AVAILABLE_PRE_ACTION'


class SimulatorJournal:
    """In-memory simulator-only explicit linkage; never claims live attestation."""
    def __init__(self, identity):
        self.identity = identity
        self.bindings = {}
        self.actions = {}

    def bind(self, step):
        t = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * step)
        decision = t.isoformat()
        at = (t + timedelta(microseconds=1)).isoformat()
        action_id = f'{self.identity}:{step}'
        hashes = {}
        for component, ref, unit in [('temperature', 30., '°C'), ('RH', 90., '%RH')]:
            d = dict(objective_version='cucumber_profile_a_offline.v1', target_version='cucumber72_interview_a/v1',
                     component=component, active=True, inactive_reason=None, reference_value=ref, unit=unit,
                     reference_semantics='PROJECT_OPERATIONAL_POINT_DISCREPANCY', activation_context={'explicit_profile_hash': PROFILE_HASH},
                     activation_rule_version='explicit_fixed_offline.v1', source='interview_profile_a', provenance_class='SIMULATOR_PREDICTED',
                     available_at=datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat(), effective_from=datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat(),
                     decision_time=decision, action_id=action_id, action_time=at, component_mask=['temperature', 'RH'], evaluator_version=VERSION,
                     scope='OFFLINE_SIMULATOR_ONLY', semantic_hash=SEMANTIC_HASH)
            b = SimulatorBinding(canonical(d)); self.bindings[b.target_hash] = b; hashes[component] = b.target_hash
        self.actions['action:' + action_id] = (step + 1, at, canonical(hashes))
        return hashes, dict(action_id=action_id, binding_hashes=hashes,
                            observed_at=(t + timedelta(minutes=5)).isoformat(), ambiguous=False)

    def lookup(self, h):
        return self.bindings.get(h)

    def get(self, key):
        return self.actions.get(key)

    def verifies_operational_binding(self, b):
        d = b.data
        return self.lookup(b.target_hash) == b and d['scope'] == 'OFFLINE_SIMULATOR_ONLY' and d['semantic_hash'] == SEMANTIC_HASH and d['activation_context'] == {'explicit_profile_hash': PROFILE_HASH}


class ProfileAReward:
    def __init__(self, contract_path, identity):
        self.evaluator = QualityEvaluator.from_file(contract_path)
        self.journal = SimulatorJournal(identity)
        self.binding_identity = digest({'identity': identity, 'profile': PROFILE_HASH, 'semantic': SEMANTIC_HASH})

    def bind(self, step):
        return self.journal.bind(step)

    def evaluate(self, bound, prediction):
        hashes, outcome = bound
        contexts = {k: TargetContext.from_binding(self.journal.lookup(v)) for k, v in hashes.items()}
        observations = {ch: prediction.get(key) for ch, key in zip(['T1', 'T2', 'RH1', 'RH2'], TARGETS)}
        return self.evaluator.evaluate(self.journal, hashes, observations, outcome, contexts, contexts)


def project(a, local):
    out = a.copy(); empty = False
    for j, c in enumerate(ACT[:3]):
        out[j] = np.clip(out[j], *local['continuous'][c])
    for j, c in enumerate(ACT[3:], 3):
        values = local['binary'][c]
        if not values:
            empty = True
        elif out[j] not in values:
            out[j] = min(values, key=lambda v: abs(v - out[j]))
    return out, empty


def fallback(state, local):
    t, rh = state[TARGETS[0]], state[TARGETS[2]]
    low, high = state['obs_current_target_temp_min_c'], state['obs_current_target_temp_max_c']
    day = state['obs_is_daytime'] > .5; vpd = state['obs_current_vpd_kpa']
    raw = np.array([.2 if t > high or rh > CFG.rh_max_pct else 0, float(day and t > high),
                    float(not day and t < low), float(t < low), float(t > high), float(rh > CFG.rh_max_pct or vpd < CFG.vpd_min_kpa)])
    out, empty = project(raw, local)
    return raw, out, empty


class Cohort:
    def __init__(self, root, mode):
        if mode not in {'train', 'development'}:
            raise ValueError('COHORT_MODE_FORBIDDEN')
        self.root = Path(root); self.mode = mode
        a = self.root / 'experiments/rl_policy_training/artifacts'
        self.semantic = locked_json(a/'cucumber_profile_a_ppo_semantics_v1/ppo_semantics_contract.json', 'semantic_payload_sha256', SEMANTIC_HASH)
        self.registry = locked_json(a/'cucumber_profile_a_ppo_semantics_v1/ppo_cohort_registry.json', 'registry_payload_sha256', COHORT_HASH)
        for r in self.semantic['evidence']:
            p = self.root/r['path']
            if not p.resolve().is_relative_to(self.root.resolve()) or file_hash(p) != r['sha256']:
                raise ValueError('FROZEN_EVIDENCE_MUTATION:' + r['path'])
        self.dataset = a/'transition_4ch_phase0_canonical_v1/dataset/past_only_qc_hlv15_b60_r6_4ch_v1'
        # Explicit Train files only. No path/split input can redirect this loader.
        frame = pd.read_parquet(self.dataset/'cucumber/train_physical.parquet')
        for split, count, total in [('train', 43, 11380), ('development', 11, 2811)]:
            ids = self.registry[split + '_episode_ids']; part = frame[frame.episode_id.astype(str).isin(ids)]
            if len(ids) != count or len(part) != total or digest(sorted(part.row_id.astype(str))) != self.registry['membership_hashes'][split]:
                raise ValueError('COHORT_MEMBERSHIP')
        if set(self.registry['train_episode_ids']) & set(self.registry['development_episode_ids']):
            raise ValueError('COHORT_OVERLAP')
        self.history = pd.read_parquet(self.dataset/'cucumber/train_sensor_prehistory.parquet')
        starts = frame.mdp_v1_rollout_id.ne(frame.mdp_v1_rollout_id.shift()) | frame.reg_date.ne(frame.target_timestamp.shift())
        self.runs = {}
        allowed = set(self.registry[mode + '_episode_ids'])
        for _, g in frame.groupby(starts.cumsum(), sort=False):
            if str(g.iloc[0].episode_id) not in allowed:
                continue
            for key in ['crop', 'series_id', 'segment_id', 'episode_id', 'source_confidence']:
                if g[key].nunique() != 1:
                    raise ValueError('RUN_IDENTITY')
            if not (g.target_timestamp - g.reg_date).eq(pd.Timedelta(minutes=5)).all() or (len(g) > 1 and not np.array_equal(g.target_timestamp.iloc[:-1], g.reg_date.iloc[1:])):
                raise ValueError('RUN_GAP')
            self.runs[str(g.iloc[0].row_id)] = g.reset_index(drop=True)


class ProfileAOfflineEnv:
    """Versioned one-worker, fixed-56 observation adapter; no policy optimization."""
    def __init__(self, root, mode='train', num_workers=1):
        if num_workers != 1:
            raise ValueError('SINGLE_WORKER_ONLY')
        self.cohort = Cohort(root, mode); self.root = Path(root)
        a = self.root/'experiments/rl_policy_training/artifacts'
        binding = a/'step15_hlv15_b60_r6_4ch_v1/cucumber/binding.json'
        self.runtime, self.support = load_bound_support(binding, manifest_sha256=file_hash(binding), crop='cucumber', root=root)
        self.obs = self.runtime.obs
        self.quality_path = a/'future_f_quality_scalarization_v1/quality_scalarization.json'
        self.done = True; self.run = None
        self.fixed_metadata = dict(implementation_hash=file_hash(__file__), profile_id='cucumber72_interview_a', profile_version='v1', profile_hash=PROFILE_HASH,
            scale_hash=SCALE_HASH, scalarization_hash=SCALAR_HASH, semantic_hash=SEMANTIC_HASH, cohort_payload_hash=COHORT_HASH,
            cohort_file_hash=file_hash(a/'cucumber_profile_a_ppo_semantics_v1/ppo_cohort_registry.json'),
            schema_hash=self.runtime.manifest['bundle_hashes']['schema.json'], scaler_hash=self.runtime.manifest['bundle_hashes']['observation_scaler.json'],
            model_hash=self.runtime.manifest['bundle_hashes']['model.pkl'], support_bundle_hash=file_hash(binding),
            action_order_hash=file_hash(a/'step16_hlv15_b60_r6_4ch_v1/protocol.json'),
            distribution_hash=digest({'order':ACT,'continuous':'Beta hard [0,1]','binary':'Bernoulli {0,1}','density':'proposal'}),
            gamma=.99, gae_lambda=.95, budget_protocol_hash=None, training_authorized=False)

    def reset(self, run_id=None, seed=42):
        if not self.done:
            raise ValueError('ACTIVE_EPISODE_RESET_FORBIDDEN')
        if seed not in [42, 43, 44]:
            raise ValueError('UNREGISTERED_SEED')
        ids = sorted(self.cohort.runs)
        run_id = run_id if run_id is not None else ids[int(np.random.default_rng(seed).integers(len(ids)))]
        if run_id not in self.cohort.runs:
            raise ValueError('RUN_OUTSIDE_MODE')
        self.run = self.cohort.runs[run_id]; self.cursor = 0; self.counter = 0; self.done = False
        self.identity = f'{self.cohort.mode}:{run_id}:seed{seed}'
        self.reward_adapter = ProfileAReward(self.quality_path, self.identity)
        self.staging = EpisodeStaging(self.identity)
        self.runtime.reset(self.run.iloc[0], self.cohort.history)
        self.seed = seed
        self.last_valid_observation = self.observation().copy()
        return self.observation(), {'identity':self.identity, 'profile_hash':PROFILE_HASH, 'binding_identity':self.reward_adapter.binding_identity}

    def observation(self):
        out = self.runtime.scaler.transform_frame(self.runtime.adapter.current.to_frame().T)[self.obs].iloc[0].to_numpy(float)
        if out.shape != (56,) or not np.isfinite(out).all():
            raise ValueError('OBSERVATION_INVALID')
        return out

    def _end(self, category, info):
        self.staging.finalize(category); self.done = True; info['category'] = category
        return self.last_valid_observation.copy(), None, True, False, info

    def step(self, proposal):
        if self.done:
            raise ValueError('RESET_REQUIRED')
        try:
            return self._step(proposal)
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            return self._end('SYSTEM_INVALID', {'identity': self.identity, 'step': self.cursor,
                'reason': type(exc).__name__ + ':' + str(exc), 'final_action': None})

    def _step(self, proposal):
        if self.done:
            raise ValueError('RESET_REQUIRED')
        a = np.asarray(proposal, dtype=float)
        if a.shape != (6,) or not np.isfinite(a).all() or ((a < 0)|(a > 1)).any() or not np.isin(a[3:], [0, 1]).all():
            raise ValueError('INVALID_PROPOSED_ACTION')
        s = self.runtime.adapter.current.copy(); scaled = pd.DataFrame([self.observation()], columns=self.obs)
        scores = self.support.scores(scaled, pd.DataFrame([a], columns=ACT)); local = self.support.local_action_support(scaled)[0]
        ss, js = float(scores['state'][0]), float(scores['joint'][0]); thr = self.support.config['thresholds']
        info = dict(identity=self.identity, step=self.cursor, proposed=a.tolist(), state_distance=ss, joint_distance=js,
                    support=copy.deepcopy(local), order=['proposal','joint_ood'], category=None, final_action=None)
        if not np.isfinite([ss, js]).all():
            return self._end('SYSTEM_INVALID', info)
        if ss > thr['state_severe']:
            info['reason'] = 'severe_state_ood'; return self._end('FAILURE', info)
        projected = a.copy(); fb = None; empty = False
        if js > thr['joint_warning']:
            projected, empty = project(a, local)
        if js > thr['joint_severe']:
            _, fb, bad = fallback(s, local); empty |= bad; self.counter += 1
        else:
            self.counter = 0
        if empty:
            info['reason'] = 'unsupported_local_support_failure'; return self._end('FAILURE', info)
        intermediate = fb if fb is not None else projected
        weather, _ = project_mdp_v1_action_constraints(dict(zip(ACT, intermediate)), s.to_dict(), CFG)
        final = np.array([weather[k] for k in ACT])
        info.update(projected_action=projected.tolist(), fallback_action=None if fb is None else fb.tolist(),
                    intermediate_action=intermediate.tolist(), final_action=final.tolist(),
                    intervention='fallback' if fb is not None else ('projection' if js > thr['joint_warning'] else 'none'))
        info['order'] += ['intervention','weather_cap','transition']
        try:
            bound = self.reward_adapter.bind(self.cursor)  # before prediction/outcome; simulator-only clock
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            info['reason'] = 'TARGET_BINDING_INVALID:' + str(exc)
            return self._end('UNSCORABLE', info)
        row = self.run.iloc[self.cursor].copy()
        future = row.copy(); future[self.obs] = row[['next_'+c for c in self.obs]].to_numpy(float)
        future['reg_date'] = row.target_timestamp
        exogenous = self.runtime.adapter.exogenous(future)
        try:
            pred, state = self.runtime.advance(dict(zip(ACT, final)), exogenous)
            obs = self.observation()
            self.last_valid_observation = obs.copy()
        except (ValueError, FloatingPointError, ArithmeticError) as exc:
            info['reason'] = str(exc); return self._end('SYSTEM_INVALID', info)
        info['order'] += ['derived_history_carry','quality','accounting']
        try:
            result = self.reward_adapter.evaluate(bound, pred.to_dict())
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            info['reason'] = 'QUALITY_BINDING_INVALID:' + str(exc)
            return self._end('UNSCORABLE', info)
        info.update(prediction=pred.to_dict(), prediction_provenance='OFFLINE_MODEL_PREDICTION_NOT_RAW_SENSOR', quality=result, target_binding_hashes=bound[0])
        if result['evaluation_status'] != 'VALID' or result['reward'] is None:
            info['reason'] = 'QUALITY_UNSCORABLE'; return self._end('UNSCORABLE', info)
        if not np.isfinite(result['reward']):
            info['reason'] = 'NONFINITE_REWARD'; return self._end('SYSTEM_INVALID', info)
        self.cursor += 1
        category = 'COMPLETE' if self.cursor == len(self.run) else ('FAILURE' if self.counter >= 3 else None)
        info['category'] = category
        if category:
            self.done = True
        # Collector stages the real transition before finalizing this category.
        return obs, result['reward'], self.done, False, info

    def checkpoint_metadata(self):
        return dict(self.fixed_metadata, seed=self.seed, simulator_binding_identity=self.reward_adapter.binding_identity,
                    simulator_binding_registry_hash=digest({'bindings':sorted(self.reward_adapter.journal.bindings),
                        'actions':self.reward_adapter.journal.actions}))

    def validate_resume(self, metadata):
        if metadata != self.checkpoint_metadata():
            raise ValueError('CHECKPOINT_BINDING_MISMATCH')
        if not self.done:
            raise ValueError('RESUME_REQUIRES_CLEAN_BOUNDARY')
        return True

    def authorize_training(self):
        raise RuntimeError('BUDGET_PREREGISTRATION_REQUIRED')


class OfflineCollector:
    """Policy callback interface only. No optimizer/trainer import or update."""
    def __init__(self, env, num_workers=1):
        if num_workers != 1:
            raise ValueError('SINGLE_WORKER_ONLY')
        self.env = env

    def step(self, policy_action, policy_log_prob, value, value_fn):
        if not np.isfinite([policy_log_prob, value]).all():
            return self.env._end('SYSTEM_INVALID', {'identity': self.env.identity, 'reason': 'POLICY_NUMERIC_INVALID', 'final_action': None})
        obs = self.env.observation().copy()
        nxt, reward, done, truncated, info = self.env.step(policy_action)
        if reward is not None:
            try:
                nv = float(value_fn(nxt.copy()))
            except (ValueError, TypeError, ArithmeticError):
                nv = float('nan')
            if not np.isfinite(nv):
                self.env.staging.finalize('SYSTEM_INVALID'); self.env.done = True
                return nxt, None, True, False, dict(info, category='SYSTEM_INVALID')
            row = dict(identity=self.env.identity, next_identity=self.env.identity, step=info['step'], observation=obs,
                       next_observation=nxt.copy(), policy_action=np.asarray(policy_action).copy(), policy_log_prob=policy_log_prob,
                       value=value, next_value=nv, reward=reward, bootstrap_mask=1, trace_mask=1, info=info)
            self.env.staging.append(row)
            if info['category']:
                self.env.staging.finalize(info['category'])
        return nxt, reward, done, truncated, info

    def truncate(self, final_observation, identity):
        self.env.staging.truncate(final_observation, identity)


def development_rank(system_invalid_rate, unscorable_rate, failure_rate, complete_mean, checkpoint_index):
    return (system_invalid_rate + unscorable_rate, failure_rate,
            -complete_mean if complete_mean is not None else float('inf'), checkpoint_index)
