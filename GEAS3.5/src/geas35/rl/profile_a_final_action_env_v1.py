"""Explicit final-action termination v1. Legacy implementation/dispatch stays intact.

Only repeated-joint authority changes. Inherited counter is proposal diagnostic.
Cross-environment selected policies are evaluation-only; no training authorization.
"""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
from geas35.rl import profile_a_offline_v1 as legacy
from geas35.rl.profile_a_offline_v1 import (ProfileAOfflineEnv, ACT, CFG, project, fallback,
    project_mdp_v1_action_constraints, file_hash, locked_json)

VERSION = 'geas35.rl.profile_a.final_action_termination.v1'
MIGRATION_HASH = '73516c9044969a1825fe6782fe1696bd5d0bdc23efd05207c26f3bd7062aa343'
INTENT_HASH = '65e88c02ecef1c89f33a6e8950a19c130a742e6f470348a4b42647cef1c0d5b6'
SHADOW_HASH = 'b78010c734ab6952569633d762bae8e903e2c8af354f482cce9fc54ddf41a64d'

class FinalActionProfileAEnv(ProfileAOfflineEnv):
    def __init__(self, root, *, semantic_version, mode='development', num_workers=1):
        if semantic_version != VERSION:
            raise ValueError('EXPLICIT_ENVIRONMENT_VERSION_REQUIRED')
        if mode != 'development':
            raise ValueError('EVALUATION_ONLY_NOT_TRAINING_AUTHORIZED')
        root = Path(root)
        a = root/'experiments/rl_policy_training/artifacts'
        migration_path = a/'cucumber_profile_a_final_action_migration_review_v1/migration_contract.json'
        contract = locked_json(migration_path, 'contract_payload_sha256', MIGRATION_HASH)
        for record in contract['inputs']:
            path = root/record['path']
            if not path.resolve().is_relative_to(root.resolve()) or file_hash(path) != record['sha256']:
                raise ValueError('MIGRATION_INPUT_MISMATCH:' + record['path'])
        intent_path = a/'cucumber_profile_a_joint_termination_intent_v1/joint_termination_intent_contract.json'
        shadow_path = a/'cucumber_profile_a_final_action_shadow_v1/final_action_shadow_contract.json'
        locked_json(intent_path, 'semantic_payload_sha256', INTENT_HASH)
        locked_json(shadow_path, 'contract_payload_sha256', SHADOW_HASH)
        super().__init__(root, mode=mode, num_workers=num_workers)
        self.legacy_fixed_metadata = copy.deepcopy(self.fixed_metadata)
        self.fixed_metadata.update(
            environment_semantic_version=VERSION, implementation_hash=file_hash(__file__),
            legacy_implementation_hash=file_hash(legacy.__file__),
            migration_contract_hash=file_hash(migration_path), migration_payload_hash=MIGRATION_HASH,
            intent_contract_hash=file_hash(intent_path), intent_payload_hash=INTENT_HASH,
            shadow_preregistration_hash=file_hash(shadow_path), shadow_payload_hash=SHADOW_HASH,
            shadow_execution_manifest_hash=file_hash(a/'cucumber_profile_a_final_action_shadow_v1/execution_v1/execution_manifest.json'),
            shadow_execution_result_hash=file_hash(a/'cucumber_profile_a_final_action_shadow_v1/execution_v1/aggregate_shadow_summary.json'),
            repeated_joint_counter_authority='post_weather_final_action',
            training_authorized=False)
        self.selected_policies = contract['selected_policies']

    def reset(self, run_id=None, seed=42):
        observation, info = super().reset(run_id, seed)
        self.final_counter = 0
        info['environment_semantic_version'] = VERSION
        return observation, info

    def _joint_status(self, distance):
        if distance is None or not np.isfinite(distance):
            return None
        thresholds = self.support.config['thresholds']
        return ('severe' if distance > thresholds['joint_severe'] else
                'warning' if distance > thresholds['joint_warning'] else 'in_support')

    def step(self, proposal):
        pb, fb = self.counter, self.final_counter
        obs, reward, done, truncated, info = super().step(proposal)
        info.update(environment_semantic_version=VERSION,
                    proposal_joint_distance=info.get('joint_distance'),
                    proposal_joint_status=self._joint_status(info.get('joint_distance')),
                    proposal_severe_streak=self.counter,
                    proposal_streak_before=pb, final_streak_before=fb,
                    final_streak_last_valid=self.final_counter,
                    counter_authority='post_weather_final_action')
        if 'final_joint_distance' not in info:
            info.update(final_joint_distance=None, final_joint_status=None,
                        final_severe_streak=None,
                        final_support_unavailable_reason=info.get('reason', 'NO_VALID_FINAL_QUERY'))
        return obs, reward, done, truncated, info

    def validate_resume(self, metadata):
        if metadata.get('environment_semantic_version') != VERSION:
            raise ValueError('CHECKPOINT_ENVIRONMENT_SEMANTIC_MISMATCH')
        return super().validate_resume(metadata)

    def frozen_policy_evaluation_binding(self, checkpoint_path, *, seed, update,
                                        original_training_binding):
        expected = next((x for x in self.selected_policies
                         if x['seed'] == seed and x['update'] == update), None)
        if expected is None:
            raise ValueError('UNSELECTED_CHECKPOINT')
        checkpoint_path = Path(checkpoint_path)
        if (checkpoint_path.resolve() != (self.root/expected['checkpoint']['path']).resolve()
                or file_hash(checkpoint_path) != expected['checkpoint']['sha256']):
            raise ValueError('FROZEN_CHECKPOINT_MISMATCH')
        if not original_training_binding:
            raise ValueError('MISSING_ORIGINAL_TRAINING_BINDING')
        for key in ('implementation_hash', 'semantic_hash', 'model_hash', 'schema_hash',
                    'scaler_hash', 'support_bundle_hash', 'profile_hash', 'scale_hash',
                    'scalarization_hash', 'cohort_payload_hash', 'distribution_hash'):
            if original_training_binding.get(key) != self.legacy_fixed_metadata[key]:
                raise ValueError('ORIGINAL_TRAINING_BINDING_MISMATCH:' + key)
        return dict(mode='EXPLICIT_CROSS_ENVIRONMENT_FROZEN_POLICY_EVALUATION',
                    checkpoint=expected['checkpoint'], seed=seed, update=update,
                    original_training_binding=copy.deepcopy(original_training_binding),
                    evaluation_environment_binding=copy.deepcopy(self.fixed_metadata),
                    original_training_semantics='legacy proposal-based',
                    training_authorized=False, optimizer_state_used=False)

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
        final_scores = self.support.scores(scaled, pd.DataFrame([final], columns=ACT))
        final_state, final_joint = float(final_scores['state'][0]), float(final_scores['joint'][0])
        if not np.isfinite([final_state, final_joint]).all():
            info['reason'] = 'NONFINITE_FINAL_SUPPORT'
            return self._end('SYSTEM_INVALID', info)
        if final_state != ss:
            info['reason'] = 'FINAL_QUERY_STATE_MISMATCH'
            return self._end('SYSTEM_INVALID', info)
        self.final_counter = self.final_counter + 1 if final_joint > thr['joint_severe'] else 0
        info.update(final_joint_distance=final_joint,
                    final_joint_status=self._joint_status(final_joint),
                    final_severe_streak=self.final_counter,
                    final_support_unavailable_reason=None)
        info['order'] += ['intervention','weather_cap','final_joint_ood','transition']
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
        category = 'COMPLETE' if self.cursor == len(self.run) else ('FAILURE' if self.final_counter >= 3 else None)
        info['category'] = category
        if category == 'FAILURE':
            info['reason'] = 'executed_joint_support_failure'
        if category:
            self.done = True
        # Collector stages the real transition before finalizing this category.
        return obs, result['reward'], self.done, False, info
