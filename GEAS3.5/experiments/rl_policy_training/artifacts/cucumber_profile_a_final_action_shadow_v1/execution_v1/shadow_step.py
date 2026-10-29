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
    self._observe_final(scaled, final)
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
    category = 'COMPLETE' if self.cursor == len(self.run) else ('FAILURE' if self.final_counter >= 3 else None)
    info['category'] = category
    if category:
        self.done = True
    # Collector stages the real transition before finalizing this category.
    return obs, result['reward'], self.done, False, info
