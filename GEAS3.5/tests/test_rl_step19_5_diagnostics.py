from pathlib import Path
import json
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geas35.experiments.rl import step19_5_diagnostics as d


def record(reward,info,done=False):
    return {'reward':reward,'info':info,'reward_components':d.reward_components(reward,info),
            'trajectory_hash':'x','terminated':done,'truncated':False}


@pytest.mark.parametrize('split',['test','Test','all'])
def test_test_input_rejected(split):
    with pytest.raises(ValueError,match='locked'):d.check_split(split)


def test_training_subset_and_checkpoint_grid():
    assert len(d.indices(100,'train'))==24
    assert d.indices(100,'train')[0]==0 and d.indices(100,'train')[-1]==99
    assert d.indices(3,'validation')==[0,1,2]
    with pytest.raises(ValueError,match='unavailable'):d.checkpoint(Path('.'),'melon','representative_0',42,262144)
    assert len(d.stage_jobs('learning-analysis',['strawberry'],'cpu'))==30
    assert len(d.stage_jobs('rollout-diagnosis',['melon'],'cpu'))==20


def test_normal_and_first_step_state_ood_accounting():
    normal=d.summarize_trace([record(-.2,{'base_reward':-.2,'termination_reason':'episode_boundary'},True)],{'scheduled_valid_steps':1})
    assert normal['executed_steps']==1 and normal['reward_sum']==-.2
    early=d.summarize_trace([record(-10,{'pessimistic_terminal_cost':-10,'state_ood_level':'severe',
        'termination_reason':'severe_state_ood'},True)],{'scheduled_valid_steps':10})
    assert early['mean_reward_per_scheduled_valid_step']==-1
    assert early['severe_ood_rate']==1 and early['severe_ood_per_scheduled_step']==.1
    assert early['first_severe_step']==0
    assert early['reward_components']['terminal']==-10


def test_fallback_and_terminal_cost_are_not_double_counted():
    info={'base_reward':-.2,'warning_ood_penalty':.1,'pessimistic_terminal_cost':-8,
        'joint_ood_level':'severe','safety_fallback_applied':True,'termination_reason':'repeated_severe_action_ood'}
    row=d.summarize_trace([record(-9,info,True)],{'scheduled_valid_steps':9})
    assert row['fallback_rate']==1 and row['severe_ood_rate']==1
    assert sum(row['reward_components'].values())==pytest.approx(-9)
    assert row['reward_components']['clamp_or_override']==pytest.approx(-.7)


def test_episode_resume_and_tamper_detection(tmp_path):
    p=tmp_path/'episode.json';calls=[]
    def op(stream):calls.append(1);stream.write('{}\n');return {'value':2}
    assert d.atomic_episode(p,'a',op)['value']==2
    assert d.atomic_episode(p,'a',op)['value']==2 and len(calls)==1
    with pytest.raises(ValueError,match='Stale'):d.atomic_episode(p,'b',op)
    p.with_suffix('.jsonl.gz').write_bytes(b'broken')
    with pytest.raises(ValueError,match='hash mismatch'):d.atomic_episode(p,'a',op)


def test_failed_episode_is_not_committed(tmp_path):
    p=tmp_path/'episode.json'
    def fail(stream):stream.write('partial');raise RuntimeError('fail')
    with pytest.raises(RuntimeError):d.atomic_episode(p,'a',fail)
    assert not p.exists()
    assert d.atomic_episode(p,'a',lambda s:{'value':1})['value']==1


def test_duplicate_job_lock(tmp_path):
    with d.lock(tmp_path/'job.lock'):
        with pytest.raises(RuntimeError,match='Already running'):
            with d.lock(tmp_path/'job.lock'):pass


def test_wrapper_is_observation_only():
    class Env:
        def __init__(self):
            self.episodes=[{}];self.observation_columns=('x',)
            self._current_scaled=pd.Series({'x':1.});self._current_physical=self._current_scaled.copy()
            self._position=0;self.state_updater=SimpleNamespace(_history=[self._current_physical]);self.calls=0
        def reset(self,**kwargs):return np.array([1.]),{'scheduled_valid_steps':1}
        def step(self,action):
            self.calls+=1
            return np.array([2.]),-.2,True,False,{'base_reward':-.2,'termination_reason':'episode_boundary'}
    env=Env();wrapped=d.TraceEnv(env,0);wrapped.reset()
    got=wrapped.step(np.array([.2]));expected=Env().step(np.array([.2]))
    assert np.array_equal(got[0],expected[0]) and got[1:]==expected[1:]
    assert env.calls==1 and len(wrapped.records)==1


def test_input_mutation_is_detected(tmp_path):
    p=tmp_path/'source';p.write_text('original')
    manifest={'files':[{'path':'source','sha256':d.sha256(p)}]}
    assert d.verify_inputs(tmp_path,manifest)==1
    p.write_text('changed')
    with pytest.raises(ValueError,match='inputs changed'):d.verify_inputs(tmp_path,manifest)


def test_repeat_comparison_checks_terminal_events():
    a=d.summarize_trace([record(-1,{'termination_reason':'episode_boundary'},True)],{'scheduled_valid_steps':1})
    b=dict(a,termination_reason='severe_state_ood')
    assert 'termination_reason' in d.compare_episodes(a,b)


def test_kth_neighbor_components_reproduce_weighted_distance():
    from sklearn.neighbors import NearestNeighbors
    reference=np.array([[0.,0.],[2.,0.],[0.,4.]])
    support=SimpleNamespace(state_weights=np.array([2.,1.]),_state_reference=reference,
        _state_nn=NearestNeighbors(n_neighbors=2).fit(reference))
    env=SimpleNamespace(support=support,observation_columns=('a','b'))
    result=d.state_contributions(env,np.array([.5,1.]))
    assert result['component_sum_matches_distance']
    assert sum(result['squared_distance_components'].values())==pytest.approx(result['state_score']**2)


def test_feature_sensitivity_does_not_mutate_state():
    import importlib.util
    p=Path(__file__).resolve().parents[1]/'experiments/rl_policy_training/scripts/analyze_step19_5_feature_consistency.py'
    spec=importlib.util.spec_from_file_location('feature_sensitivity',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Support:
        def scores(self,state,action):return {'state':np.linalg.norm(state.to_numpy(),axis=1)}
    state=pd.DataFrame([{'light':4.,'temp':3.}]);before=state.copy(deep=True)
    value=module.score_replacement(Support(),state,pd.Series({'light':0.}),pd.DataFrame(),['light'])
    assert value==3.
    pd.testing.assert_frame_equal(state,before)


def test_final_review_requires_all_jobs_and_handles_undefined_correlations(tmp_path,monkeypatch):
    import importlib.util
    p=Path(__file__).resolve().parents[1]/'experiments/rl_policy_training/scripts/finalize_step19_5_diagnosis_review.py'
    spec=importlib.util.spec_from_file_location('final_review',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.correlation([1,2,3],[2,4,6])==pytest.approx(1.)
    assert module.correlation([1,1,1],[2,3,4]) is None
    assert module.correlation([1,2,float('nan')],[2,4,6]) is None
    monkeypatch.setattr(module,'ROOT',tmp_path)
    report=tmp_path/module.OUT/'report';report.mkdir(parents=True)
    (report/'evidence.json').write_text(json.dumps({'status':'incomplete'}))
    with pytest.raises(ValueError,match='all planned jobs'):module.main()
    assert not (report/'diagnosis_decision.json').exists()
