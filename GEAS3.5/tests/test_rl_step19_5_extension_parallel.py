"""Regression checks for preserving completed and resumable extension jobs."""
import fcntl
import json
from pathlib import Path
import sys
from concurrent.futures import Future

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from geas35.experiments.rl import step19_5_runner_v2_2 as runner


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_resume_copy_preserves_existing_progress(tmp_path):
    source, target = tmp_path / 'source.pt', tmp_path / 'target.pt'
    source.write_bytes(b'initial')
    target.write_bytes(b'advanced')
    write(source.with_suffix('.training_history.json'), {'history': [1]})
    write(target.with_suffix('.training_history.json'), {'history': [1, 2]})
    runner._copy_resume_state(source, target)
    assert target.read_bytes() == b'advanced'
    assert json.loads(target.with_suffix('.training_history.json').read_text())['history'] == [1, 2]


def test_completed_cache_checks_hashes_and_rebuilds_row(tmp_path):
    source, target = tmp_path / 'source.pt', tmp_path / 'target.pt'
    source.write_bytes(b'initial'); target.write_bytes(b'complete')
    write(target.with_suffix('.completed.json'), {
        'status': 'complete', 'training_steps': 100, 'crop': 'strawberry',
        'config_id': 'a', 'seed': 42, 'source_checkpoint_sha256': runner.sha256(source),
        'checkpoint_sha256': runner.sha256(target)})
    write(target.with_suffix('.v2_2_metrics.json'), {
        'source_checkpoint_sha256': runner.sha256(source), 'extended_checkpoint_sha256': runner.sha256(target),
        'safety_gate_passed': False, 'validation': {'mean_reward_per_scheduled_valid_step': 0.25}})
    latest = dict.fromkeys(('policy_loss', 'value_loss', 'approx_kl', 'entropy', 'explained_variance',
        'gradient_norm', 'in_support_rate', 'warning_ood_rate', 'severe_ood_rate', 'projection_rate',
        'fallback_rate', 'severe_state_termination_rate'), 0.1)
    write(target.with_suffix('.training_history.json'), {'history': [{'training_steps': 100, **latest}]})
    row = runner._completed_extension_row(source, target, 'strawberry', 'a', 42, 100)
    assert row['mean_reward_per_scheduled_valid_step'] == 0.25
    assert row['safety_gate_passed'] is False
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        runner._completed_extension_row(source, target, 'strawberry', 'a', 42, 100)


def test_parallel_coordinator_skips_completed_and_schedules_unique_jobs(tmp_path, monkeypatch):
    source, output, config = runner._paths(tmp_path)
    write(config, {'seeds': [42, 43, 44], 'extension_budget_steps': 100,
        'configuration_eligibility': {'minimum_eligible_configurations': 2}})
    write(output / 'safety_audit/strawberry_safety_audit.json', {'eligible_configurations': ['a', 'b']})
    def row(config_id, seed):
        return {'config_id': config_id, 'seed': seed}
    monkeypatch.setattr(runner, '_completed_extension_row', lambda s, t, crop, c, seed, budget:
        row(c, seed) if c == 'a' else None)
    scheduled = []
    class Pool:
        def __init__(self, max_workers, mp_context):
            assert max_workers == 3
            assert mp_context.get_start_method() == 'spawn'
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, *args):
            scheduled.append((args[2], args[3]))
            f = Future(); f.set_result(row(args[2], args[3])); return f
    monkeypatch.setattr(runner, 'ProcessPoolExecutor', Pool)
    status = runner.extend_crop(tmp_path, 'strawberry', workers=3)
    assert scheduled == [('b', 42), ('b', 43), ('b', 44)]
    assert json.loads(status.read_text())['completed_runs'] == 6
    curves = json.loads((status.parent / 'learning_curves_extension.json').read_text())
    assert len(curves['rows']) == 6
    assert json.loads(status.read_text())['status'] == 'extension_complete'


def test_duplicate_coordinator_and_job_are_rejected(tmp_path):
    _, output, _ = runner._paths(tmp_path)
    for name, call in [
        ('coordinator.lock', lambda: runner.extend_crop(tmp_path, 'strawberry', workers=3)),
        ('a/seed_42.lock', lambda: runner._extend_one_job(str(tmp_path), 'strawberry', 'a', 42, 'cpu', 1)),
    ]:
        p = output / 'extended_training/strawberry' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(RuntimeError, match='already running'):
                call()
