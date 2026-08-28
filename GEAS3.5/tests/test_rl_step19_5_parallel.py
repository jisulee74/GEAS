from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import sys
import time
import threading

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geas35.experiments.rl.step19_5_parallel import CalibrationLeaseStore
import geas35.experiments.rl.step19_5_runner as runner
from geas35.experiments.rl.step19_5_runner import _job_payloads
from geas35.experiments.rl.checkpoint_v2 import load_ppo_checkpoint_v2
from geas35.rl.hybrid_ppo import HybridActorCritic, HybridPPOConfig, HybridPPOTrainer, save_ppo_checkpoint


def _claim_worker(path: Path, owner: str, delay: float) -> None:
    store = CalibrationLeaseStore(path, "strawberry")
    while True:
        claim = store.claim(owner, 131072, lease_seconds=5)
        if claim is None:
            return
        time.sleep(delay)
        store.complete(owner, 131072, claim["job_id"], {"owner": owner})


def test_eight_workers_can_be_extended_by_seven_without_duplicate_jobs(tmp_path: Path) -> None:
    path = tmp_path / "calibration.sqlite3"
    store = CalibrationLeaseStore(path, "strawberry")
    payloads = {
        index: {"config_id": f"representative_{index // 3}", "seed": 42 + index % 3}
        for index in range(15)
    }
    store.initialize(131072, payloads)
    context = mp.get_context("spawn")
    first = [context.Process(target=_claim_worker, args=(path, f"primary-{i}", .05)) for i in range(8)]
    for process in first:
        process.start()
    time.sleep(.02)
    extra = [context.Process(target=_claim_worker, args=(path, f"extra-{i}", .05)) for i in range(7)]
    for process in extra:
        process.start()
    for process in [*first, *extra]:
        process.join()
    assert all(process.exitcode == 0 for process in [*first, *extra])
    rows = store.rows(131072)
    assert len(rows) == 15 and all(row["status"] == "complete" for row in rows)
    assert len({row["job_id"] for row in rows}) == 15
    assert any(row["result"]["owner"].startswith("extra-") for row in rows)


def test_failed_and_expired_jobs_are_safely_reclaimed(tmp_path: Path) -> None:
    store = CalibrationLeaseStore(tmp_path / "calibration.sqlite3", "strawberry")
    store.initialize(131072, {0: {"config_id": "representative_0", "seed": 42}})
    claim = store.claim("failed", 131072)
    store.fail("failed", 131072, claim["job_id"], "synthetic")
    assert store.retry_failed(131072) == 1
    retried = store.claim("replacement", 131072, lease_seconds=.01)
    assert retried["attempt"] == 2
    time.sleep(.02)
    reclaimed = store.claim("after-expiry", 131072)
    assert reclaimed["attempt"] == 3
    with pytest.raises(ValueError, match="unowned"):
        store.complete("replacement", 131072, 0, {"ok": False})
    store.complete("after-expiry", 131072, 0, {"ok": True})


def test_parallel_payloads_keep_checkpoint_identity_unique() -> None:
    representatives = [
        {"config_id": f"representative_{i}", "parameters": {
            "learning_rate": 3e-4, "rollout_length": 2048, "minibatch_size": 64,
            "update_epochs": 2, "gamma": .99, "gae_lambda": .95,
            "clip_range": .2, "entropy_coefficient": .01, "value_coefficient": .5,
            "target_kl": .03, "network_width": 16, "network_depth": 1,
            "beta_concentration_floor": 1.1,
        }} for i in range(5)
    ]
    payloads = _job_payloads(representatives, (42, 43, 44), 59, 131072)
    identities = {(value["config_id"], value["seed"]) for value in payloads.values()}
    assert len(payloads) == len(identities) == 15
    assert all(value["budget_steps"] == 131072 for value in payloads.values())


def test_second_coordinator_is_rejected_but_worker_attachment_remains_separate(tmp_path: Path, monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()

    def fake_coordinator(*args, **kwargs):
        entered.set()
        release.wait(5)
        return tmp_path / "done.json"

    monkeypatch.setattr(runner, "_run_crop_calibration_coordinator", fake_coordinator)
    first = threading.Thread(
        target=runner.run_crop_calibration, args=(tmp_path, "strawberry"), daemon=True
    )
    first.start()
    assert entered.wait(2)
    with pytest.raises(RuntimeError, match="already running"):
        runner.run_crop_calibration(tmp_path, "strawberry")
    release.set()
    first.join(2)
    assert not first.is_alive()


def test_v2_checkpoint_loader_normalizes_rng_without_mutating_v1(tmp_path: Path) -> None:
    config = HybridPPOConfig(observation_dim=3, hidden_sizes=(8,), update_epochs=1, minibatch_size=2)
    trainer = HybridPPOTrainer(HybridActorCritic(config), config)
    hashes = {"observation_scaler": "s", "environment": "e", "transition_model": "t", "support": "o"}
    path = save_ppo_checkpoint(tmp_path / "policy.pt", trainer, artifact_hashes=hashes)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["rng_state"]["torch_cpu"] = payload["rng_state"]["torch_cpu"].to(torch.int64)
    torch.save(payload, path)
    restored, _ = load_ppo_checkpoint_v2(path, expected_artifact_hashes=hashes, restore_rng=True)
    assert isinstance(restored, HybridPPOTrainer)
