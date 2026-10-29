from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import socket
import threading
import time
from typing import Any

from analysis import analyze
from geas35.experiments.rl.checkpoint_v2 import load_ppo_checkpoint_v2
from geas35.experiments.rl.step19_5_parallel import CalibrationLeaseStore
import geas35.experiments.rl.step20_runner as step20_runner
from geas35.experiments.rl.step20_runner import _evaluate
from geas35.rl.hybrid_ppo import HybridPPOConfig
from geas35.rl.model_driven_env import load_model_driven_env_from_step15

VERSION = "geas35.rl.step19_5.v2_1"
CROPS = ("strawberry", "melon", "cucumber")


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _base(root):
    return root / "experiments/rl_policy_training_v2_1"


def _out(root):
    return _base(root) / "artifacts/step19_5_v2_1"


def _config(root):
    return _read(_base(root) / "configs/step19_5_calibration_v2_1.json")


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False


def prepare(project_root):
    root = Path(project_root).resolve()
    source = root / "experiments/rl_policy_training/artifacts/step19_5_v2"
    destination = _out(root)
    live = []
    for crop in CROPS:
        state_path = source / crop / "parallel_state.json"
        if state_path.exists():
            state = _read(state_path)
            if state.get("status") == "running" and _pid_alive(state.get("coordinator_pid")):
                live.append(f"{crop}:{state.get('coordinator_pid')}")
    if live:
        raise RuntimeError("Stop strict v2 coordinators before inheritance: " + ", ".join(live))
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("representative_configurations.json", "v1_preservation_snapshot.json"):
        shutil.copy2(source / name, destination / name)
    inherited = {}
    for crop in CROPS:
        target = destination / crop
        target.mkdir(parents=True, exist_ok=True)
        for pattern in ("learning_curves.json", "representative_*/seed_*.pt",
                        "representative_*/seed_*.training_history.json"):
            for item in (source / crop).glob(pattern):
                relative = item.relative_to(source / crop)
                output = target / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, output)
        curves_path = target / "learning_curves.json"
        rows = _read(curves_path)["rows"] if curves_path.exists() else []
        result = analyze(rows, _config(root)) if rows else None
        if result is not None:
            _write(target / "calibration_analysis.json", result)
        inherited[crop] = {
            "curve_rows": len(rows),
            "completed_budgets": sorted({int(row["steps"]) for row in rows}),
            "checkpoint_files": len(list(target.glob("representative_*/seed_*.pt"))),
        }
    manifest = destination / "inheritance_manifest.json"
    _write(manifest, {
        "schema_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "experiments/rl_policy_training/artifacts/step19_5_v2",
        "prior_validation_curves_seen": True,
        "test_results_seen": False,
        "reason": "pre-Test protocol amendment motivated by unbounded compute and noise-sensitive stopping",
        "crops": inherited, "test_accessed": False,
    })
    _write(destination / "readiness.json", {
        "schema_version": VERSION, "status": "ready_to_run", "test_accessed": False,
        "inheritance_manifest_sha256": _sha(manifest),
    })
    return manifest


def _payload(parameters, observation_dim, seed):
    values = dict(parameters)
    width = int(values.pop("network_width"))
    depth = int(values.pop("network_depth"))
    rollout = int(values.pop("rollout_length"))
    ppo = HybridPPOConfig(
        observation_dim=observation_dim, hidden_sizes=tuple([width] * depth), seed=seed,
        learning_rate=float(values.pop("learning_rate")),
        minibatch_size=int(values.pop("minibatch_size")), update_epochs=int(values.pop("update_epochs")),
        gamma=float(values.pop("gamma")), gae_lambda=float(values.pop("gae_lambda")),
        clip_range=float(values.pop("clip_range")),
        entropy_coefficient=float(values.pop("entropy_coefficient")),
        value_coefficient=float(values.pop("value_coefficient")), target_kl=float(values.pop("target_kl")),
        beta_concentration_floor=float(values.pop("beta_concentration_floor")),
    )
    if values:
        raise ValueError(f"Unused parameters: {sorted(values)}")
    return {"ppo": json.loads(json.dumps(asdict(ppo))), "rollout_length": rollout}


def _train(**kwargs):
    original = step20_runner.load_ppo_checkpoint
    step20_runner.load_ppo_checkpoint = load_ppo_checkpoint_v2
    try:
        return step20_runner._train_to_budget(**kwargs)
    finally:
        step20_runner.load_ppo_checkpoint = original


def _payloads(representatives, seeds, observation_dim, budget):
    result = {}
    job_id = 0
    for representative in representatives:
        for seed in seeds:
            result[job_id] = {
                "config_id": representative["config_id"], "seed": int(seed),
                "budget_steps": int(budget),
                "config": _payload(representative["parameters"], observation_dim, int(seed)),
            }
            job_id += 1
    return result


def _execute(root, crop, budget, claim, owner, store, device, transition_n_jobs):
    payload = claim["payload"]
    job_id = int(claim["job_id"])
    seed = int(payload["seed"])
    config_id = str(payload["config_id"])
    checkpoint = _out(root) / crop / config_id / f"seed_{seed}.pt"
    gates = _read(root / "experiments/rl_policy_training/artifacts/step20/train_derived_safety_gates.json")["crops"][crop]
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(60):
            store.heartbeat(owner, budget, job_id, lease_seconds=180)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        trainer, history = _train(
            root=root, crop=crop, config_payload=payload["config"], seed=seed,
            budget_steps=budget, checkpoint=checkpoint, device=device, resume=True,
            transition_n_jobs=transition_n_jobs,
        )
        metrics, feasible, _, _ = _evaluate(
            root=root, crop=crop, trainer=trainer, validation_episode_indices=(),
            evaluation_seed=42, gates=gates, transition_n_jobs=transition_n_jobs,
        )
        latest = history[-1]
        row = {
            "crop": crop, "config_id": config_id, "seed": seed, "steps": budget,
            "split": "validation", "test_accessed": False, "safety_gate_passed": feasible,
            "mean_reward_per_scheduled_valid_step": metrics["mean_reward_per_scheduled_valid_step"],
            "policy_loss": latest["policy_loss"], "value_loss": latest["value_loss"],
            "approx_kl": latest["approx_kl"], "entropy": latest["entropy"],
            "explained_variance": latest["explained_variance"], "gradient_norm": latest["gradient_norm"],
            "in_support_rate": latest["in_support_rate"], "warning_ood_rate": latest["warning_ood_rate"],
            "severe_ood_rate": latest["severe_ood_rate"], "projection_rate": latest["projection_rate"],
            "fallback_rate": latest["fallback_rate"],
            "termination_rate": latest["severe_state_termination_rate"],
        }
        store.complete(owner, budget, job_id, {"row": row, "checkpoint": str(checkpoint), "test_accessed": False})
    except BaseException as exc:
        store.fail(owner, budget, job_id, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop.set()
        thread.join(timeout=2)


def _fixed_worker(root_text, crop, budget, device, transition_n_jobs, index):
    root = Path(root_text)
    store = CalibrationLeaseStore(_out(root) / crop / "calibration_jobs.sqlite3", crop)
    owner = f"{socket.gethostname()}:{os.getpid()}:{index}"
    while True:
        claim = store.claim(owner, budget, lease_seconds=180)
        if claim is None:
            return
        _execute(root, crop, budget, claim, owner, store, device, transition_n_jobs)


def _following_worker(root_text, crop, device, transition_n_jobs, idle_timeout, index):
    root = Path(root_text)
    output = _out(root)
    store = CalibrationLeaseStore(output / crop / "calibration_jobs.sqlite3", crop)
    owner = f"{socket.gethostname()}:{os.getpid()}:extra-{index}"
    idle_since = time.monotonic()
    while True:
        state = _read(output / crop / "parallel_state.json")
        if state.get("status") in {"passed", "failed", "stopped"}:
            return
        budget = int(state["active_budget"])
        claim = store.claim(owner, budget, lease_seconds=180)
        if claim is not None:
            _execute(root, crop, budget, claim, owner, store, device, transition_n_jobs)
            idle_since = time.monotonic()
        elif time.monotonic() - idle_since >= idle_timeout:
            return
        else:
            time.sleep(2)


def _spawn(target, args, workers):
    context = mp.get_context("spawn")
    processes = [context.Process(target=target, args=(*args, index)) for index in range(workers)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"v2.1 workers failed: {failed}")


def add_workers(project_root, crop, workers, device="cpu", transition_n_jobs=1, idle_timeout=3600):
    root = Path(project_root).resolve()
    state_path = _out(root) / crop / "parallel_state.json"
    if not state_path.exists() or _read(state_path).get("status") != "running":
        raise RuntimeError(f"{crop}: no running v2.1 coordinator")
    _spawn(_following_worker, (str(root), crop, device, transition_n_jobs, idle_timeout), workers)
    return state_path


def run_crop(project_root, crop, device="cpu", transition_n_jobs=1, workers=1):
    root = Path(project_root).resolve()
    output = _out(root)
    if not (output / "inheritance_manifest.json").exists():
        raise RuntimeError("Run --prepare after stopping strict v2 before starting v2.1")
    lock_path = output / crop / "coordinator.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"{crop}: v2.1 coordinator already running") from exc
        representatives = _read(output / "representative_configurations.json")["configurations"]
        config = _config(root)
        seeds = config["seeds"]
        probe = load_model_driven_env_from_step15(project_root=root, crop=crop, split="train")
        base = max(int(item["parameters"]["rollout_length"]) for item in representatives)
        curves_path = output / crop / "learning_curves.json"
        curves = _read(curves_path)["rows"] if curves_path.exists() else []
        completed = sorted({int(row["steps"]) for row in curves})
        budget = base if not completed else completed[-1] * 2
        maximum = int(config["checkpoint_grid"]["maximum_environment_steps"])
        store = CalibrationLeaseStore(output / crop / "calibration_jobs.sqlite3", crop)
        while True:
            if budget > maximum:
                raise RuntimeError(f"{crop}: next budget {budget} exceeds v2.1 maximum {maximum}")
            payloads = _payloads(representatives, seeds, probe.observation_shape[0], budget)
            store.initialize(budget, payloads)
            store.retry_failed(budget)
            state_path = output / crop / "parallel_state.json"
            _write(state_path, {
                "schema_version": VERSION, "status": "running", "crop": crop,
                "active_budget": budget, "jobs": len(payloads), "coordinator_pid": os.getpid(),
                "primary_workers": workers, "test_accessed": False,
            })
            _spawn(_fixed_worker, (str(root), crop, budget, device, transition_n_jobs), workers)
            rows = store.rows(budget)
            failed = [row for row in rows if row["status"] == "failed"]
            if failed:
                _write(state_path, {**_read(state_path), "status": "failed"})
                raise RuntimeError(f"{crop} budget {budget}: worker failure")
            if not rows or not all(row["status"] == "complete" for row in rows):
                raise RuntimeError(f"{crop} budget {budget}: incomplete worker grid")
            existing = {(int(row["steps"]), str(row["config_id"]), int(row["seed"])) for row in curves}
            for job in rows:
                row = job["result"]["row"]
                key = (int(row["steps"]), str(row["config_id"]), int(row["seed"]))
                if key not in existing:
                    curves.append(row)
                    existing.add(key)
            curves.sort(key=lambda row: (int(row["steps"]), str(row["config_id"]), int(row["seed"])))
            _write(curves_path, {"schema_version": VERSION, "crop": crop, "rows": curves, "test_accessed": False})
            result = analyze(curves, config)
            _write(output / crop / "calibration_analysis.json", result)
            if result["status"] == "passed":
                _write(state_path, {**_read(state_path), "status": "passed",
                                    "decision": result["decision"], "rung_budgets": result["rung_budgets"]})
                return output / crop / "calibration_analysis.json"
            budget *= 2


def finalize(project_root):
    root = Path(project_root).resolve()
    output = _out(root)
    crops = {}
    for crop in CROPS:
        analysis = _read(output / crop / "calibration_analysis.json")
        if analysis["status"] != "passed":
            raise RuntimeError(f"{crop}: v2.1 calibration not passed")
        rows = _read(output / crop / "learning_curves.json")["rows"]
        with (output / crop / "learning_curves.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        crops[crop] = {
            "status": "passed", "decision": analysis["decision"],
            "rung_budgets": analysis["rung_budgets"],
        }
    budget = output / "calibrated_budget.json"
    _write(budget, {
        "schema_version": "geas35.rl.step19_5.budget.v2_1", "protocol_version": VERSION,
        "status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "budget_decision_source": "v2.1 robust stability with predeclared maximum budget",
        "numeric_budget_precommitted": False, "crops": crops, "test_accessed": False,
    })
    config_path = _base(root) / "configs/step19_5_calibration_v2_1.json"
    _write(output / "step19_5_integrity_manifest.json", {
        "schema_version": VERSION, "status": "passed", "config_sha256": _sha(config_path),
        "budget": {"path": str(budget.relative_to(root)), "sha256": _sha(budget)},
        "test_accessed": False, "step20_started": False,
    })
    return budget
