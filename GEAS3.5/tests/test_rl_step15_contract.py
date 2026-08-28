from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.rl.step15_contract import (
    STEP15_VERSION, Step15ContractError, authorize_test_access,
    run_step15_contract,
)
from geas35.rl.support_v1 import (
    StateActionSupportModel, SupportBuildConfig, build_support_artifact,
    conformal_quantile,
)
from geas35.rl import MDP_V1_ACTION_COLUMNS


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows=[]
    for episode in range(10):
        for step in range(20):
            x=episode/10 + step/100
            rows.append({
                "episode_id": f"e{episode}", "s_cont": x,
                "s_binary": episode % 2,
                "vent_pct": np.clip(x,0,1),
                "shade_curtain_pct": rng.uniform(0,1),
                "thermal_curtain_pct": rng.uniform(0,1),
                "heat_run": int(x < .5), "cool_run": int(x > .7),
                "fan_run": episode % 2,
            })
    return pd.DataFrame(rows)


def test_conformal_support_is_deterministic_and_queryable(tmp_path: Path) -> None:
    frame=_frame()
    cfg=SupportBuildConfig(seed=42, neighbors=5, max_reference_rows=100,
                           max_calibration_rows=50)
    a1,_=build_support_artifact(
        frame,crop="synthetic",state_columns=["s_cont","s_binary"],
        action_columns=MDP_V1_ACTION_COLUMNS,episode_column="episode_id",
        output_dir=tmp_path/"one",config=cfg,
    )
    a2,_=build_support_artifact(
        frame,crop="synthetic",state_columns=["s_cont","s_binary"],
        action_columns=MDP_V1_ACTION_COLUMNS,episode_column="episode_id",
        output_dir=tmp_path/"two",config=cfg,
    )
    d1=json.loads(a1.read_text()); d2=json.loads(a2.read_text())
    assert d1["thresholds"] == d2["thresholds"]
    assert d1["fit_split"] == "train_only"
    model=StateActionSupportModel(a1)
    state=frame.loc[[0],["s_cont","s_binary"]]
    action=frame.loc[[0],MDP_V1_ACTION_COLUMNS]
    near=model.scores(state,action)
    far_state=state.copy(); far_state["s_cont"]=100.0
    far=model.scores(far_state,action)
    assert far["state"][0] > near["state"][0]
    support=model.local_action_support(state)[0]
    assert set(support) == {"continuous","binary"}
    assert set(support["continuous"]) == set(MDP_V1_ACTION_COLUMNS[:3])


def test_conformal_quantile_rejects_invalid_input() -> None:
    assert conformal_quantile(np.arange(100), .95) == 95.0
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), .95)


def test_test_access_is_fail_closed(tmp_path: Path) -> None:
    test_path=tmp_path/"test.parquet"; test_path.write_bytes(b"fixture")
    split_policy=tmp_path/"split_access_policy.json"
    split_policy.write_text(json.dumps({
        "datasets":{"synthetic":{"test":{"path":str(test_path)}}}
    }))
    split_hash=hashlib.sha256(split_policy.read_bytes()).hexdigest()
    protocol=tmp_path/"protocol.json"
    protocol.write_text(json.dumps({
        "schema_version":STEP15_VERSION,
        "split_access_policy":{"path":str(split_policy),"sha256":split_hash},
    }))
    auth=tmp_path/"auth.json"
    auth.write_text(json.dumps({
        "schema_version":"geas35.rl.test_access_authorization.v1",
        "purpose":"final_test", "authorized":True,
        "protocol_sha256":"wrong", "test_path":str(test_path),
    }))
    with pytest.raises(Step15ContractError, match="protocol hash"):
        authorize_test_access(protocol,auth)
    digest=hashlib.sha256(protocol.read_bytes()).hexdigest()
    payload=json.loads(auth.read_text()); payload["protocol_sha256"]=digest
    auth.write_text(json.dumps(payload))
    assert authorize_test_access(protocol,auth) == test_path.resolve()
    other=tmp_path/"other.parquet"; other.write_bytes(b"fixture")
    payload["test_path"]=str(other); auth.write_text(json.dumps(payload))
    with pytest.raises(Step15ContractError, match="not a frozen Test dataset"):
        authorize_test_access(protocol,auth)



def test_changed_config_requires_new_protocol_version(tmp_path: Path) -> None:
    config = tmp_path / "changed.json"
    source = PROJECT_ROOT / "experiments/rl_policy_training/configs/step15_protocol_v1.json"
    payload = json.loads(source.read_text())
    payload["support"]["neighbors"] = 49
    config.write_text(json.dumps(payload))
    output = tmp_path / "step15"
    output.mkdir()
    (output / "rl_protocol_manifest.json").write_text(json.dumps({
        "protocol_config": {"sha256": "frozen-different-hash"}
    }))
    with pytest.raises(Step15ContractError, match="config changed"):
        run_step15_contract(
            project_root=PROJECT_ROOT, output_root=output, config_path=config
        )

def test_official_step15_artifacts_are_complete() -> None:
    root=PROJECT_ROOT/"experiments/rl_policy_training/artifacts/step15"
    integrity=json.loads((root/"step15_integrity_manifest.json").read_text())
    protocol=json.loads((root/"rl_protocol_manifest.json").read_text())
    assert integrity["status"] == "passed"
    assert integrity["checks"]["step16_environment_implemented"] is False
    assert protocol["hpo_primary_metric"] == "mean_reward_per_scheduled_valid_step"
    assert set(protocol["crops"]) == {"strawberry","melon","cucumber"}
    for crop,record in protocol["crops"].items():
        assert record["candidate_name"] == "extra_trees"
        assert record["validation_rows_used_for_support"] == 0
        assert record["test_rows_used_for_support"] == 0
        support_path=PROJECT_ROOT/record["state_action_support"]["path"]
        support=json.loads(support_path.read_text())
        assert support["fit_split"] == "train_only"
        assert support["empirical_calibration_coverage"]["state_warning"] >= .95
        assert support["empirical_calibration_coverage"]["state_severe"] >= .99
