from __future__ import annotations

from typing import Dict, Any
from pathlib import Path
import os
import sys

_UPDATED_ROOT = Path(__file__).resolve().parents[2]
_INNER_ROOT = _UPDATED_ROOT / "inner_layer"
for _path in (_UPDATED_ROOT, _INNER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from core.config import DbConfig
from controller_layer.runtime import run_once
from utilities.actuator_mapping import map_controller_output_to_legacy
from utilities.control_io import (
    insert_control_to_monitoring,  
    insert_controls,               
    send_control_api,              
)
from controller_layer.control_params import (
    insert_params_to_log,
)


def debug_actuation_from_result(cfg: DbConfig, result: Dict[str, Any]) -> None:
    now = result.get("now")             
    ctrl_res = result.get("control")     
    outer_res = result.get("outer") or {}
    policy_state = result.get("policy_state", None)  # 필요하면 나중에 활용

    if ctrl_res is None:
        print("[DEBUG] no control result; skip debug_actuation_from_result.")
        return

    kpi = outer_res.get("kpi", {}) or {}

    print(
        "[DEBUG CTRL] time={time}, window={w}, curtain={c}, fcu={f}, fan={fan}".format(
            time=now,
            w=ctrl_res.get("window"),
            c=ctrl_res.get("curtain"),
            f=ctrl_res.get("fcu"),
            fan=ctrl_res.get("fan"),
        )
    )
    print(
        "[DEBUG POLICY] updated={updated} score={score} cov_day={cov_day} cov_night={cov_night}".format(
            updated=outer_res.get("updated"),
            score=outer_res.get("score"),
            cov_day=kpi.get("cov_day", 0.0),
            cov_night=kpi.get("cov_night", 0.0),
        )
    )

    legacy_control = map_controller_output_to_legacy(ctrl_res)
    print("[LEGACY CONTROL]", legacy_control)

    try:
        insert_control_to_monitoring(legacy_control, cfg)
        
    except Exception as e:
        print("[ERROR] insert_control_to_monitoring failed:", e)

    try:
        insert_params_to_log(result, cfg)
        
    except Exception as e:
        print("[ERROR] insert_params_to_log failed:", e)

    try:
        insert_controls(legacy_control) 
        
    except Exception as e:
        print("[ERROR] insert_controls failed:", e)

    try:
        send_control_api(legacy_control)
        
    except Exception as e:
        print("[ERROR] send_control_api failed:", e)


def debug_actuation_from_env() -> None:
    cfg = DbConfig.from_env()

    lat = float(os.getenv("DEBUG_LAT", os.getenv("LAT", "36.46")))
    lon = float(os.getenv("DEBUG_LON", os.getenv("LON", "128.22")))
    fcu_mode = os.getenv("DEBUG_FCU_MODE", os.getenv("FCU_MODE", "heat"))

    stage_name = os.getenv("DEBUG_STAGE_NAME", os.getenv("STAGE_NAME", "stage3"))
    profile_name = os.getenv("DEBUG_PROFILE_NAME", os.getenv("PROFILE_NAME", "safe_default"))
    policy_state_path = os.getenv("DEBUG_POLICY_STATE_PATH", os.getenv("POLICY_STATE_PATH", str(_INNER_ROOT / "policy_state.json")))

    control_table = os.getenv("CONTROL_TABLE", "control_log")
    policy_table = os.getenv("POLICY_TABLE", "policy_log")

    print(
        "[DEBUG] run_once with "
        f"lat={lat}, lon={lon}, fcu_mode={fcu_mode}, "
        f"stage={stage_name}, profile={profile_name}"
    )

    result = run_once(
        lat=lat,
        lon=lon,
        fcu_mode=fcu_mode,
        stage_name=stage_name,
        profile_name=profile_name,
        policy_state_path=policy_state_path,
        control_table=control_table,
        policy_table=policy_table,
    )

    if result is None:
        print("[DEBUG] run_once returned None; skip debug_actuation_from_result.")
        return

    debug_actuation_from_result(cfg, result)
