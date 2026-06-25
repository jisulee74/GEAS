# main.py
from __future__ import annotations

from dotenv import load_dotenv

from pathlib import Path
import os
import sys

_UPDATED_ROOT = Path(__file__).resolve().parents[1]
_INNER_ROOT = _UPDATED_ROOT / "inner_layer"
for _path in (_UPDATED_ROOT, _INNER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from core.config import DbConfig
from controller_layer.runtime import run_once, run_daily_batch
from utilities.control_io import send_control_api

def main() -> None:
    load_dotenv(dotenv_path=_INNER_ROOT / ".env")

    cfg = DbConfig.from_env()
    print(
        "[DB CONFIG] host={host}, port={port}, user={user}, db={db}, table={table}, farm_sn={farm_sn}".format(
            host=cfg.host,
            port=cfg.port,
            user=cfg.user,
            db=cfg.name,
            table=cfg.main_table,
            farm_sn=cfg.farm_sn,
        )
    )

    FCU_MODE = "heat"
    LAT = 36.46
    LON = 128.22

    STAGE_NAME = "stage3" # policy/stage.py 코드로부터 stage 3의 주간/야간 목표온도(25/10)와 목표최대일사량(1200)을 가져오는 키
    PROFILE_NAME = "safe_default"
    POLICY_STATE_PATH = str(_INNER_ROOT / "policy_state.json")

    CONTROL_TABLE = "control_log"
    POLICY_TABLE = "control_log"
    RUN_MODE = os.getenv("GEAS_RUN_MODE", "realtime").strip().lower()

    if RUN_MODE == "daily_batch":
        run_daily_batch(
            lat=LAT,
            lon=LON,
            stage_name=STAGE_NAME,
            profile_name=PROFILE_NAME,
            policy_state_path=POLICY_STATE_PATH,
            policy_table=POLICY_TABLE,
        )
        return

    res = run_once( # controller_layer/runtime.py의 run_once 함수를 호출하여 실시간 제어값과 로그를 얻음
        lat=LAT,
        lon=LON,
        fcu_mode=FCU_MODE,
        stage_name=STAGE_NAME,
        profile_name=PROFILE_NAME,
        policy_state_path=POLICY_STATE_PATH,
        control_table=CONTROL_TABLE,
        policy_table=POLICY_TABLE,
    )

    send_control_api(res)

if __name__ == "__main__":
    main()
