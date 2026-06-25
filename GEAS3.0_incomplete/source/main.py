# main.py
from __future__ import annotations

from dotenv import load_dotenv

from core.config import DbConfig
from utilities.runtime import run_once
from utilities.control_io import send_control_api
from debug.debug_actuation import debug_actuation_from_env

def main() -> None:
    load_dotenv(dotenv_path="/home/farmstom/os/module/ver3.0/.env")

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

    STAGE_NAME = "stage3"
    PROFILE_NAME = "safe_default"
    POLICY_STATE_PATH = "policy_state.json"

    CONTROL_TABLE = "control_log"
    POLICY_TABLE = "control_params"   

    res = run_once(
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
