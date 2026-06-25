from __future__ import annotations

import json
import time
import datetime
from typing import Dict, Any, List, Tuple

import requests

from core.config import DbConfig
from infra.repository import Repository
from utilities.db_utils import ensure_db_and_table, get_db_connection

ENABLE_API_SEND = True

def _map_window_lr(window: Tuple[Any, Any, Any]) -> Tuple[int, int]:

    value, side, _ = window
    if value == 5:
        return value, value

    if side == "left":
        return value, 20
    if side == "right":
        return 20, value

    return value, value


def _map_curtain_values(curtain: Tuple[str, int]) -> Tuple[int, int]:

    mode, value = curtain
    if mode == "cha_gwang":
        return value, 0
    if mode == "bo_on":
        return 0, value
    return value, 0


def _map_fcu_values(fcu: Tuple[str, int]) -> Tuple[int, int, int, int, int, int]:
    mode, value = fcu

    if isinstance(value, str):
        value = 100 if value.lower() == "on" else 0
    else:
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 0

    m = str(mode).lower()
    if m in ("cool", "cooling"):
        mode = "cooling"
        
    elif m in ("heat", "heating") and value != 0:
        mode = "heating"
        
    elif m == "off" or value == 0:
        mode = "off"
        
    else:
        mode = "off" if value == 0 else m

    if mode == "cooling":
        pred_cooler = 0   
        pred_heater = 0   # 26
        pred_cp1 = 0      # 24
        pred_cp2 = 0      # 25
        pred_tw1 = 0      # 27
        pred_tw2 = 0      # 29
        
    elif mode == "heating":
        pred_cooler = 0         # 126 OFF
        pred_heater = value     # 26 ON
        pred_cp1 = 1            # 24 ON
        pred_cp2 = 0            # 25 OFF
        pred_tw1 = 100          # 27 ON
        pred_tw2 = 0            # 29 OFF
        
    elif mode == "off":
        pred_cooler = 0
        pred_heater = 0
        pred_cp1 = 0
        pred_cp2 = 0
        pred_tw1 = 0
        pred_tw2 = 0
        
    else:
        pred_cooler = 0
        pred_heater = 0
        pred_cp1 = 0
        pred_cp2 = 0
        pred_tw1 = 0
        pred_tw2 = 0

    return pred_cooler, pred_heater, pred_cp1, pred_cp2, pred_tw1, pred_tw2

def _map_fan_value(fan: Any) -> int:
    if isinstance(fan, tuple) and len(fan) > 0:
        fan = fan[0]

    if isinstance(fan, str):
        return 1 if fan.lower() == "on" else 0

    try:
        return int(fan)
    
    except (TypeError, ValueError):
        return 0

def build_control_payloads(control: Dict[str, Any]) -> List[Dict[str, str]]:

    payload_list: List[Dict[str, str]] = []

    # --- WINDOW ---
    window = control.get("window")
    if window:
        pred_ltw, pred_rtw = _map_window_lr(window)

        payload_list.append({
            "controller_node_id": "10",
            "device_id": "16",
            "controller_id": "100",
            "target": str(pred_ltw),
        })
        payload_list.append({
            "controller_node_id": "10",
            "device_id": "18",
            "controller_id": "100",
            "target": str(pred_rtw),
        })

    # --- CURTAIN ---
    curtain = control.get("curtain")
    if curtain:
        pred_pc1, pred_pc2 = _map_curtain_values(curtain)

        payload_list.append({
            "controller_node_id": "10",
            "device_id": "120",
            "controller_id": "100",
            "target": str(pred_pc1),
        })

    # --- THERMAL CURTAIN ---
    thermal = control.get("thermal_curtain")
    if thermal:
        _, value = thermal
        try:
            v = int(value)
        except (TypeError, ValueError):
            v = 0

        payload_list.append({
            "controller_node_id": "10",
            "device_id": "122",
            "controller_id": "100",
            "target": str(v),
        })

    # --- FCU ---
    fcu = control.get("fcu")
    if fcu:

        if isinstance(fcu, (list, tuple)) and len(fcu) >= 2:
            mode, raw_value = fcu[0], fcu[1]
        else:
            mode, raw_value = fcu, 0

        if isinstance(raw_value, str):
            value: int = 100 if raw_value.lower() == "on" else 0
        else:
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                value = 0

        mode_str = str(mode).lower()
        if mode_str in ("cool", "cooling"):
            legacy_mode = "cooling"
        elif mode_str in ("heat", "heating"):
            legacy_mode = "heating"
        elif mode_str == "off" or value == 0:
            legacy_mode = "off"
        else:

            legacy_mode = mode_str

        (
            pred_cooler,
            pred_heater,
            pred_cp1,
            pred_cp2,
            pred_tw1,
            pred_tw2,
        ) = _map_fcu_values((legacy_mode, value))

        payload_list.append({
            "controller_node_id": "10",
            "device_id": "26",   # heater
            "controller_id": "100",
            "target": str(pred_heater),
        })
        payload_list.append({
            "controller_node_id": "10",
            "device_id": "126",  # cooler
            "controller_id": "100",
            "target": str(pred_cooler),
        })
        payload_list.append({
            "controller_node_id": "10",
            "device_id": "27",   # TW1
            "controller_id": "100",
            "target": str(pred_tw1),
        })
        payload_list.append({
            "controller_node_id": "10",
            "device_id": "29",   # TW2
            "controller_id": "100",
            "target": str(pred_tw2),
        })
        payload_list.append({
            "controller_node_id": "10",
            "device_id": "24",   # CP1
            "controller_id": "100",
            "target": str(pred_cp1),
        })
        payload_list.append({
            "controller_node_id": "10",
            "device_id": "25",   # CP2
            "controller_id": "100",
            "target": str(pred_cp2),
        })

    # --- FAN ---
    fan = control.get("fan")
    if fan is not None:
        pred_fan = _map_fan_value(fan)
        payload_list.append({
            "controller_node_id": "10",
            "device_id": "31",
            "controller_id": "100",
            "target": str(pred_fan),
        })

    return payload_list

def send_control_api(control: Dict[str, Any]) -> None:

    payload_list = build_control_payloads(control)

    print(f"[API DRY-RUN] {len(payload_list)} payload(s) would be sent:")
    for p in payload_list:
        print("   ", p)

    if not ENABLE_API_SEND:
        return

    login_payload = json.dumps({
        "user_id": "user",
        "password": "user"
    })

    conn = requests.Session()
    try:
        conn.post(
            "http://theimc-sj1.iptime.org:8080/api/auth/login",
            data=login_payload,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )

        repo = Repository(DbConfig.from_env())

        for payload in payload_list:
            res = conn.post(
                "http://theimc-sj1.iptime.org:8080/api/override/target",
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )

            if res.status_code == 200:
                log_param = {
                    "device_id": payload["device_id"],
                    "target": payload["target"],
                }
                try:
                    repo.logging(log_param) 
                except Exception as e:
                    print("[ERROR] logging failed:", e)
            else:
                print("[ERROR] override failed:", res.status_code, res.text)
            
            time.sleep(0.5)

    except Exception as e:
        print("[ERROR] send_control_api:", e)
    finally:
        try:
            conn.post(
                "http://theimc-sj1.iptime.org:8080/api/auth/logout",
                timeout=3,
            )
        except Exception:
            pass
        conn.close()

def _legacy_control_to_values(control: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}

    # WINDOW
    window = control.get("window")
    if window:
        pred_ltw, pred_rtw = _map_window_lr(window)
        values["pred_ltw"] = pred_ltw
        values["pred_rtw"] = pred_rtw

    # CURTAIN
    curtain = control.get("curtain")
    if curtain:
        pred_pc1, pred_pc2 = _map_curtain_values(curtain)
        values["pred_pc1"] = pred_pc1
        values["pred_pc2"] = pred_pc2

    # FCU
    fcu = control.get("fcu")
    if fcu:
        (
            pred_cooler,
            pred_heater,
            pred_cp1,
            pred_cp2,
            pred_tw1,
            pred_tw2,
        ) = _map_fcu_values(fcu)
        values["pred_cooler"] = pred_cooler
        values["pred_heater"] = pred_heater
        values["pred_cp1"] = pred_cp1
        values["pred_cp2"] = pred_cp2
        values["pred_tw1"] = pred_tw1
        values["pred_tw2"] = pred_tw2

    # FAN
    fan = control.get("fan")
    if fan is not None:
        values["pred_fan"] = _map_fan_value(fan)

    return values


def insert_controls(control: Dict[str, Any]) -> None:

    values = _legacy_control_to_values(control)

    values["reg_date"] = datetime.datetime.now().replace(
        second=0, microsecond=0
    ).strftime("%Y-%m-%d %H:%M:%S")

    print("[DEBUG] control row (would be inserted to DB):")
    print("         ", values)


def insert_control_to_monitoring(control: Dict[str, Any], cfg: DbConfig) -> None:
    values = _legacy_control_to_values(control)
    values["reg_date"] = datetime.datetime.now().replace(
        second=0, microsecond=0
    ).strftime("%Y-%m-%d %H:%M:%S")

    print("[DEBUG] control row (would be inserted to monitoring DB):")
    print("         ", values)

    db_name = cfg.name          # 예: "farmstom"
    table_name = "control_actuation"

    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
            reg_date DATETIME NOT NULL,
            pred_ltw INT NULL,
            pred_rtw INT NULL,
            pred_pc1 INT NULL,
            pred_pc2 INT NULL,
            pred_cooler INT NULL,
            pred_heater INT NULL,
            pred_cp1 INT NULL,
            pred_cp2 INT NULL,
            pred_tw1 INT NULL,
            pred_tw2 INT NULL,
            pred_fan INT NULL,
            PRIMARY KEY (id),
            KEY idx_reg_date (reg_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """

    try:
        ensure_db_and_table(cfg, db_name, create_table_sql)

        conn = get_db_connection(cfg, db_name=db_name, autocommit=True)
        try:
            cols = ", ".join(f"`{c}`" for c in values.keys())
            placeholders = ", ".join(["%s"] * len(values))
            sql = f"INSERT INTO `{table_name}` ({cols}) VALUES ({placeholders})"
            params = list(values.values())
            
            with conn.cursor() as cur:
                cur.execute(sql, params)

            print(f"[MONITOR LOG] inserted into {db_name}.{table_name}")
        finally:
            conn.close()

    except Exception as e:
        print("[ERROR] insert_control_to_monitoring:", e)
