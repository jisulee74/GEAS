from __future__ import annotations

import json
import os
import runpy
import sys
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

_UPDATED_ROOT = Path(__file__).resolve().parents[1]
_INNER_ROOT = _UPDATED_ROOT / "inner_layer"
for _path in (_UPDATED_ROOT, _INNER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd
import pymysql

from core.config import DbConfig
from core.timeutils import kst_now, today_window_kst, slice_today
from core.preprocessing import normalize_for_derived, latest_row_for_controller
from core.solar_eta import compute_eta, integrate_measured_to_jcm2
from core.features import compute_all_features
from infra.repository import Repository
from utilities.sensor_qc_runtime import apply_sensor_calibration
from policy.controller import Controller
from controller_layer.policy_common import PolicyState
from controller_layer.control_params import insert_params_to_log
from policy.profiles import get_profile
from policy.stage import stage as STAGE_CONFIG
from outer_layer.daily_policy_update import DailyPolicyUpdater
from outer_layer.daily_modeling_batch import run_daily_modeling_batch

_POLICY_STATE_FIELDS = set(PolicyState.__dataclass_fields__.keys())


def db_connect_for_write(cfg: DbConfig) -> pymysql.connections.Connection:
    conn = pymysql.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        database=cfg.name,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.Cursor,
    )
    with conn.cursor() as cur:
        cur.execute("SET time_zone = %s", ("+09:00",))
        cur.execute("SET NAMES utf8mb4")
    return conn


def fetch_today_dataframe(repo: Repository) -> pd.DataFrame:
    now = kst_now()
    sod, _ = today_window_kst(now)

    limit = int(os.getenv("FETCH_LIMIT_TODAY", "5000"))
    rows = repo.fetch_since(since_ts=sod, limit=limit, inclusive=True)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def fetch_today_named_table_dataframe(cfg: DbConfig, table_name: str) -> pd.DataFrame:
    now = kst_now()
    sod, _ = today_window_kst(now)
    limit = int(os.getenv("FETCH_LIMIT_TODAY", "5000"))
    sql = f"""
        SELECT *
        FROM `{table_name}`
        WHERE farm_sn = %s AND reg_date >= %s AND reg_date <= %s
        ORDER BY reg_date ASC
        LIMIT %s
    """
    conn = db_connect_for_write(cfg)
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(sql, (cfg.farm_sn, sod.to_pydatetime(), now.to_pydatetime(), limit))
            rows = cur.fetchall()
    except pymysql.err.ProgrammingError:
        return pd.DataFrame()
    except pymysql.err.OperationalError:
        return pd.DataFrame()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


_CASE_CLASSIFIER_API: Optional[Dict[str, Any]] = None


def _load_case_classifier_api() -> Optional[Dict[str, Any]]:
    global _CASE_CLASSIFIER_API
    if _CASE_CLASSIFIER_API is not None:
        return _CASE_CLASSIFIER_API

    model_path = _UPDATED_ROOT / "outer_layer" / "data_case_classifier.py"
    if not model_path.exists():
        _CASE_CLASSIFIER_API = None
        return None

    try:
        _CASE_CLASSIFIER_API = runpy.run_path(str(model_path))
    except Exception as exc:
        print(f"[CASE CLASSIFIER] load failed: {exc}")
        _CASE_CLASSIFIER_API = None
    return _CASE_CLASSIFIER_API


def fetch_recent_history_dataframe(repo: Repository) -> pd.DataFrame:
    now = kst_now()
    lookback_days = int(os.getenv("PHYSICS_CASE_LOOKBACK_DAYS", "35"))
    since_ts = now - pd.Timedelta(days=lookback_days)
    limit = int(os.getenv("FETCH_LIMIT_PHYSICS_CASE", "50000"))
    rows = repo.fetch_since(since_ts=since_ts, limit=limit, inclusive=True)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def detect_physics_case(df_recent: pd.DataFrame) -> Dict[str, Any]:
    api = _load_case_classifier_api()
    if api and "classify_data_availability" in api:
        classify = api["classify_data_availability"]
        decision = classify(
            df_recent,
            time_col="reg_date",
            sufficient_days=int(os.getenv("PHYSICS_CASE_SUFFICIENT_DAYS", "30")),
            limited_days=int(os.getenv("PHYSICS_CASE_LIMITED_DAYS", "3")),
        )
        return {
            "data_case": decision.data_case,
            "metrics": dict(decision.metrics),
            "reasons": list(decision.reasons),
        }

    row_count = int(len(df_recent.index))
    limited_threshold = int(os.getenv("PHYSICS_LIMITED_ROWS", "24"))
    if row_count <= 0:
        return {"data_case": "no_data", "metrics": {"row_count": 0}, "reasons": ["empty_dataframe"]}
    if row_count < limited_threshold:
        return {"data_case": "limited", "metrics": {"row_count": row_count}, "reasons": ["row_count_below_threshold"]}
    return {"data_case": "sufficient", "metrics": {"row_count": row_count}, "reasons": []}


def load_policy_state(path: str) -> PolicyState:
    if not os.path.exists(path):
        return PolicyState()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return PolicyState()

        # Support both legacy flat JSON and nested "params" JSON.
        source = data.get("params") if isinstance(data.get("params"), dict) else data
        policy_data = {k: v for k, v in source.items() if k in _POLICY_STATE_FIELDS}
        return PolicyState(**policy_data)
    except Exception:
        return PolicyState()


def save_policy_state(path: str, st: PolicyState) -> None:
    data = st.to_dict()
    meta: Dict[str, Any] = {}
    use_nested_params = False

    # Preserve non-policy keys (e.g., comments/description) and keep existing shape.
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict):
                use_nested_params = isinstance(old.get("params"), dict)
                meta = {k: v for k, v in old.items() if k not in _POLICY_STATE_FIELDS and k != "params"}
        except Exception:
            meta = {}

    if use_nested_params:
        payload = {**meta, "params": data}
    else:
        payload = {**meta, **data}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def prepare_today_context(
    df_today_raw: pd.DataFrame,
    lat: float,
    lon: float,
    target_jcm2: float,
    T_day: float,
    T_night: float,
) -> Optional[Dict[str, Any]]:

    if df_today_raw.empty or "reg_date" not in df_today_raw.columns:
        return None

    now = kst_now()

    df_calibrated = apply_sensor_calibration(df_today_raw)
    df_norm = normalize_for_derived(df_calibrated)
    df_today = slice_today(df_norm, now, col="reg_date")
    if df_today.empty:
        return None

    df_today = df_today.copy()

    df_today["out_light_sum"] = integrate_measured_to_jcm2(
        df_today, col="out_light", time_col="reg_date"
    )

    eta, sunrise, sunset, cs_df = compute_eta(
        lat=lat,
        lon=lon,
        target_jcm2=target_jcm2,
        df_all=df_today,
        now_ts=now,
    )

    def _to_naive(ts: pd.Timestamp | None) -> pd.Timestamp | None:
        if ts is None or pd.isna(ts):
            return ts
        if ts.tzinfo is None:
            return ts
        return ts.tz_convert("Asia/Seoul").tz_localize(None)

    sunrise_naive = _to_naive(sunrise)
    sunset_naive = _to_naive(sunset)
    eta_naive = _to_naive(eta) if eta is not None else None

    cs_sum = cs_df["sum_jcm2"].to_numpy(dtype=float)

    agro_kpis = compute_all_features(
        df_day=df_today,
        sunrise=sunrise_naive,
        sunset=sunset_naive,
        cs_sum_jcm2=cs_sum,
        target_jcm2=target_jcm2,
        light_eta=eta_naive,
        T_day=T_day,
        T_night=T_night,
    )

    latest = latest_row_for_controller(df_today)

    return {
        "now": now,
        "df_today": df_today,
        "latest": latest,
        "sunrise": sunrise_naive,
        "sunset": sunset_naive,
        "eta": eta_naive,
        "cs_sum": cs_sum,
        "agro_kpis": agro_kpis,
    }


def log_control_to_db(
    cfg: DbConfig,
    ctrl_res: Dict[str, Any],
    now: datetime,
    farm_sn: int,
    control_table: str,
) -> None:

    (win_pct, _, win_cmd) = ctrl_res["window"]
    (cur_mode, cur_pct) = ctrl_res["curtain"]
    (fcu_mode, fcu_state) = ctrl_res["fcu"]
    (fan_state,) = ctrl_res["fan"]

    sql = f"""
        INSERT INTO {control_table} (
            reg_date, farm_sn,
            window_pct, window_cmd,
            curtain_mode, curtain_pct,
            fcu_mode, fcu_state,
            fan_state
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    params = (
        now,
        farm_sn,
        int(win_pct),
        str(win_cmd),
        str(cur_mode),
        int(cur_pct),
        str(fcu_mode),
        str(fcu_state),
        str(fan_state),
    )

    conn = db_connect_for_write(cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        print(
            f"[CTRL LOG] inserted into {cfg.name}.{control_table}: "
            f"reg_date={now}, farm_sn={farm_sn}, "
            f"window_pct={win_pct}, curtain_mode={cur_mode}, "
            f"fcu_mode={fcu_mode}, fan_state={fan_state}"
        )
    finally:
        conn.close()
def log_policy_to_db(
    cfg: DbConfig,
    policy_state: PolicyState,
    outer_res: Dict[str, Any],
    now: datetime,
    farm_sn: int,
    policy_table: str,
) -> None:

    kpi = outer_res.get("kpi", {}) or {}

  
    conn = db_connect_for_write(cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SHOW COLUMNS FROM `{policy_table}`")
            rows = cur.fetchall()
            cols_in_db = {row[0] for row in rows}

        candidate_values: Dict[str, Any] = {
            "reg_date": now,
            "farm_sn": farm_sn,
            "cov_day": float(kpi.get("cov_day", 0.0)),
            "cov_night": float(kpi.get("cov_night", 0.0)),
            "mdev_day": float(kpi.get("mdev_day", 0.0)),
            "mdev_night": float(kpi.get("mdev_night", 0.0)),
            "temp_viol_rate": float(kpi.get("temp_viol_rate", 0.0)),
            "cond_viol_rate": float(kpi.get("cond_viol_rate", 0.0)),
            "rh_viol_rate": float(kpi.get("rh_viol_rate", 0.0)),
            "vpd_viol_rate": float(kpi.get("vpd_viol_rate", 0.0)),
            "hv_ineff_rate": float(kpi.get("hv_ineff_rate", 0.0)),
            "max_consec_temp_viol": int(kpi.get("max_consec_temp_viol", 0) or 0),
            "max_consec_cond_viol": int(kpi.get("max_consec_cond_viol", 0) or 0),
            "max_consec_rh_viol": int(kpi.get("max_consec_rh_viol", 0) or 0),
            "max_consec_vpd_viol": int(kpi.get("max_consec_vpd_viol", 0) or 0),
            "T_day_bias": float(policy_state.T_day_bias),
            "T_night_bias": float(policy_state.T_night_bias),
            "DB_day": float(policy_state.DB_day),
            "PB_day": float(policy_state.PB_day),
            "DB_night": float(policy_state.DB_night),
            "PB_night": float(policy_state.PB_night),
            "K_window_day": float(policy_state.K_window_day),
            "K_window_night": float(policy_state.K_window_night),
            "alpha_curtain": float(policy_state.alpha_curtain),
            "fcu_mode": str(policy_state.fcu_mode),
            "updated": int(bool(outer_res.get("updated", False))),
            "score": outer_res.get("score", None),
            "sunlight_ratio": outer_res.get("sunlight_ratio", None),
            "evaluation_mode": outer_res.get("evaluation_mode", None),
            "batch_due": int(bool(outer_res.get("batch_due", False))),
            "batch_executed": int(bool(outer_res.get("batch_executed", False))),
        }

        values: Dict[str, Any] = {}
        for col, v in candidate_values.items():
            if col in cols_in_db:
                values[col] = v

        if not values:
            print(
                f"[POLICY LOG] skip: no matching columns in {cfg.name}.{policy_table}"
            )
            return

        cols = ", ".join(f"`{c}`" for c in values.keys())
        placeholders = ", ".join(["%s"] * len(values))
        sql = f"INSERT INTO `{policy_table}` ({cols}) VALUES ({placeholders})"
        params = list(values.values())

        with conn.cursor() as cur:
            cur.execute(sql, params)

        print(
            f"[POLICY LOG] inserted into {cfg.name}.{policy_table}: "
            f"reg_date={now}, farm_sn={farm_sn}"
        )
    finally:
        conn.close()

def ensure_control_table(cfg: DbConfig, control_table: str) -> None:

    conn = db_connect_for_write(cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {control_table} (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    reg_date DATETIME NOT NULL,
                    farm_sn INT NOT NULL,
                    window_pct INT,
                    window_cmd VARCHAR(32),
                    curtain_mode VARCHAR(32),
                    curtain_pct INT,
                    fcu_mode VARCHAR(32),
                    fcu_state VARCHAR(32),
                    fan_state VARCHAR(32),
                    PRIMARY KEY (id),
                    KEY idx_reg_date (reg_date),
                    KEY idx_farm_reg (farm_sn, reg_date)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        print(f"[INIT] ensured table {cfg.name}.{control_table}")
    finally:
        conn.close()

def ensure_policy_table(cfg: DbConfig, policy_table: str) -> None:

    conn = db_connect_for_write(cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS `{policy_table}` (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    reg_date DATETIME NOT NULL,
                    farm_sn INT NOT NULL,
                    updated TINYINT(1),
                    score DOUBLE,
                    sunlight_ratio DOUBLE,
                    evaluation_mode VARCHAR(32),
                    batch_due TINYINT(1),
                    batch_executed TINYINT(1),
                    cov_day DOUBLE,
                    cov_night DOUBLE,
                    mdev_day DOUBLE,
                    mdev_night DOUBLE,
                    temp_viol_rate DOUBLE,
                    cond_viol_rate DOUBLE,
                    rh_viol_rate DOUBLE,
                    vpd_viol_rate DOUBLE,
                    hv_ineff_rate DOUBLE,
                    max_consec_temp_viol INT,
                    max_consec_cond_viol INT,
                    max_consec_rh_viol INT,
                    max_consec_vpd_viol INT,
                    T_day_bias DOUBLE,
                    T_night_bias DOUBLE,
                    DB_day DOUBLE,
                    PB_day DOUBLE,
                    DB_night DOUBLE,
                    PB_night DOUBLE,
                    K_window_day DOUBLE,
                    K_window_night DOUBLE,
                    alpha_curtain DOUBLE,
                    fcu_mode VARCHAR(32),
                    PRIMARY KEY (id),
                    KEY idx_reg_date (reg_date),
                    KEY idx_farm_reg (farm_sn, reg_date)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            alter_stmts = [
                f"ALTER TABLE `{policy_table}` ADD COLUMN IF NOT EXISTS sunlight_ratio DOUBLE NULL AFTER score",
                f"ALTER TABLE `{policy_table}` ADD COLUMN IF NOT EXISTS evaluation_mode VARCHAR(32) NULL AFTER sunlight_ratio",
                f"ALTER TABLE `{policy_table}` ADD COLUMN IF NOT EXISTS batch_due TINYINT(1) NULL AFTER evaluation_mode",
                f"ALTER TABLE `{policy_table}` ADD COLUMN IF NOT EXISTS batch_executed TINYINT(1) NULL AFTER batch_due",
                f"ALTER TABLE `{policy_table}` ADD COLUMN IF NOT EXISTS temp_viol_rate DOUBLE NULL AFTER mdev_night",
                f"ALTER TABLE `{policy_table}` ADD COLUMN IF NOT EXISTS cond_viol_rate DOUBLE NULL AFTER temp_viol_rate",
                f"ALTER TABLE `{policy_table}` ADD COLUMN IF NOT EXISTS rh_viol_rate DOUBLE NULL AFTER cond_viol_rate",
                f"ALTER TABLE `{policy_table}` ADD COLUMN IF NOT EXISTS vpd_viol_rate DOUBLE NULL AFTER rh_viol_rate",
                f"ALTER TABLE `{policy_table}` ADD COLUMN IF NOT EXISTS hv_ineff_rate DOUBLE NULL AFTER vpd_viol_rate",
                f"ALTER TABLE `{policy_table}` ADD COLUMN IF NOT EXISTS max_consec_temp_viol INT NULL AFTER hv_ineff_rate",
                f"ALTER TABLE `{policy_table}` ADD COLUMN IF NOT EXISTS max_consec_cond_viol INT NULL AFTER max_consec_temp_viol",
                f"ALTER TABLE `{policy_table}` ADD COLUMN IF NOT EXISTS max_consec_rh_viol INT NULL AFTER max_consec_cond_viol",
                f"ALTER TABLE `{policy_table}` ADD COLUMN IF NOT EXISTS max_consec_vpd_viol INT NULL AFTER max_consec_rh_viol",
            ]
            for stmt in alter_stmts:
                cur.execute(stmt)
        print(f"[INIT] ensured table {cfg.name}.{policy_table}")
    finally:
        conn.close()

def _map_window_lr(window: Tuple[Any, Any, Any]) -> Tuple[int, int]:
    _, side, value = window
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = 0
    if v == 5:
        return v, v
    if side == "left":
        return v, 20
    if side == "right":
        return 20, v
    return v, v

def _map_curtain_values(curtain: Tuple[str, int]) -> Tuple[int, int]:
    mode, value = curtain
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = 0
    if mode == "cha_gwang":
        return v, 0
    if mode == "bo_on":
        return 0, v
    return v, 0

def _map_fcu_values(fcu: Tuple[str, int]) -> Tuple[int, int, int, int, int, int]:
    mode, value = fcu
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = 0
    if mode == "cooling":
        pred_cooler = v
        pred_heater = 0
        pred_cp1 = 0
        pred_cp2 = 0
        pred_tw1 = 0
        pred_tw2 = 0
    elif mode == "heating":
        pred_cooler = 0
        pred_heater = v
        pred_cp1 = 1
        pred_cp2 = 0
        pred_tw1 = 100
        pred_tw2 = 0
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

def _map_fan_value(fan: int) -> int:
    try:
        return int(fan)
    except (TypeError, ValueError):
        return 0


def map_controller_output_to_legacy(control_res: Dict[str, Any]) -> Dict[str, Any]:

    legacy: Dict[str, Any] = {}

    # WINDOW
    window: Optional[Tuple[Any, Any, Any]] = control_res.get("window")
    if window:
        pct_final, _, _ = window
        try:
            pct = int(pct_final)
        except (TypeError, ValueError):
            pct = 0
        legacy["window"] = (0, "both", pct)

    # CURTAIN
    curtain: Optional[Tuple[Any, Any]] = control_res.get("curtain")
    if curtain:
        mode, size = curtain
        try:
            pct = int(size)
        except (TypeError, ValueError):
            pct = 0
        legacy_mode = "cha_gwang"
        legacy["curtain"] = (legacy_mode, pct)
        
    # THERMAL CURTAIN
    thermal: Optional[Tuple[Any, Any]] = control_res.get("thermal_curtain")
    if thermal:
        mode, size = thermal
        try:
            pct = int(size)
        except (TypeError, ValueError):
            pct = 0
        legacy["thermal_curtain"] = ("thermal", pct)

    # FCU
    fcu: Optional[Tuple[Any, Any]] = control_res.get("fcu")
    if fcu:
        mode, state = fcu
        if state == "off":
            legacy["fcu"] = ("off", 0)
        elif mode == "cool":
            legacy["fcu"] = ("cooling", 1)
        elif mode == "heat":
            legacy["fcu"] = ("heating", 1)
        else:
            legacy["fcu"] = ("off", 0)

    # FAN
    fan: Optional[Tuple[Any]] = control_res.get("fan")
    if fan:
        state = fan[0]
        if state == "on":
            legacy["fan"] = 1
        elif state == "off":
            legacy["fan"] = 0

    return legacy


def _legacy_control_to_values(control: Dict[str, Any]) -> Dict[str, Any]:
    """
    legacy control dict → control_actuation 테이블 컬럼 dict
    """
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

    # THERMAL CURTAIN
    thermal = control.get("thermal_curtain")
    if thermal:
        _, value = thermal
        try:
            v = int(value)
        except (TypeError, ValueError):
            v = 0

        values["pred_pc2"] = v
        
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


def ensure_control_actuation_table(cfg: DbConfig) -> None:
    table_name = "control_actuation"
    conn = db_connect_for_write(cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS `{table_name}` (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    reg_date DATETIME NOT NULL,
                    farm_sn INT NULL,
                    window_pct INT NULL,
                    window_cmd VARCHAR(32) NULL,
                    curtain_mode VARCHAR(32) NULL,
                    curtain_pct INT NULL,
                    fcu_mode VARCHAR(32) NULL,
                    fcu_state VARCHAR(32) NULL,
                    fan_state VARCHAR(32) NULL,
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
                    KEY idx_reg_date (reg_date),
                    KEY idx_farm_reg (farm_sn, reg_date)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            alter_stmts = [
                f"ALTER TABLE `{table_name}` ADD COLUMN IF NOT EXISTS farm_sn INT NULL AFTER reg_date",
                f"ALTER TABLE `{table_name}` ADD COLUMN IF NOT EXISTS window_pct INT NULL AFTER farm_sn",
                f"ALTER TABLE `{table_name}` ADD COLUMN IF NOT EXISTS window_cmd VARCHAR(32) NULL AFTER window_pct",
                f"ALTER TABLE `{table_name}` ADD COLUMN IF NOT EXISTS curtain_mode VARCHAR(32) NULL AFTER window_cmd",
                f"ALTER TABLE `{table_name}` ADD COLUMN IF NOT EXISTS curtain_pct INT NULL AFTER curtain_mode",
                f"ALTER TABLE `{table_name}` ADD COLUMN IF NOT EXISTS fcu_mode VARCHAR(32) NULL AFTER curtain_pct",
                f"ALTER TABLE `{table_name}` ADD COLUMN IF NOT EXISTS fcu_state VARCHAR(32) NULL AFTER fcu_mode",
                f"ALTER TABLE `{table_name}` ADD COLUMN IF NOT EXISTS fan_state VARCHAR(32) NULL AFTER fcu_state",
            ]
            for stmt in alter_stmts:
                cur.execute(stmt)
        print(f"[INIT] ensured table {cfg.name}.{table_name}")
    finally:
        conn.close()


def log_control_actuation_to_db(
    cfg: DbConfig,
    ctrl_res: Dict[str, Any],
    now: datetime,
) -> None:

    table_name = "control_actuation"
    legacy = map_controller_output_to_legacy(ctrl_res)
    values = _legacy_control_to_values(legacy)
    (win_pct, _, win_cmd) = ctrl_res["window"]
    (cur_mode, cur_pct) = ctrl_res["curtain"]
    (fcu_mode, fcu_state) = ctrl_res["fcu"]
    (fan_state,) = ctrl_res["fan"]
    values["reg_date"] = now
    values["farm_sn"] = cfg.farm_sn
    values["window_pct"] = int(win_pct)
    values["window_cmd"] = str(win_cmd)
    values["curtain_mode"] = str(cur_mode)
    values["curtain_pct"] = int(cur_pct)
    values["fcu_mode"] = str(fcu_mode)
    values["fcu_state"] = str(fcu_state)
    values["fan_state"] = str(fan_state)

    cols = ", ".join(f"`{c}`" for c in values.keys())
    placeholders = ", ".join(["%s"] * len(values))
    sql = f"INSERT INTO `{table_name}` ({cols}) VALUES ({placeholders})"
    params = list(values.values())
    print(params)
    conn = db_connect_for_write(cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        print(f"[MONITOR LOG] inserted into {cfg.name}.{table_name}")
    finally:
        conn.close()

def run_once(
    *,
    lat: float,
    lon: float,
    fcu_mode: str = "heat",
    stage_name: str = "stage3",
    profile_name: str = "safe_default",
    policy_state_path: str = "policy_state.json",
    control_table: str = "control_log",
    policy_table: str = "control_log",
) -> Dict[str, Any] | None:

    cfg = DbConfig.from_env()

    try:
        ensure_policy_table(cfg, policy_table)
        ensure_control_actuation_table(cfg)
    except Exception as e:
        print(
            f"[WARN] ensure_*_table failed "
            f"(control_table={control_table}, policy_table={policy_table}): {e}"
        )

    repo = Repository(cfg)

    stage_cfg = STAGE_CONFIG.get(stage_name)
    if not stage_cfg:
        raise ValueError(f"Unknown stage name: {stage_name}")

    T_day = float(stage_cfg["temp"]["day"])
    T_night = float(stage_cfg["temp"]["night"])
    target_jcm2 = float(stage_cfg.get("max_sun_light", 0.0))

    print(
        f"[INIT] stage={stage_name}, profile={profile_name}, fcu_mode={fcu_mode}"
    )
    print(
        f"[INIT] base_temp=(day={T_day}, night={T_night}), "
        f"target_jcm2={target_jcm2}"
    )
    print(
        f"[INIT] farm_sn={cfg.farm_sn}, lat={lat}, lon={lon}"
    )

    df_recent = fetch_recent_history_dataframe(repo)
    case_info = detect_physics_case(df_recent)
    print(f"[INIT] physics_case={case_info['data_case']} metrics={case_info['metrics']} reasons={case_info['reasons']}")

    df_today_raw = fetch_today_dataframe(repo)
    if df_today_raw.empty:
        print("[INFO] No data for today. Exit.")
        return None

    ctx = prepare_today_context(
        df_today_raw=df_today_raw,
        lat=lat,
        lon=lon,
        target_jcm2=target_jcm2,
        T_day=T_day,
        T_night=T_night,
    )
    if ctx is None:
        print("[INFO] Today context not available. Exit.")
        return None

    now = ctx["now"]
    df_today = ctx["df_today"]
    latest = ctx["latest"]
    sunrise = ctx["sunrise"]
    sunset = ctx["sunset"]
    eta = ctx["eta"]
    cs_sum = ctx["cs_sum"]
    agro_kpis = ctx["agro_kpis"]

    out_light_info = (sunrise, sunset, eta, cs_sum, target_jcm2)
    policy_state_candidate = Path(policy_state_path)
    if not policy_state_candidate.is_absolute():
        inner_candidate = _INNER_ROOT / policy_state_candidate
        if inner_candidate.exists() or policy_state_path == "policy_state.json":
            policy_state_path = str(inner_candidate)
    policy_state = load_policy_state(policy_state_path)
    profile_params = get_profile(profile_name) or {}
    params: Dict[str, Any] = dict(profile_params)
    params["fcu_mode"] = fcu_mode
    params["farm_sn"] = cfg.farm_sn
    params["stage_name"] = stage_name
    params["physics_case"] = case_info["data_case"]
    params["physics_case_info"] = case_info
    params["physics_model_name"] = str(params.get("physics_model_name", "default"))
    params["physics_store_path"] = str(
        _UPDATED_ROOT / "controller_layer" / "physics_store" / "physics_params_store.json"
    )

    ctrl = Controller(
        dataframe=latest,
        out_light_info=out_light_info,
        temp=(T_day, T_night),
        params=params,
        policy_state=policy_state,
        df_today=df_today,
        derived_result=None,
        phys_defaults=None,
        agro_kpis=agro_kpis,
    )

    res = ctrl.run()
    res["agro_kpis"] = dict(agro_kpis or {})

    print(
        f"[CTRL] time={now}, "
        f"window={res['window']}, curtain={res['curtain']}, "
        f"fcu={res['fcu']}, fan={res['fan']}"
    )

    try:
        log_control_actuation_to_db(cfg, res, now.to_pydatetime())
    except Exception as e:
        print(f"[ERROR] log_control_actuation_to_db failed: {e}")

    try:
        df_state_today = fetch_today_named_table_dataframe(cfg, "control_params")
    except Exception as e:
        print(f"[WARN] fetch control_params for policy update failed: {e}")
        df_state_today = pd.DataFrame()

    try:
        df_actuation_today = fetch_today_named_table_dataframe(cfg, "control_actuation")
    except Exception as e:
        print(f"[WARN] fetch control_actuation for policy update failed: {e}")
        df_actuation_today = pd.DataFrame()

    try:
        outer_updater = DailyPolicyUpdater(
            df_today=df_today,
            out_light_info=out_light_info,
            base_temp=(T_day, T_night),
            policy_state=policy_state,
            params=params,
            df_state=df_state_today,
            df_actuation=df_actuation_today,
        )
        outer_res = outer_updater.run()
        res["outer"] = outer_res

        new_policy_dict = outer_res.get("policy", None)
        if new_policy_dict is not None:
            try:
                policy_state = PolicyState(**new_policy_dict)
            except TypeError:
                for k, v in new_policy_dict.items():
                    if hasattr(policy_state, k):
                        setattr(policy_state, k, v)

            try:
                save_policy_state(policy_state_path, policy_state)
            except Exception as e:
                print(f"[ERROR] save_policy_state failed: {e}")

            try:
                log_policy_to_db(
                    cfg,
                    policy_state,
                    outer_res,
                    now.to_pydatetime(),
                    cfg.farm_sn,
                    policy_table,
                )
            except Exception as e:
                print(f"[ERROR] log_policy_to_db failed: {e}")
    except Exception as e:
        print(f"[ERROR] DailyPolicyUpdater failed: {e}")

    try:
        insert_params_to_log(res, cfg)
    except Exception as e:
        print(f"[ERROR] insert_params_to_log failed: {e}")

    return res


def run_daily_batch(
    *,
    lat: float,
    lon: float,
    stage_name: str = "stage3",
    profile_name: str = "safe_default",
    policy_state_path: str = "policy_state.json",
    policy_table: str = "control_log",
) -> Dict[str, Any]:
    cfg = DbConfig.from_env()

    try:
        ensure_policy_table(cfg, policy_table)
    except Exception as e:
        print(f"[WARN] ensure_policy_table failed (policy_table={policy_table}): {e}")

    repo = Repository(cfg)
    stage_cfg = STAGE_CONFIG.get(stage_name)
    if not stage_cfg:
        raise ValueError(f"Unknown stage name: {stage_name}")

    T_day = float(stage_cfg["temp"]["day"])
    T_night = float(stage_cfg["temp"]["night"])
    target_jcm2 = float(stage_cfg.get("max_sun_light", 0.0))

    policy_state_candidate = Path(policy_state_path)
    if not policy_state_candidate.is_absolute():
        inner_candidate = _INNER_ROOT / policy_state_candidate
        if inner_candidate.exists() or policy_state_path == "policy_state.json":
            policy_state_path = str(inner_candidate)

    profile_params = get_profile(profile_name) or {}
    params: Dict[str, Any] = dict(profile_params)
    params["FORCE_DAILY_EVAL"] = True
    params["farm_sn"] = cfg.farm_sn
    params["stage_name"] = stage_name

    physics_store_path = str(_UPDATED_ROOT / "controller_layer" / "physics_store" / "physics_params_store.json")
    df_recent = fetch_recent_history_dataframe(repo)
    modeling_res = run_daily_modeling_batch(
        df_recent,
        farm_sn=cfg.farm_sn,
        stage_name=stage_name,
        store_path=physics_store_path,
        area_m2=float(params.get("GREENHOUSE_AREA_M2", 360.0)),
        cover_type=str(params.get("GREENHOUSE_COVER_TYPE", "single_film")),
        height_m=float(params.get("GREENHOUSE_HEIGHT_M", 4.0)),
    )

    df_today_raw = fetch_today_dataframe(repo)
    if df_today_raw.empty:
        return {
            "modeling": modeling_res,
            "outer": {
                "updated": False,
                "kpi": {},
                "intraday_kpi": {},
                "sunlight_ratio": 0.0,
                "score": None,
                "batch_due": True,
                "batch_executed": False,
                "evaluation_mode": "no_data",
            },
        }

    ctx = prepare_today_context(
        df_today_raw=df_today_raw,
        lat=lat,
        lon=lon,
        target_jcm2=target_jcm2,
        T_day=T_day,
        T_night=T_night,
    )
    if ctx is None:
        return {
            "modeling": modeling_res,
            "outer": {
                "updated": False,
                "kpi": {},
                "intraday_kpi": {},
                "sunlight_ratio": 0.0,
                "score": None,
                "batch_due": True,
                "batch_executed": False,
                "evaluation_mode": "no_context",
            },
        }

    now = ctx["now"]
    df_today = ctx["df_today"]
    sunrise = ctx["sunrise"]
    sunset = ctx["sunset"]
    eta = ctx["eta"]
    cs_sum = ctx["cs_sum"]
    df_state_today = fetch_today_named_table_dataframe(cfg, "control_params")
    df_actuation_today = fetch_today_named_table_dataframe(cfg, "control_actuation")

    out_light_info = (sunrise, sunset, eta, cs_sum, target_jcm2)
    policy_state = load_policy_state(policy_state_path)
    outer_updater = DailyPolicyUpdater(
        df_today=df_today,
        out_light_info=out_light_info,
        base_temp=(T_day, T_night),
        policy_state=policy_state,
        params=params,
        df_state=df_state_today,
        df_actuation=df_actuation_today,
    )
    outer_res = outer_updater.run()

    new_policy_dict = outer_res.get("policy", None)
    if new_policy_dict is not None:
        try:
            policy_state = PolicyState(**new_policy_dict)
        except TypeError:
            for k, v in new_policy_dict.items():
                if hasattr(policy_state, k):
                    setattr(policy_state, k, v)

        try:
            save_policy_state(policy_state_path, policy_state)
        except Exception as e:
            print(f"[ERROR] save_policy_state failed: {e}")

        try:
            log_policy_to_db(
                cfg,
                policy_state,
                outer_res,
                now.to_pydatetime(),
                cfg.farm_sn,
                policy_table,
            )
        except Exception as e:
            print(f"[ERROR] log_policy_to_db failed: {e}")

    return {
        "modeling": modeling_res,
        "outer": outer_res,
    }
