from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import pymysql

from core.config import DbConfig
from core.timeutils import kst_now, today_window_kst, slice_today
from core.preprocessing import normalize_for_derived, latest_row_for_controller
from core.solar_eta import compute_eta, integrate_measured_to_jcm2
from core.features import compute_all_features
from infra.repository import Repository
from policy.controller import Controller, PolicyState
from policy.profiles import get_profile
from policy.stage import stage as STAGE_CONFIG

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

    df_norm = normalize_for_derived(df_today_raw)
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
                CREATE TABLE IF NOT EXISTS {policy_table} (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    reg_date DATETIME NOT NULL,
                    farm_sn INT NOT NULL,
                    updated TINYINT(1),
                    score DOUBLE,
                    cov_day DOUBLE,
                    cov_night DOUBLE,
                    mdev_day DOUBLE,
                    mdev_night DOUBLE,
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
            )
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
    values["reg_date"] = now

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
    policy_table: str = "policy_log",
) -> Dict[str, Any] | None:

    cfg = DbConfig.from_env()

    try:
        ensure_control_table(cfg, control_table)
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
    policy_state = load_policy_state(policy_state_path)
    profile_params = get_profile(profile_name) or {}
    params: Dict[str, Any] = dict(profile_params)
    params["fcu_mode"] = fcu_mode

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

    print(
        f"[CTRL] time={now}, "
        f"window={res['window']}, curtain={res['curtain']}, "
        f"fcu={res['fcu']}, fan={res['fan']}"
    )

    try:
        log_control_to_db(cfg, res, now.to_pydatetime(), cfg.farm_sn, control_table)
        
    except Exception as e:
        print(f"[ERROR] log_control_to_db failed: {e}")

    try:
        log_control_actuation_to_db(cfg, res, now.to_pydatetime())
        
    except Exception as e:
        print(f"[ERROR] log_control_actuation_to_db failed: {e}")

    outer_res = res.get("outer", {})
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

    return res
