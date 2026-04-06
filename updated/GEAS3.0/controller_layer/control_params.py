
from __future__ import annotations

import datetime
from typing import Dict, Any
from pathlib import Path
import sys

_UPDATED_ROOT = Path(__file__).resolve().parents[1]
_INNER_ROOT = _UPDATED_ROOT / "inner_layer"
for _path in (_UPDATED_ROOT, _INNER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pymysql

from core.config import DbConfig
from utilities.db_utils import ensure_db_and_table, get_db_connection

LOG_TABLE_NAME = "control_params"


def _result_sections(result: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    outer = result.get("outer", {})
    if not isinstance(outer, dict):
        outer = {}

    control_section = result.get("control", {})
    if isinstance(control_section, dict):
        effective = control_section.get("effective_params", {})
        physics = control_section.get("physics", {})
    else:
        effective = result.get("effective_params", {})
        physics = result.get("physics", {})

    if not isinstance(effective, dict):
        effective = {}
    if not isinstance(physics, dict):
        physics = {}

    return effective, physics, outer

def build_params_row_from_result(
    result: Dict[str, Any],
    cfg: DbConfig,
    ) -> Dict[str, Any]:

    effective, physics, outer = _result_sections(result)
    agro_kpis = result.get("agro_kpis", {}) or {}
    kpi = outer.get("kpi", {}) or {}

    raw_ts = effective.get("timestamp")
    if hasattr(raw_ts, "to_pydatetime"):
        raw_ts = raw_ts.to_pydatetime()
    if isinstance(raw_ts, datetime.datetime):
        reg_date = raw_ts.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(raw_ts, str) and raw_ts.strip():
        reg_date = raw_ts.strip()
    else:
        reg_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row: Dict[str, Any] = {
        "reg_date": reg_date,
        "farm_sn": cfg.farm_sn,
        "is_daytime": 1 if effective.get("is_daytime") else 0,
        "fcu_mode": effective.get("fcu_mode"),
    }

    targets = effective.get("targets", {}) or {}
    bands = effective.get("bands", {}) or {}
    window_gain = effective.get("window_gain", {}) or {}
    curtain = effective.get("curtain", {}) or {}
    phys = effective.get("physics", {}) or {}
    if not isinstance(phys, dict) or not phys:
        phys = physics

    row.update(
        {
            "T_day_eff": targets.get("T_day_eff"),
            "T_night_eff": targets.get("T_night_eff"),
            "T_current_eff": targets.get("T_current_eff"),
            "DB_day": bands.get("DB_day"),
            "PB_day": bands.get("PB_day"),
            "DB_night": bands.get("DB_night"),
            "PB_night": bands.get("PB_night"),
            "DB_used_now": bands.get("DB_used_now"),
            "PB_used_now": bands.get("PB_used_now"),
            "K_window_day": window_gain.get("K_window_day"),
            "K_window_night": window_gain.get("K_window_night"),
            "K_used_now": window_gain.get("K_used_now"),
            "alpha_curtain": curtain.get("alpha_curtain"),
            "target_jcm2": curtain.get("target_jcm2"),
            "measured_sum_jcm2": curtain.get("measured_sum_jcm2"),
            "sunlight_ratio": curtain.get("sunlight_ratio"),
            "UA": phys.get("UA"),
            "C": phys.get("C"),
            "g_solar": phys.get("g_solar"),
            "physics_mode": phys.get("mode"),
        }
    )

    lam = phys.get("lambda", {}) or {}
    row.update(
        {
            "lambda_UA": lam.get("UA"),
            "lambda_C": lam.get("C"),
            "lambda_ACH": lam.get("ACH"),
            "lambda_g_solar": lam.get("g_solar"),
        }
    )

    safety = effective.get("safety", {}) or {}
    row.update(
        {
            "delta_cond": safety.get("delta_cond"),
            "vpd": safety.get("vpd"),
            "tdew": agro_kpis.get("Tdew"),
            "cond_risk_10m": agro_kpis.get("cond_risk_10m"),
            "cond_dTcond_10m": agro_kpis.get("cond_dTcond_10m"),
            "rh90_min_1h": agro_kpis.get("rh90_min_1h"),
            "vpdlo_min_1h": agro_kpis.get("vpdlo_min_1h"),
            "ach_min_day": agro_kpis.get("ACH_min_day"),
            "ach_min_night": agro_kpis.get("ACH_min_night"),
            "ramp_lim": agro_kpis.get("RampLim"),
            "dT": agro_kpis.get("dT"),
            "cs_sum_jcm2": agro_kpis.get("cs_sum_jcm2"),
            "light_eta": agro_kpis.get("LightETA"),
            "vent_loss_proxy": agro_kpis.get("vent_loss_proxy"),
            "candidate_generated": (effective.get("candidate_search", {}) or {}).get("generated"),
            "candidate_feasible": (effective.get("candidate_search", {}) or {}).get("feasible"),
            "selected_cost": (effective.get("candidate_search", {}) or {}).get("selected_cost"),
        }
    )

    # KPI + policy score
    row.update(
        {
            "cov_day": float(kpi.get("cov_day", 0.0)),
            "cov_night": float(kpi.get("cov_night", 0.0)),
            "mdev_day": float(kpi.get("mdev_day", 0.0)),
            "mdev_night": float(kpi.get("mdev_night", 0.0)),
            "policy_score": outer.get("score"),
            "policy_updated": 1 if outer.get("updated") else 0,
            "policy_sunlight_ratio": outer.get("sunlight_ratio"),
        }
    )

    return row


def insert_params_to_log(result: Dict[str, Any], cfg: DbConfig) -> None:

    row = build_params_row_from_result(result, cfg)

    db_name = cfg.name

    print(f"[DEBUG LOG] params row → {db_name}.{LOG_TABLE_NAME}:")
    print("            ", row)

    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `{LOG_TABLE_NAME}` (
            id           BIGINT AUTO_INCREMENT PRIMARY KEY,
            reg_date     DATETIME NOT NULL,
            farm_sn      INT      NOT NULL,

            is_daytime   TINYINT  NULL,
            fcu_mode     VARCHAR(16) NULL,

            T_day_eff    DOUBLE   NULL,
            T_night_eff  DOUBLE   NULL,
            T_current_eff DOUBLE  NULL,

            DB_day       DOUBLE   NULL,
            PB_day       DOUBLE   NULL,
            DB_night     DOUBLE   NULL,
            PB_night     DOUBLE   NULL,
            DB_used_now  DOUBLE   NULL,
            PB_used_now  DOUBLE   NULL,

            K_window_day   DOUBLE NULL,
            K_window_night DOUBLE NULL,
            K_used_now     DOUBLE NULL,

            alpha_curtain DOUBLE NULL,

            target_jcm2       DOUBLE NULL,
            measured_sum_jcm2 DOUBLE NULL,
            sunlight_ratio    DOUBLE NULL,

            UA           DOUBLE NULL,
            C            DOUBLE NULL,
            g_solar      DOUBLE NULL,
            physics_mode VARCHAR(32) NULL,

            lambda_UA    DOUBLE NULL,
            lambda_C     DOUBLE NULL,
            lambda_ACH   DOUBLE NULL,
            lambda_g_solar DOUBLE NULL,
            delta_cond   DOUBLE NULL,
            vpd          DOUBLE NULL,
            tdew         DOUBLE NULL,
            cond_risk_10m TINYINT NULL,
            cond_dTcond_10m DOUBLE NULL,
            rh90_min_1h  DOUBLE NULL,
            vpdlo_min_1h DOUBLE NULL,
            ach_min_day  DOUBLE NULL,
            ach_min_night DOUBLE NULL,
            ramp_lim     DOUBLE NULL,
            dT           DOUBLE NULL,
            cs_sum_jcm2  DOUBLE NULL,
            light_eta    DATETIME NULL,
            vent_loss_proxy DOUBLE NULL,
            candidate_generated INT NULL,
            candidate_feasible INT NULL,
            selected_cost DOUBLE NULL,

            cov_day      DOUBLE NULL,
            cov_night    DOUBLE NULL,
            mdev_day     DOUBLE NULL,
            mdev_night   DOUBLE NULL,

            policy_score          DOUBLE NULL,
            policy_updated        TINYINT NULL,
            policy_sunlight_ratio DOUBLE NULL,

            KEY idx_reg_date (reg_date),
            KEY idx_farm_sn_reg_date (farm_sn, reg_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """

    try:
        ensure_db_and_table(cfg, db_name, create_table_sql)

        conn = get_db_connection(cfg, db_name=db_name, autocommit=True)
        try:
            alter_stmts = [
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS tdew DOUBLE NULL AFTER vpd",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS cond_risk_10m TINYINT NULL AFTER tdew",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS cond_dTcond_10m DOUBLE NULL AFTER cond_risk_10m",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS rh90_min_1h DOUBLE NULL AFTER cond_dTcond_10m",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS vpdlo_min_1h DOUBLE NULL AFTER rh90_min_1h",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS ach_min_day DOUBLE NULL AFTER vpdlo_min_1h",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS ach_min_night DOUBLE NULL AFTER ach_min_day",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS ramp_lim DOUBLE NULL AFTER ach_min_night",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS dT DOUBLE NULL AFTER ramp_lim",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS cs_sum_jcm2 DOUBLE NULL AFTER dT",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS light_eta DATETIME NULL AFTER cs_sum_jcm2",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS vent_loss_proxy DOUBLE NULL AFTER light_eta",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS candidate_generated INT NULL AFTER vent_loss_proxy",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS candidate_feasible INT NULL AFTER candidate_generated",
                "ALTER TABLE `control_params` ADD COLUMN IF NOT EXISTS selected_cost DOUBLE NULL AFTER candidate_feasible",
            ]
            with conn.cursor() as cur:
                for stmt in alter_stmts:
                    cur.execute(stmt)
        finally:
            conn.close()

        conn = get_db_connection(cfg, db_name=db_name, autocommit=True)
        try:
            cols = [
                "reg_date",
                "farm_sn",
                "is_daytime",
                "fcu_mode",
                "T_day_eff",
                "T_night_eff",
                "T_current_eff",
                "DB_day",
                "PB_day",
                "DB_night",
                "PB_night",
                "DB_used_now",
                "PB_used_now",
                "K_window_day",
                "K_window_night",
                "K_used_now",
                "alpha_curtain",
                "target_jcm2",
                "measured_sum_jcm2",
                "sunlight_ratio",
                "UA",
                "C",
                "g_solar",
                "physics_mode",
                "lambda_UA",
                "lambda_C",
                "lambda_ACH",
                "lambda_g_solar",
                "delta_cond",
                "vpd",
                "tdew",
                "cond_risk_10m",
                "cond_dTcond_10m",
                "rh90_min_1h",
                "vpdlo_min_1h",
                "ach_min_day",
                "ach_min_night",
                "ramp_lim",
                "dT",
                "cs_sum_jcm2",
                "light_eta",
                "vent_loss_proxy",
                "candidate_generated",
                "candidate_feasible",
                "selected_cost",
                "cov_day",
                "cov_night",
                "mdev_day",
                "mdev_night",
                "policy_score",
                "policy_updated",
                "policy_sunlight_ratio",
            ]

            values = [row.get(c) for c in cols]
            placeholders = ", ".join(["%s"] * len(cols))

            sql = f"""
                INSERT INTO `{LOG_TABLE_NAME}` ({", ".join(cols)})
                VALUES ({placeholders})
            """

            with conn.cursor() as cur:
                cur.execute(sql, values)

            print(f"[PARAM LOG] inserted into {db_name}.{LOG_TABLE_NAME}")
        finally:
            conn.close()

    except pymysql.err.OperationalError as e:
        print("[ERROR] insert_params_to_log:", e)
    except Exception as e:
        print("[ERROR] insert_params_to_log:", e)
