
from __future__ import annotations

import datetime
from typing import Dict, Any

import pymysql

from core.config import DbConfig
from utilities.db_utils import ensure_db_and_table, get_db_connection

LOG_TABLE_NAME = "control_params"

def build_params_row_from_result(
    result: Dict[str, Any],
    cfg: DbConfig,
    ) -> Dict[str, Any]:

    ctrl = result.get("control", {})
    outer = result.get("outer", {})
    policy_state = result.get("policy_state", None)
    agro_kpis = result.get("agro_kpis", {}) or {}

    effective = ctrl.get("effective_params", {}) if isinstance(ctrl, dict) else {}
    physics = ctrl.get("physics", {}) if isinstance(ctrl, dict) else {}

    kpi = outer.get("kpi", {}) or {}

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row: Dict[str, Any] = {
        "reg_date": now_str,
        "farm_sn": cfg.farm_sn,
        "is_daytime": 1 if effective.get("is_daytime") else 0,
        "fcu_mode": effective.get("fcu_mode"),
    }

    targets = effective.get("targets", {}) or {}
    bands = effective.get("bands", {}) or {}
    window_gain = effective.get("window_gain", {}) or {}
    curtain = effective.get("curtain", {}) or {}
    phys = effective.get("physics", {}) or {}

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
