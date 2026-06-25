# core/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Tuple, Optional
import os

@dataclass
class DbConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = ""
    name: str = "farmstom"

    main_table: str = "data_silla_enc"
    id_idx: str = "farm_sn"
    farm_sn: int = 97
    main_cols: str = (
        "reg_date, in_temp, in_hum, out_temp, out_hum, "
        "out_winddirec, out_windsp, out_light"
    )

    time_index: str = "reg_date"
    
    sub_table: str = ""
    sub_cols: str = ""

    out_env_cols: str = ""
    in_env_cols: str = ""

    insert_table: str = ""
    insert_cols: str = ""

    pk_col: str = ""
    poll_interval_sec: float = 2.0

    @classmethod
    def from_env(cls) -> "DbConfig":
        return cls(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            name=os.getenv("DB_NAME", "farmstom"),

            time_index=os.getenv("DB_TIME_INDEX", "reg_date"),

            main_table=os.getenv("MAIN_TABLE_NAME", "data_silla_enc"),
            sub_table=os.getenv("SUB_TABLE_NAME", ""),

            id_idx=os.getenv("ID_IDX", "farm_sn"),
            farm_sn=int(os.getenv("FARM_SN", "97")),

            main_cols=os.getenv("MAIN_COLS", "reg_date, out_light"),
            sub_cols=os.getenv("SUB_COLS", ""),

            out_env_cols=os.getenv("OUT_ENV_COL", ""),
            in_env_cols=os.getenv("IN_ENV_COL", ""),

            insert_table=os.getenv("INSERT_TABLE", ""),
            insert_cols=os.getenv("INSERT_COLS", ""),

            pk_col=os.getenv("PK_COL", "").strip(),
            poll_interval_sec=float(os.getenv("POLL_INTERVAL_SEC", "2.0")),
        )

    def to_config_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {
            "DB_HOST": d["host"],
            "DB_PORT": d["port"],
            "DB_USER": d["user"],
            "DB_PASSWORD": d["password"],
            "DB_NAME": d["name"],

            "DB_TIME_INDEX": d["time_index"],

            "MAIN_TABLE_NAME": d["main_table"],
            "SUB_TABLE_NAME": d["sub_table"],

            "ID_IDX": d["id_idx"],
            "FARM_SN": d["farm_sn"],

            "MAIN_COLS": d["main_cols"],
            "SUB_COLS": d["sub_cols"],

            "OUT_ENV_COL": d["out_env_cols"],
            "IN_ENV_COL": d["in_env_cols"],

            "INSERT_TABLE": d["insert_table"],
            "INSERT_COLS": d["insert_cols"],

            "PK_COL": d["pk_col"],
            "POLL_INTERVAL_SEC": d["poll_interval_sec"],
        }

@dataclass
class ControllerParams:
    USE_WIND_CAP: bool = True
    USE_MIN_ACH: bool = True
    USE_RAMP_LIMIT: bool = True
    USE_VPD_FOR_MIN_ACH: bool = False
    USE_ETA_FOR_CURTAIN: bool = False

    WIND_CAP_TH: float = 5.0
    WIND_CAP_OPEN: int = 20
    ACH_MIN_DAY: float = 0.10
    ACH_MIN_NIGHT: float = 0.05
    RAMP_LIMIT: int = 15

    fcu_mode: str = "heat"

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_layers(cls, *layers: Dict[str, Any]) -> "ControllerParams":
        base = asdict(cls())
        base.pop("extra", None)

        merged_extra: Dict[str, Any] = {}

        for layer in layers:
            if not layer:
                continue
            for k, v in layer.items():
                if k in base:
                    base[k] = v
                else:
                    merged_extra[k] = v

        cp = cls(**base)
        cp.extra.update(merged_extra)
        return cp

@dataclass
class PhysicsDefaults:
    UA_default: float = 1.0
    C_default: float = 1.0
    g_solar_default: float = 0.0
    ACH_table: Tuple[Tuple[float, float], ...] = (
        (0, 0.1), (20, 0.3), (50, 1.0), (100, 3.0)
    )

@dataclass
class EstimatorConfig:
    light_threshold: float = 10.0
    min_points_ua_c: int = 200
    min_points_ach: int = 300
    n_ref: int = 1000
    r2_min: float = 0.3
    pi_ref: float = 0.3
    sig_ref: float = 0.15
    rho_air: float = 1.2
    cp_air: float = 1005.0
    volume_m3: Optional[float] = None
