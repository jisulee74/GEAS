"""Database adapter for crop cultivation metadata.

``crop_info`` is the source of truth for greenhouse crop, transplant date, and
crop-end date. The query functions here are read-only and accept either an
existing PyMySQL connection or a config created from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from geas35.core import GrowthStage, find_growth_stage, normalize_crop


@dataclass(frozen=True)
class DbConfig:
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "farmstom"
    charset: str = "utf8mb4"

    @classmethod
    def from_env(cls) -> "DbConfig":
        return cls(
            host=os.getenv("GEAS_DB_HOST", os.getenv("DB_HOST", cls.host)),
            port=int(os.getenv("GEAS_DB_PORT", os.getenv("DB_PORT", str(cls.port)))),
            user=os.getenv("GEAS_DB_USER", os.getenv("DB_USER", cls.user)),
            password=os.getenv(
                "GEAS_DB_PASSWORD",
                os.getenv("DB_PASSWORD", cls.password),
            ),
            database=os.getenv("GEAS_DB_NAME", os.getenv("DB_NAME", cls.database)),
            charset=os.getenv("GEAS_DB_CHARSET", cls.charset),
        )


@dataclass(frozen=True)
class CropInfoRecord:
    idx: int
    iot_data_idx: int
    crop_nm: str | None
    subj_cd: str | None
    kind_cd: str | None
    trans_crop_date: datetime
    crop_end_date: datetime | None
    end_yn: str | None
    del_yn: str | None

    @property
    def crop(self) -> str | None:
        return normalize_crop(self.subj_cd)

    def growth_stage(self, target_ts: datetime) -> GrowthStage:
        return find_growth_stage(
            target_ts=target_ts,
            transplant_date=self.trans_crop_date,
            crop=self.crop,
            subj_cd=self.subj_cd,
        )


def connect(config: DbConfig | None = None):
    """Open a PyMySQL connection from config."""

    import pymysql

    cfg = config or DbConfig.from_env()
    return pymysql.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        database=cfg.database,
        charset=cfg.charset,
        cursorclass=pymysql.cursors.DictCursor,
    )


def crop_info_record_from_row(row: dict[str, Any]) -> CropInfoRecord:
    trans_crop_date = row.get("trans_crop_date")
    if trans_crop_date is None:
        raise ValueError(f"crop_info row has no trans_crop_date: {row!r}")
    return CropInfoRecord(
        idx=int(row["idx"]),
        iot_data_idx=int(row["iot_data_idx"]),
        crop_nm=row.get("crop_nm"),
        subj_cd=None if row.get("subj_cd") is None else str(row.get("subj_cd")),
        kind_cd=None if row.get("kind_cd") is None else str(row.get("kind_cd")),
        trans_crop_date=trans_crop_date,
        crop_end_date=row.get("crop_end_date"),
        end_yn=row.get("end_yn"),
        del_yn=row.get("del_yn"),
    )


def fetch_crop_info_at(
    connection,
    *,
    iot_data_idx: int,
    target_ts: datetime,
) -> CropInfoRecord | None:
    """Fetch the active crop-info row for a greenhouse at ``target_ts``."""

    query = """
        SELECT
            idx,
            iot_data_idx,
            crop_nm,
            subj_cd,
            kind_cd,
            trans_crop_date,
            crop_end_date,
            end_yn,
            del_yn
        FROM crop_info
        WHERE iot_data_idx = %s
          AND COALESCE(del_yn, 'N') = 'N'
          AND trans_crop_date IS NOT NULL
          AND trans_crop_date <= %s
          AND (crop_end_date IS NULL OR crop_end_date >= %s)
        ORDER BY trans_crop_date DESC, idx DESC
        LIMIT 1
    """
    with connection.cursor() as cur:
        cur.execute(query, (iot_data_idx, target_ts, target_ts))
        row = cur.fetchone()
    return crop_info_record_from_row(row) if row else None


def fetch_latest_active_crop_info(
    connection,
    *,
    iot_data_idx: int,
) -> CropInfoRecord | None:
    """Fetch the latest non-ended crop row for realtime operation."""

    query = """
        SELECT
            idx,
            iot_data_idx,
            crop_nm,
            subj_cd,
            kind_cd,
            trans_crop_date,
            crop_end_date,
            end_yn,
            del_yn
        FROM crop_info
        WHERE iot_data_idx = %s
          AND COALESCE(del_yn, 'N') = 'N'
          AND COALESCE(end_yn, 'N') = 'N'
          AND trans_crop_date IS NOT NULL
        ORDER BY trans_crop_date DESC, idx DESC
        LIMIT 1
    """
    with connection.cursor() as cur:
        cur.execute(query, (iot_data_idx,))
        row = cur.fetchone()
    return crop_info_record_from_row(row) if row else None


def fetch_growth_stage_at(
    connection,
    *,
    iot_data_idx: int,
    target_ts: datetime,
) -> GrowthStage | None:
    """Fetch crop-info metadata and calculate the current growth stage."""

    record = fetch_crop_info_at(
        connection,
        iot_data_idx=iot_data_idx,
        target_ts=target_ts,
    )
    return None if record is None else record.growth_stage(target_ts)


def fetch_growth_stage_at_from_db(
    *,
    iot_data_idx: int,
    target_ts: datetime,
    config: DbConfig | None = None,
) -> GrowthStage | None:
    """Open a DB connection, fetch crop metadata, and calculate growth stage."""

    with connect(config) as connection:
        return fetch_growth_stage_at(
            connection,
            iot_data_idx=iot_data_idx,
            target_ts=target_ts,
        )


__all__ = [
    "CropInfoRecord",
    "DbConfig",
    "connect",
    "crop_info_record_from_row",
    "fetch_crop_info_at",
    "fetch_growth_stage_at",
    "fetch_growth_stage_at_from_db",
    "fetch_latest_active_crop_info",
]
