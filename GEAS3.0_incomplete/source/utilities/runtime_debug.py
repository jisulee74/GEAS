# utilities/runtime_debug.py
from __future__ import annotations

from typing import Any

import pymysql
from pymysql.err import ProgrammingError

from utilities.runtime import run_once
from core.config import DbConfig


def _print_last_row(conn, table: str, label: str) -> None:
    with conn.cursor() as cur:
        try:
            cur.execute(f"SELECT * FROM {table} ORDER BY reg_date DESC LIMIT 1")
        except ProgrammingError as e:
            if e.args and e.args[0] == 1146:
                print(f"[{label}] table '{table}' does not exist; skip debug print.")
                return
            raise

        row = cur.fetchone()
        if row is None:
            print(f"[{label}] no rows in {table}")
        else:
            cols = [d[0] for d in cur.description]
            data = dict(zip(cols, row))
            print(f"[{label}] latest from {table}:", data)


def run_once_and_print(
    *,
    lat: float,
    lon: float,
    fcu_mode: str,
    stage_name: str,
    profile_name: str,
    policy_state_path: str,
    control_table: str,
    policy_table: str,
) -> None:

    run_once(
        lat=lat,
        lon=lon,
        fcu_mode=fcu_mode,
        stage_name=stage_name,
        profile_name=profile_name,
        policy_state_path=policy_state_path,
        control_table=control_table,
        policy_table=policy_table,
    )

    # 2) DB 접속해서 최신 제어/정책 레코드 출력
    cfg = DbConfig.from_env()

    conn = pymysql.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        db=cfg.name,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
    )

    try:
        _print_last_row(conn, control_table, "CTRL")
        _print_last_row(conn, policy_table, "POLICY")
    finally:
        conn.close()
