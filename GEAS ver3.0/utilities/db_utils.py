from __future__ import annotations

from typing import Optional
import pymysql

from core.config import DbConfig


def get_server_connection(cfg: DbConfig) -> pymysql.connections.Connection:

    conn = pymysql.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.Cursor,
    )
    with conn.cursor() as cur:
        cur.execute("SET time_zone = %s", ("+09:00",))
        cur.execute("SET NAMES utf8mb4")
    return conn


def get_db_connection(
    cfg: DbConfig,
    db_name: Optional[str] = None,
    autocommit: bool = True,
    ) -> pymysql.connections.Connection:

    target_db = db_name or cfg.name
    conn = pymysql.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        database=target_db,
        charset="utf8mb4",
        autocommit=autocommit,
        cursorclass=pymysql.cursors.Cursor,
    )
    with conn.cursor() as cur:
        cur.execute("SET time_zone = %s", ("+09:00",))
        cur.execute("SET NAMES utf8mb4")
    return conn


def ensure_database(cfg: DbConfig, db_name: str) -> None:

    conn = get_server_connection(cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE DATABASE IF NOT EXISTS `{db_name}`
                DEFAULT CHARACTER SET utf8mb4
                COLLATE utf8mb4_unicode_ci
                """
            )
    finally:
        conn.close()


def ensure_table(
    cfg: DbConfig,
    db_name: str,
    create_table_sql: str,
    ) -> None:

    conn = get_db_connection(cfg, db_name=db_name, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
    finally:
        conn.close()


def ensure_db_and_table(
    cfg: DbConfig,
    db_name: str,
    create_table_sql: str,
    ) -> None:

    ensure_database(cfg, db_name)
    ensure_table(cfg, db_name, create_table_sql)
