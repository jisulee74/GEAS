from __future__ import annotations
import pymysql
import time
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Iterator
from datetime import datetime
from core.config import DbConfig
from core.timeutils import normalize_timestamp_kst

class Repository:
    def __init__(self, cfg: DbConfig):
        self.cfg = cfg

        self.host = cfg.host
        self.port = cfg.port
        self.user = cfg.user
        self.password = cfg.password
        self.dbname = cfg.name

        self.table = cfg.main_table
        self.cols = cfg.main_cols
        self.id_idx = cfg.id_idx
        self.farm_sn = cfg.farm_sn

        self.pk_col = cfg.pk_col
        self.poll_sec = cfg.poll_interval_sec

    def connect(self):
        conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.dbname,
            charset="utf8mb4",
            autocommit=False,
            cursorclass=pymysql.cursors.Cursor,
        )
        with conn.cursor() as cur:
            cur.execute("SET time_zone = %s", ("+09:00",))
            cur.execute("SET NAMES utf8mb4")
        return conn

    def _run_read(self, sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
        conn = self.connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                cols = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(cols, row)) for row in rows]
        finally:
            conn.close()

    def fetch(self, limit: int = 288, order: str = "asc"):
        if order not in ("asc", "desc"):
            raise ValueError("order must be 'asc' or 'desc'")

        if order == "desc":
            sql = f"""
                SELECT {self.cols}
                FROM {self.table}
                WHERE {self.id_idx} = %s
                ORDER BY reg_date DESC
                LIMIT %s
            """
            params = (self.farm_sn, limit)
        else:
            sql = f"""
                SELECT * FROM (
                    SELECT {self.cols}
                    FROM {self.table}
                    WHERE {self.id_idx} = %s
                    ORDER BY reg_date DESC
                    LIMIT %s
                ) AS t
                ORDER BY reg_date ASC
            """
            params = (self.farm_sn, limit)

        return self._run_read(sql, params)

    def fetch_since(
        self,
        since_ts,
        limit: int = 1000,
        inclusive: bool = False
    ) -> List[Dict[str, Any]]:

        since_dt = normalize_timestamp_kst(since_ts)
        op = ">=" if inclusive else ">"

        if self.pk_col:
            sql = f"""
                SELECT {self.cols}, {self.pk_col}
                FROM {self.table}
                WHERE {self.id_idx} = %s
                  AND (reg_date {op} %s)
                ORDER BY reg_date ASC, {self.pk_col} ASC
                LIMIT %s
            """
            params = (self.farm_sn, since_dt, limit)
        else:
            sql = f"""
                SELECT {self.cols}
                FROM {self.table}
                WHERE {self.id_idx} = %s
                  AND reg_date {op} %s
                ORDER BY reg_date ASC
                LIMIT %s
            """
            params = (self.farm_sn, since_dt, limit)

        return self._run_read(sql, params)

    def stream(
        self,
        start: Optional[datetime] = None,
        max_rows: Optional[int] = None,
        on_error_sleep: float = 5.0
    ) -> Iterator[Dict[str, Any]]:

        if start is None:
            last_ts, last_pk = self._discover_tail()
        else:
            last_ts = normalize_timestamp_kst(start)
            last_pk = -1 if self.pk_col else None

        yielded = 0

        while True:
            try:
                rows = self._poll(last_ts, last_pk)
                if rows:
                    for r in rows:
                        yield r
                        yielded += 1

                        last_ts = r["reg_date"]
                        if self.pk_col in r:
                            last_pk = int(r[self.pk_col])

                        if max_rows and yielded >= max_rows:
                            return

                else:
                    time.sleep(self.poll_sec)

            except pymysql.err.OperationalError:
                time.sleep(on_error_sleep)
            except Exception:
                time.sleep(on_error_sleep)

    def _discover_tail(self) -> Tuple[datetime, Optional[int]]:
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                if self.pk_col:
                    cur.execute(
                        f"""
                        SELECT reg_date, {self.pk_col}
                        FROM {self.table}
                        WHERE {self.id_idx} = %s
                        ORDER BY reg_date DESC, {self.pk_col} DESC
                        LIMIT 1
                        """,
                        (self.farm_sn,)
                    )
                    row = cur.fetchone()
                    if row:
                        return row[0], int(row[1])
                else:
                    cur.execute(
                        f"""
                        SELECT MAX(reg_date)
                        FROM {self.table}
                        WHERE {self.id_idx} = %s
                        """,
                        (self.farm_sn,)
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        return row[0], None

                # empty DB fallback
                cur.execute("SELECT NOW()")
                return cur.fetchone()[0], None

        finally:
            conn.close()

    def _poll(self, last_ts: datetime, last_pk: Optional[int]):
        if self.pk_col:
            sql = f"""
                SELECT {self.cols}, {self.pk_col}
                FROM {self.table}
                WHERE {self.id_idx} = %s
                  AND (
                        reg_date > %s
                        OR (reg_date = %s AND {self.pk_col} > %s)
                  )
                ORDER BY reg_date ASC, {self.pk_col} ASC
            """
            params = (self.farm_sn, last_ts, last_ts, last_pk or -1)

        else:
            sql = f"""
                SELECT {self.cols}
                FROM {self.table}
                WHERE {self.id_idx} = %s
                  AND reg_date > %s
                ORDER BY reg_date ASC
            """
            params = (self.farm_sn, last_ts)

        return self._run_read(sql, params)

    # yjchoi: API 호출 로그
    def logging(self, log_param):
        try:
            conn = self.connect()
            cursor = conn.cursor()

            sql = f"""
                INSERT INTO
                    send_control_api_history(
                        iot_data_idx,
                        device_id,
                        target,
                        reg_date
                    ) VALUES (
                        {self.farm_sn},
                        {log_param['device_id']},
                        '{log_param['target']}',
                        now()
                    )
            """
            
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            print("[ERROR] logging failed:", e)
        finally:
            conn.close()
