"""Crash-safe, dynamically extensible worker leases for Step 19.5."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping


class CalibrationLeaseStore:
    """One independent job per (crop, budget, representative, seed)."""

    def __init__(self, path: str | Path, crop: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.crop = crop
        with self.connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS jobs(
                crop TEXT, budget INTEGER, job_id INTEGER, config_id TEXT,
                seed INTEGER, status TEXT, owner TEXT, lease_until REAL,
                heartbeat REAL, attempt INTEGER DEFAULT 0, payload TEXT,
                result TEXT, error TEXT,
                PRIMARY KEY(crop, budget, job_id))"""
            )

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        db.execute("PRAGMA busy_timeout=60000")
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def initialize(self, budget: int, payloads: Mapping[int, Mapping[str, Any]]) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for job_id, payload in payloads.items():
                db.execute(
                    """INSERT OR IGNORE INTO jobs(
                    crop,budget,job_id,config_id,seed,status,payload)
                    VALUES(?,?,?,?,?,'pending',?)""",
                    (
                        self.crop,
                        int(budget),
                        int(job_id),
                        str(payload["config_id"]),
                        int(payload["seed"]),
                        json.dumps(dict(payload)),
                    ),
                )
            db.commit()

    def retry_failed(self, budget: int) -> int:
        with self.connect() as db:
            return db.execute(
                """UPDATE jobs SET status='pending',owner=NULL,lease_until=NULL,error=NULL
                WHERE crop=? AND budget=? AND status='failed'""",
                (self.crop, int(budget)),
            ).rowcount

    def claim(self, owner: str, budget: int, lease_seconds: float = 300) -> dict[str, Any] | None:
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """SELECT job_id,payload,attempt FROM jobs
                WHERE crop=? AND budget=? AND
                (status='pending' OR (status='running' AND lease_until<?))
                ORDER BY job_id LIMIT 1""",
                (self.crop, int(budget), now),
            ).fetchone()
            if row is None:
                db.commit()
                return None
            db.execute(
                """UPDATE jobs SET status='running',owner=?,lease_until=?,heartbeat=?,attempt=?
                WHERE crop=? AND budget=? AND job_id=?""",
                (owner, now + lease_seconds, now, row[2] + 1, self.crop, int(budget), row[0]),
            )
            db.commit()
            return {"job_id": row[0], "payload": json.loads(row[1]), "attempt": row[2] + 1}

    def heartbeat(self, owner: str, budget: int, job_id: int, lease_seconds: float = 300) -> None:
        now = time.time()
        with self.connect() as db:
            changed = db.execute(
                """UPDATE jobs SET heartbeat=?,lease_until=?
                WHERE crop=? AND budget=? AND job_id=? AND status='running' AND owner=?""",
                (now, now + lease_seconds, self.crop, int(budget), int(job_id), owner),
            ).rowcount
            if changed != 1:
                raise ValueError("Lease ownership lost")

    def complete(self, owner: str, budget: int, job_id: int, result: Mapping[str, Any]) -> None:
        with self.connect() as db:
            changed = db.execute(
                """UPDATE jobs SET status='complete',result=?,lease_until=NULL
                WHERE crop=? AND budget=? AND job_id=? AND owner=? AND status='running'""",
                (json.dumps(dict(result)), self.crop, int(budget), int(job_id), owner),
            ).rowcount
            if changed != 1:
                raise ValueError("Cannot complete an unowned lease")

    def fail(self, owner: str, budget: int, job_id: int, error: str) -> None:
        with self.connect() as db:
            db.execute(
                """UPDATE jobs SET status='failed',error=?,lease_until=NULL
                WHERE crop=? AND budget=? AND job_id=? AND owner=?""",
                (error, self.crop, int(budget), int(job_id), owner),
            )

    def rows(self, budget: int | None = None) -> list[dict[str, Any]]:
        query = """SELECT budget,job_id,config_id,seed,status,owner,lease_until,attempt,payload,result,error
                   FROM jobs WHERE crop=?"""
        values: tuple[Any, ...] = (self.crop,)
        if budget is not None:
            query += " AND budget=?"
            values += (int(budget),)
        query += " ORDER BY budget,job_id"
        with self.connect() as db:
            rows = db.execute(query, values).fetchall()
        return [
            {
                "budget": row[0], "job_id": row[1], "config_id": row[2],
                "seed": row[3], "status": row[4], "owner": row[5],
                "lease_until": row[6], "attempt": row[7], "payload": json.loads(row[8]),
                "result": None if row[9] is None else json.loads(row[9]),
                "error": row[10],
            }
            for row in rows
        ]


__all__ = ["CalibrationLeaseStore"]
