"""SQLite-backed persistence for review records."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone


class ReviewStore:
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS reviews ("
            " id TEXT PRIMARY KEY,"
            " created_at TEXT NOT NULL,"
            " payload TEXT NOT NULL)"
        )
        self._conn.commit()

    def save(self, record: dict) -> str:
        review_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO reviews (id, created_at, payload) VALUES (?, ?, ?)",
                (review_id, created_at, json.dumps(record, ensure_ascii=False)),
            )
            self._conn.commit()
        return review_id

    def get(self, review_id: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT id, created_at, payload FROM reviews WHERE id = ?",
                (review_id,),
            ).fetchone()
        if row is None:
            return None
        record = json.loads(row[2])
        return {"id": row[0], "created_at": row[1], **record}

    def close(self) -> None:
        self._conn.close()
