from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path

from core.schemas import SafetyEvent


class EventStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                reason TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )""")

    def add(self, event: SafetyEvent) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO events (event_type,status,confidence,source,timestamp,image_path,reason,payload_json) VALUES (?,?,?,?,?,?,?,?)",
                (event.event_type, event.status, event.confidence, event.source, event.timestamp, event.image_path, event.reason, json.dumps(event.payload, ensure_ascii=False)),
            )
            return int(cursor.lastrowid)

    def list(self, limit: int = 200) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def count_by_type(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT event_type,status,COUNT(*) AS count FROM events GROUP BY event_type,status ORDER BY count DESC").fetchall()
        return [dict(row) for row in rows]

    def export_csv(self, limit: int = 10000) -> str:
        output = io.StringIO()
        fields = ["id", "event_type", "status", "confidence", "source", "timestamp", "image_path", "reason", "payload"]
        writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader()
        for row in self.list(limit): writer.writerow(row)
        return output.getvalue()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        data = dict(row); data["payload"] = json.loads(data.pop("payload_json")); return data
