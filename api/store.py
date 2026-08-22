"""SQLite store for prediction logs + user feedback.

Prediction logs  -> fuel for Evidently drift reports (Phase 6).
Feedback rows    -> fuel for the weekly retrain job (Phase 6).
"""
import os
import sqlite3
import time
from pathlib import Path

_conn: sqlite3.Connection | None = None


def _db_path() -> str:
    return os.getenv("DB_PATH", "data/logs/predictions.db")


def init() -> None:
    global _conn
    Path(_db_path()).parent.mkdir(parents=True, exist_ok=True)
    _conn = sqlite3.connect(_db_path(), check_same_thread=False)
    _conn.execute(
        """CREATE TABLE IF NOT EXISTS predictions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               ts REAL, text_hash TEXT, text TEXT,
               label TEXT, confidence REAL,
               model_version TEXT, latency_ms REAL)"""
    )
    _conn.execute(
        """CREATE TABLE IF NOT EXISTS feedback (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               ts REAL, comment_id TEXT, text TEXT,
               predicted TEXT, actual TEXT, model_version TEXT)"""
    )
    _conn.commit()


def log_prediction(text_hash: str, text: str, label: str, confidence: float,
                   model_version: str, latency_ms: float) -> None:
    _conn.execute(
        "INSERT INTO predictions (ts, text_hash, text, label, confidence, model_version, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (time.time(), text_hash, text[:500], label, confidence, model_version, latency_ms),
    )
    _conn.commit()


def log_feedback(comment_id: str, text: str, predicted: str,
                 actual: str, model_version: str) -> None:
    _conn.execute(
        "INSERT INTO feedback (ts, comment_id, text, predicted, actual, model_version) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (time.time(), comment_id, text[:500], predicted, actual, model_version),
    )
    _conn.commit()


def counts() -> dict:
    p = _conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    f = _conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    return {"predictions": p, "feedback": f}
