"""SQLite schema for enrolled drivers and their calibration baselines."""

import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parent / "adapticurve.db"

DRIVERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS drivers (
    driver_id TEXT PRIMARY KEY,
    face_embedding BLOB NOT NULL,
    baseline_ear REAL NOT NULL,
    baseline_mar REAL NOT NULL,
    baseline_perclos REAL NOT NULL,
    baseline_blink_rate REAL NOT NULL,
    baseline_blink_duration REAL NOT NULL,
    baseline_yawn_frequency REAL NOT NULL,
    baseline_head_pose_deviation REAL NOT NULL,
    enrolled_at TEXT NOT NULL
)
"""


def initialize_database(db_path: Path | str = DATABASE_PATH) -> None:
    """Create the database and drivers table without altering existing rows."""

    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    try:
        with connection:
            connection.execute(DRIVERS_TABLE_SQL)
    finally:
        connection.close()
