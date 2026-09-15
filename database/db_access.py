"""Reusable SQLite access functions for driver enrollment and baselines."""

import io
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from database.schema import DATABASE_PATH, initialize_database as create_schema


@dataclass
class DriverRecord:
    """One enrolled driver, including their SFace embedding and baselines."""

    driver_id: str
    face_embedding: np.ndarray
    baseline_ear: float
    baseline_mar: float
    baseline_perclos: float
    baseline_blink_rate: float
    baseline_blink_duration: float
    baseline_yawn_frequency: float
    baseline_head_pose_deviation: float
    enrolled_at: str


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize an embedding as a non-pickle NumPy .npy byte stream for SQLite."""

    if not isinstance(embedding, np.ndarray):
        raise TypeError("face_embedding must be a NumPy array.")

    buffer = io.BytesIO()
    np.save(buffer, embedding, allow_pickle=False)
    return buffer.getvalue()


def deserialize_embedding(serialized_embedding: bytes) -> np.ndarray:
    """Restore an embedding saved by serialize_embedding, preserving dtype and shape."""

    with io.BytesIO(serialized_embedding) as buffer:
        return np.load(buffer, allow_pickle=False)


def initialize_database(db_path: Path | str = DATABASE_PATH) -> None:
    """Create the database/table when absent; retain all existing driver records."""

    create_schema(db_path)


@contextmanager
def _connect(db_path: Path | str):
    """Yield a transaction connection and always release its file handle."""

    connection = sqlite3.connect(Path(db_path))
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def insert_driver(
    driver_id: str,
    face_embedding: np.ndarray,
    baseline_ear: float,
    baseline_mar: float,
    baseline_perclos: float,
    baseline_blink_rate: float,
    baseline_blink_duration: float,
    baseline_yawn_frequency: float,
    baseline_head_pose_deviation: float,
    enrolled_at: Optional[str] = None,
    db_path: Path | str = DATABASE_PATH,
) -> None:
    """Insert a newly enrolled driver and their initial calibration baselines."""

    timestamp = enrolled_at or datetime.now(timezone.utc).isoformat()
    embedding_blob = serialize_embedding(face_embedding)

    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO drivers (
                driver_id, face_embedding, baseline_ear, baseline_mar,
                baseline_perclos, baseline_blink_rate, baseline_blink_duration,
                baseline_yawn_frequency, baseline_head_pose_deviation, enrolled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                driver_id,
                embedding_blob,
                baseline_ear,
                baseline_mar,
                baseline_perclos,
                baseline_blink_rate,
                baseline_blink_duration,
                baseline_yawn_frequency,
                baseline_head_pose_deviation,
                timestamp,
            ),
        )


def _row_to_driver(row: sqlite3.Row) -> DriverRecord:
    return DriverRecord(
        driver_id=row["driver_id"],
        face_embedding=deserialize_embedding(row["face_embedding"]),
        baseline_ear=row["baseline_ear"],
        baseline_mar=row["baseline_mar"],
        baseline_perclos=row["baseline_perclos"],
        baseline_blink_rate=row["baseline_blink_rate"],
        baseline_blink_duration=row["baseline_blink_duration"],
        baseline_yawn_frequency=row["baseline_yawn_frequency"],
        baseline_head_pose_deviation=row["baseline_head_pose_deviation"],
        enrolled_at=row["enrolled_at"],
    )


def get_driver(
    driver_id: str,
    db_path: Path | str = DATABASE_PATH,
) -> Optional[DriverRecord]:
    """Return one driver record, or None when the ID is not enrolled."""

    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM drivers WHERE driver_id = ?",
            (driver_id,),
        ).fetchone()

    return _row_to_driver(row) if row is not None else None


def get_all_drivers(db_path: Path | str = DATABASE_PATH) -> list[DriverRecord]:
    """Return all enrolled drivers without applying recognition logic."""

    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM drivers ORDER BY driver_id"
        ).fetchall()

    return [_row_to_driver(row) for row in rows]


def driver_exists(
    driver_id: str,
    db_path: Path | str = DATABASE_PATH,
) -> bool:
    """Return whether a driver ID is present in the database."""

    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT 1 FROM drivers WHERE driver_id = ?",
            (driver_id,),
        ).fetchone()

    return row is not None


def update_driver_baseline(
    driver_id: str,
    baseline_ear: float,
    baseline_mar: float,
    baseline_perclos: float,
    baseline_blink_rate: float,
    baseline_blink_duration: float,
    baseline_yawn_frequency: float,
    baseline_head_pose_deviation: float,
    db_path: Path | str = DATABASE_PATH,
) -> bool:
    """Update baselines only; the stored face embedding is intentionally untouched."""

    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE drivers
            SET baseline_ear = ?, baseline_mar = ?, baseline_perclos = ?,
                baseline_blink_rate = ?, baseline_blink_duration = ?,
                baseline_yawn_frequency = ?, baseline_head_pose_deviation = ?
            WHERE driver_id = ?
            """,
            (
                baseline_ear,
                baseline_mar,
                baseline_perclos,
                baseline_blink_rate,
                baseline_blink_duration,
                baseline_yawn_frequency,
                baseline_head_pose_deviation,
                driver_id,
            ),
        )

    return cursor.rowcount == 1


def delete_driver(
    driver_id: str,
    db_path: Path | str = DATABASE_PATH,
) -> bool:
    """Delete a driver record and report whether a matching ID existed."""

    with _connect(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM drivers WHERE driver_id = ?",
            (driver_id,),
        )

    return cursor.rowcount == 1
