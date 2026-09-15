"""Bridge standalone SFace recognition with enrolled-driver baseline profiles."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from database import db_access
from database.schema import DATABASE_PATH
from src.driver_recognition import DriverRecognizer


@dataclass
class DriverBaselines:
    """The seven stored calibration baselines for a recognized driver."""

    baseline_ear: float
    baseline_mar: float
    baseline_perclos: float
    baseline_blink_rate: float
    baseline_blink_duration: float
    baseline_yawn_frequency: float
    baseline_head_pose_deviation: float


@dataclass
class DriverSessionResult:
    """Recognition outcome and, when available, the matched driver baselines."""

    status: str
    driver_id: Optional[str] = None
    similarity: Optional[float] = None
    baselines: Optional[DriverBaselines] = None


class DriverSessionRecognizer:
    """Recognize a frame against enrolled drivers without issuing SQL directly."""

    def __init__(
        self,
        recognizer: Optional[DriverRecognizer] = None,
        db_path: Path | str = DATABASE_PATH,
    ) -> None:
        self.recognizer = recognizer or DriverRecognizer()
        self.db_path = Path(db_path)

    def load_enrolled_drivers(self) -> list[db_access.DriverRecord]:
        """Load all stored embeddings and profiles through the database access layer."""

        return db_access.get_all_drivers(self.db_path)

    @staticmethod
    def _baselines_from_driver(
        driver: db_access.DriverRecord,
    ) -> DriverBaselines:
        return DriverBaselines(
            baseline_ear=driver.baseline_ear,
            baseline_mar=driver.baseline_mar,
            baseline_perclos=driver.baseline_perclos,
            baseline_blink_rate=driver.baseline_blink_rate,
            baseline_blink_duration=driver.baseline_blink_duration,
            baseline_yawn_frequency=driver.baseline_yawn_frequency,
            baseline_head_pose_deviation=driver.baseline_head_pose_deviation,
        )

    def recognize_embedding(self, query_embedding: np.ndarray) -> DriverSessionResult:
        """Recognize an embedding and return the matched driver's stored baselines."""

        drivers = self.load_enrolled_drivers()
        stored_embeddings = {
            driver.driver_id: driver.face_embedding
            for driver in drivers
        }
        recognition = self.recognizer.recognize(
            query_embedding,
            stored_embeddings,
        )

        if not recognition.recognized or recognition.driver_id is None:
            return DriverSessionResult(
                status="unknown",
                similarity=recognition.similarity,
            )

        matched_driver = next(
            driver for driver in drivers
            if driver.driver_id == recognition.driver_id
        )
        return DriverSessionResult(
            status="recognized",
            driver_id=matched_driver.driver_id,
            similarity=recognition.similarity,
            baselines=self._baselines_from_driver(matched_driver),
        )

    def recognize_frame(self, frame: np.ndarray) -> DriverSessionResult:
        """Extract one SFace embedding from a BGR frame and recognize it."""

        embedding_result = self.recognizer.extract_embedding(frame)
        if embedding_result.status != "ok" or embedding_result.embedding is None:
            return DriverSessionResult(status=embedding_result.status)

        return self.recognize_embedding(embedding_result.embedding)
