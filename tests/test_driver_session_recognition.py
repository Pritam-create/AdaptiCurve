import tempfile
import unittest
from pathlib import Path

import numpy as np

from database import db_access
from src.driver_recognition import EmbeddingResult, RecognitionResult
from src.driver_session_recognition import DriverSessionRecognizer


class FakeDriverRecognizer:
    """Deterministic recognizer used to isolate service/database tests."""

    def __init__(self, frame_result: EmbeddingResult) -> None:
        self.frame_result = frame_result

    def extract_embedding(self, frame: np.ndarray) -> EmbeddingResult:
        return self.frame_result

    def recognize(
        self,
        query_embedding: np.ndarray,
        stored_embeddings: dict[str, np.ndarray],
    ) -> RecognitionResult:
        if not stored_embeddings:
            return RecognitionResult(None, None, False)

        def cosine_similarity(embedding: np.ndarray) -> float:
            numerator = float(np.dot(query_embedding.ravel(), embedding.ravel()))
            denominator = float(
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            return numerator / denominator if denominator else 0.0

        driver_id, similarity = max(
            (
                (driver_id, cosine_similarity(embedding))
                for driver_id, embedding in stored_embeddings.items()
            ),
            key=lambda match: match[1],
        )
        return RecognitionResult(driver_id, similarity, similarity >= 0.363)


class DriverSessionRecognizerTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "test_adapticurve.db"
        db_access.initialize_database(self.db_path)
        self.driver_a_embedding = np.array([[1.0, 0.0]], dtype=np.float32)
        self.driver_b_embedding = np.array([[0.0, 1.0]], dtype=np.float32)
        self._insert_driver("driver_a", self.driver_a_embedding, 0.25)
        self._insert_driver("driver_b", self.driver_b_embedding, 0.30)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _insert_driver(
        self,
        driver_id: str,
        embedding: np.ndarray,
        baseline_ear: float,
    ) -> None:
        db_access.insert_driver(
            driver_id=driver_id,
            face_embedding=embedding,
            baseline_ear=baseline_ear,
            baseline_mar=0.10,
            baseline_perclos=5.0,
            baseline_blink_rate=12.0,
            baseline_blink_duration=0.15,
            baseline_yawn_frequency=1.0,
            baseline_head_pose_deviation=8.0,
            db_path=self.db_path,
        )

    def test_loads_all_enrolled_drivers_and_recognizes_best_match(self):
        recognizer = DriverSessionRecognizer(
            recognizer=FakeDriverRecognizer(EmbeddingResult("ok", self.driver_a_embedding)),
            db_path=self.db_path,
        )

        loaded_drivers = recognizer.load_enrolled_drivers()
        result = recognizer.recognize_frame(np.zeros((1, 1, 3), dtype=np.uint8))

        self.assertEqual([driver.driver_id for driver in loaded_drivers], ["driver_a", "driver_b"])
        self.assertEqual(result.status, "recognized")
        self.assertEqual(result.driver_id, "driver_a")
        self.assertAlmostEqual(result.similarity, 1.0)
        self.assertEqual(result.baselines.baseline_ear, 0.25)
        self.assertEqual(result.baselines.baseline_head_pose_deviation, 8.0)

    def test_below_threshold_returns_unknown_without_baselines(self):
        unknown_embedding = np.array([[-1.0, 0.0]], dtype=np.float32)
        recognizer = DriverSessionRecognizer(
            recognizer=FakeDriverRecognizer(EmbeddingResult("ok", unknown_embedding)),
            db_path=self.db_path,
        )

        result = recognizer.recognize_frame(np.zeros((1, 1, 3), dtype=np.uint8))

        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.driver_id)
        self.assertIsNone(result.baselines)

    def test_no_face_frame_does_not_attempt_recognition(self):
        recognizer = DriverSessionRecognizer(
            recognizer=FakeDriverRecognizer(EmbeddingResult("no_face")),
            db_path=self.db_path,
        )

        result = recognizer.recognize_frame(np.zeros((1, 1, 3), dtype=np.uint8))

        self.assertEqual(result.status, "no_face")
        self.assertIsNone(result.driver_id)
        self.assertIsNone(result.baselines)


if __name__ == "__main__":
    unittest.main()
