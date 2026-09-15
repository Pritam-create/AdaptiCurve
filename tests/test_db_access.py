import tempfile
import unittest
from pathlib import Path

import numpy as np

from database import db_access


class DatabaseAccessTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "test_adapticurve.db"
        db_access.initialize_database(self.db_path)
        self.embedding = np.arange(128, dtype=np.float32).reshape(1, 128)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _insert_test_driver(self):
        db_access.insert_driver(
            driver_id="test_driver",
            face_embedding=self.embedding,
            baseline_ear=0.25,
            baseline_mar=0.10,
            baseline_perclos=5.0,
            baseline_blink_rate=12.0,
            baseline_blink_duration=0.15,
            baseline_yawn_frequency=1.0,
            baseline_head_pose_deviation=8.0,
            enrolled_at="2026-01-01T00:00:00+00:00",
            db_path=self.db_path,
        )

    def test_initialization_creates_drivers_table(self):
        self.assertTrue(self.db_path.is_file())
        self.assertFalse(db_access.driver_exists("test_driver", self.db_path))

    def test_insert_retrieve_and_deserialize_embedding(self):
        self._insert_test_driver()

        driver = db_access.get_driver("test_driver", self.db_path)

        self.assertIsNotNone(driver)
        self.assertEqual(driver.driver_id, "test_driver")
        self.assertTrue(np.array_equal(driver.face_embedding, self.embedding))
        self.assertEqual(driver.baseline_ear, 0.25)
        self.assertEqual(driver.enrolled_at, "2026-01-01T00:00:00+00:00")
        self.assertTrue(db_access.driver_exists("test_driver", self.db_path))
        self.assertEqual(len(db_access.get_all_drivers(self.db_path)), 1)

    def test_embedding_serialization_round_trip(self):
        serialized = db_access.serialize_embedding(self.embedding)
        deserialized = db_access.deserialize_embedding(serialized)

        self.assertTrue(np.array_equal(deserialized, self.embedding))
        self.assertEqual(deserialized.dtype, self.embedding.dtype)
        self.assertEqual(deserialized.shape, self.embedding.shape)

    def test_update_baseline_keeps_embedding_and_delete_driver(self):
        self._insert_test_driver()

        updated = db_access.update_driver_baseline(
            "test_driver",
            baseline_ear=0.30,
            baseline_mar=0.12,
            baseline_perclos=7.0,
            baseline_blink_rate=10.0,
            baseline_blink_duration=0.20,
            baseline_yawn_frequency=2.0,
            baseline_head_pose_deviation=5.0,
            db_path=self.db_path,
        )
        driver = db_access.get_driver("test_driver", self.db_path)

        self.assertTrue(updated)
        self.assertEqual(driver.baseline_ear, 0.30)
        self.assertEqual(driver.baseline_yawn_frequency, 2.0)
        self.assertTrue(np.array_equal(driver.face_embedding, self.embedding))
        self.assertTrue(db_access.delete_driver("test_driver", self.db_path))
        self.assertFalse(db_access.driver_exists("test_driver", self.db_path))
        self.assertIsNone(db_access.get_driver("test_driver", self.db_path))


if __name__ == "__main__":
    unittest.main()
