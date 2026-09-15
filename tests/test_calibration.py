import unittest
import tempfile
from pathlib import Path

import numpy as np

from database import db_access
from src.calibration import (
    CalibrationAccumulator,
    CalibrationFeatureMonitor,
    calibration_window_expired,
    can_collect_sample,
    run_calibration,
)


class CalibrationAccumulatorTest(unittest.TestCase):
    def test_baseline_calculations(self):
        accumulator = CalibrationAccumulator(start_time=100.0)
        accumulator.add_sample(
            ear=0.20,
            mar=0.10,
            perclos=10.0,
            head_pose="CENTER",
        )
        accumulator.add_sample(
            ear=0.30,
            mar=0.20,
            perclos=20.0,
            head_pose="LEFT",
            blink_completed=True,
            blink_duration=0.20,
            yawn_detected=True,
        )

        baselines = accumulator.finalize(end_time=110.0)

        self.assertAlmostEqual(baselines.baseline_ear, 0.25)
        self.assertAlmostEqual(baselines.baseline_mar, 0.15)
        self.assertAlmostEqual(baselines.baseline_perclos, 15.0)
        self.assertAlmostEqual(baselines.baseline_blink_rate, 6.0)
        self.assertAlmostEqual(baselines.baseline_blink_duration, 0.20)
        self.assertAlmostEqual(baselines.baseline_yawn_frequency, 6.0)
        self.assertAlmostEqual(baselines.baseline_head_pose_deviation, 50.0)

    def test_invalid_or_no_face_status_does_not_update_accumulator(self):
        accumulator = CalibrationAccumulator(start_time=10.0)

        for status in ("invalid_frame", "no_face", "multiple_faces"):
            if can_collect_sample(status, has_landmarks=False):
                accumulator.add_sample(0.25, 0.10, 5.0, "CENTER")

        self.assertFalse(accumulator.has_samples())
        self.assertEqual(accumulator.face_frames, 0)
        self.assertEqual(accumulator.finalize(20.0).baseline_ear, 0.0)

    def test_post_window_sample_and_event_are_excluded(self):
        accumulator = CalibrationAccumulator(start_time=100.0)
        accumulator.add_sample(0.25, 0.10, 5.0, "CENTER")

        post_window_time = 110.01
        if not calibration_window_expired(accumulator.start_time, post_window_time):
            accumulator.add_sample(
                0.10,
                0.30,
                80.0,
                "LEFT",
                blink_completed=True,
                blink_duration=0.25,
                yawn_detected=True,
            )

        baselines = accumulator.finalize(110.0)
        self.assertEqual(accumulator.face_frames, 1)
        self.assertEqual(baselines.baseline_blink_rate, 0.0)
        self.assertEqual(baselines.baseline_yawn_frequency, 0.0)
        self.assertEqual(baselines.baseline_head_pose_deviation, 0.0)

    def test_multiple_face_status_is_skipped(self):
        self.assertFalse(can_collect_sample("multiple_faces", has_landmarks=True))
        self.assertTrue(can_collect_sample("ok", has_landmarks=True))

    def test_face_loss_resets_partial_detector_state(self):
        monitor = CalibrationFeatureMonitor()
        monitor.blink_detector.update(0.10, 1.0)
        monitor.yawn_detector.update(0.60, 1.0)
        monitor.perclos_tracker.update(True, 1.0)
        monitor.smoother.update(0.20, 0.10)

        monitor.reset_after_face_loss()

        self.assertFalse(monitor.blink_detector.eye_closed)
        self.assertEqual(monitor.yawn_detector.yawn_count, 0)
        self.assertEqual(len(monitor.perclos_tracker.history), 0)
        self.assertEqual(len(monitor.smoother.ear_values), 0)

    def test_zero_blinks_and_yawns_are_safe(self):
        accumulator = CalibrationAccumulator(start_time=0.0)
        accumulator.add_sample(0.25, 0.10, 0.0, "CENTER")

        baselines = accumulator.finalize(end_time=10.0)

        self.assertEqual(baselines.baseline_blink_rate, 0.0)
        self.assertEqual(baselines.baseline_blink_duration, 0.0)
        self.assertEqual(baselines.baseline_yawn_frequency, 0.0)

    def test_duplicate_driver_id_refuses_enrollment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            db_path = Path(temporary_directory) / "test_adapticurve.db"
            db_access.initialize_database(db_path)
            db_access.insert_driver(
                driver_id="existing_driver",
                face_embedding=np.ones((1, 128), dtype=np.float32),
                baseline_ear=0.25,
                baseline_mar=0.10,
                baseline_perclos=0.0,
                baseline_blink_rate=0.0,
                baseline_blink_duration=0.0,
                baseline_yawn_frequency=0.0,
                baseline_head_pose_deviation=0.0,
                db_path=db_path,
            )

            self.assertFalse(run_calibration("existing_driver", db_path=db_path))


if __name__ == "__main__":
    unittest.main()
