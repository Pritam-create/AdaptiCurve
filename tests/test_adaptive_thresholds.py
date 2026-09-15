import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from database import db_access
from src.adaptive_thresholds import (
    calculate_adaptive_thresholds,
    decide_drowsiness,
    is_elevated_blink_rate,
    is_elevated_head_pose_deviation,
    is_elevated_yawn_frequency,
    is_unusually_long_blink,
)
from src.driver_recognition import EmbeddingResult, RecognitionResult
from src.driver_session_recognition import (
    DriverBaselines,
    DriverSessionRecognizer,
    DriverSessionResult,
)
from src.adaptive_drowsiness_demo import (
    LiveFeatureMonitor,
    create_session_profile,
    start_session,
)


class SyntheticRecognizer:
    def __init__(self, embedding_result: EmbeddingResult, recognition: RecognitionResult):
        self.embedding_result = embedding_result
        self.recognition = recognition

    def extract_embedding(self, frame):
        return self.embedding_result

    def recognize(self, query_embedding, stored_embeddings):
        return self.recognition


class AdaptiveThresholdsTest(unittest.TestCase):
    def test_personalized_threshold_formulas(self):
        baselines = DriverBaselines(
            baseline_ear=0.40,
            baseline_mar=0.10,
            baseline_perclos=25.0,
            baseline_blink_rate=12.0,
            baseline_blink_duration=0.15,
            baseline_yawn_frequency=1.0,
            baseline_head_pose_deviation=8.0,
        )

        thresholds = calculate_adaptive_thresholds(baselines)

        self.assertTrue(thresholds.personalized)
        self.assertAlmostEqual(thresholds.ear_threshold, 0.30)
        self.assertAlmostEqual(thresholds.mar_threshold, 1.0)
        self.assertAlmostEqual(thresholds.perclos_threshold, 35.0)

    def test_fallback_thresholds_are_absolute(self):
        thresholds = calculate_adaptive_thresholds(None)

        self.assertFalse(thresholds.personalized)
        self.assertAlmostEqual(thresholds.ear_threshold, 0.21)
        self.assertAlmostEqual(thresholds.mar_threshold, 0.50)

    def test_recognized_driver_profile_loads_all_baselines(self):
        baselines = DriverBaselines(
            baseline_ear=0.40,
            baseline_mar=0.028,
            baseline_perclos=12.0,
            baseline_blink_rate=8.0,
            baseline_blink_duration=0.18,
            baseline_yawn_frequency=1.5,
            baseline_head_pose_deviation=10.0,
        )
        profile = create_session_profile(
            DriverSessionResult(
                status="recognized",
                driver_id="driver_01",
                similarity=0.91,
                baselines=baselines,
            )
        )

        self.assertEqual(profile.driver_id, "driver_01")
        self.assertFalse(profile.using_fallback_thresholds)
        self.assertEqual(profile.baselines, baselines)
        self.assertAlmostEqual(profile.eye_closed_threshold, 0.30)
        self.assertAlmostEqual(profile.yawn_threshold, 0.28)

    def test_failed_recognition_uses_fallback_without_enrollment(self):
        class FailingRecognizer:
            def recognize_frame(self, frame):
                return DriverSessionResult(status="unknown")

        class OneFrameCapture:
            def read(self):
                return False, None

        with patch("database.db_access.insert_driver") as insert_driver:
            profile = start_session(
                FailingRecognizer(),
                OneFrameCapture(),
                startup_seconds=1.0,
            )

        self.assertTrue(profile.using_fallback_thresholds)
        self.assertAlmostEqual(profile.eye_closed_threshold, 0.21)
        self.assertAlmostEqual(profile.yawn_threshold, 0.50)
        insert_driver.assert_not_called()

    def test_start_session_returns_recognized_profile(self):
        baselines = DriverBaselines(
            baseline_ear=0.40,
            baseline_mar=0.028,
            baseline_perclos=12.0,
            baseline_blink_rate=8.0,
            baseline_blink_duration=0.18,
            baseline_yawn_frequency=1.5,
            baseline_head_pose_deviation=10.0,
        )

        class RecognizingService:
            def recognize_frame(self, frame):
                return DriverSessionResult(
                    status="recognized",
                    driver_id="driver_01",
                    similarity=0.91,
                    baselines=baselines,
                )

        class OneFrameCapture:
            def read(self):
                return True, object()

        profile = start_session(
            RecognizingService(),
            OneFrameCapture(),
            startup_seconds=1.0,
        )

        self.assertEqual(profile.recognition_status, "recognized")
        self.assertEqual(profile.driver_id, "driver_01")
        self.assertAlmostEqual(profile.yawn_threshold, 0.28)

    def test_unknown_driver_uses_absolute_fallback_and_no_face_is_safe(self):
        thresholds = calculate_adaptive_thresholds(None)
        decision = decide_drowsiness(
            perclos=0.0,
            prolonged_closure=False,
            yawn_detected=False,
            thresholds=thresholds,
        )

        self.assertFalse(thresholds.personalized)
        self.assertAlmostEqual(thresholds.ear_threshold, 0.21)
        self.assertAlmostEqual(thresholds.mar_threshold, 0.50)
        self.assertEqual(decision.status, "ALERT")

    def test_completed_yawn_produces_caution(self):
        decision = decide_drowsiness(
            perclos=0.0,
            prolonged_closure=False,
            yawn_detected=True,
            thresholds=calculate_adaptive_thresholds(None),
        )

        self.assertEqual(decision.status, "CAUTION")

    def test_no_new_yawn_is_alert_when_otherwise_normal(self):
        decision = decide_drowsiness(
            perclos=0.0,
            prolonged_closure=False,
            yawn_detected=False,
            thresholds=calculate_adaptive_thresholds(None),
        )

        self.assertEqual(decision.status, "ALERT")

    def test_drowsy_condition_overrides_yawn_caution(self):
        decision = decide_drowsiness(
            perclos=100.0,
            prolonged_closure=False,
            yawn_detected=True,
            thresholds=calculate_adaptive_thresholds(None),
        )

        self.assertEqual(decision.status, "DROWSY")

    def test_personalized_blink_rate_rule_including_zero_baseline(self):
        self.assertTrue(is_elevated_blink_rate(19.0, 12.0))
        self.assertFalse(is_elevated_blink_rate(18.0, 12.0))
        self.assertTrue(is_elevated_blink_rate(4.0, 0.0))
        self.assertFalse(is_elevated_blink_rate(3.0, 0.0))

    def test_personalized_blink_duration_rule_including_zero_baseline(self):
        self.assertTrue(is_unusually_long_blink(0.5, 0.2))
        self.assertFalse(is_unusually_long_blink(0.4, 0.2))
        self.assertTrue(is_unusually_long_blink(0.3, 0.0))
        self.assertFalse(is_unusually_long_blink(0.2, 0.0))

    def test_personalized_yawn_frequency_rule_including_zero_baseline(self):
        self.assertTrue(is_elevated_yawn_frequency(2.1, 1.0))
        self.assertFalse(is_elevated_yawn_frequency(1.0, 1.0))
        self.assertTrue(is_elevated_yawn_frequency(2.0, 0.0))
        self.assertFalse(is_elevated_yawn_frequency(1.0, 0.0))

    def test_head_pose_deviation_rule_including_zero_baseline(self):
        self.assertTrue(is_elevated_head_pose_deviation(1.0, 0.0))
        self.assertFalse(is_elevated_head_pose_deviation(0.0, 0.0))
        self.assertFalse(is_elevated_head_pose_deviation(5.0, None))

    def test_caution_signals_do_not_override_drowsy_priority(self):
        decision = decide_drowsiness(
            perclos=0.0,
            prolonged_closure=True,
            yawn_detected=False,
            thresholds=calculate_adaptive_thresholds(None),
            blink_rate_elevated=True,
            repeated_long_blinks=True,
            yawn_frequency_elevated=True,
        )

        self.assertEqual(decision.status, "DROWSY")

    def test_one_long_blink_does_not_produce_caution(self):
        decision = decide_drowsiness(
            perclos=0.0,
            prolonged_closure=False,
            yawn_detected=False,
            thresholds=calculate_adaptive_thresholds(None),
            repeated_long_blinks=False,
        )

        self.assertEqual(decision.status, "ALERT")

    def test_live_monitor_passes_only_new_yawn_count_to_decision(self):
        monitor = LiveFeatureMonitor()
        monitor.smoother.update = lambda ear, mar: (0.30, mar)
        monitor.blink_detector.update = lambda ear, current_time: (
            False,
            0,
            0.0,
            False,
            0.0,
        )
        monitor.blink_detector.last_blink_completed = False
        monitor.perclos_tracker.update = lambda eye_closed, current_time: (0.0, False)
        monitor.yawn_detector.update = unittest.mock.Mock(
            side_effect=[
                (True, 1.0, True, 1),
                (True, 1.5, False, 1),
            ]
        )

        with patch("src.adaptive_drowsiness_demo.calculate_head_pose", return_value="CENTER"), patch(
            "src.adaptive_drowsiness_demo.calculate_average_ear",
            return_value=(0.30, 0.30, 0.30),
        ), patch("src.adaptive_drowsiness_demo.calculate_mar", return_value=0.60):
            first_metrics = monitor.process([], 1.0)
            second_metrics = monitor.process([], 1.5)

        self.assertTrue(first_metrics["yawn_detected"])
        self.assertEqual(first_metrics["yawn_count"], 1)
        self.assertFalse(second_metrics["yawn_detected"])
        self.assertEqual(second_metrics["yawn_count"], 1)

    def test_recognition_baseline_loading_to_adaptive_decision(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            db_path = Path(temporary_directory) / "test_adapticurve.db"
            db_access.initialize_database(db_path)
            embedding = np.array([[1.0, 0.0]], dtype=np.float32)
            db_access.insert_driver(
                driver_id="driver_a",
                face_embedding=embedding,
                baseline_ear=0.369,
                baseline_mar=0.028,
                baseline_perclos=25.0,
                baseline_blink_rate=12.0,
                baseline_blink_duration=0.15,
                baseline_yawn_frequency=1.0,
                baseline_head_pose_deviation=8.0,
                db_path=db_path,
            )
            session_recognizer = DriverSessionRecognizer(
                recognizer=SyntheticRecognizer(
                    EmbeddingResult("ok", embedding),
                    RecognitionResult("driver_a", 0.99, True),
                ),
                db_path=db_path,
            )

            session_result = session_recognizer.recognize_frame(
                np.zeros((1, 1, 3), dtype=np.uint8)
            )
            thresholds = calculate_adaptive_thresholds(session_result.baselines)
            decision = decide_drowsiness(
                perclos=36.0,
                prolonged_closure=False,
                yawn_detected=False,
                thresholds=thresholds,
            )

            self.assertEqual(session_result.status, "recognized")
            self.assertEqual(session_result.driver_id, "driver_a")
            self.assertAlmostEqual(thresholds.ear_threshold, 0.27675)
            self.assertEqual(decision.status, "DROWSY")
            self.assertIn("high PERCLOS", decision.reasons)

    def test_no_face_result_does_not_load_or_apply_baselines(self):
        recognizer = DriverSessionRecognizer(
            recognizer=SyntheticRecognizer(
                EmbeddingResult("no_face"),
                RecognitionResult(None, None, False),
            ),
            db_path=Path("unused.db"),
        )

        result = recognizer.recognize_frame(np.zeros((1, 1, 3), dtype=np.uint8))

        self.assertEqual(result.status, "no_face")
        self.assertIsNone(result.baselines)

    def test_blink_and_yawn_detectors_persist_and_accumulate(self):
        monitor = LiveFeatureMonitor()
        blink_detector = monitor.blink_detector
        yawn_detector = monitor.yawn_detector
        personalized_thresholds = calculate_adaptive_thresholds(
            DriverBaselines(
                baseline_ear=0.369,
                baseline_mar=0.028,
                baseline_perclos=5.0,
                baseline_blink_rate=0.0,
                baseline_blink_duration=0.0,
                baseline_yawn_frequency=0.0,
                baseline_head_pose_deviation=0.0,
            )
        )
        monitor.set_thresholds(personalized_thresholds, driver_id="driver_a")
        monitor.set_thresholds(personalized_thresholds, driver_id="driver_a")

        self.assertIs(monitor.blink_detector, blink_detector)
        self.assertIs(monitor.yawn_detector, yawn_detector)
        self.assertAlmostEqual(monitor.blink_detector.ear_threshold, 0.27675)

        monitor.blink_detector.update(0.10, 0.0)
        monitor.blink_detector.update(0.30, 0.1)
        monitor.blink_detector.update(0.10, 0.2)
        monitor.blink_detector.update(0.30, 0.3)
        monitor.yawn_detector.update(0.60, 0.0)
        monitor.yawn_detector.update(0.60, 1.1)
        monitor.yawn_detector.update(0.10, 1.2)
        monitor.yawn_detector.update(0.60, 2.0)
        monitor.yawn_detector.update(0.60, 3.1)

        monitor.set_thresholds(monitor.thresholds, driver_id=None)
        monitor.reset_visibility_buffers()

        self.assertIs(monitor.blink_detector, blink_detector)
        self.assertIs(monitor.yawn_detector, yawn_detector)
        self.assertEqual(monitor.blink_detector.blink_count, 2)
        self.assertEqual(monitor.yawn_detector.yawn_count, 2)
        self.assertEqual(monitor.yawn_detector.update(0.10, 3.2)[3], 2)
        self.assertEqual(monitor.blink_detector.update(0.30, 3.2)[4], 2)

    def test_active_profile_reaches_detector_thresholds(self):
        profile = create_session_profile(
            DriverSessionResult(
                status="recognized",
                driver_id="driver_01",
                baselines=DriverBaselines(
                    baseline_ear=0.40,
                    baseline_mar=0.028,
                    baseline_perclos=0.0,
                    baseline_blink_rate=0.0,
                    baseline_blink_duration=0.0,
                    baseline_yawn_frequency=0.0,
                    baseline_head_pose_deviation=0.0,
                ),
            )
        )
        monitor = LiveFeatureMonitor()

        monitor.set_session_profile(profile)

        self.assertAlmostEqual(
            monitor.blink_detector.ear_threshold,
            profile.eye_closed_threshold,
        )
        self.assertAlmostEqual(
            monitor.yawn_detector.mar_threshold,
            profile.yawn_threshold,
        )


if __name__ == "__main__":
    unittest.main()
