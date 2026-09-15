"""Standalone webcam enrollment and baseline calibration for one driver."""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import db_access
from database.schema import DATABASE_PATH
from src.driver_recognition import DriverRecognizer
from src.feature_extraction import (
    BlinkDetector,
    FeatureSmoother,
    PERCLOS,
    YawnDetector,
    calculate_average_ear,
    calculate_head_pose,
    calculate_mar,
)
from src.landmark_config import (
    CALIBRATION_DURATION_SECONDS,
    CENTERED_HEAD_POSES,
    LEFT_EYE_EAR,
    MAR_THRESHOLD,
    MOUTH_HORIZONTAL,
    MOUTH_VERTICAL_1,
    MOUTH_VERTICAL_2,
    PERCLOS_WINDOW_SECONDS,
    RIGHT_EYE_EAR,
    SMOOTHING_WINDOW,
    YAWN_MIN_DURATION,
)


# src/webcam_test.py establishes index 1 as the project working webcam.
WEBCAM_INDEX = 0


@dataclass
class CalibrationBaselines:
    """The seven baseline values persisted for a successfully enrolled driver."""

    baseline_ear: float
    baseline_mar: float
    baseline_perclos: float
    baseline_blink_rate: float
    baseline_blink_duration: float
    baseline_yawn_frequency: float
    baseline_head_pose_deviation: float


@dataclass
class CalibrationAccumulator:
    """Accumulate only valid single-face samples during one calibration period."""

    start_time: float
    ear_values: list[float] = field(default_factory=list)
    mar_values: list[float] = field(default_factory=list)
    perclos_values: list[float] = field(default_factory=list)
    blink_durations: list[float] = field(default_factory=list)
    completed_blinks: int = 0
    completed_yawns: int = 0
    deviated_frames: int = 0
    face_frames: int = 0

    def add_sample(
        self,
        ear: float,
        mar: float,
        perclos: float,
        head_pose: str,
        blink_completed: bool = False,
        blink_duration: float = 0.0,
        yawn_detected: bool = False,
    ) -> None:
        """Record one processed frame already confirmed to contain exactly one face."""

        self.ear_values.append(ear)
        self.mar_values.append(mar)
        self.perclos_values.append(perclos)
        self.face_frames += 1

        if head_pose not in CENTERED_HEAD_POSES:
            self.deviated_frames += 1
        if blink_completed:
            self.completed_blinks += 1
            self.blink_durations.append(blink_duration)
        if yawn_detected:
            self.completed_yawns += 1

    def has_samples(self) -> bool:
        return self.face_frames > 0

    def finalize(self, end_time: float) -> CalibrationBaselines:
        """Calculate baseline values using the same event definitions as Day 3."""

        duration = max(end_time - self.start_time, 0.0)
        mean_blink_duration = (
            sum(self.blink_durations) / len(self.blink_durations)
            if self.blink_durations
            else 0.0
        )

        return CalibrationBaselines(
            baseline_ear=sum(self.ear_values) / len(self.ear_values)
            if self.ear_values else 0.0,
            baseline_mar=sum(self.mar_values) / len(self.mar_values)
            if self.mar_values else 0.0,
            baseline_perclos=sum(self.perclos_values) / len(self.perclos_values)
            if self.perclos_values else 0.0,
            baseline_blink_rate=(self.completed_blinks / duration) * 60.0
            if duration > 0 else 0.0,
            baseline_blink_duration=mean_blink_duration,
            baseline_yawn_frequency=(self.completed_yawns / duration) * 60.0
            if duration > 0 else 0.0,
            baseline_head_pose_deviation=(
                self.deviated_frames / self.face_frames
            ) * 100.0 if self.face_frames else 0.0,
        )


class CalibrationFeatureMonitor:
    """Reuse Day 1-3 feature components for calibration-only aggregation."""

    def __init__(self) -> None:
        self._create_detectors()
        self.smoother = FeatureSmoother(window_size=SMOOTHING_WINDOW)
        self.perclos_tracker = PERCLOS(window_seconds=PERCLOS_WINDOW_SECONDS)

    def _create_detectors(self) -> None:
        self.blink_detector = BlinkDetector()
        self.yawn_detector = YawnDetector(
            mar_threshold=MAR_THRESHOLD,
            minimum_duration=YAWN_MIN_DURATION,
        )

    def reset_after_face_loss(self) -> None:
        """Avoid carrying a partial blink/yawn across a no-face interruption."""

        self._create_detectors()
        self.smoother.reset()
        self.perclos_tracker.reset()

    def process(
        self,
        face_landmarks: list,
        current_time: float,
    ) -> tuple[float, float, float, str, bool, float, bool, int, int, float]:
        """Return existing Day 1-3 feature values for one valid face frame."""

        head_pose = calculate_head_pose(face_landmarks)
        average_ear, _, _ = calculate_average_ear(
            face_landmarks,
            LEFT_EYE_EAR,
            RIGHT_EYE_EAR,
        )
        mar = calculate_mar(
            face_landmarks,
            MOUTH_HORIZONTAL,
            MOUTH_VERTICAL_1,
            MOUTH_VERTICAL_2,
        )
        average_ear, mar = self.smoother.update(average_ear, mar)

        eye_closed, blink_count, _, _, blink_rate = self.blink_detector.update(
            average_ear,
            current_time,
        )
        perclos, _ = self.perclos_tracker.update(eye_closed, current_time)
        _, _, yawn_detected, yawn_count = self.yawn_detector.update(
            mar,
            current_time,
        )

        return (
            average_ear,
            mar,
            perclos,
            head_pose,
            self.blink_detector.last_blink_completed,
            self.blink_detector.last_blink_duration,
            yawn_detected,
            blink_count,
            yawn_count,
            blink_rate,
        )


def calibration_window_expired(
    calibration_start: Optional[float],
    current_time: float,
) -> bool:
    """Return whether the fixed calibration window has reached its endpoint."""

    return (
        calibration_start is not None
        and current_time - calibration_start >= CALIBRATION_DURATION_SECONDS
    )


def can_collect_sample(embedding_status: str, has_landmarks: bool) -> bool:
    """Allow samples only after YuNet/SFace and MediaPipe both confirm one face."""

    return embedding_status == "ok" and has_landmarks


def _draw_status(
    frame: np.ndarray,
    driver_id: str,
    face_status: str,
    remaining_seconds: Optional[float],
    feature_values: Optional[tuple] = None,
    accumulator: Optional[CalibrationAccumulator] = None,
) -> None:
    """Draw the small calibration status overlay without changing the Day 3 UI."""

    lines = ["Calibration", f"Driver ID: {driver_id}"]
    if remaining_seconds is None:
        lines.append("Waiting for exactly one face")
    else:
        lines.append(f"Remaining: {remaining_seconds:.1f} s")
    lines.append(f"Face: {face_status}")

    if feature_values is not None:
        ear, mar, perclos, head_pose, _, _, _, _, _, blink_rate = (
            feature_values
        )
        completed_blinks = accumulator.completed_blinks if accumulator else 0
        completed_yawns = accumulator.completed_yawns if accumulator else 0
        lines.extend(
            [
                f"EAR: {ear:.3f}",
                f"MAR: {mar:.3f}",
                f"PERCLOS: {perclos:.1f}%",
                f"Blinks: {completed_blinks} ({blink_rate}/min)",
                f"Yawns: {completed_yawns}",
                f"Head pose: {head_pose}",
            ]
        )

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, 35 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


def run_calibration(
    driver_id: str,
    db_path: Path | str = DATABASE_PATH,
) -> bool:
    """Run one 10-second enrollment/calibration session and save its baselines."""

    driver_id = driver_id.strip()
    if not driver_id:
        print("Calibration cancelled: driver ID cannot be empty.")
        return False

    db_access.initialize_database(db_path)
    if db_access.driver_exists(driver_id, db_path):
        print(f"Calibration cancelled: driver '{driver_id}' already exists.")
        return False

    recognizer = DriverRecognizer()
    feature_monitor = CalibrationFeatureMonitor()
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {WEBCAM_INDEX}.")
        return False

    embedding: Optional[np.ndarray] = None
    calibration_start: Optional[float] = None
    accumulator: Optional[CalibrationAccumulator] = None
    session_start = time.monotonic()
    last_timestamp_ms = -1
    face_was_visible = False
    completed = False

    try:
        base_options = mp.tasks.BaseOptions
        face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options(
                model_asset_path=str(PROJECT_ROOT / "models" / "face_landmarker.task")
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
        )
        with mp.tasks.vision.FaceLandmarker.create_from_options(
            face_landmarker_options
        ) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read webcam frame.")
                    break

                current_time = time.monotonic()
                # Do not process a post-window frame or its events.
                if calibration_window_expired(calibration_start, current_time):
                    completed = True
                    break

                embedding_result = recognizer.extract_embedding(frame)
                face_status = embedding_result.status
                feature_values = None

                if embedding_result.status == "ok":
                    timestamp_ms = max(
                        int((current_time - session_start) * 1000),
                        last_timestamp_ms + 1,
                    )
                    last_timestamp_ms = timestamp_ms
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mediapipe_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )
                    landmark_result = landmarker.detect_for_video(
                        mediapipe_image,
                        timestamp_ms,
                    )

                    if can_collect_sample(
                        embedding_result.status,
                        bool(landmark_result.face_landmarks),
                    ):
                        if embedding is None:
                            embedding = embedding_result.embedding
                            calibration_start = current_time
                            accumulator = CalibrationAccumulator(calibration_start)

                        feature_values = feature_monitor.process(
                            landmark_result.face_landmarks[0],
                            current_time,
                        )
                        accumulator.add_sample(
                            ear=feature_values[0],
                            mar=feature_values[1],
                            perclos=feature_values[2],
                            head_pose=feature_values[3],
                            blink_completed=feature_values[4],
                            blink_duration=feature_values[5],
                            yawn_detected=feature_values[6],
                        )
                        face_status = "exactly one face"
                        face_was_visible = True
                    else:
                        face_status = "MediaPipe face unavailable"

                if face_status != "exactly one face" and face_was_visible:
                    feature_monitor.reset_after_face_loss()
                    face_was_visible = False

                remaining = None
                if calibration_start is not None:
                    remaining = max(
                        CALIBRATION_DURATION_SECONDS
                        - (current_time - calibration_start),
                        0.0,
                    )
                _draw_status(
                    frame,
                    driver_id,
                    face_status,
                    remaining,
                    feature_values,
                    accumulator,
                )
                cv2.imshow("Driver Calibration", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Calibration cancelled by user.")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not completed or accumulator is None or embedding is None:
        print("Calibration failed: no completed calibration data was saved.")
        return False
    if not accumulator.has_samples():
        print("Calibration failed: no usable face data was collected.")
        return False

    calibration_end = accumulator.start_time + CALIBRATION_DURATION_SECONDS
    baselines = accumulator.finalize(calibration_end)
    db_access.insert_driver(
        driver_id=driver_id,
        face_embedding=embedding,
        baseline_ear=baselines.baseline_ear,
        baseline_mar=baselines.baseline_mar,
        baseline_perclos=baselines.baseline_perclos,
        baseline_blink_rate=baselines.baseline_blink_rate,
        baseline_blink_duration=baselines.baseline_blink_duration,
        baseline_yawn_frequency=baselines.baseline_yawn_frequency,
        baseline_head_pose_deviation=baselines.baseline_head_pose_deviation,
        db_path=db_path,
    )
    print(f"Calibration completed and saved for driver '{driver_id}'.")
    return True


def main() -> None:
    """Request an ID and run the standalone calibration application."""

    driver_id = input("Enter driver ID for calibration: ")
    run_calibration(driver_id)


if __name__ == "__main__":
    main()
