"""Standalone live demo combining driver recognition and personalized monitoring."""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.schema import DATABASE_PATH
from src.adaptive_thresholds import (
    AdaptiveThresholds,
    calculate_adaptive_thresholds,
    decide_drowsiness,
    is_elevated_blink_rate,
    is_elevated_head_pose_deviation,
    is_elevated_yawn_frequency,
    is_unusually_long_blink,
)
from src.driver_session_recognition import (
    DriverBaselines,
    DriverSessionRecognizer,
    DriverSessionResult,
)
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
    LEFT_EYE_EAR,
    MOUTH_HORIZONTAL,
    MOUTH_VERTICAL_1,
    MOUTH_VERTICAL_2,
    PERCLOS_WINDOW_SECONDS,
    RIGHT_EYE_EAR,
    SMOOTHING_WINDOW,
    YAWN_MIN_DURATION,
)


WEBCAM_INDEX = 0
STARTUP_RECOGNITION_SECONDS = 3.0
STARTUP_DIAGNOSTIC_MAX_FRAMES = 120


@dataclass(frozen=True)
class ActiveSessionProfile:
    """Identity, complete baselines, and thresholds fixed for one live session."""

    driver_id: Optional[str]
    recognition_status: str
    similarity: Optional[float]
    baselines: Optional[DriverBaselines]
    eye_closed_threshold: float
    yawn_threshold: float
    using_fallback_thresholds: bool
    thresholds: AdaptiveThresholds


def create_session_profile(
    recognition_result: DriverSessionResult,
) -> ActiveSessionProfile:
    """Build a personalized or documented fallback profile from recognition."""

    recognized = (
        recognition_result.status == "recognized"
        and recognition_result.baselines is not None
        and recognition_result.driver_id is not None
    )
    baselines = recognition_result.baselines if recognized else None
    thresholds = calculate_adaptive_thresholds(baselines)
    recognition_status = (
        recognition_result.status
        if recognized
        else "unknown"
        if recognition_result.status == "recognized"
        else recognition_result.status
    )
    return ActiveSessionProfile(
        driver_id=recognition_result.driver_id if recognized else None,
        recognition_status=recognition_status,
        similarity=recognition_result.similarity,
        baselines=baselines,
        eye_closed_threshold=thresholds.ear_threshold,
        yawn_threshold=thresholds.mar_threshold,
        using_fallback_thresholds=not recognized,
        thresholds=thresholds,
    )


def start_session(
    session_recognizer: DriverSessionRecognizer,
    capture,
    startup_seconds: float = STARTUP_RECOGNITION_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> ActiveSessionProfile:
    """Collect startup evidence while retaining the prior first-match behavior."""

    deadline = clock() + startup_seconds
    last_result = DriverSessionResult(status="unknown")
    first_recognized_result: Optional[DriverSessionResult] = None
    frame_number = 0
    while (
        clock() < deadline
        and frame_number < STARTUP_DIAGNOSTIC_MAX_FRAMES
    ):
        ret, frame = capture.read()
        if not ret:
            break
        frame_number += 1
        last_result = session_recognizer.recognize_frame(frame)
        candidate = (
            last_result.driver_id
            if last_result.status == "recognized"
            else "-"
        )
        similarity = (
            f"{last_result.similarity:.3f}"
            if last_result.similarity is not None
            else "N/A"
        )
        print(
            f"Startup recognition frame {frame_number}: "
            f"status={last_result.status}, similarity={similarity}, "
            f"candidate={candidate}"
        )
        if (
            first_recognized_result is None
            and last_result.status == "recognized"
            and last_result.driver_id is not None
            and last_result.baselines is not None
        ):
            first_recognized_result = last_result

    selected_result = first_recognized_result or last_result
    profile = create_session_profile(selected_result)
    if first_recognized_result is not None:
        print(
            f"Startup candidate retained: {profile.driver_id} "
            f"(first recognized frame; latest observed status={last_result.status})."
        )
        return profile

    print(
        "Session driver could not be confirmed during startup; "
        f"using fallback EAR {profile.eye_closed_threshold:.3f} and "
        f"MAR {profile.yawn_threshold:.3f}."
    )
    return profile


class LiveFeatureMonitor:
    """Apply supplied thresholds through the existing Day 2-3 detectors."""

    def __init__(self) -> None:
        self.thresholds = calculate_adaptive_thresholds(None)
        self.active_driver_id: Optional[str] = None
        self.smoother = FeatureSmoother(window_size=SMOOTHING_WINDOW)
        self.perclos_tracker = PERCLOS(window_seconds=PERCLOS_WINDOW_SECONDS)
        # These session detectors retain completed blink/yawn counts.
        self.blink_detector = BlinkDetector(ear_threshold=self.thresholds.ear_threshold)
        self.yawn_detector = YawnDetector(
            mar_threshold=self.thresholds.mar_threshold,
            minimum_duration=YAWN_MIN_DURATION,
        )
        self.session_blink_count = 0
        self.session_yawn_count = 0
        self.long_blink_count = 0
        self.yawn_timestamps = []
        self.session_head_pose_frames = 0
        self.session_deviated_head_pose_frames = 0

    def set_thresholds(
        self,
        thresholds: AdaptiveThresholds,
        driver_id: Optional[str],
    ) -> None:
        """Update thresholds in place without discarding session detector state."""

        if driver_id is None:
            return

        self.thresholds = thresholds
        self.active_driver_id = driver_id
        # BlinkDetector/YawnDetector are persistent. Their configurable
        # thresholds may change with the recognized driver, but their counts
        # and rolling blink timestamps must remain session-wide.
        self.blink_detector.ear_threshold = thresholds.ear_threshold
        self.yawn_detector.mar_threshold = thresholds.mar_threshold

    def set_session_profile(self, profile: ActiveSessionProfile) -> None:
        """Apply the startup profile without recreating persistent detectors."""

        self.set_thresholds(profile.thresholds, profile.driver_id)
        if profile.using_fallback_thresholds:
            self.thresholds = profile.thresholds
            self.active_driver_id = None

    def reset_visibility_buffers(self) -> None:
        """Match existing no-face smoothing/PERCLOS handling without clearing counts."""

        self.smoother.reset()
        self.perclos_tracker.reset()

    def process(self, face_landmarks, current_time: float) -> dict:
        """Return existing feature outputs for the current valid face frame."""

        head_pose = calculate_head_pose(face_landmarks)
        ear, _, _ = calculate_average_ear(
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
        ear, mar = self.smoother.update(ear, mar)
        eye_closed, blink_count, closure_duration, prolonged_closure, blink_rate = (
            self.blink_detector.update(ear, current_time)
        )
        perclos, warming_up = self.perclos_tracker.update(eye_closed, current_time)
        mouth_open, yawn_duration, _, yawn_count = self.yawn_detector.update(
            mar,
            current_time,
        )
        if self.blink_detector.last_blink_completed:
            self.session_blink_count += 1
            if is_unusually_long_blink(
                self.blink_detector.last_blink_duration,
                self.thresholds.baseline_blink_duration,
            ):
                self.long_blink_count += 1
        yawn_detected = yawn_count > self.session_yawn_count
        self.session_yawn_count = yawn_count
        if yawn_detected:
            self.yawn_timestamps.append(current_time)
        self.yawn_timestamps = [
            timestamp
            for timestamp in self.yawn_timestamps
            if current_time - timestamp <= 60.0
        ]
        self.session_head_pose_frames += 1
        if head_pose != "CENTER":
            self.session_deviated_head_pose_frames += 1
        head_pose_deviation = (
            self.session_deviated_head_pose_frames
            / self.session_head_pose_frames
            * 100.0
        )
        yawn_frequency = len(self.yawn_timestamps)

        return {
            "ear": ear,
            "mar": mar,
            "perclos": perclos,
            "perclos_warming_up": warming_up,
            "blink_count": self.session_blink_count,
            "blink_rate": blink_rate,
            "closure_duration": closure_duration,
            "prolonged_closure": prolonged_closure,
            "yawn_count": self.session_yawn_count,
            "yawn_duration": yawn_duration,
            "yawn_detected": yawn_detected,
            "blink_rate_elevated": is_elevated_blink_rate(
                blink_rate,
                self.thresholds.baseline_blink_rate,
            ),
            "repeated_long_blinks": self.long_blink_count >= 2,
            "yawn_frequency": yawn_frequency,
            "yawn_frequency_elevated": is_elevated_yawn_frequency(
                yawn_frequency,
                self.thresholds.baseline_yawn_frequency,
            ),
            "head_pose_deviation": head_pose_deviation,
            "head_pose_deviation_elevated": is_elevated_head_pose_deviation(
                head_pose_deviation,
                self.thresholds.baseline_head_pose_deviation,
            ),
            "mouth_open": mouth_open,
            "head_pose": head_pose,
        }


def _put_text(frame, text: str, line: int, color=(255, 255, 255)) -> None:
    cv2.putText(
        frame,
        text,
        (20, 30 + line * 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def draw_demo_status(
    frame,
    recognition_result,
    thresholds: Optional[AdaptiveThresholds],
    metrics: Optional[dict],
) -> None:
    """Render recognition, current metrics, thresholds, and drowsiness status."""

    recognized = recognition_result.status == "recognized"
    driver_labels = {
        "no_face": "No face detected",
        "multiple_faces": "Multiple faces detected",
        "unknown": "Unknown",
    }
    driver_id = (
        recognition_result.driver_id
        if recognized
        else driver_labels.get(recognition_result.status, "Unknown")
    )
    similarity = (
        f"{recognition_result.similarity:.3f}"
        if recognition_result.similarity is not None
        else "N/A"
    )
    _put_text(frame, f"Driver: {driver_id}", 0, (0, 255, 0) if recognized else (0, 0, 255))
    _put_text(frame, f"Similarity: {similarity}", 1)
    _put_text(frame, f"Recognition: {recognition_result.status}", 2)

    if thresholds is None or metrics is None:
        return

    threshold_label = "Personalized" if thresholds.personalized else "Default fallback"
    _put_text(frame, f"Thresholds: {threshold_label}", 3)
    _put_text(
        frame,
        f"EAR: {metrics['ear']:.3f} (threshold {thresholds.ear_threshold:.3f})",
        4,
    )
    _put_text(
        frame,
        f"MAR: {metrics['mar']:.3f} (threshold {thresholds.mar_threshold:.3f})",
        5,
    )
    perclos_text = "Warming up" if metrics["perclos_warming_up"] else f"{metrics['perclos']:.1f}%"
    _put_text(
        frame,
        f"PERCLOS: {perclos_text} (alert {thresholds.perclos_threshold:.1f}%)",
        6,
    )
    _put_text(frame, f"Blinks: {metrics['blink_count']} ({metrics['blink_rate']}/min)", 7)
    _put_text(frame, f"Yawns: {metrics['yawn_count']}", 8)
    _put_text(
        frame,
        f"Head pose: {metrics['head_pose']} ({metrics['head_pose_deviation']:.1f}%)",
        9,
    )

    decision = decide_drowsiness(
        metrics["perclos"],
        metrics["prolonged_closure"],
        metrics["yawn_detected"],
        thresholds,
        metrics["blink_rate_elevated"],
        metrics["repeated_long_blinks"],
        metrics["yawn_frequency_elevated"],
    )
    reasons = ", ".join(decision.reasons) if decision.reasons else "normal"
    _put_text(
        frame,
        f"Drowsiness: {decision.status} ({reasons})",
        10,
        (0, 0, 255) if decision.status == "DROWSY" else (0, 255, 255),
    )


def run_demo() -> None:
    """Run startup-only recognition and personalized drowsiness monitoring."""

    if not DATABASE_PATH.is_file():
        print(f"Error: enrolled-driver database not found: {DATABASE_PATH}")
        return

    session_recognizer = DriverSessionRecognizer(db_path=DATABASE_PATH)
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {WEBCAM_INDEX}.")
        return

    active_profile = start_session(session_recognizer, cap)
    active_result = DriverSessionResult(
        status=active_profile.recognition_status,
        driver_id=active_profile.driver_id,
        similarity=active_profile.similarity,
        baselines=active_profile.baselines,
    )
    monitor = LiveFeatureMonitor()
    monitor.set_session_profile(active_profile)
    session_start = time.monotonic()
    last_timestamp_ms = -1

    try:
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(PROJECT_ROOT / "models" / "face_landmarker.task")
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
        )
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            print("Adaptive drowsiness demo started. Press 'q' to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read webcam frame.")
                    break

                current_time = time.monotonic()
                frame_result = active_result
                thresholds = active_profile.thresholds
                metrics = None

                timestamp_ms = max(
                    int((current_time - session_start) * 1000),
                    last_timestamp_ms + 1,
                )
                last_timestamp_ms = timestamp_ms
                mediapipe_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                landmark_result = landmarker.detect_for_video(
                    mediapipe_image,
                    timestamp_ms,
                )
                if landmark_result.face_landmarks:
                    metrics = monitor.process(
                        landmark_result.face_landmarks[0],
                        current_time,
                    )
                else:
                    monitor.reset_visibility_buffers()
                    frame_result = DriverSessionResult(status="no_face")

                draw_demo_status(frame, frame_result, thresholds, metrics)
                cv2.imshow("Adaptive Drowsiness Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
