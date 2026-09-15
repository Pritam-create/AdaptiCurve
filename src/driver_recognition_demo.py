"""Read-only live YuNet/SFace driver-recognition demonstration."""

import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.schema import DATABASE_PATH
from src.driver_session_recognition import DriverSessionResult, DriverSessionRecognizer


WEBCAM_INDEX = 0


def _put_text(frame, text: str, line_number: int, color=(255, 255, 255)) -> None:
    cv2.putText(
        frame,
        text,
        (20, 35 + line_number * 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def draw_recognition_result(frame, result: DriverSessionResult) -> None:
    """Display a read-only recognition outcome and matched baseline profile."""

    recognized = result.status == "recognized"
    driver_name = result.driver_id if recognized else "Unknown"
    similarity_text = (
        f"{result.similarity:.3f}"
        if result.similarity is not None
        else "N/A"
    )

    _put_text(
        frame,
        f"Driver: {driver_name}",
        0,
        (0, 255, 0) if recognized else (0, 0, 255),
    )
    _put_text(frame, f"Similarity: {similarity_text}", 1)

    if not recognized or result.baselines is None:
        _put_text(frame, f"Face status: {result.status}", 2)
        return

    baselines = result.baselines
    _put_text(frame, f"EAR baseline: {baselines.baseline_ear:.3f}", 2)
    _put_text(frame, f"MAR baseline: {baselines.baseline_mar:.3f}", 3)
    _put_text(frame, f"PERCLOS baseline: {baselines.baseline_perclos:.2f}%", 4)
    _put_text(frame, f"Blink rate baseline: {baselines.baseline_blink_rate:.2f}/min", 5)
    _put_text(frame, f"Yawn frequency baseline: {baselines.baseline_yawn_frequency:.2f}/min", 6)
    _put_text(
        frame,
        "Head-pose baseline: "
        f"{baselines.baseline_head_pose_deviation:.2f}%",
        7,
    )


def run_demo() -> None:
    """Open webcam index 0 and show recognition for each captured frame."""

    if not DATABASE_PATH.is_file():
        print(f"Error: enrolled-driver database not found: {DATABASE_PATH}")
        return

    session_recognizer = DriverSessionRecognizer(db_path=DATABASE_PATH)
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {WEBCAM_INDEX}.")
        return

    print("Driver recognition demo started. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read webcam frame.")
                break

            # DriverSessionRecognizer performs YuNet detection and SFace matching.
            result = session_recognizer.recognize_frame(frame)
            draw_recognition_result(frame, result)
            cv2.imshow("Driver Recognition Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
