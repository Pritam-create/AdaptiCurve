import cv2
import mediapipe as mp
import time

from landmark_config import (
    LEFT_EYE_EAR,
    RIGHT_EYE_EAR,
    MOUTH_HORIZONTAL,
    MOUTH_VERTICAL_1,
    MOUTH_VERTICAL_2
)
from feature_extraction import (
    calculate_average_ear,
    calculate_mar,
    BlinkDetector,
    YawnDetector,
    FeatureSmoother,
    calculate_head_pose
)


# ============================================================
# 1. MediaPipe Face Landmarker Setup
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)


# ============================================================
# 2. Function to Draw Landmarks
# ============================================================

def draw_landmarks(frame, face_landmarks):

    height, width, _ = frame.shape

    # --------------------------------------------------------
    # Draw all face landmarks in gray
    # --------------------------------------------------------

    for landmark in face_landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            1,
            (150, 150, 150),
            -1
        )

    # --------------------------------------------------------
    # Draw LEFT EYE landmarks in green
    # --------------------------------------------------------

    for index in LEFT_EYE_EAR:

        landmark = face_landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

    # --------------------------------------------------------
    # Draw RIGHT EYE landmarks in green
    # --------------------------------------------------------

    for index in RIGHT_EYE_EAR:

        landmark = face_landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

    # --------------------------------------------------------
    # Draw MOUTH landmarks in red
    # --------------------------------------------------------

    mouth_indices = (
        MOUTH_HORIZONTAL
        + MOUTH_VERTICAL_1
        + MOUTH_VERTICAL_2
    )

    for index in mouth_indices:

        landmark = face_landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )


# ============================================================
# 3. Start MediaPipe Face Landmarker
# ============================================================

with FaceLandmarker.create_from_options(options) as landmarker:

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Error: Could not open webcam.")
        exit()

    print("Face Landmarker started.")
    print("Green = eye landmarks")
    print("Red = mouth landmarks")
    print("Press 'q' to quit.")

    # --------------------------------------------------------
    # Use real elapsed time for MediaPipe timestamps
    # --------------------------------------------------------

    start_time = time.monotonic()

    blink_detector = BlinkDetector(
        ear_threshold=0.21
    )

    yawn_detector = YawnDetector(
        mar_threshold=0.50,
        minimum_duration=1.0
    )


    smoother = FeatureSmoother(
        window_size=5
    )

    while True:

        # ----------------------------------------------------
        # Read webcam frame
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print("Error: Could not read frame.")
            break

        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Convert OpenCV image → MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # ----------------------------------------------------
        # Calculate actual timestamp in milliseconds
        # ----------------------------------------------------

        frame_timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        # ----------------------------------------------------
        # Detect face landmarks
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        # ----------------------------------------------------
        # If face detected
        # ----------------------------------------------------

        if result.face_landmarks:

            face_landmarks = result.face_landmarks[0]

            head_pose = calculate_head_pose(face_landmarks)

            # Calculate EAR
            average_ear, left_ear, right_ear = calculate_average_ear(
                face_landmarks,
                LEFT_EYE_EAR,
                RIGHT_EYE_EAR
            )


            
            mar = calculate_mar(
                face_landmarks,
                MOUTH_HORIZONTAL,
                MOUTH_VERTICAL_1,
                MOUTH_VERTICAL_2
            )

            # Smooth EAR and MAR
            average_ear, mar = smoother.update(
                average_ear,
                mar
            )

            eye_closed, blink_count, closure_duration, prolonged_closure, blink_rate = (
                blink_detector.update(
                    average_ear,
                    time.monotonic()
                )
            )

            mouth_open, yawn_duration, yawn_detected, yawn_count = (
                yawn_detector.update(
                    mar,
                    time.monotonic()
                )
            )

            
            # Draw landmarks
            draw_landmarks(
                frame,
                face_landmarks
            )

            # Display face status
            cv2.putText(
                frame,
                "Face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display EAR values
            cv2.putText(
                frame,
                f"EAR: {average_ear: .3f}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"Left: {left_ear: .3f} Right: {right_ear: .3f}",
                (20,110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"MAR: {mar:.3f}",
                (20, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Display blink count
            cv2.putText(
                frame,
                f"Blinks: {blink_count}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Head pose: {head_pose}",
                (20, 410),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Blink rate: {blink_rate}/min",
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Display current eye state
            eye_status = "CLOSED" if eye_closed else "OPEN"

            cv2.putText(
                frame,
                f"Eyes: {eye_status}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Closure: {closure_duration:.2f} s",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            closure_status = (
                "PROLONGED"
                if prolonged_closure
                else "Normal"
            )

            cv2.putText(
                frame,
                f"Closure status: {closure_status}",
                (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if prolonged_closure else (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Yawn duration: {yawn_duration:.2f} s",
                (20, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Yawns: {yawn_count}",
                (20, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            mouth_status = "OPEN" if mouth_open else "CLOSED"

            cv2.putText(
                frame,
                f"Mouth: {mouth_status}",
                (20, 380),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if mouth_open else (255, 255, 255),
                2
            )
        # ----------------------------------------------------
        # If no face detected
        # ----------------------------------------------------

        else:

            smoother.reset()

            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # Display webcam
        # ----------------------------------------------------

        cv2.imshow(
            "Face Landmark Visualization",
            frame
        )

        # ----------------------------------------------------
        # Press Q to quit
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    # ========================================================
    # 4. Cleanup
    # ========================================================

    cap.release()
    cv2.destroyAllWindows()