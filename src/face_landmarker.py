import cv2
import mediapipe as mp
import time

from landmark_config import LEFT_EYE, RIGHT_EYE, MOUTH


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

    for index in LEFT_EYE:

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

    for index in RIGHT_EYE:

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

    for index in MOUTH:

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

            # Draw landmarks
            draw_landmarks(
                frame,
                face_landmarks
            )

            # Display status
            cv2.putText(
                frame,
                "Face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # If no face detected
        # ----------------------------------------------------

        else:

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