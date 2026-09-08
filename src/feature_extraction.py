import math
from collections import deque

def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two facial landmarks.

    Each point must have .x and .y attributes.
    """

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR).

    Expected landmark order:

        p2       p3
         ↓       ↓
      ┌───────────┐
    p1               p4
      └───────────┘
         ↑       ↑
        p6       p5

    EAR =
        (distance(p2,p6) + distance(p3,p5))
        -----------------------------------
              2 * distance(p1,p4)
    """

    p1, p2, p3, p4, p5, p6 = eye_landmarks

    vertical_distance_1 = euclidean_distance(p2, p6)
    vertical_distance_2 = euclidean_distance(p3, p5)

    horizontal_distance = euclidean_distance(p1, p4)

    # Prevent division by zero
    if horizontal_distance == 0:
        return 0.0

    ear = (
        vertical_distance_1 + vertical_distance_2
    ) / (2.0 * horizontal_distance)

    return ear


def extract_eye_landmarks(face_landmarks, eye_indices):
    """
    Extract the six landmarks required for EAR.
    """

    return [
        face_landmarks[index]
        for index in eye_indices
    ]


def calculate_average_ear(
    face_landmarks,
    left_eye_indices,
    right_eye_indices
):
    """
    Calculate EAR for both eyes and return their average.

    Returns:
        average_ear, left_ear, right_ear
    """

    left_eye = extract_eye_landmarks(
        face_landmarks,
        left_eye_indices
    )

    right_eye = extract_eye_landmarks(
        face_landmarks,
        right_eye_indices
    )

    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)

    average_ear = (left_ear + right_ear) / 2.0

    return average_ear, left_ear, right_ear



def calculate_mar(
    face_landmarks,
    mouth_horizontal,
    mouth_vertical_1,
    mouth_vertical_2
):
    """
    Calculate Mouth Aspect Ratio (MAR).

    MAR = average vertical mouth opening
          / horizontal mouth width
    """

    # Horizontal mouth points
    left_corner = face_landmarks[mouth_horizontal[0]]
    right_corner = face_landmarks[mouth_horizontal[1]]

    # Vertical pair 1
    upper_1 = face_landmarks[mouth_vertical_1[0]]
    lower_1 = face_landmarks[mouth_vertical_1[1]]

    # Vertical pair 2
    upper_2 = face_landmarks[mouth_vertical_2[0]]
    lower_2 = face_landmarks[mouth_vertical_2[1]]

    # Horizontal mouth width
    horizontal_distance = math.sqrt(
        (left_corner.x - right_corner.x) ** 2 +
        (left_corner.y - right_corner.y) ** 2
    )

    # First vertical distance
    vertical_distance_1 = math.sqrt(
        (upper_1.x - lower_1.x) ** 2 +
        (upper_1.y - lower_1.y) ** 2
    )

    # Second vertical distance
    vertical_distance_2 = math.sqrt(
        (upper_2.x - lower_2.x) ** 2 +
        (upper_2.y - lower_2.y) ** 2
    )

    # Average vertical opening
    average_vertical_distance = (
        vertical_distance_1 + vertical_distance_2
    ) / 2

    # MAR
    mar = (
        average_vertical_distance /
        horizontal_distance
    )

    return mar



class FeatureSmoother:
    """
    Smooth EAR and MAR values using a moving average.
    """

    def __init__(self, window_size=5):
        self.window_size = window_size

        self.ear_values = deque(maxlen=window_size)
        self.mar_values = deque(maxlen=window_size)

    def update(self, ear, mar):
        """
        Add new EAR/MAR values and return their averages.
        """

        self.ear_values.append(ear)
        self.mar_values.append(mar)

        smoothed_ear = sum(self.ear_values) / len(self.ear_values)
        smoothed_mar = sum(self.mar_values) / len(self.mar_values)

        return smoothed_ear, smoothed_mar

    def reset(self):
        """
        Clear stored values.
        """
        self.ear_values.clear()
        self.mar_values.clear()



class YawnDetector:
    """
    Detect yawns using MAR and mouth-open duration.

    A yawn is detected when:
        MAR > threshold
        continuously for at least the minimum duration.
    """

    def __init__(
        self,
        mar_threshold=0.50,
        minimum_duration=1.0
    ):
        self.mar_threshold = mar_threshold
        self.minimum_duration = minimum_duration

        # Current mouth state
        self.mouth_open = False

        # Time when current mouth opening started
        self.yawn_start_time = None

        # Total number of detected yawns
        self.yawn_count = 0

        # Prevent counting the same yawn repeatedly
        self.yawn_already_counted = False

    def update(self, mar, current_time):
        """
        Process one MAR value.

        Returns:
            mouth_open
            yawn_duration
            yawn_detected
            yawn_count
        """

        # ----------------------------------------------------
        # Mouth is open
        # ----------------------------------------------------

        if mar > self.mar_threshold:

            # First frame where mouth becomes open
            if not self.mouth_open:

                self.mouth_open = True
                self.yawn_start_time = current_time
                self.yawn_already_counted = False

            # Calculate how long mouth has remained open
            yawn_duration = (
                current_time - self.yawn_start_time
            )

        # ----------------------------------------------------
        # Mouth is closed
        # ----------------------------------------------------

        else:

            self.mouth_open = False
            self.yawn_start_time = None
            self.yawn_already_counted = False

            yawn_duration = 0.0

        # ----------------------------------------------------
        # Check whether this opening lasted long enough
        # ----------------------------------------------------

        yawn_detected = False

        if (
            self.mouth_open
            and yawn_duration >= self.minimum_duration
            and not self.yawn_already_counted
        ):

            self.yawn_count += 1
            yawn_detected = True
            self.yawn_already_counted = True

        return (
            self.mouth_open,
            yawn_duration,
            yawn_detected,
            self.yawn_count
        )


class BlinkDetector:
    """
    Detect blinks and measure continuous eye-closure duration.

    Blink:
        OPEN → CLOSED → OPEN

    Prolonged closure:
        OPEN → CLOSED → ... → OPEN
                    ↑
             measure duration
    """

    def __init__(
        self,
        ear_threshold=0.21,
        prolonged_closure_duration=1.0
    ):
        self.ear_threshold = ear_threshold

        # Duration after which closure is considered prolonged
        self.prolonged_closure_duration = prolonged_closure_duration

        # Current eye state
        self.eye_closed = False

        # Total completed blinks
        self.blink_count = 0

        # Time when the current eye closure started
        self.closure_start_time = None

        # Store timestamps of recent blinks
        self.blink_timestamps = deque()

        # Rolling window = 60 seconds
        self.blink_rate_window = 60.0

    def update(self, ear, current_time):
        """
        Process one EAR measurement.

        Parameters:
            ear:
                Current average EAR.

            current_time:
                Current time from time.monotonic().

        Returns:
            eye_closed
            blink_count
            closure_duration
            prolonged_closure
        """

        # ----------------------------------------------------
        # CASE 1: Eyes are closed
        # ----------------------------------------------------

        if ear < self.ear_threshold:

            # This is the first frame of the closure
            if not self.eye_closed:

                self.eye_closed = True

                # Start the closure timer
                self.closure_start_time = current_time

            # Calculate how long eyes have remained closed
            closure_duration = (
                current_time - self.closure_start_time
            )

        # ----------------------------------------------------
        # CASE 2: Eyes are open
        # ----------------------------------------------------

        else:

            # If eyes were previously closed,
            # the closure has just ended.
            if self.eye_closed:

                self.blink_count += 1

                # Record when this blink happened
                self.blink_timestamps.append(current_time)

            self.eye_closed = False

            # No active closure
            closure_duration = 0.0

            # Reset start time
            self.closure_start_time = None

        # ----------------------------------------------------
        # Determine prolonged closure
        # ----------------------------------------------------

        prolonged_closure = (
            self.eye_closed
            and closure_duration >= self.prolonged_closure_duration
        )

        # Remove blink timestamps older than 60 seconds
        while (
            self.blink_timestamps
            and current_time - self.blink_timestamps[0]
            > self.blink_rate_window
        ):
            self.blink_timestamps.popleft()

        # Number of blinks in the last 60 seconds
        blink_rate = len(self.blink_timestamps)

        return (
            self.eye_closed,
            self.blink_count,
            closure_duration,
            prolonged_closure,
            blink_rate
        )


def calculate_head_pose(face_landmarks):
    """
    Estimate coarse horizontal head pose using facial landmarks.

    Returns:
        "LEFT", "CENTER", or "RIGHT"
    """

    nose = face_landmarks[1]
    left_cheek = face_landmarks[234]
    right_cheek = face_landmarks[454]

    face_width = right_cheek.x - left_cheek.x

    if face_width == 0:
        return "CENTER"

    nose_position = (nose.x - left_cheek.x) / face_width

    if nose_position < 0.40:
        return "RIGHT"
    elif nose_position > 0.60:
        return "LEFT"
    else:
        return "CENTER"
