import math
from collections import deque
from dataclasses import dataclass



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

        self.last_blink_completed = False
        self.last_blink_duration = 0.0

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

        self.last_blink_completed = False
        self.last_blink_duration = 0.0

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

                # Record the completed blink event
                self.last_blink_completed = True
                self.last_blink_duration = (
                    current_time - self.closure_start_time
                )

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






class PERCLOS:
    """
    Calculate rolling PERCLOS (Percentage of Eyelid Closure Over time).

    PERCLOS represents the percentage of processed frames in the
    rolling window where the eyes were classified as closed.
    """

    def __init__(self, window_seconds=60.0, minimum_frames=30):
        self.window_seconds = window_seconds
        self.minimum_frames = minimum_frames

        # Stores (timestamp, eye_closed)
        self.history = deque()

    def update(self, eye_closed, current_time):
        """
        Add the current eye state and calculate rolling PERCLOS.

        Returns:
            perclos: PERCLOS percentage
            warming_up: True if there is not enough data yet
        """


        # Add current observation
        self.history.append(
            (current_time, eye_closed)
        )

        # Remove observations older than the rolling window
        cutoff_time = current_time - self.window_seconds

        while self.history and self.history[0][0] < cutoff_time:
            self.history.popleft()

        # Not enough data yet
        if len(self.history) < self.minimum_frames:
            return 0.0, True

        # Count closed-eye frames
        closed_frames = sum(
            1 for _, is_closed in self.history
            if is_closed
        )

        # Calculate PERCLOS
        perclos = (
            closed_frames / len(self.history)
        ) * 100.0

        return perclos, False

    def reset(self):
        """Clear the rolling PERCLOS history."""
        self.history.clear()




@dataclass
class EpochSummary:
    """
    Summary statistics for one completed epoch.
    Basically final result of one epoch.
    """

    start_time: float
    end_time: float

    mean_perclos: float
    mean_blink_rate: float
    mean_blink_duration: float
    yawn_count: int
    mean_open_eye_ear: float
    head_pose_deviation_pct: float



class EpochAccumulator:
    """
    Accumulate frame-level features during one epoch.
    """

    def __init__(self, start_time, centered_head_poses=("CENTER",)):
        self.start_time = start_time
        self.centered_head_poses = centered_head_poses

        # PERCLOS
        self.perclos_sum = 0.0
        self.perclos_count = 0

        # Blinks
        self.blink_count = 0
        self.blink_durations = []

        # Yawns
        self.yawn_count = 0

        # Open-eye EAR
        self.open_eye_ear_sum = 0.0
        self.open_eye_ear_count = 0

        # Head pose
        self.deviated_frames = 0
        self.total_frames = 0

    def update(
        self,
        perclos,
        eye_closed,
        ear,
        head_pose,
        blink_completed=False,
        blink_duration=0.0,
        yawn_detected=False
    ):
        """
        Add one frame's feature values to the current epoch.
        """

        # PERCLOS
        self.perclos_sum += perclos
        self.perclos_count += 1

        # Blink event
        if blink_completed:
            self.blink_count += 1
            self.blink_durations.append(blink_duration)

        # Yawn event
        if yawn_detected:
            self.yawn_count += 1

        # Open-eye EAR
        if not eye_closed:
            self.open_eye_ear_sum += ear
            self.open_eye_ear_count += 1

        # Head pose
        self.total_frames += 1

        if head_pose not in self.centered_head_poses:
            self.deviated_frames += 1

    def finalize(self, end_time):
        """
        Create an EpochSummary from the accumulated values.
        """

        duration = end_time - self.start_time

        # Mean PERCLOS
        if self.perclos_count > 0:
            mean_perclos = (
                self.perclos_sum / self.perclos_count
            )
        else:
            mean_perclos = 0.0

        # Blink rate per minute
        if duration > 0:
            mean_blink_rate = (
                self.blink_count / duration
            ) * 60.0
        else:
            mean_blink_rate = 0.0

        # Mean blink duration
        if self.blink_durations:
            mean_blink_duration = (
                sum(self.blink_durations)
                / len(self.blink_durations)
            )
        else:
            mean_blink_duration = 0.0

        # Mean open-eye EAR
        if self.open_eye_ear_count > 0:
            mean_open_eye_ear = (
                self.open_eye_ear_sum
                / self.open_eye_ear_count
            )
        else:
            mean_open_eye_ear = 0.0

        # Head pose deviation percentage
        if self.total_frames > 0:
            head_pose_deviation_pct = (
                self.deviated_frames
                / self.total_frames
            ) * 100.0
        else:
            head_pose_deviation_pct = 0.0

        return EpochSummary(
            start_time=self.start_time,
            end_time=end_time,
            mean_perclos=mean_perclos,
            mean_blink_rate=mean_blink_rate,
            mean_blink_duration=mean_blink_duration,
            yawn_count=self.yawn_count,
            mean_open_eye_ear=mean_open_eye_ear,
            head_pose_deviation_pct=head_pose_deviation_pct
        )
