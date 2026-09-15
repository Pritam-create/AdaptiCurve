# ============================================================
# MediaPipe Face Landmarker Configuration
# ============================================================

# ------------------------------------------------------------
# Eye landmarks used for EAR
#
# Order:
#
#        p2       p3
#         ↓       ↓
#      ┌───────────┐
#   p1 →             ← p4
#      └───────────┘
#         ↑       ↑
#        p6       p5
#
# EAR = (distance(p2,p6) + distance(p3,p5))
#       / (2 * distance(p1,p4))
# ------------------------------------------------------------

LEFT_EYE_EAR = [
    362, 385, 387, 263, 373, 380
]

RIGHT_EYE_EAR = [
    33, 160, 158, 133, 153, 144
]


# ------------------------------------------------------------
# Mouth landmarks used for MAR
#
# Horizontal:
#   left corner  →  right corner
#
# Vertical:
#   upper lip    ↕    lower lip
# ------------------------------------------------------------

MOUTH_HORIZONTAL = [
    61,    # left mouth corner
    291    # right mouth corner
]

MOUTH_VERTICAL_1 = [
    13,    # upper inner lip
    14     # lower inner lip
]

MOUTH_VERTICAL_2 = [
    82,    # upper/lower mouth region
    87     # corresponding lower region
]


# ------------------------------------------------------------
# Detection thresholds
#
# These are INITIAL values.
# We will calibrate them using your own webcam data.
# ------------------------------------------------------------

EAR_THRESHOLD = 0.21

MAR_THRESHOLD = 0.50

# Day 5 personalized thresholds are relative to each driver's calibrated
# baseline. Absolute EAR/MAR thresholds above remain the fallback when identity
# cannot be confirmed.
EAR_CLOSED_RATIO = 0.75
# A calibrated MAR near 0.028 therefore maps to about 0.28, close to the
# existing personalized yawn threshold while avoiding an implausibly tiny value.
MAR_YAWN_RATIO = 10.0
PERCLOS_DROWSINESS_THRESHOLD = 20.0
PERSONALIZED_PERCLOS_BASELINE_MARGIN = 10.0


# ------------------------------------------------------------
# Temporal thresholds
# ------------------------------------------------------------

# Minimum duration for a prolonged eye closure
EYE_CLOSURE_MIN_DURATION = 1.0  # seconds

# Minimum duration for a yawn
YAWN_MIN_DURATION = 1.0  # seconds


# ------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------

SMOOTHING_WINDOW = 5


# ------------------------------------------------------------
# Blink-rate calculation
# ------------------------------------------------------------

BLINK_RATE_WINDOW = 60.0  # seconds



# ============================================================
# Day 3 - PERCLOS and Epoch Configuration
# ============================================================

# Rolling window used for PERCLOS calculation.
# PERCLOS measures the percentage of time the eyes are closed
# within this rolling time window.
PERCLOS_WINDOW_SECONDS = 60.0

# Length of one epoch for aggregating features.
# 30 seconds is intentionally short for live demonstration.
# A real deployment would typically use a much longer window
# such as 15-30 minutes.
EPOCH_DURATION_SECONDS = 30.0

# Length of the standalone Day 4 driver enrollment calibration window.
CALIBRATION_DURATION_SECONDS = 10.0

# Head pose is considered centered when the detected pose
# remains within this range.
CENTERED_HEAD_POSES = ["CENTER"]
