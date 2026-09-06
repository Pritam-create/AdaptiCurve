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