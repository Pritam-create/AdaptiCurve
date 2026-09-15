"""Explainable baseline-relative thresholds for live drowsiness monitoring."""

from dataclasses import dataclass
from typing import Optional

from src.driver_session_recognition import DriverBaselines
from src.landmark_config import (
    EAR_CLOSED_RATIO,
    EAR_THRESHOLD,
    MAR_YAWN_RATIO,
    MAR_THRESHOLD,
    PERSONALIZED_PERCLOS_BASELINE_MARGIN,
    PERCLOS_DROWSINESS_THRESHOLD,
)


@dataclass(frozen=True)
class AdaptiveThresholds:
    """Detector thresholds derived from a recognized driver's baseline."""

    ear_threshold: float
    mar_threshold: float
    perclos_threshold: float
    personalized: bool
    baseline_blink_rate: Optional[float] = None
    baseline_blink_duration: Optional[float] = None
    baseline_yawn_frequency: Optional[float] = None
    baseline_head_pose_deviation: Optional[float] = None


@dataclass(frozen=True)
class DrowsinessDecision:
    """A small explainable status for the integrated live demo."""

    status: str
    reasons: tuple[str, ...]


def calculate_adaptive_thresholds(
    baselines: Optional[DriverBaselines],
) -> AdaptiveThresholds:
    """Return personalized thresholds or the existing absolute fallback values."""

    if baselines is None:
        return AdaptiveThresholds(
            ear_threshold=EAR_THRESHOLD,
            mar_threshold=MAR_THRESHOLD,
            perclos_threshold=PERCLOS_DROWSINESS_THRESHOLD,
            personalized=False,
        )

    # Relative thresholds preserve each driver's calibrated eye and mouth scale.
    ear_threshold = baselines.baseline_ear * EAR_CLOSED_RATIO
    mar_threshold = baselines.baseline_mar * MAR_YAWN_RATIO
    # Sustained closure is concerning above the driver's baseline plus a margin.
    perclos_threshold = max(
        PERCLOS_DROWSINESS_THRESHOLD,
        baselines.baseline_perclos + PERSONALIZED_PERCLOS_BASELINE_MARGIN,
    )

    return AdaptiveThresholds(
        ear_threshold=ear_threshold,
        mar_threshold=mar_threshold,
        perclos_threshold=perclos_threshold,
        personalized=True,
        baseline_blink_rate=baselines.baseline_blink_rate,
        baseline_blink_duration=baselines.baseline_blink_duration,
        baseline_yawn_frequency=baselines.baseline_yawn_frequency,
        baseline_head_pose_deviation=baselines.baseline_head_pose_deviation,
    )


def is_elevated_blink_rate(
    blink_rate: float,
    baseline_blink_rate: Optional[float],
) -> bool:
    """Return whether the live rolling blink rate exceeds its baseline rule."""

    if baseline_blink_rate is None:
        return False
    return blink_rate > max(1.5 * baseline_blink_rate, baseline_blink_rate + 3.0)


def is_unusually_long_blink(
    blink_duration: float,
    baseline_blink_duration: Optional[float],
) -> bool:
    """Return whether one completed blink exceeds its personalized duration rule."""

    if baseline_blink_duration is None:
        return False
    return blink_duration > max(
        1.5 * baseline_blink_duration,
        baseline_blink_duration + 0.2,
    )


def is_elevated_yawn_frequency(
    yawn_frequency: float,
    baseline_yawn_frequency: Optional[float],
) -> bool:
    """Return whether session yawns/min exceeds a conservative baseline rule."""

    if baseline_yawn_frequency is None:
        return False
    return yawn_frequency > max(1.5 * baseline_yawn_frequency, baseline_yawn_frequency + 1.0)


def is_elevated_head_pose_deviation(
    head_pose_deviation: float,
    baseline_head_pose_deviation: Optional[float],
) -> bool:
    """Return whether informational head-pose deviation exceeds its baseline."""

    if baseline_head_pose_deviation is None:
        return False
    return head_pose_deviation > baseline_head_pose_deviation


def decide_drowsiness(
    perclos: float,
    prolonged_closure: bool,
    yawn_detected: bool,
    thresholds: AdaptiveThresholds,
    blink_rate_elevated: bool = False,
    repeated_long_blinks: bool = False,
    yawn_frequency_elevated: bool = False,
) -> DrowsinessDecision:
    """Classify current evidence without altering underlying detector behavior."""

    reasons = []
    if prolonged_closure:
        reasons.append("prolonged eye closure")
    if perclos >= thresholds.perclos_threshold:
        reasons.append("high PERCLOS")

    if reasons:
        return DrowsinessDecision("DROWSY", tuple(reasons))
    caution_reasons = []
    if yawn_detected:
        caution_reasons.append("yawn detected")
    if blink_rate_elevated:
        caution_reasons.append("elevated blink rate")
    if repeated_long_blinks:
        caution_reasons.append("repeated long blinks")
    if yawn_frequency_elevated:
        caution_reasons.append("elevated yawn frequency")
    if caution_reasons:
        return DrowsinessDecision("CAUTION", tuple(caution_reasons))
    return DrowsinessDecision("ALERT", ())
