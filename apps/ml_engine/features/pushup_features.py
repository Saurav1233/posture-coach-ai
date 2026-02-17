"""
Push-Up Feature Engineering
=============================
Biomechanical feature set for the standard Push-Up exercise.

Academic Rationale
──────────────────
Push-up mechanics focus on (Youdas et al. 2010; Calatayud et al. 2015):

1. Elbow Flexion Angle       – primary ROM indicator; ~90° at bottom
2. Body Alignment (plank)    – hip sag / pike; shoulder–hip–ankle linearity
3. Shoulder Width            – hand placement alignment
4. Head/Neck Alignment       – cervical spine neutral position
5. Elbow Flare Angle         – optimal elbow track (30–45° from torso)
6. Hip Sag Angle             – measure of core failure

The push-up plank position demands that shoulder–hip–ankle form a nearly
straight line. We measure the deviation of the hip from this line as the
primary alignment indicator.
"""

from __future__ import annotations
import numpy as np
from apps.ml_engine.biomechanics import (
    joint_angle_2d, vertical_deviation_angle, bilateral_symmetry_ratio,
    midpoint, horizontal_deviation_angle
)
from apps.ml_engine.pose_extractor import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    NOSE,
)

EXERCISE_NAME = "pushup"

FEATURE_NAMES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "body_alignment_angle",    # shoulder-hip-ankle linearity
    "hip_sag_deviation",       # how far hip deviates from shoulder-ankle line
    "head_alignment_angle",    # nose vs shoulder alignment
    "elbow_symmetry_ratio",
    "shoulder_width_ratio",    # wrist span / shoulder span (form check)
    "left_shoulder_angle",     # shoulder flexion
    "right_shoulder_angle",
]

FEEDBACK_RULES = {
    "left_elbow_angle": {
        "ideal_range": (70, 100),
        "messages": {
            "too_high": "Lower yourself further — your left elbow isn't at 90°",
            "too_low":  "Left elbow over-bent — press back up",
        }
    },
    "right_elbow_angle": {
        "ideal_range": (70, 100),
        "messages": {
            "too_high": "Lower yourself further — your right elbow isn't at 90°",
            "too_low":  "Right elbow over-bent — press back up",
        }
    },
    "body_alignment_angle": {
        "ideal_range": (0, 15),
        "messages": {
            "too_high": "Keep your body straight — avoid sagging hips or piking",
        }
    },
    "hip_sag_deviation": {
        "ideal_range": (0, 0.15),
        "messages": {
            "too_high": "Engage your core — your hips are dropping or rising",
        }
    },
    "head_alignment_angle": {
        "ideal_range": (0, 20),
        "messages": {
            "too_high": "Keep your head neutral — don't crane your neck up or tuck chin",
        }
    },
    "elbow_symmetry_ratio": {
        "ideal_range": (0, 0.15),
        "messages": {
            "too_high": "Uneven arm bend — check that both elbows flex equally",
        }
    },
}


def extract_features(lm: np.ndarray) -> dict[str, float]:
    """
    Compute all biomechanical features for one push-up frame.
    """
    # ── Elbow angles ──────────────────────────────────────────────────────
    left_elbow_angle = joint_angle_2d(
        lm[LEFT_SHOULDER], lm[LEFT_ELBOW], lm[LEFT_WRIST]
    )
    right_elbow_angle = joint_angle_2d(
        lm[RIGHT_SHOULDER], lm[RIGHT_ELBOW], lm[RIGHT_WRIST]
    )

    # ── Shoulder angles ───────────────────────────────────────────────────
    left_shoulder_angle = joint_angle_2d(
        lm[LEFT_ELBOW], lm[LEFT_SHOULDER], lm[LEFT_HIP]
    )
    right_shoulder_angle = joint_angle_2d(
        lm[RIGHT_ELBOW], lm[RIGHT_SHOULDER], lm[RIGHT_HIP]
    )

    # ── Body alignment (shoulder→hip→ankle) ───────────────────────────────
    shoulder_mid = midpoint(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER])
    hip_mid = midpoint(lm[LEFT_HIP], lm[RIGHT_HIP])
    ankle_mid = midpoint(lm[LEFT_ANKLE], lm[RIGHT_ANKLE])
    body_alignment = joint_angle_2d(shoulder_mid, hip_mid, ankle_mid)
    # 180° = perfect straight line; deviation = 180 - angle
    body_alignment_dev = abs(180.0 - body_alignment)

    # ── Hip sag: perpendicular distance from hip to shoulder-ankle line ───
    sa_vec = ankle_mid[:2] - shoulder_mid[:2]
    sa_len = np.linalg.norm(sa_vec)
    if sa_len > 1e-6:
        sa_unit = sa_vec / sa_len
        sh_vec = hip_mid[:2] - shoulder_mid[:2]
        proj = np.dot(sh_vec, sa_unit)
        perp_vec = sh_vec - proj * sa_unit
        hip_sag = float(np.linalg.norm(perp_vec))
    else:
        hip_sag = 0.0

    # ── Head/neck alignment ───────────────────────────────────────────────
    head_alignment = vertical_deviation_angle(lm[NOSE], shoulder_mid)

    # ── Symmetry ──────────────────────────────────────────────────────────
    elbow_sym = bilateral_symmetry_ratio(left_elbow_angle, right_elbow_angle)

    # ── Hand width vs shoulder width ──────────────────────────────────────
    shoulder_span = np.linalg.norm(lm[LEFT_SHOULDER][:2] - lm[RIGHT_SHOULDER][:2])
    wrist_span = np.linalg.norm(lm[LEFT_WRIST][:2] - lm[RIGHT_WRIST][:2])
    shoulder_width_ratio = float(wrist_span / (shoulder_span + 1e-8))

    return {
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "body_alignment_angle": body_alignment_dev,
        "hip_sag_deviation": hip_sag,
        "head_alignment_angle": head_alignment,
        "elbow_symmetry_ratio": elbow_sym,
        "shoulder_width_ratio": shoulder_width_ratio,
        "left_shoulder_angle": left_shoulder_angle,
        "right_shoulder_angle": right_shoulder_angle,
    }
