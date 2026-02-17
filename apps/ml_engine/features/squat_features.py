"""
Squat Feature Engineering
==========================
Biomechanical feature set for the Barbell Back Squat / Bodyweight Squat.

Academic Rationale
──────────────────
Squats are assessed on five primary axes in sports biomechanics literature
(Myer et al. 2014; Hewett et al. 2005):

1. Knee Flexion Angle        – primary depth indicator; target ~90°
2. Hip Flexion Angle         – hip hinge quality; parallel to knee angle
3. Trunk Lean Angle          – forward lean < 45° for healthy mechanics
4. Knee Valgus Angle         – frontal plane collapse indicator
5. Knee-over-Toe Alignment   – sagittal plane knee position
6. Bilateral Symmetry        – left/right knee angle difference

All features are computed from normalized landmarks so they are
scale-independent.
"""

from __future__ import annotations
import numpy as np
from apps.ml_engine.biomechanics import (
    joint_angle_2d, vertical_deviation_angle, knee_valgus_angle,
    bilateral_symmetry_ratio, midpoint
)
from apps.ml_engine.pose_extractor import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
)

EXERCISE_NAME = "squat"

# Feature names — must match JSON profile keys exactly
FEATURE_NAMES = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "trunk_lean_angle",
    "left_knee_valgus",
    "right_knee_valgus",
    "knee_symmetry_ratio",
    "hip_symmetry_ratio",
    "knee_depth_score",      # derived: min(left, right) knee angle proximity to 90°
]

# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK THRESHOLDS  (used by posture scorer for explainable feedback)
# ─────────────────────────────────────────────────────────────────────────────
FEEDBACK_RULES = {
    "left_knee_angle": {
        "ideal_range": (80, 110),
        "messages": {
            "too_high": "Squat deeper — your left knee isn't reaching full depth",
            "too_low": "Left knee over-flexed — reduce range slightly",
        }
    },
    "right_knee_angle": {
        "ideal_range": (80, 110),
        "messages": {
            "too_high": "Squat deeper — your right knee isn't reaching full depth",
            "too_low": "Right knee over-flexed — reduce range slightly",
        }
    },
    "trunk_lean_angle": {
        "ideal_range": (0, 45),
        "messages": {
            "too_high": "Reduce forward lean — keep your chest up",
        }
    },
    "left_knee_valgus": {
        "ideal_range": (-10, 10),
        "messages": {
            "too_high": "Left knee caving inward — push your knees out over your toes",
            "too_low": "Left knee bowing outward — realign with your toes",
        }
    },
    "right_knee_valgus": {
        "ideal_range": (-10, 10),
        "messages": {
            "too_high": "Right knee caving inward — push your knees out over your toes",
            "too_low": "Right knee bowing outward — realign with your toes",
        }
    },
    "knee_symmetry_ratio": {
        "ideal_range": (0, 0.15),
        "messages": {
            "too_high": "Uneven squat depth — distribute weight equally on both legs",
        }
    },
}


def extract_features(lm: np.ndarray) -> dict[str, float]:
    """
    Compute all biomechanical features for one frame.

    Parameters
    ──────────
    lm : np.ndarray  shape (33, 3)  — normalised landmark coordinates

    Returns
    ───────
    dict mapping feature_name → float value
    """
    # ── Joint angles ──────────────────────────────────────────────────────
    left_knee_angle = joint_angle_2d(
        lm[LEFT_HIP], lm[LEFT_KNEE], lm[LEFT_ANKLE]
    )
    right_knee_angle = joint_angle_2d(
        lm[RIGHT_HIP], lm[RIGHT_KNEE], lm[RIGHT_ANKLE]
    )
    left_hip_angle = joint_angle_2d(
        lm[LEFT_SHOULDER], lm[LEFT_HIP], lm[LEFT_KNEE]
    )
    right_hip_angle = joint_angle_2d(
        lm[RIGHT_SHOULDER], lm[RIGHT_HIP], lm[RIGHT_KNEE]
    )

    # ── Trunk lean ────────────────────────────────────────────────────────
    shoulder_center = midpoint(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER])
    hip_center = midpoint(lm[LEFT_HIP], lm[RIGHT_HIP])
    trunk_lean = vertical_deviation_angle(shoulder_center, hip_center)

    # ── Knee valgus/varus ─────────────────────────────────────────────────
    l_valgus = knee_valgus_angle(lm[LEFT_HIP], lm[LEFT_KNEE], lm[LEFT_ANKLE])
    r_valgus = knee_valgus_angle(lm[RIGHT_HIP], lm[RIGHT_KNEE], lm[RIGHT_ANKLE])

    # ── Symmetry ──────────────────────────────────────────────────────────
    knee_sym = bilateral_symmetry_ratio(left_knee_angle, right_knee_angle)
    hip_sym = bilateral_symmetry_ratio(left_hip_angle, right_hip_angle)

    # ── Depth score (proximity to 90° at knee) ────────────────────────────
    min_knee = min(left_knee_angle, right_knee_angle)
    knee_depth = abs(min_knee - 90.0)   # 0 = perfect depth

    return {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "trunk_lean_angle": trunk_lean,
        "left_knee_valgus": l_valgus,
        "right_knee_valgus": r_valgus,
        "knee_symmetry_ratio": knee_sym,
        "hip_symmetry_ratio": hip_sym,
        "knee_depth_score": knee_depth,
    }
