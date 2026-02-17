"""
Shoulder Press Feature Engineering (Fixed)
===========================================
Fixed shoulder abduction calculation to work even when hips are not visible.
Fixed elbow angle to use correct landmark ordering.
"""

from __future__ import annotations
import numpy as np
from apps.ml_engine.biomechanics import (
    joint_angle_2d, vertical_deviation_angle, bilateral_symmetry_ratio,
    midpoint
)
from apps.ml_engine.pose_extractor import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    NOSE,
)

EXERCISE_NAME = "shoulder_press"

FEATURE_NAMES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_abduction",
    "right_shoulder_abduction",
    "trunk_lean_angle",
    "left_wrist_over_elbow",
    "right_wrist_over_elbow",
    "elbow_symmetry_ratio",
    "shoulder_symmetry_ratio",
    "head_forward_lean",
]

FEEDBACK_RULES = {
    "left_elbow_angle": {
        "ideal_range": (80, 185),
        "messages": {
            "too_high": "Press your left arm higher — full extension at the top",
            "too_low":  "Lower your left arm to starting position before pressing",
        }
    },
    "right_elbow_angle": {
        "ideal_range": (80, 185),
        "messages": {
            "too_high": "Press your right arm higher — full extension at the top",
            "too_low":  "Lower your right arm to starting position before pressing",
        }
    },
    "trunk_lean_angle": {
        "ideal_range": (0, 20),
        "messages": {
            "too_high": "You're leaning back — engage your core to stay upright",
        }
    },
    "left_wrist_over_elbow": {
        "ideal_range": (0, 25),
        "messages": {
            "too_high": "Left wrist drifting — keep wrist directly above elbow",
        }
    },
    "right_wrist_over_elbow": {
        "ideal_range": (0, 25),
        "messages": {
            "too_high": "Right wrist drifting — keep wrist directly above elbow",
        }
    },
    "elbow_symmetry_ratio": {
        "ideal_range": (0, 0.15),
        "messages": {
            "too_high": "Uneven press — both arms should extend equally",
        }
    },
    "left_shoulder_abduction": {
        "ideal_range": (60, 120),
        "messages": {
            "too_high": "Left elbow too wide — bring it slightly forward",
            "too_low":  "Left elbow too close — flare it to shoulder level",
        }
    },
    "right_shoulder_abduction": {
        "ideal_range": (60, 120),
        "messages": {
            "too_high": "Right elbow too wide — bring it slightly forward",
            "too_low":  "Right elbow too close — flare it to shoulder level",
        }
    },
}


def extract_features(lm: np.ndarray) -> dict:
    """Compute all biomechanical features for one shoulder press frame."""

    # ── Elbow angles (shoulder → elbow → wrist) ──────────────────────────
    left_elbow_angle  = joint_angle_2d(lm[LEFT_SHOULDER],  lm[LEFT_ELBOW],  lm[LEFT_WRIST])
    right_elbow_angle = joint_angle_2d(lm[RIGHT_SHOULDER], lm[RIGHT_ELBOW], lm[RIGHT_WRIST])

    # ── Shoulder abduction ────────────────────────────────────────────────
    # Use shoulder-to-shoulder line as reference instead of hips
    # This works even when lower body is not visible
    shoulder_mid = midpoint(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER])

    # Shoulder abduction = angle at shoulder between elbow and torso center
    left_shoulder_abduction  = joint_angle_2d(lm[LEFT_ELBOW],  lm[LEFT_SHOULDER],  shoulder_mid)
    right_shoulder_abduction = joint_angle_2d(lm[RIGHT_ELBOW], lm[RIGHT_SHOULDER], shoulder_mid)

    # ── Trunk lean ────────────────────────────────────────────────────────
    hip_mid = midpoint(lm[LEFT_HIP], lm[RIGHT_HIP])
    hip_visible = (
        float(np.linalg.norm(lm[LEFT_HIP][:2]))  > 0.01 and
        float(np.linalg.norm(lm[RIGHT_HIP][:2])) > 0.01
    )
    if hip_visible:
        trunk_lean = vertical_deviation_angle(shoulder_mid, hip_mid)
    else:
        trunk_lean = 5.0  # assume neutral if hips not visible

    # ── Wrist over elbow (vertical alignment) ─────────────────────────────
    shoulder_width = float(np.linalg.norm(
        lm[LEFT_SHOULDER][:2] - lm[RIGHT_SHOULDER][:2]
    )) + 1e-8
    l_wrist_drift = abs(float(lm[LEFT_WRIST][0]  - lm[LEFT_ELBOW][0]))
    r_wrist_drift = abs(float(lm[RIGHT_WRIST][0] - lm[RIGHT_ELBOW][0]))
    left_wrist_over_elbow  = (l_wrist_drift / shoulder_width) * 45.0
    right_wrist_over_elbow = (r_wrist_drift / shoulder_width) * 45.0

    # ── Symmetry ──────────────────────────────────────────────────────────
    elbow_sym    = bilateral_symmetry_ratio(left_elbow_angle,         right_elbow_angle)
    shoulder_sym = bilateral_symmetry_ratio(left_shoulder_abduction,  right_shoulder_abduction)

    # ── Head forward lean ─────────────────────────────────────────────────
    head_forward = vertical_deviation_angle(lm[NOSE], shoulder_mid)

    return {
        "left_elbow_angle":        left_elbow_angle,
        "right_elbow_angle":       right_elbow_angle,
        "left_shoulder_abduction":  left_shoulder_abduction,
        "right_shoulder_abduction": right_shoulder_abduction,
        "trunk_lean_angle":         trunk_lean,
        "left_wrist_over_elbow":    left_wrist_over_elbow,
        "right_wrist_over_elbow":   right_wrist_over_elbow,
        "elbow_symmetry_ratio":     elbow_sym,
        "shoulder_symmetry_ratio":  shoulder_sym,
        "head_forward_lean":        head_forward,
    }