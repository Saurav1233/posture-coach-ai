"""
Barbell Curl Feature Engineering
==================================
Biomechanical feature set for the Standing Barbell Bicep Curl.

Academic Rationale
──────────────────
Barbell curl mechanics focus on (Oliveira et al. 2009; Marcolin et al. 2018):

1. Elbow Flexion Angle        – primary ROM; 30°–150° full arc
2. Shoulder Stability         – elbow must stay at side (no shoulder swing)
3. Wrist Alignment            – wrists should stay neutral (no flexion/extension)
4. Trunk Uprightness          – no backward lean for momentum
5. Elbow Symmetry             – equal bilateral ROM
6. Upper Arm Vertical Angle   – elbow drift indicator (should stay vertical)

The most common barbell curl error is using shoulder flexion (swinging elbows
forward) to lift heavier weight. We detect this as an increase in shoulder
flexion angle combined with a change in elbow position relative to torso.
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
)

EXERCISE_NAME = "barbell_curl"

FEATURE_NAMES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "trunk_lean_angle",
    "left_upper_arm_vertical",   # upper arm deviation from vertical
    "right_upper_arm_vertical",
    "left_shoulder_angle",       # shoulder flexion (elbow swing indicator)
    "right_shoulder_angle",
    "elbow_symmetry_ratio",
    "wrist_alignment_left",      # wrist deviation (flexion/extension)
    "wrist_alignment_right",
]

FEEDBACK_RULES = {
    "left_elbow_angle": {
        "ideal_range": (25, 160),
        "messages": {
            "too_high": "Curl your left arm higher — full ROM to the top",
            "too_low":  "Left arm over-curled — lower slightly",
        }
    },
    "right_elbow_angle": {
        "ideal_range": (25, 160),
        "messages": {
            "too_high": "Curl your right arm higher — full ROM to the top",
            "too_low":  "Right arm over-curled — lower slightly",
        }
    },
    "trunk_lean_angle": {
        "ideal_range": (0, 15),
        "messages": {
            "too_high": "You're leaning back for momentum — keep your torso upright",
        }
    },
    "left_upper_arm_vertical": {
        "ideal_range": (0, 20),
        "messages": {
            "too_high": "Left elbow drifting forward — pin your elbows to your sides",
        }
    },
    "right_upper_arm_vertical": {
        "ideal_range": (0, 20),
        "messages": {
            "too_high": "Right elbow drifting forward — pin your elbows to your sides",
        }
    },
    "elbow_symmetry_ratio": {
        "ideal_range": (0, 0.12),
        "messages": {
            "too_high": "Uneven curl — both arms should lift together symmetrically",
        }
    },
}


def extract_features(lm: np.ndarray) -> dict[str, float]:
    """
    Compute all biomechanical features for one barbell curl frame.
    """
    # ── Primary elbow angles ──────────────────────────────────────────────
    left_elbow_angle = joint_angle_2d(
        lm[LEFT_SHOULDER], lm[LEFT_ELBOW], lm[LEFT_WRIST]
    )
    right_elbow_angle = joint_angle_2d(
        lm[RIGHT_SHOULDER], lm[RIGHT_ELBOW], lm[RIGHT_WRIST]
    )

    # ── Shoulder angle (elbow swing detector) ─────────────────────────────
    left_shoulder_angle = joint_angle_2d(
        lm[LEFT_ELBOW], lm[LEFT_SHOULDER], lm[LEFT_HIP]
    )
    right_shoulder_angle = joint_angle_2d(
        lm[RIGHT_ELBOW], lm[RIGHT_SHOULDER], lm[RIGHT_HIP]
    )

    # ── Trunk lean ────────────────────────────────────────────────────────
    shoulder_mid = midpoint(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER])
    hip_mid = midpoint(lm[LEFT_HIP], lm[RIGHT_HIP])
    trunk_lean = vertical_deviation_angle(shoulder_mid, hip_mid)

    # ── Upper arm vertical angle (should stay near vertical during curl) ──
    left_upper_arm_vert = vertical_deviation_angle(lm[LEFT_SHOULDER], lm[LEFT_ELBOW])
    right_upper_arm_vert = vertical_deviation_angle(lm[RIGHT_SHOULDER], lm[RIGHT_ELBOW])

    # ── Elbow symmetry ────────────────────────────────────────────────────
    elbow_sym = bilateral_symmetry_ratio(left_elbow_angle, right_elbow_angle)

    # ── Wrist alignment (deviation from forearm line) ─────────────────────
    # Approximate: angle at wrist joint (elbow-wrist-MCP not available,
    # so we use elbow→wrist vertical deviation as wrist extension proxy)
    wrist_align_left = vertical_deviation_angle(lm[LEFT_ELBOW], lm[LEFT_WRIST])
    wrist_align_right = vertical_deviation_angle(lm[RIGHT_ELBOW], lm[RIGHT_WRIST])

    return {
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "trunk_lean_angle": trunk_lean,
        "left_upper_arm_vertical": left_upper_arm_vert,
        "right_upper_arm_vertical": right_upper_arm_vert,
        "left_shoulder_angle": left_shoulder_angle,
        "right_shoulder_angle": right_shoulder_angle,
        "elbow_symmetry_ratio": elbow_sym,
        "wrist_alignment_left": wrist_align_left,
        "wrist_alignment_right": wrist_align_right,
    }
