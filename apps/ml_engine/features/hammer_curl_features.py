"""
Hammer Curl Feature Engineering
=================================
Biomechanical feature set for the Hammer Curl (neutral grip dumbbell curl).

Academic Rationale
──────────────────
Hammer curls target the brachialis and brachioradialis with a neutral grip.
Key differences from standard barbell curl (Marcolin et al. 2018):

1. Forearm should remain NEUTRAL (thumb-up) throughout ROM
   - Tracked via vertical alignment of forearm axis
2. Unilateral execution is common (alternating or simultaneous)
3. Elbow must stay fixed at the side

We share most metrics with barbell_curl but track
forearm pronation/supination differently via wrist-to-elbow vector.
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

EXERCISE_NAME = "hammer_curl"

FEATURE_NAMES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "trunk_lean_angle",
    "left_upper_arm_vertical",
    "right_upper_arm_vertical",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "elbow_symmetry_ratio",
    "left_forearm_neutral",     # forearm planarity with vertical (neutral grip check)
    "right_forearm_neutral",
]

FEEDBACK_RULES = {
    "left_elbow_angle": {
        "ideal_range": (25, 160),
        "messages": {
            "too_high": "Hammer curl your left arm higher — bring it fully up",
            "too_low":  "Left arm too bent — control the descent",
        }
    },
    "right_elbow_angle": {
        "ideal_range": (25, 160),
        "messages": {
            "too_high": "Hammer curl your right arm higher — bring it fully up",
            "too_low":  "Right arm too bent — control the descent",
        }
    },
    "trunk_lean_angle": {
        "ideal_range": (0, 15),
        "messages": {
            "too_high": "Lean back detected — keep your torso stationary and upright",
        }
    },
    "left_upper_arm_vertical": {
        "ideal_range": (0, 20),
        "messages": {
            "too_high": "Left elbow swinging — keep it pinned to your side",
        }
    },
    "right_upper_arm_vertical": {
        "ideal_range": (0, 20),
        "messages": {
            "too_high": "Right elbow swinging — keep it pinned to your side",
        }
    },
    "elbow_symmetry_ratio": {
        "ideal_range": (0, 0.15),
        "messages": {
            "too_high": "Uneven hammer curl — both arms should move symmetrically",
        }
    },
}


def extract_features(lm: np.ndarray) -> dict[str, float]:
    """Compute all biomechanical features for one hammer curl frame."""
    left_elbow_angle = joint_angle_2d(
        lm[LEFT_SHOULDER], lm[LEFT_ELBOW], lm[LEFT_WRIST]
    )
    right_elbow_angle = joint_angle_2d(
        lm[RIGHT_SHOULDER], lm[RIGHT_ELBOW], lm[RIGHT_WRIST]
    )
    left_shoulder_angle = joint_angle_2d(
        lm[LEFT_ELBOW], lm[LEFT_SHOULDER], lm[LEFT_HIP]
    )
    right_shoulder_angle = joint_angle_2d(
        lm[RIGHT_ELBOW], lm[RIGHT_SHOULDER], lm[RIGHT_HIP]
    )
    shoulder_mid = midpoint(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER])
    hip_mid = midpoint(lm[LEFT_HIP], lm[RIGHT_HIP])
    trunk_lean = vertical_deviation_angle(shoulder_mid, hip_mid)
    left_upper_arm_vert = vertical_deviation_angle(lm[LEFT_SHOULDER], lm[LEFT_ELBOW])
    right_upper_arm_vert = vertical_deviation_angle(lm[RIGHT_SHOULDER], lm[RIGHT_ELBOW])
    elbow_sym = bilateral_symmetry_ratio(left_elbow_angle, right_elbow_angle)

    # Neutral grip proxy: forearm should travel in sagittal plane
    # Approximate by measuring z-component magnitude of wrist-elbow vector
    left_forearm_vec = lm[LEFT_WRIST] - lm[LEFT_ELBOW]
    right_forearm_vec = lm[RIGHT_WRIST] - lm[RIGHT_ELBOW]
    lf_norm = np.linalg.norm(left_forearm_vec)
    rf_norm = np.linalg.norm(right_forearm_vec)
    left_forearm_neutral = float(abs(left_forearm_vec[2]) / (lf_norm + 1e-8)) * 90.0
    right_forearm_neutral = float(abs(right_forearm_vec[2]) / (rf_norm + 1e-8)) * 90.0

    return {
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "trunk_lean_angle": trunk_lean,
        "left_upper_arm_vertical": left_upper_arm_vert,
        "right_upper_arm_vertical": right_upper_arm_vert,
        "left_shoulder_angle": left_shoulder_angle,
        "right_shoulder_angle": right_shoulder_angle,
        "elbow_symmetry_ratio": elbow_sym,
        "left_forearm_neutral": left_forearm_neutral,
        "right_forearm_neutral": right_forearm_neutral,
    }
