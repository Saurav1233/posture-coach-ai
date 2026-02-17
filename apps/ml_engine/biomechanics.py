"""
Biomechanical Geometry Utilities
=================================
Pure-function helpers for computing joint angles, alignment angles,
and symmetry metrics from normalized MediaPipe landmark arrays.

All functions accept np.ndarray slices from the (33, 3) normalized
landmark matrix. They return scalar float values in degrees unless
noted otherwise.

Mathematical Conventions
─────────────────────────
Joint angle at B (vertex) given three points A-B-C:
  θ = arccos( dot(BA, BC) / (|BA| × |BC|) )

Alignment angle of segment A→B with respect to vertical (Y-axis):
  θ = arctan2(Δx, Δy)  → deviation from vertical, in degrees

Symmetry ratio: abs(left_val - right_val) / ((left_val + right_val) / 2)
  → 0 = perfect symmetry, higher = more asymmetric
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CORE ANGLE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the interior angle at vertex B formed by rays B→A and B→C.

    Parameters
    ──────────
    a, b, c : np.ndarray  shape (3,)  — 3D normalised landmark coordinates

    Returns
    ───────
    angle : float  — degrees in [0, 180]
    """
    ba = a - b
    bc = c - b

    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)

    if ba_norm < 1e-6 or bc_norm < 1e-6:
        return 0.0

    cos_theta = np.dot(ba, bc) / (ba_norm * bc_norm)
    # Clamp to [-1, 1] to guard against floating-point drift
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def joint_angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Same as joint_angle but uses only (x, y) — useful when depth (z)
    is noisy from a monocular camera.
    """
    return joint_angle(a[:2], b[:2], c[:2])


# ─────────────────────────────────────────────────────────────────────────────
# ALIGNMENT / DEVIATION ANGLES
# ─────────────────────────────────────────────────────────────────────────────

def vertical_deviation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """
    Angle (degrees) between segment A→B and the vertical axis (downward Y).

    In MediaPipe image coordinates Y increases downward, so a perfectly
    vertical segment has Δx=0 and Δy>0.

    Returns value in [0, 90] — 0 = perfectly vertical.
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return float(abs(np.degrees(np.arctan2(abs(dx), abs(dy)))))


def horizontal_deviation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """
    Angle (degrees) between segment A→B and the horizontal axis.
    Returns value in [0, 90] — 0 = perfectly horizontal.
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return float(abs(np.degrees(np.arctan2(abs(dy), abs(dx)))))


def trunk_lean_angle(shoulder_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    """
    Forward/backward lean of the trunk.
    Computed as vertical_deviation_angle of the shoulder→hip segment.
    0° = upright, larger = more lean.
    """
    return vertical_deviation_angle(shoulder_mid, hip_mid)


# ─────────────────────────────────────────────────────────────────────────────
# MIDPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the 3D midpoint between two landmarks."""
    return (a + b) / 2.0


def shoulder_mid(lm: np.ndarray, l_idx: int, r_idx: int) -> np.ndarray:
    return midpoint(lm[l_idx], lm[r_idx])


# ─────────────────────────────────────────────────────────────────────────────
# SYMMETRY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def bilateral_symmetry_ratio(left_val: float, right_val: float) -> float:
    """
    Normalised asymmetry ratio (NAR).

    Formula: |L - R| / mean(L, R)

    Interpretation
    ───────────────
    0.0 = perfect bilateral symmetry
    0.1 = 10% asymmetry between sides
    > 0.2 = notable asymmetry, likely deserves feedback

    Returns float in [0, ∞), practically bounded at ~2.0.
    """
    mean_val = (left_val + right_val) / 2.0
    if mean_val < 1e-6:
        return 0.0
    return float(abs(left_val - right_val) / mean_val)


def shoulder_level_ratio(lm: np.ndarray, l_shoulder_idx: int, r_shoulder_idx: int) -> float:
    """
    Asymmetry of shoulder heights (y-coordinate difference, normalized).
    0 = shoulders perfectly level.
    """
    dy = abs(lm[l_shoulder_idx][1] - lm[r_shoulder_idx][1])
    mean_y = (abs(lm[l_shoulder_idx][1]) + abs(lm[r_shoulder_idx][1])) / 2.0
    return float(dy / (mean_y + 1e-8))


def hip_level_ratio(lm: np.ndarray, l_hip_idx: int, r_hip_idx: int) -> float:
    """Asymmetry of hip heights."""
    dy = abs(lm[l_hip_idx][1] - lm[r_hip_idx][1])
    mean_y = (abs(lm[l_hip_idx][1]) + abs(lm[r_hip_idx][1])) / 2.0
    return float(dy / (mean_y + 1e-8))


# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_distance_3d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def euclidean_distance_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


# ─────────────────────────────────────────────────────────────────────────────
# KNEE VALGUS / VARUS PROXY
# ─────────────────────────────────────────────────────────────────────────────

def knee_valgus_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    """
    Approximates knee alignment in the frontal plane.
    Uses 2D (x, y) only — checks lateral deviation of knee from hip-ankle line.

    Returns signed angle in degrees:
      positive → knee caves inward (valgus)
      negative → knee bows outward (varus)
    """
    # Vector from hip to ankle (ideal knee direction)
    ha = ankle[:2] - hip[:2]
    # Vector from hip to knee
    hk = knee[:2] - hip[:2]

    ha_len = np.linalg.norm(ha)
    if ha_len < 1e-6:
        return 0.0

    ha_unit = ha / ha_len
    # Component of hk perpendicular to ha
    proj = np.dot(hk, ha_unit) * ha_unit
    perp = hk - proj
    # Signed lateral deviation (positive = medial/inward for right leg)
    lateral = float(perp[0])
    return float(np.degrees(np.arctan2(lateral, ha_len)))


# ─────────────────────────────────────────────────────────────────────────────
# ELBOW FLARE / SHOULDER ABDUCTION
# ─────────────────────────────────────────────────────────────────────────────

def elbow_flare_angle(shoulder: np.ndarray, elbow: np.ndarray, torso_normal: np.ndarray) -> float:
    """
    Approximate elbow flare: angle between upper-arm vector and torso sagittal normal.
    Used for push-up and shoulder press elbow alignment.
    """
    upper_arm = elbow - shoulder
    ua_norm = np.linalg.norm(upper_arm)
    tn_norm = np.linalg.norm(torso_normal)
    if ua_norm < 1e-6 or tn_norm < 1e-6:
        return 0.0
    cos_theta = np.dot(upper_arm, torso_normal) / (ua_norm * tn_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))
