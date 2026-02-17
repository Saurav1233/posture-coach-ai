"""
Posture Scorer — Deviation-Based One-Class Assessment
======================================================

Mathematical Foundation
────────────────────────
Since ONLY correct posture data is available, we use a statistical
posture profiling approach grounded in robust statistics theory.

PROFILE CONSTRUCTION (Training Phase)
═══════════════════════════════════════
For each exercise, we compute:
  • μ_i  = mean of feature_i across all correct-posture frames
  • σ_i  = standard deviation of feature_i across all correct-posture frames

This defines a multi-dimensional Gaussian envelope of correct posture.

DEVIATION SCORE (Inference Phase)
═══════════════════════════════════
For each feature_i at inference time:

  z_i = (x_i - μ_i) / σ_i              [z-score / standard score]

  clamped_z_i = min(|z_i|, Z_MAX)       [Z_MAX = 3.0 by default]

  feature_score_i = 100 × (1 - clamped_z_i / Z_MAX)

  global_score = 100 × (1 - Σ(w_i × clamped_z_i / Z_MAX) / Σ(w_i))

Where w_i are per-feature weights reflecting biomechanical importance.

MATHEMATICAL PROPERTIES
• At μ_i → score = 100 (on distribution center)
• At μ_i ± 1σ → z = 1.0 → score = 100 × (1 - 1/3) ≈ 67
• At μ_i ± 2σ → z = 2.0 → score = 100 × (1 - 2/3) ≈ 33
• At μ_i ± 3σ → z = 3.0 → score = 0 (outside 99.7% of training data)

ACADEMIC JUSTIFICATION
This is equivalent to a one-class Gaussian classifier operating in
feature space, analogous to:
• One-Class SVM (Schölkopf et al. 1999)
• Isolation Forest (Liu et al. 2008)
• Gaussian Envelope Model (Hodge & Austin 2004)

The key advantage over binary classification is that it requires only
positive (correct) examples and provides continuous, interpretable
deviation scores rather than discrete labels.

VALIDATION WITHOUT INCORRECT SAMPLES
────────────────────────────────────────
Robustness is validated by:
1. Leave-One-Video-Out cross-validation on correct data
   → Mean/std should remain stable across folds
2. Synthetic perturbation testing: artificially perturb features by
   known amounts and verify score decreases proportionally
3. Expert annotation: physio expert reviews borderline (score 40-60) cases
4. Calibration check: score histograms on training data should cluster at 75-100
"""

from __future__ import annotations
import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger('posture_coach')

# Maximum z-score before feature is considered fully incorrect
Z_MAX = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT FEATURE WEIGHTS  (per exercise, biomechanically motivated)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "squat": {
        "left_knee_angle":       3.0,   # Primary ROM — high weight
        "right_knee_angle":      3.0,
        "left_hip_angle":        2.0,
        "right_hip_angle":       2.0,
        "trunk_lean_angle":      2.5,   # Injury risk — high weight
        "left_knee_valgus":      3.0,   # Injury risk
        "right_knee_valgus":     3.0,
        "knee_symmetry_ratio":   1.5,
        "hip_symmetry_ratio":    1.0,
        "knee_depth_score":      2.0,
    },
    "pushup": {
        "left_elbow_angle":      3.0,
        "right_elbow_angle":     3.0,
        "body_alignment_angle":  3.5,   # Core integrity — critical
        "hip_sag_deviation":     3.5,
        "head_alignment_angle":  1.5,
        "elbow_symmetry_ratio":  1.5,
        "shoulder_width_ratio":  1.0,
        "left_shoulder_angle":   1.5,
        "right_shoulder_angle":  1.5,
    },
    "barbell_curl": {
        "left_elbow_angle":          3.0,
        "right_elbow_angle":         3.0,
        "trunk_lean_angle":          3.0,   # Cheat prevention
        "left_upper_arm_vertical":   2.5,
        "right_upper_arm_vertical":  2.5,
        "left_shoulder_angle":       2.0,
        "right_shoulder_angle":      2.0,
        "elbow_symmetry_ratio":      1.5,
        "wrist_alignment_left":      1.0,
        "wrist_alignment_right":     1.0,
    },
    "hammer_curl": {
        "left_elbow_angle":          3.0,
        "right_elbow_angle":         3.0,
        "trunk_lean_angle":          3.0,
        "left_upper_arm_vertical":   2.5,
        "right_upper_arm_vertical":  2.5,
        "left_shoulder_angle":       2.0,
        "right_shoulder_angle":      2.0,
        "elbow_symmetry_ratio":      1.5,
        "left_forearm_neutral":      1.5,
        "right_forearm_neutral":     1.5,
    },
    "shoulder_press": {
        "left_elbow_angle":           3.0,
        "right_elbow_angle":          3.0,
        "left_shoulder_abduction":    2.5,
        "right_shoulder_abduction":   2.5,
        "trunk_lean_angle":           3.5,   # Lumbar hyperextension risk
        "left_wrist_over_elbow":      2.0,
        "right_wrist_over_elbow":     2.0,
        "elbow_symmetry_ratio":       1.5,
        "shoulder_symmetry_ratio":    1.5,
        "head_forward_lean":          1.0,
    },
}


class PostureScorer:
    """
    Loads a posture profile JSON and scores incoming feature dicts.

    Profile JSON schema:
    {
      "exercise": "squat",
      "n_frames": 4200,
      "features": {
        "left_knee_angle": { "mean": 91.3, "std": 8.7 },
        ...
      }
    }
    """

    def __init__(self, exercise_name: str, profiles_dir: Path | str):
        self.exercise = exercise_name
        self.profiles_dir = Path(profiles_dir)
        self.profile: dict = {}
        self.weights: dict = DEFAULT_WEIGHTS.get(exercise_name, {})
        self._load_profile()

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def score(self, features: dict[str, float]) -> dict:
        """
        Compute posture score and per-feature deviation analysis.

        Returns
        ───────
        {
          "score": float,           # 0–100 global posture score
          "status": str,            # "Good" | "Warning" | "Poor"
          "color": str,             # "green" | "yellow" | "red"
          "feature_scores": {...},  # per-feature sub-scores (0–100)
          "z_scores": {...},        # per-feature z-scores
          "feedback": [str, ...],   # list of actionable feedback messages
          "worst_features": [str],  # top 3 most deviant features
        }
        """
        if not self.profile:
            return self._no_profile_response()

        feature_data = self.profile.get("features", {})
        feature_scores = {}
        z_scores = {}
        weighted_sum = 0.0
        weight_total = 0.0

        for feat_name, feat_val in features.items():
            if feat_name not in feature_data:
                continue

            mu = feature_data[feat_name]["mean"]
            sigma = feature_data[feat_name]["std"]

            # Robust z-score
            if sigma < 1e-6:
                sigma = 1.0  # Prevent division by zero for constant features

            z = abs((feat_val - mu) / sigma)
            clamped_z = min(z, Z_MAX)
            feat_score = 100.0 * (1.0 - clamped_z / Z_MAX)

            w = self.weights.get(feat_name, 1.0)
            weighted_sum += w * clamped_z / Z_MAX
            weight_total += w

            feature_scores[feat_name] = round(feat_score, 1)
            z_scores[feat_name] = round(z, 2)

        if weight_total < 1e-6:
            global_score = 100.0
        else:
            global_score = 100.0 * (1.0 - weighted_sum / weight_total)

        global_score = max(0.0, min(100.0, global_score))

        # Identify worst features for targeted feedback
        sorted_feats = sorted(z_scores.items(), key=lambda x: -x[1])
        worst_features = [f for f, z in sorted_feats if z > 1.0][:3]

        # Generate actionable feedback
        feedback = self._generate_feedback(features, feature_data, worst_features)

        # Determine status
        status, color = self._status_from_score(global_score)

        return {
            "score": round(global_score, 1),
            "status": status,
            "color": color,
            "feature_scores": feature_scores,
            "z_scores": z_scores,
            "feedback": feedback,
            "worst_features": worst_features,
        }

    def is_profile_loaded(self) -> bool:
        return bool(self.profile)

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _load_profile(self):
        profile_path = self.profiles_dir / f"{self.exercise}_profile.json"
        if not profile_path.exists():
            logger.warning("Profile not found: %s", profile_path)
            return
        try:
            with open(profile_path, "r") as f:
                self.profile = json.load(f)
            logger.info(
                "Loaded profile for '%s' (%d frames)",
                self.exercise,
                self.profile.get("n_frames", 0)
            )
        except Exception as e:
            logger.error("Failed to load profile %s: %s", profile_path, e)

    def _generate_feedback(
        self,
        features: dict,
        profile_features: dict,
        worst_features: list[str],
    ) -> list[str]:
        """Generate human-readable feedback for the most deviant features."""
        from apps.ml_engine.features import get_feature_module

        messages = []
        try:
            module = get_feature_module(self.exercise)
            feedback_rules = getattr(module, "FEEDBACK_RULES", {})
        except Exception:
            return ["Keep focusing on your form!"]

        for feat_name in worst_features:
            if feat_name not in feedback_rules:
                continue
            rule = feedback_rules[feat_name]
            ideal_min, ideal_max = rule.get("ideal_range", (-999, 999))
            val = features.get(feat_name, None)
            if val is None:
                continue
            msgs = rule.get("messages", {})
            if val > ideal_max and "too_high" in msgs:
                messages.append(msgs["too_high"])
            elif val < ideal_min and "too_low" in msgs:
                messages.append(msgs["too_low"])

        if not messages:
            messages.append("Excellent form! Keep it up.")

        return messages[:3]  # Cap at 3 simultaneous feedback items

    @staticmethod
    def _status_from_score(score: float) -> tuple[str, str]:
        if score >= 75:
            return "Good", "green"
        elif score >= 50:
            return "Warning", "yellow"
        else:
            return "Poor", "red"

    @staticmethod
    def _no_profile_response() -> dict:
        return {
            "score": 0.0,
            "status": "No Profile",
            "color": "grey",
            "feature_scores": {},
            "z_scores": {},
            "feedback": ["Profile not loaded — run training first."],
            "worst_features": [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# SCORER REGISTRY  (cache loaded scorers to avoid repeated disk reads)
# ─────────────────────────────────────────────────────────────────────────────

_scorer_cache: dict[str, PostureScorer] = {}


def get_scorer(exercise_name: str, profiles_dir: Path | str) -> PostureScorer:
    """Return a cached PostureScorer for the exercise."""
    key = f"{exercise_name}:{profiles_dir}"
    if key not in _scorer_cache:
        _scorer_cache[key] = PostureScorer(exercise_name, profiles_dir)
    return _scorer_cache[key]


def clear_scorer_cache():
    """Force reload of all profiles (useful after retraining)."""
    _scorer_cache.clear()
