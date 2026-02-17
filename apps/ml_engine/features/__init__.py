"""
Feature Registry
================
Central registry mapping exercise identifiers to their feature modules.
Importing this package gives access to all exercise feature extractors.
"""

from apps.ml_engine.features import (
    squat_features,
    pushup_features,
    barbell_curl_features,
    hammer_curl_features,
    shoulder_press_features,
)

FEATURE_REGISTRY = {
    "squat":           squat_features,
    "pushup":          pushup_features,
    "barbell_curl":    barbell_curl_features,
    "hammer_curl":     hammer_curl_features,
    "shoulder_press":  shoulder_press_features,
}


def get_feature_module(exercise_name: str):
    """Return the feature module for the given exercise name."""
    name = exercise_name.lower().replace("-", "_").replace(" ", "_")
    if name not in FEATURE_REGISTRY:
        raise ValueError(
            f"Unknown exercise '{exercise_name}'. "
            f"Supported: {list(FEATURE_REGISTRY.keys())}"
        )
    return FEATURE_REGISTRY[name]


def extract_features(exercise_name: str, normalized_landmarks):
    """Convenience wrapper: extract features for given exercise + landmarks."""
    module = get_feature_module(exercise_name)
    return module.extract_features(normalized_landmarks)
