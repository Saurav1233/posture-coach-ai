"""
Generate demo posture profiles with realistic values
based on actual observed mediapipe angles.
"""

import json
from pathlib import Path

PROFILES_DIR = Path(__file__).parent.parent / "data" / "posture_profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

# ── Realistic values based on observed mediapipe output ─────────────────────
# Elbow angles: ~160-170° extended, ~90° bent, ~30-40° fully curled
# Shoulder abduction: ~80-100° at goal post for shoulder press
# Knee angles: ~165-175° standing, ~90-100° squatted

PROFILES = {

    "squat": {
        "exercise": "squat",
        "feature_weights": {
            "left_knee_angle":    3.0,
            "right_knee_angle":   3.0,
            "left_hip_angle":     2.0,
            "right_hip_angle":    2.0,
            "trunk_lean_angle":   2.5,
            "left_knee_valgus":   3.0,
            "right_knee_valgus":  3.0,
            "bilateral_symmetry": 1.5,
        },
        "feature_stats": {
            "left_knee_angle":    {"mean": 95.0,  "std": 12.0},
            "right_knee_angle":   {"mean": 95.0,  "std": 12.0},
            "left_hip_angle":     {"mean": 85.0,  "std": 12.0},
            "right_hip_angle":    {"mean": 85.0,  "std": 12.0},
            "trunk_lean_angle":   {"mean": 25.0,  "std": 8.0},
            "left_knee_valgus":   {"mean": 8.0,   "std": 5.0},
            "right_knee_valgus":  {"mean": 8.0,   "std": 5.0},
            "bilateral_symmetry": {"mean": 0.05,  "std": 0.04},
        }
    },

    "pushup": {
        "exercise": "pushup",
        "feature_weights": {
            "left_elbow_angle":   3.0,
            "right_elbow_angle":  3.0,
            "body_alignment":     2.5,
            "hip_sag":            2.5,
            "left_shoulder_angle":  1.5,
            "right_shoulder_angle": 1.5,
            "bilateral_symmetry": 1.0,
        },
        "feature_stats": {
            "left_elbow_angle":   {"mean": 90.0,  "std": 15.0},
            "right_elbow_angle":  {"mean": 90.0,  "std": 15.0},
            "body_alignment":     {"mean": 8.0,   "std": 5.0},
            "hip_sag":            {"mean": 5.0,   "std": 4.0},
            "left_shoulder_angle":  {"mean": 45.0, "std": 10.0},
            "right_shoulder_angle": {"mean": 45.0, "std": 10.0},
            "bilateral_symmetry": {"mean": 0.05,  "std": 0.04},
        }
    },

    "barbell_curl": {
        "exercise": "barbell_curl",
        "feature_weights": {
            "left_elbow_angle":   3.0,
            "right_elbow_angle":  3.0,
            "left_elbow_swing":   2.5,
            "right_elbow_swing":  2.5,
            "trunk_lean_angle":   2.0,
            "bilateral_symmetry": 1.5,
        },
        "feature_stats": {
            "left_elbow_angle":   {"mean": 80.0,  "std": 20.0},
            "right_elbow_angle":  {"mean": 80.0,  "std": 20.0},
            "left_elbow_swing":   {"mean": 5.0,   "std": 4.0},
            "right_elbow_swing":  {"mean": 5.0,   "std": 4.0},
            "trunk_lean_angle":   {"mean": 5.0,   "std": 4.0},
            "bilateral_symmetry": {"mean": 0.05,  "std": 0.04},
        }
    },

    "hammer_curl": {
        "exercise": "hammer_curl",
        "feature_weights": {
            "left_elbow_angle":   3.0,
            "right_elbow_angle":  3.0,
            "left_elbow_swing":   2.5,
            "right_elbow_swing":  2.5,
            "trunk_lean_angle":   2.0,
            "bilateral_symmetry": 1.5,
        },
        "feature_stats": {
            "left_elbow_angle":   {"mean": 80.0,  "std": 20.0},
            "right_elbow_angle":  {"mean": 80.0,  "std": 20.0},
            "left_elbow_swing":   {"mean": 5.0,   "std": 4.0},
            "right_elbow_swing":  {"mean": 5.0,   "std": 4.0},
            "trunk_lean_angle":   {"mean": 5.0,   "std": 4.0},
            "bilateral_symmetry": {"mean": 0.05,  "std": 0.04},
        }
    },

    "shoulder_press": {
        "exercise": "shoulder_press",
        "feature_weights": {
            "left_elbow_angle":        3.0,
            "right_elbow_angle":       3.0,
            "left_shoulder_abduction":  2.0,
            "right_shoulder_abduction": 2.0,
            "trunk_lean_angle":        2.5,
            "left_wrist_over_elbow":   1.5,
            "right_wrist_over_elbow":  1.5,
            "elbow_symmetry_ratio":    1.5,
            "shoulder_symmetry_ratio": 1.0,
            "head_forward_lean":       1.0,
        },
        "feature_stats": {
            "left_elbow_angle":        {"mean": 120.0, "std": 30.0},
            "right_elbow_angle":       {"mean": 120.0, "std": 30.0},
            "left_shoulder_abduction":  {"mean": 90.0,  "std": 20.0},
            "right_shoulder_abduction": {"mean": 90.0,  "std": 20.0},
            "trunk_lean_angle":        {"mean": 8.0,   "std": 6.0},
            "left_wrist_over_elbow":   {"mean": 10.0,  "std": 8.0},
            "right_wrist_over_elbow":  {"mean": 10.0,  "std": 8.0},
            "elbow_symmetry_ratio":    {"mean": 0.05,  "std": 0.05},
            "shoulder_symmetry_ratio": {"mean": 0.05,  "std": 0.05},
            "head_forward_lean":       {"mean": 10.0,  "std": 8.0},
        }
    },
}


def main():
    for name, profile in PROFILES.items():
        path = PROFILES_DIR / f"{name}_profile.json"
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"  ✓ {name}")

    print(f"\nAll 5 demo profiles saved to: {PROFILES_DIR}")
    print("Run 'python training/scripts/train.py --exercise <name> --video_dir ...'")
    print("to replace these with profiles trained on your real videos.")


if __name__ == "__main__":
    main()