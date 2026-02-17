"""
Generate Demo Posture Profiles
================================
Creates realistic synthetic posture profiles for all 5 exercises.
Used for testing the Django app WITHOUT having trained on real videos.

These profiles are based on published biomechanical literature values.
Replace them with real trained profiles when your MP4 videos are ready.

Usage
─────
python training/scripts/generate_demo_profiles.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Profiles based on biomechanics literature (mean ± std)
DEMO_PROFILES = {
    "squat": {
        "exercise": "squat",
        "n_frames": 3200,
        "n_videos": 8,
        "note": "Demo profile — replace with trained profile from real videos",
        "features": {
            "left_knee_angle":       {"mean": 92.0,  "std": 10.0, "median": 91.5, "p5": 73.0, "p95": 110.0, "n_samples": 3200, "n_outliers_removed": 0},
            "right_knee_angle":      {"mean": 93.0,  "std": 10.0, "median": 92.5, "p5": 74.0, "p95": 111.0, "n_samples": 3200, "n_outliers_removed": 0},
            "left_hip_angle":        {"mean": 85.0,  "std": 12.0, "median": 84.0, "p5": 63.0, "p95": 107.0, "n_samples": 3200, "n_outliers_removed": 0},
            "right_hip_angle":       {"mean": 86.0,  "std": 12.0, "median": 85.0, "p5": 64.0, "p95": 108.0, "n_samples": 3200, "n_outliers_removed": 0},
            "trunk_lean_angle":      {"mean": 28.0,  "std": 7.0,  "median": 27.5, "p5": 15.0, "p95": 41.0, "n_samples": 3200, "n_outliers_removed": 0},
            "left_knee_valgus":      {"mean": 2.0,   "std": 5.0,  "median": 2.0,  "p5": -6.0, "p95": 9.5,  "n_samples": 3200, "n_outliers_removed": 0},
            "right_knee_valgus":     {"mean": -1.5,  "std": 5.0,  "median": -1.5, "p5": -9.0, "p95": 7.0,  "n_samples": 3200, "n_outliers_removed": 0},
            "knee_symmetry_ratio":   {"mean": 0.03,  "std": 0.04, "median": 0.02, "p5": 0.0,  "p95": 0.10, "n_samples": 3200, "n_outliers_removed": 0},
            "hip_symmetry_ratio":    {"mean": 0.03,  "std": 0.04, "median": 0.02, "p5": 0.0,  "p95": 0.10, "n_samples": 3200, "n_outliers_removed": 0},
            "knee_depth_score":      {"mean": 8.0,   "std": 8.0,  "median": 7.0,  "p5": 0.5,  "p95": 22.0, "n_samples": 3200, "n_outliers_removed": 0},
        }
    },
    "pushup": {
        "exercise": "pushup",
        "n_frames": 2800,
        "n_videos": 7,
        "note": "Demo profile — replace with trained profile from real videos",
        "features": {
            "left_elbow_angle":       {"mean": 110.0, "std": 25.0, "median": 108.0, "p5": 68.0,  "p95": 160.0, "n_samples": 2800, "n_outliers_removed": 0},
            "right_elbow_angle":      {"mean": 111.0, "std": 25.0, "median": 109.0, "p5": 69.0,  "p95": 161.0, "n_samples": 2800, "n_outliers_removed": 0},
            "body_alignment_angle":   {"mean": 7.0,   "std": 5.0,  "median": 6.5,   "p5": 1.0,   "p95": 16.0,  "n_samples": 2800, "n_outliers_removed": 0},
            "hip_sag_deviation":      {"mean": 0.06,  "std": 0.05, "median": 0.05,  "p5": 0.01,  "p95": 0.15,  "n_samples": 2800, "n_outliers_removed": 0},
            "head_alignment_angle":   {"mean": 10.0,  "std": 6.0,  "median": 9.5,   "p5": 2.0,   "p95": 20.0,  "n_samples": 2800, "n_outliers_removed": 0},
            "elbow_symmetry_ratio":   {"mean": 0.03,  "std": 0.04, "median": 0.02,  "p5": 0.0,   "p95": 0.10,  "n_samples": 2800, "n_outliers_removed": 0},
            "shoulder_width_ratio":   {"mean": 1.35,  "std": 0.15, "median": 1.35,  "p5": 1.08,  "p95": 1.62,  "n_samples": 2800, "n_outliers_removed": 0},
            "left_shoulder_angle":    {"mean": 55.0,  "std": 10.0, "median": 54.5,  "p5": 37.0,  "p95": 72.0,  "n_samples": 2800, "n_outliers_removed": 0},
            "right_shoulder_angle":   {"mean": 56.0,  "std": 10.0, "median": 55.5,  "p5": 38.0,  "p95": 73.0,  "n_samples": 2800, "n_outliers_removed": 0},
        }
    },
    "barbell_curl": {
        "exercise": "barbell_curl",
        "n_frames": 2400,
        "n_videos": 6,
        "note": "Demo profile — replace with trained profile from real videos",
        "features": {
            "left_elbow_angle":          {"mean": 90.0,  "std": 40.0, "median": 88.0,  "p5": 30.0,  "p95": 155.0, "n_samples": 2400, "n_outliers_removed": 0},
            "right_elbow_angle":         {"mean": 91.0,  "std": 40.0, "median": 89.0,  "p5": 31.0,  "p95": 156.0, "n_samples": 2400, "n_outliers_removed": 0},
            "trunk_lean_angle":          {"mean": 6.0,   "std": 4.0,  "median": 5.5,   "p5": 1.0,   "p95": 13.0,  "n_samples": 2400, "n_outliers_removed": 0},
            "left_upper_arm_vertical":   {"mean": 8.0,   "std": 5.0,  "median": 7.5,   "p5": 1.5,   "p95": 17.0,  "n_samples": 2400, "n_outliers_removed": 0},
            "right_upper_arm_vertical":  {"mean": 8.0,   "std": 5.0,  "median": 7.5,   "p5": 1.5,   "p95": 17.0,  "n_samples": 2400, "n_outliers_removed": 0},
            "left_shoulder_angle":       {"mean": 38.0,  "std": 8.0,  "median": 37.5,  "p5": 24.0,  "p95": 52.0,  "n_samples": 2400, "n_outliers_removed": 0},
            "right_shoulder_angle":      {"mean": 38.0,  "std": 8.0,  "median": 37.5,  "p5": 24.0,  "p95": 52.0,  "n_samples": 2400, "n_outliers_removed": 0},
            "elbow_symmetry_ratio":      {"mean": 0.03,  "std": 0.04, "median": 0.02,  "p5": 0.0,   "p95": 0.10,  "n_samples": 2400, "n_outliers_removed": 0},
            "wrist_alignment_left":      {"mean": 12.0,  "std": 6.0,  "median": 11.5,  "p5": 3.0,   "p95": 23.0,  "n_samples": 2400, "n_outliers_removed": 0},
            "wrist_alignment_right":     {"mean": 12.0,  "std": 6.0,  "median": 11.5,  "p5": 3.0,   "p95": 23.0,  "n_samples": 2400, "n_outliers_removed": 0},
        }
    },
    "hammer_curl": {
        "exercise": "hammer_curl",
        "n_frames": 2200,
        "n_videos": 6,
        "note": "Demo profile — replace with trained profile from real videos",
        "features": {
            "left_elbow_angle":          {"mean": 90.0,  "std": 40.0, "median": 88.0,  "p5": 30.0,  "p95": 155.0, "n_samples": 2200, "n_outliers_removed": 0},
            "right_elbow_angle":         {"mean": 91.0,  "std": 40.0, "median": 89.0,  "p5": 31.0,  "p95": 156.0, "n_samples": 2200, "n_outliers_removed": 0},
            "trunk_lean_angle":          {"mean": 6.0,   "std": 4.0,  "median": 5.5,   "p5": 1.0,   "p95": 13.0,  "n_samples": 2200, "n_outliers_removed": 0},
            "left_upper_arm_vertical":   {"mean": 9.0,   "std": 5.0,  "median": 8.5,   "p5": 2.0,   "p95": 18.0,  "n_samples": 2200, "n_outliers_removed": 0},
            "right_upper_arm_vertical":  {"mean": 9.0,   "std": 5.0,  "median": 8.5,   "p5": 2.0,   "p95": 18.0,  "n_samples": 2200, "n_outliers_removed": 0},
            "left_shoulder_angle":       {"mean": 38.0,  "std": 8.0,  "median": 37.5,  "p5": 24.0,  "p95": 52.0,  "n_samples": 2200, "n_outliers_removed": 0},
            "right_shoulder_angle":      {"mean": 38.0,  "std": 8.0,  "median": 37.5,  "p5": 24.0,  "p95": 52.0,  "n_samples": 2200, "n_outliers_removed": 0},
            "elbow_symmetry_ratio":      {"mean": 0.04,  "std": 0.04, "median": 0.03,  "p5": 0.0,   "p95": 0.11,  "n_samples": 2200, "n_outliers_removed": 0},
            "left_forearm_neutral":      {"mean": 8.0,   "std": 5.0,  "median": 7.5,   "p5": 1.5,   "p95": 17.0,  "n_samples": 2200, "n_outliers_removed": 0},
            "right_forearm_neutral":     {"mean": 8.0,   "std": 5.0,  "median": 7.5,   "p5": 1.5,   "p95": 17.0,  "n_samples": 2200, "n_outliers_removed": 0},
        }
    },
    "shoulder_press": {
        "exercise": "shoulder_press",
        "n_frames": 2600,
        "n_videos": 7,
        "note": "Demo profile — replace with trained profile from real videos",
        "features": {
            "left_elbow_angle":           {"mean": 130.0, "std": 35.0, "median": 128.0, "p5": 78.0,  "p95": 180.0, "n_samples": 2600, "n_outliers_removed": 0},
            "right_elbow_angle":          {"mean": 131.0, "std": 35.0, "median": 129.0, "p5": 79.0,  "p95": 180.0, "n_samples": 2600, "n_outliers_removed": 0},
            "left_shoulder_abduction":    {"mean": 88.0,  "std": 12.0, "median": 87.5,  "p5": 67.0,  "p95": 109.0, "n_samples": 2600, "n_outliers_removed": 0},
            "right_shoulder_abduction":   {"mean": 89.0,  "std": 12.0, "median": 88.5,  "p5": 68.0,  "p95": 110.0, "n_samples": 2600, "n_outliers_removed": 0},
            "trunk_lean_angle":           {"mean": 7.0,   "std": 4.0,  "median": 6.5,   "p5": 1.0,   "p95": 14.0,  "n_samples": 2600, "n_outliers_removed": 0},
            "left_wrist_over_elbow":      {"mean": 8.0,   "std": 6.0,  "median": 7.5,   "p5": 1.0,   "p95": 19.0,  "n_samples": 2600, "n_outliers_removed": 0},
            "right_wrist_over_elbow":     {"mean": 8.0,   "std": 6.0,  "median": 7.5,   "p5": 1.0,   "p95": 19.0,  "n_samples": 2600, "n_outliers_removed": 0},
            "elbow_symmetry_ratio":       {"mean": 0.03,  "std": 0.04, "median": 0.02,  "p5": 0.0,   "p95": 0.10,  "n_samples": 2600, "n_outliers_removed": 0},
            "shoulder_symmetry_ratio":    {"mean": 0.03,  "std": 0.04, "median": 0.02,  "p5": 0.0,   "p95": 0.10,  "n_samples": 2600, "n_outliers_removed": 0},
            "head_forward_lean":          {"mean": 10.0,  "std": 6.0,  "median": 9.5,   "p5": 2.0,   "p95": 21.0,  "n_samples": 2600, "n_outliers_removed": 0},
        }
    },
}


def main():
    output_dir = Path(__file__).resolve().parent.parent / "data" / "posture_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    for exercise, profile in DEMO_PROFILES.items():
        out_path = output_dir / f"{exercise}_profile.json"
        with open(out_path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"  ✓ {exercise:20s} → {out_path}")

    print(f"\nAll 5 demo profiles saved to: {output_dir}")
    print("Run 'python training/scripts/train.py --exercise <name> --video_dir ...'")
    print("to replace these with profiles trained on your real videos.")


if __name__ == "__main__":
    main()
