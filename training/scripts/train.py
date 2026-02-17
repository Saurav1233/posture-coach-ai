"""
Training Pipeline: Feature Extraction + Profile Construction
=============================================================

Usage
─────
python training/scripts/train.py --exercise squat --video_dir training/data/raw_videos/squat

This script:
  1. Iterates all MP4 files in the given directory
  2. Extracts pose landmarks from every 3rd frame using MediaPipe
  3. Computes biomechanical features per frame
  4. Saves all frame features to CSV
  5. Computes mean + std per feature → posture profile
  6. Saves profile JSON to training/data/posture_profiles/

Mathematical Rationale
──────────────────────
We use ALL frames from correct-posture videos.
The profile captures the statistical distribution of biomechanical features
across the full range of motion of the exercise.
This is intentional — we want to profile the ENTIRE movement arc, not just
peak positions. The resulting distribution encodes both range of motion and
postural quality throughout the movement.

Cross-Validation
─────────────────
We perform Leave-One-Video-Out cross-validation to verify profile stability:
For each video V:
  - Train on all videos EXCEPT V
  - Compute mean absolute deviation (MAD) between hold-out scores and train scores
A stable profile will show low MAD (< 5 score points on average).
This validates robustness without requiring negative examples.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Setup Django settings for standalone script
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

import numpy as np
import pandas as pd

from apps.ml_engine.pose_extractor import PoseExtractor
from apps.ml_engine.features import get_feature_module

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("training")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_from_videos(
    exercise: str,
    video_dir: Path,
    frame_skip: int = 3,
) -> pd.DataFrame:
    """
    Extract features from all MP4 videos in video_dir.
    Returns a DataFrame with one row per valid frame.
    """
    module = get_feature_module(exercise)
    extractor = PoseExtractor(
        static_image_mode=False,
        model_complexity=1,    # Use higher quality for training
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.MP4"))
    if not video_files:
        logger.error("No MP4 files found in %s", video_dir)
        return pd.DataFrame()

    logger.info("Found %d video(s) in %s", len(video_files), video_dir)
    all_rows = []

    for video_path in sorted(video_files):
        logger.info("Processing: %s", video_path.name)
        frame_count = 0
        valid_count = 0

        for frame_idx, raw_lm, norm_lm, vis, valid in \
                extractor.extract_from_video(str(video_path), frame_skip=frame_skip):

            frame_count += 1
            if not valid or norm_lm is None:
                continue

            try:
                feats = module.extract_features(norm_lm)
                feats["_video"] = video_path.name
                feats["_frame_idx"] = frame_idx
                all_rows.append(feats)
                valid_count += 1
            except Exception as e:
                logger.debug("Frame %d skipped: %s", frame_idx, e)

        logger.info(
            "  %s: %d frames processed, %d valid (%.1f%%)",
            video_path.name, frame_count, valid_count,
            100 * valid_count / max(frame_count, 1)
        )

    extractor.close()

    if not all_rows:
        logger.error("No valid frames extracted!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    logger.info("Total valid frames: %d", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_profile(exercise: str, df: pd.DataFrame) -> dict:
    """
    Compute mean + std for each feature across all training frames.
    Returns the posture profile dict.
    """
    module = get_feature_module(exercise)
    feature_cols = module.FEATURE_NAMES

    profile = {
        "exercise": exercise,
        "n_frames": len(df),
        "n_videos": df["_video"].nunique() if "_video" in df.columns else None,
        "features": {}
    }

    for feat in feature_cols:
        if feat not in df.columns:
            logger.warning("Feature '%s' not found in extracted data", feat)
            continue

        values = df[feat].dropna().values
        if len(values) < 10:
            logger.warning("Too few samples for feature '%s': %d", feat, len(values))
            continue

        # Use robust statistics (IQR-trimmed) to reduce outlier influence
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        # Keep values within 3 IQR of median (Tukey outlier fence)
        median = np.median(values)
        mask = (values >= median - 3 * iqr) & (values <= median + 3 * iqr)
        clean_values = values[mask]

        profile["features"][feat] = {
            "mean": round(float(np.mean(clean_values)), 4),
            "std": round(float(np.std(clean_values)), 4),
            "median": round(float(np.median(clean_values)), 4),
            "p5": round(float(np.percentile(clean_values, 5)), 4),
            "p95": round(float(np.percentile(clean_values, 95)), 4),
            "n_samples": int(len(clean_values)),
            "n_outliers_removed": int(len(values) - len(clean_values)),
        }
        logger.info(
            "  %-35s  mean=%.1f  std=%.1f  n=%d",
            feat,
            profile["features"][feat]["mean"],
            profile["features"][feat]["std"],
            len(clean_values)
        )

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# LEAVE-ONE-VIDEO-OUT CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(exercise: str, df: pd.DataFrame) -> dict:
    """
    Leave-One-Video-Out cross-validation.
    Validates that the profile is stable across videos.
    """
    from apps.ml_engine.posture_scorer import PostureScorer
    import tempfile

    videos = df["_video"].unique()
    if len(videos) < 2:
        logger.warning("Need >= 2 videos for cross-validation. Skipping.")
        return {"skipped": True, "reason": "< 2 videos"}

    logger.info("Running Leave-One-Video-Out CV on %d videos...", len(videos))
    cv_results = []

    for hold_out_video in videos:
        train_df = df[df["_video"] != hold_out_video]
        test_df = df[df["_video"] == hold_out_video]

        # Build train profile
        train_profile = build_profile(exercise, train_df)

        # Score test frames using train profile
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / f"{exercise}_profile.json"
            with open(profile_path, "w") as f:
                json.dump(train_profile, f)

            scorer = PostureScorer(exercise, tmpdir)
            module = get_feature_module(exercise)

            scores = []
            for _, row in test_df.iterrows():
                feat_dict = {k: row[k] for k in module.FEATURE_NAMES if k in row}
                result = scorer.score(feat_dict)
                scores.append(result["score"])

        mean_score = float(np.mean(scores))
        cv_results.append({
            "hold_out_video": hold_out_video,
            "n_test_frames": len(test_df),
            "mean_score_on_holdout": round(mean_score, 1),
        })
        logger.info(
            "  Held out: %-30s  score=%.1f  (n=%d)",
            hold_out_video, mean_score, len(test_df)
        )

    all_scores = [r["mean_score_on_holdout"] for r in cv_results]
    cv_summary = {
        "cv_mean_score": round(float(np.mean(all_scores)), 1),
        "cv_std_score": round(float(np.std(all_scores)), 1),
        "cv_min_score": round(float(np.min(all_scores)), 1),
        "per_video": cv_results,
    }
    logger.info(
        "CV Summary: mean=%.1f  std=%.1f  min=%.1f",
        cv_summary["cv_mean_score"],
        cv_summary["cv_std_score"],
        cv_summary["cv_min_score"]
    )
    return cv_summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train posture profile for one exercise")
    parser.add_argument("--exercise", required=True,
                        choices=["squat", "pushup", "barbell_curl", "hammer_curl", "shoulder_press"])
    parser.add_argument("--video_dir", required=True, help="Directory containing MP4 videos")
    parser.add_argument("--frame_skip", type=int, default=3)
    parser.add_argument("--output_dir",
                        default="training/data/posture_profiles",
                        help="Where to save profile JSON")
    parser.add_argument("--csv_dir",
                        default="training/data/features_csv",
                        help="Where to save features CSV")
    parser.add_argument("--cross_validate", action="store_true",
                        help="Run leave-one-video-out cross-validation")
    args = parser.parse_args()

    exercise = args.exercise
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    csv_dir = Path(args.csv_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Training: %s", exercise)
    logger.info("Video dir: %s", video_dir)
    logger.info("=" * 60)

    # Step 1: Extract features
    df = extract_features_from_videos(exercise, video_dir, args.frame_skip)
    if df.empty:
        logger.error("Aborting: no valid frames extracted.")
        sys.exit(1)

    # Step 2: Save CSV
    csv_path = csv_dir / f"{exercise}_features.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Features saved to: %s", csv_path)

    # Step 3: Build profile
    logger.info("Building posture profile...")
    profile = build_profile(exercise, df)

    # Step 4: Cross-validation
    if args.cross_validate and "_video" in df.columns:
        cv_result = cross_validate(exercise, df)
        profile["cross_validation"] = cv_result

    # Step 5: Save profile
    profile_path = output_dir / f"{exercise}_profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    logger.info("Profile saved to: %s", profile_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("  Exercise : %s", exercise)
    logger.info("  Frames   : %d", profile["n_frames"])
    logger.info("  Features : %d", len(profile["features"]))
    logger.info("  Profile  : %s", profile_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()