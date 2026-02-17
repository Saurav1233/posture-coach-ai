"""
Inference Engine
=================
Combines PoseExtractor + FeatureExtraction + PostureScorer + RepCounter
into a single request/response handler for Django views.

This is the ONLY module that Django views should call directly.
ML logic is fully isolated from HTTP handling.

Smoothing
──────────
We maintain a rolling window of deviation scores across the last N frames
and report the smoothed score. This prevents the UI from flickering on
occasional noisy frames, providing a better user experience.
"""

from __future__ import annotations
import base64
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from apps.ml_engine.pose_extractor import PoseExtractor
from apps.ml_engine.features import extract_features as extract_biomech_features
from apps.ml_engine.posture_scorer import get_scorer
from apps.ml_engine.rep_counter import RepCounter

logger = logging.getLogger('posture_coach')


class InferenceEngine:
    """
    Stateful per-session inference engine.

    One InferenceEngine instance is maintained per user session.
    It holds the pose extractor, scorer reference, and rep counter.
    It does NOT hold Django session state directly.

    Usage
    ─────
    engine = InferenceEngine(exercise="squat", profiles_dir=...)
    result = engine.process_frame_b64(base64_jpeg_string)
    """

    SKELETON_COLORS = {
        "green":  (0, 220, 80),     # Good posture (BGR)
        "yellow": (0, 200, 255),    # Warning posture
        "red":    (0, 60, 220),     # Poor posture
        "grey":   (150, 150, 150),  # No profile / no detection
    }

    def __init__(
        self,
        exercise: str,
        profiles_dir: Path | str,
        smoothing_window: int = 5,
    ):
        self.exercise = exercise
        self.profiles_dir = Path(profiles_dir)
        self.smoothing_window = smoothing_window

        self._extractor = PoseExtractor(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
        )
        self._scorer = get_scorer(exercise, profiles_dir)
        self._rep_counter = RepCounter(exercise)
        self._score_buffer: deque[float] = deque(maxlen=smoothing_window)
        self._frame_count = 0
        self._last_score_result: Optional[dict] = None

    # ──────────────────────────────────────────────────────────────────────
    # PRIMARY PUBLIC METHODS
    # ──────────────────────────────────────────────────────────────────────

    def process_frame_b64(self, b64_data: str, draw_skeleton: bool = True) -> dict:
        """
        Process a single Base64-encoded JPEG frame from the browser.

        Parameters
        ──────────
        b64_data : str  — base64 string (with or without data URL prefix)
        draw_skeleton : bool — whether to return annotated frame

        Returns full inference result dict.
        """
        frame = self._decode_b64_frame(b64_data)
        if frame is None:
            return self._error_response("Failed to decode frame")
        return self._process_bgr_frame(frame, draw_skeleton=draw_skeleton)

    def process_video_file(self, video_path: str) -> list[dict]:
        """
        Process an entire uploaded video file.
        Returns a list of per-frame results plus a session summary.
        """
        results = []
        scorer_static = get_scorer(self.exercise, self.profiles_dir)
        rep_counter_local = RepCounter(self.exercise)
        extractor_static = PoseExtractor(static_image_mode=True)

        try:
            for frame_idx, raw_lm, norm_lm, vis, valid in \
                    extractor_static.extract_from_video(video_path, frame_skip=3):

                if not valid or norm_lm is None:
                    continue

                features = extract_biomech_features(self.exercise, norm_lm)
                score_result = scorer_static.score(features)
                rep_result = rep_counter_local.update(features)

                results.append({
                    "frame": frame_idx,
                    "score": score_result["score"],
                    "status": score_result["status"],
                    "reps": rep_result["reps"],
                    "feedback": score_result["feedback"],
                })
        finally:
            extractor_static.close()

        return self._build_video_summary(results, rep_counter_local.reps)

    def get_rep_counter_dict(self) -> dict:
        """Serialize rep counter for session storage."""
        return self._rep_counter.to_dict()

    def restore_rep_counter(self, data: dict):
        """Restore rep counter from session storage."""
        self._rep_counter = RepCounter.from_dict(data)

    def reset_session(self):
        """Reset rep counter and score buffer for a new session."""
        self._rep_counter.reset()
        self._score_buffer.clear()
        self._frame_count = 0
        self._last_score_result = None

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE PROCESSING PIPELINE
    # ──────────────────────────────────────────────────────────────────────

    def _process_bgr_frame(self, frame: np.ndarray, draw_skeleton: bool) -> dict:
        self._frame_count += 1
        t0 = time.perf_counter()

        raw_lm, norm_lm, vis, valid = self._extractor.extract(frame)

        if not valid or norm_lm is None:
            return {
                "pose_detected": False,
                "score": 0,
                "status": "No Pose",
                "color": "grey",
                "reps": self._rep_counter.reps,
                "rep_state": self._rep_counter.state.value,
                "feedback": ["Position yourself fully in frame"],
                "worst_features": [],
                "annotated_frame": self._encode_frame(frame) if draw_skeleton else None,
                "processing_ms": round((time.perf_counter() - t0) * 1000, 1),
                "frame_count": self._frame_count,
            }

        # Feature extraction
        features = extract_biomech_features(self.exercise, norm_lm)

        # Posture scoring
        score_result = self._scorer.score(features)
        self._score_buffer.append(score_result["score"])
        smoothed_score = np.mean(self._score_buffer)
        smoothed_status, smoothed_color = self._status_from_score(smoothed_score)
        self._last_score_result = score_result

        # Rep counting
        rep_result = self._rep_counter.update(features)

        # Draw skeleton with color based on posture quality
        annotated_frame = None
        if draw_skeleton:
            skel_color = self.SKELETON_COLORS.get(smoothed_color, (0, 255, 0))
            annotated_frame = self._extractor.draw_skeleton(frame, raw_lm, skel_color)
            self._overlay_hud(annotated_frame, smoothed_score, smoothed_status, rep_result["reps"])
            annotated_frame = self._encode_frame(annotated_frame)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "pose_detected": True,
            "score": round(float(smoothed_score), 1),
            "raw_score": score_result["score"],
            "status": smoothed_status,
            "color": smoothed_color,
            "reps": rep_result["reps"],
            "rep_state": rep_result["state"],
            "rep_just_counted": rep_result["rep_just_counted"],
            "feedback": score_result["feedback"],
            "worst_features": score_result["worst_features"],
            "feature_scores": score_result["feature_scores"],
            "annotated_frame": annotated_frame,
            "landmarks": raw_lm.tolist() if raw_lm is not None else None,
            "processing_ms": elapsed_ms,
            "frame_count": self._frame_count,
        }

    def _overlay_hud(
        self,
        frame: np.ndarray,
        score: float,
        status: str,
        reps: int,
    ):
        """Draw a minimal HUD on the annotated frame (server-side overlay)."""
        h, w = frame.shape[:2]
        # Score badge
        score_color = (0, 220, 80) if status == "Good" else (
            (0, 200, 255) if status == "Warning" else (0, 60, 220)
        )
        cv2.putText(
            frame, f"Score: {score:.0f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2
        )
        cv2.putText(
            frame, f"Reps: {reps}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        cv2.putText(
            frame, status,
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2
        )

    # ──────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_b64_frame(b64_data: str) -> Optional[np.ndarray]:
        """Decode a base64 JPEG/PNG string into a BGR NumPy array."""
        try:
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            raw = base64.b64decode(b64_data)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error("Frame decode error: %s", e)
            return None

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        """Encode BGR frame to base64 JPEG string for JSON response."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def _status_from_score(score: float) -> tuple[str, str]:
        if score >= 75:
            return "Good", "green"
        elif score >= 50:
            return "Warning", "yellow"
        else:
            return "Poor", "red"

    @staticmethod
    def _error_response(message: str) -> dict:
        return {
            "pose_detected": False,
            "score": 0,
            "status": "Error",
            "color": "grey",
            "reps": 0,
            "rep_state": "idle",
            "feedback": [message],
            "worst_features": [],
            "annotated_frame": None,
            "processing_ms": 0,
            "frame_count": 0,
        }

    @staticmethod
    def _build_video_summary(frame_results: list[dict], total_reps: int) -> dict:
        if not frame_results:
            return {"summary": True, "error": "No valid frames detected"}

        scores = [r["score"] for r in frame_results]
        avg_score = float(np.mean(scores))
        min_score = float(np.min(scores))

        # Collect all feedback messages
        all_feedback = []
        for r in frame_results:
            all_feedback.extend(r.get("feedback", []))

        # Count frequency of each feedback message
        from collections import Counter
        feedback_counts = Counter(all_feedback)
        top_feedback = [msg for msg, _ in feedback_counts.most_common(5)]

        return {
            "summary": True,
            "total_frames_analyzed": len(frame_results),
            "total_reps": total_reps,
            "average_score": round(avg_score, 1),
            "min_score": round(min_score, 1),
            "frame_results": frame_results,
            "top_feedback": top_feedback,
            "overall_status": "Good" if avg_score >= 75 else "Needs Work",
        }


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE REGISTRY  (per-session cache)
# ─────────────────────────────────────────────────────────────────────────────

_engine_registry: dict[str, InferenceEngine] = {}


def get_or_create_engine(
    session_key: str,
    exercise: str,
    profiles_dir: Path | str,
) -> InferenceEngine:
    """
    Get or create an InferenceEngine for a given session + exercise combo.
    Recreates the engine if the exercise changes mid-session.
    """
    cached = _engine_registry.get(session_key)
    if cached is None or cached.exercise != exercise:
        _engine_registry[session_key] = InferenceEngine(exercise, profiles_dir)
        logger.info("Created InferenceEngine for session=%s exercise=%s", session_key, exercise)
    return _engine_registry[session_key]


def clear_engine(session_key: str):
    _engine_registry.pop(session_key, None)