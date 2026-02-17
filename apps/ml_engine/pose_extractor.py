"""
Pose Extraction Module
======================
Compatible with mediapipe 0.10.30+ (Tasks API)
Downloads pose_landmarker model automatically on first run.
"""

import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger('posture_coach')

# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK INDEX CONSTANTS (hardcoded — same in all mediapipe versions)
# ─────────────────────────────────────────────────────────────────────────────
NOSE             = 0
LEFT_EYE_INNER   = 1
LEFT_EYE         = 2
LEFT_EYE_OUTER   = 3
RIGHT_EYE_INNER  = 4
RIGHT_EYE        = 5
RIGHT_EYE_OUTER  = 6
LEFT_EAR         = 7
RIGHT_EAR        = 8
MOUTH_LEFT       = 9
MOUTH_RIGHT      = 10
LEFT_SHOULDER    = 11
RIGHT_SHOULDER   = 12
LEFT_ELBOW       = 13
RIGHT_ELBOW      = 14
LEFT_WRIST       = 15
RIGHT_WRIST      = 16
LEFT_PINKY       = 17
RIGHT_PINKY      = 18
LEFT_INDEX       = 19
RIGHT_INDEX      = 20
LEFT_THUMB       = 21
RIGHT_THUMB      = 22
LEFT_HIP         = 23
RIGHT_HIP        = 24
LEFT_KNEE        = 25
RIGHT_KNEE       = 26
LEFT_ANKLE       = 27
RIGHT_ANKLE      = 28
LEFT_HEEL        = 29
RIGHT_HEEL       = 30
LEFT_FOOT_INDEX  = 31
RIGHT_FOOT_INDEX = 32

LANDMARK_NAMES = {
    0:'NOSE',1:'LEFT_EYE_INNER',2:'LEFT_EYE',3:'LEFT_EYE_OUTER',
    4:'RIGHT_EYE_INNER',5:'RIGHT_EYE',6:'RIGHT_EYE_OUTER',
    7:'LEFT_EAR',8:'RIGHT_EAR',9:'MOUTH_LEFT',10:'MOUTH_RIGHT',
    11:'LEFT_SHOULDER',12:'RIGHT_SHOULDER',13:'LEFT_ELBOW',
    14:'RIGHT_ELBOW',15:'LEFT_WRIST',16:'RIGHT_WRIST',
    17:'LEFT_PINKY',18:'RIGHT_PINKY',19:'LEFT_INDEX',
    20:'RIGHT_INDEX',21:'LEFT_THUMB',22:'RIGHT_THUMB',
    23:'LEFT_HIP',24:'RIGHT_HIP',25:'LEFT_KNEE',26:'RIGHT_KNEE',
    27:'LEFT_ANKLE',28:'RIGHT_ANKLE',29:'LEFT_HEEL',
    30:'RIGHT_HEEL',31:'LEFT_FOOT_INDEX',32:'RIGHT_FOOT_INDEX',
}

POSE_CONNECTIONS = [
    (LEFT_SHOULDER,RIGHT_SHOULDER),
    (LEFT_SHOULDER,LEFT_ELBOW),(LEFT_ELBOW,LEFT_WRIST),
    (RIGHT_SHOULDER,RIGHT_ELBOW),(RIGHT_ELBOW,RIGHT_WRIST),
    (LEFT_SHOULDER,LEFT_HIP),(RIGHT_SHOULDER,RIGHT_HIP),
    (LEFT_HIP,RIGHT_HIP),
    (LEFT_HIP,LEFT_KNEE),(LEFT_KNEE,LEFT_ANKLE),
    (RIGHT_HIP,RIGHT_KNEE),(RIGHT_KNEE,RIGHT_ANKLE),
    (LEFT_ANKLE,LEFT_HEEL),(LEFT_HEEL,LEFT_FOOT_INDEX),
    (RIGHT_ANKLE,RIGHT_HEEL),(RIGHT_HEEL,RIGHT_FOOT_INDEX),
    (NOSE,LEFT_EYE),(NOSE,RIGHT_EYE),
    (LEFT_EYE,LEFT_EAR),(RIGHT_EYE,RIGHT_EAR),
]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_DIR  = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"

def _ensure_model():
    """Download pose landmarker model if not present."""
    if MODEL_PATH.exists():
        return str(MODEL_PATH)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading pose landmarker model (~5MB)...")
    print("Downloading MediaPipe pose model (~5MB)... please wait...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")
        logger.info("Model downloaded to %s", MODEL_PATH)
        return str(MODEL_PATH)
    except Exception as e:
        logger.error("Model download failed: %s", e)
        raise RuntimeError(
            f"Failed to download pose model: {e}\n"
            f"Please manually download from:\n{MODEL_URL}\n"
            f"And place it at: {MODEL_PATH}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# POSE EXTRACTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────
class PoseExtractor:
    """
    MediaPipe Pose wrapper using Tasks API (mediapipe 0.10.30+)
    Automatically downloads the model file on first use.
    """

    MIN_VISIBILITY_THRESHOLD = 0.2
    TORSO_LANDMARKS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 0,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
    ):
        from mediapipe.tasks.python.vision import (
            PoseLandmarker, PoseLandmarkerOptions, RunningMode
        )
        from mediapipe.tasks import python as mp_tasks

        model_path = _ensure_model()

        running_mode = RunningMode.IMAGE if static_image_mode else RunningMode.IMAGE

        self._options = PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(self._options)
        self._static_mode = static_image_mode
        logger.debug("PoseExtractor (Tasks API) initialised")

    def extract(self, bgr_frame: np.ndarray):
        """
        Run pose detection on one BGR frame.
        Returns (raw_lm, norm_lm, visibility, pose_valid)
        """
        import mediapipe as mp

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Create MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        try:
            result = self._landmarker.detect(mp_image)
        except Exception as e:
            logger.error("Pose detection error: %s", e)
            return None, None, None, False

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None, None, None, False

        # Get first pose landmarks
        landmarks = result.pose_landmarks[0]

        # Build raw array (33, 4) — x, y, z, visibility
        raw = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0]
             for lm in landmarks],
            dtype=np.float32
        )

        if len(raw) < 33:
            return None, None, None, False

        visibility = raw[:, 3]

        # Check torso visibility
        pose_valid = all(
            visibility[i] >= self.MIN_VISIBILITY_THRESHOLD
            for i in self.TORSO_LANDMARKS
        )

        if not pose_valid:
            return raw, None, visibility, False

        norm = self._normalize(raw[:, :3])
        return raw, norm, visibility, True

    def draw_skeleton(self, bgr_frame: np.ndarray, raw_landmarks, color_override=None):
        """Draw colored skeleton on frame."""
        if raw_landmarks is None:
            return bgr_frame.copy()

        frame_copy = bgr_frame.copy()
        color = color_override or (0, 220, 80)
        h, w = frame_copy.shape[:2]

        pts = {}
        for i in range(min(33, len(raw_landmarks))):
            x = int(raw_landmarks[i][0] * w)
            y = int(raw_landmarks[i][1] * h)
            pts[i] = (x, y)

        for (a, b) in POSE_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame_copy, pts[a], pts[b], color, 2, cv2.LINE_AA)

        for i, (x, y) in pts.items():
            cv2.circle(frame_copy, (x, y), 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame_copy, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame_copy

    def extract_from_video(self, video_path: str, frame_skip: int = 2):
        """Generator yielding pose data for each frame of a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip == 0:
                    raw, norm, vis, valid = self.extract(frame)
                    yield frame_idx, raw, norm, vis, valid
                frame_idx += 1
        finally:
            cap.release()

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass

    def _normalize(self, xyz: np.ndarray) -> np.ndarray:
        """Torso-center + torso-length normalization."""
        torso_pts   = xyz[self.TORSO_LANDMARKS, :]
        torso_center = torso_pts.mean(axis=0)
        centered    = xyz - torso_center
        torso_length = np.mean(np.linalg.norm(torso_pts - torso_center, axis=1))
        if torso_length < 1e-6:
            return centered
        return centered / torso_length