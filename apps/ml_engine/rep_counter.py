"""
State-Machine Repetition Counter
==================================
Fixed thresholds based on observed real-world angles from mediapipe.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger('posture_coach')


@dataclass
class RepConfig:
    exercise:         str
    primary_feature:  str
    down_threshold:   float
    up_threshold:     float
    direction:        str          # "down_to_up" | "up_to_down"
    ema_alpha:        float = 0.3
    hold_frames:      int   = 2    # reduced from 3 → faster response


# ── Corrected thresholds based on real observed angles ──────────────────────
# Shoulder press: elbow angle ~90° at goal-post, ~160-170° arms fully raised
# Squat:          knee angle ~170° standing, ~90° at bottom
# Curl:           elbow angle ~160° extended, ~40° fully curled
# Push-up:        elbow angle ~160° extended, ~70° at bottom

REP_CONFIGS = {
    "squat": RepConfig(
        exercise="squat",
        primary_feature="left_knee_angle",
        down_threshold=110.0,    # < 110° = at bottom
        up_threshold=150.0,      # > 150° = standing
        direction="up_to_down",
        ema_alpha=0.3,
    ),
    "pushup": RepConfig(
        exercise="pushup",
        primary_feature="left_elbow_angle",
        down_threshold=90.0,     # < 90° = chest near floor
        up_threshold=145.0,      # > 145° = arms extended
        direction="up_to_down",
        ema_alpha=0.3,
    ),
    "barbell_curl": RepConfig(
        exercise="barbell_curl",
        primary_feature="left_elbow_angle",
        down_threshold=70.0,     # < 70° = fully curled (top)
        up_threshold=145.0,      # > 145° = arm extended (bottom)
        direction="down_to_up",
        ema_alpha=0.35,
    ),
    "hammer_curl": RepConfig(
        exercise="hammer_curl",
        primary_feature="left_elbow_angle",
        down_threshold=70.0,
        up_threshold=145.0,
        direction="down_to_up",
        ema_alpha=0.35,
    ),
    "shoulder_press": RepConfig(
        exercise="shoulder_press",
        primary_feature="left_elbow_angle",
        down_threshold=100.0,    # < 100° = goal post (start/bottom position)
        up_threshold=150.0,      # > 150° = arms raised overhead (top)
        direction="up_to_down",  # start at bottom, go up, come back = 1 rep
        ema_alpha=0.3,
    ),
}


class RepState(str, Enum):
    IDLE      = "idle"
    AT_BOTTOM = "at_bottom"
    AT_TOP    = "at_top"


class RepCounter:
    """Session-persistent exercise repetition counter."""

    def __init__(self, exercise_name: str):
        self.exercise = exercise_name
        self.config   = REP_CONFIGS.get(exercise_name)
        if not self.config:
            # Fallback config so it never crashes
            self.config = RepConfig(
                exercise=exercise_name,
                primary_feature="left_elbow_angle",
                down_threshold=90.0,
                up_threshold=150.0,
                direction="up_to_down",
            )

        self.reps: int              = 0
        self.state: RepState        = RepState.IDLE
        self._ema_angle: Optional[float] = None
        self._hold_count: int       = 0
        self._angle_history: list   = []
        self._debug_log: list       = []

    def update(self, features: dict) -> dict:
        """Process one frame of features. Returns rep count info."""
        # Try primary feature, fall back to alternatives
        angle = features.get(self.config.primary_feature)

        # Fallback to right side if left not available
        if angle is None or angle == 0:
            alt = self.config.primary_feature.replace('left_', 'right_')
            angle = features.get(alt)

        if angle is None or angle == 0:
            return self._response(False)

        smoothed    = self._ema(angle)
        rep_counted = self._advance(smoothed)

        self._angle_history.append(round(smoothed, 1))
        if len(self._angle_history) > 200:
            self._angle_history = self._angle_history[-200:]

        return self._response(rep_counted, smoothed)

    def reset(self):
        self.reps        = 0
        self.state       = RepState.IDLE
        self._ema_angle  = None
        self._hold_count = 0
        self._angle_history.clear()

    def to_dict(self) -> dict:
        return {
            "exercise":      self.exercise,
            "reps":          self.reps,
            "state":         self.state.value,
            "_ema_angle":    self._ema_angle,
            "_hold_count":   self._hold_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RepCounter":
        obj = cls(data["exercise"])
        obj.reps        = data.get("reps", 0)
        obj.state       = RepState(data.get("state", "idle"))
        obj._ema_angle  = data.get("_ema_angle")
        obj._hold_count = data.get("_hold_count", 0)
        return obj

    # ── Private ──────────────────────────────────────────────────────────────

    def _advance(self, angle: float) -> bool:
        cfg = self.config
        rep_counted = False

        if cfg.direction == "up_to_down":
            # Example: shoulder press
            # IDLE → AT_BOTTOM (angle < down_thresh, goal post position)
            # AT_BOTTOM → AT_TOP (angle > up_thresh, arms raised)
            # AT_TOP → AT_BOTTOM → rep counted

            if self.state == RepState.IDLE:
                # Accept starting from either position
                if angle < cfg.down_threshold:
                    self._goto(RepState.AT_BOTTOM)
                elif angle > cfg.up_threshold:
                    self._goto(RepState.AT_TOP)

            elif self.state == RepState.AT_BOTTOM:
                if angle > cfg.up_threshold:
                    self._hold_count += 1
                    if self._hold_count >= cfg.hold_frames:
                        self._goto(RepState.AT_TOP)
                else:
                    self._hold_count = 0

            elif self.state == RepState.AT_TOP:
                if angle < cfg.down_threshold:
                    self._hold_count += 1
                    if self._hold_count >= cfg.hold_frames:
                        self.reps += 1
                        rep_counted = True
                        self._goto(RepState.AT_BOTTOM)
                        logger.debug("Rep! %s total=%d angle=%.1f", self.exercise, self.reps, angle)
                else:
                    self._hold_count = 0

        elif cfg.direction == "down_to_up":
            # Example: curl
            # IDLE → AT_BOTTOM (arm extended, angle > up_thresh)
            # AT_BOTTOM → AT_TOP (arm curled, angle < down_thresh)
            # AT_TOP → AT_BOTTOM → rep counted

            if self.state == RepState.IDLE:
                if angle > cfg.up_threshold:
                    self._goto(RepState.AT_BOTTOM)
                elif angle < cfg.down_threshold:
                    self._goto(RepState.AT_TOP)

            elif self.state == RepState.AT_BOTTOM:
                if angle < cfg.down_threshold:
                    self._hold_count += 1
                    if self._hold_count >= cfg.hold_frames:
                        self._goto(RepState.AT_TOP)
                else:
                    self._hold_count = 0

            elif self.state == RepState.AT_TOP:
                if angle > cfg.up_threshold:
                    self._hold_count += 1
                    if self._hold_count >= cfg.hold_frames:
                        self.reps += 1
                        rep_counted = True
                        self._goto(RepState.AT_BOTTOM)
                        logger.debug("Rep! %s total=%d angle=%.1f", self.exercise, self.reps, angle)
                else:
                    self._hold_count = 0

        return rep_counted

    def _goto(self, state: RepState):
        self.state       = state
        self._hold_count = 0

    def _ema(self, angle: float) -> float:
        if self._ema_angle is None:
            self._ema_angle = angle
        else:
            a = self.config.ema_alpha
            self._ema_angle = a * angle + (1 - a) * self._ema_angle
        return self._ema_angle

    def _response(self, rep_counted: bool, angle: float = None) -> dict:
        return {
            "reps":            self.reps,
            "state":           self.state.value,
            "primary_angle":   round(angle, 1) if angle else None,
            "rep_just_counted": rep_counted,
        }