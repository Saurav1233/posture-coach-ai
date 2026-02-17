"""
State-Machine Repetition Counter
==================================
Counts exercise repetitions using a finite state machine driven by
exercise-specific joint angle thresholds.

Design Rationale
─────────────────
A simple threshold-on-angle approach suffers from jitter (rapid false
triggers at the boundary). We solve this with:

  1. State Machine  — transitions only happen when clear state boundaries
     are crossed (no immediate re-trigger).
  2. Hysteresis Gap — there is a dead-zone between "entering rep" and
     "completing rep" angles to prevent noise-driven double counts.
  3. EMA Smoothing  — exponential moving average on the primary angle
     to suppress frame-to-frame noise from MediaPipe jitter.
  4. Hold Frames    — a minimum number of consecutive frames must be
     observed in each state before transitioning.

State Diagram (example: squat)
────────────────────────────────
  IDLE ──[angle < DOWN_THRESHOLD]──► DOWN
  DOWN ──[angle > UP_THRESHOLD]────► UP  (rep += 1)
  UP   ──[angle < DOWN_THRESHOLD]──► DOWN

For exercises with upward press (shoulder press, push-up):
  IDLE ──[angle < FLEXED_THRESHOLD]─► FLEXED
  FLEXED ─[angle > EXTENDED_THRESH]─► EXTENDED (rep += 1)
  EXTENDED ─[angle < FLEXED_THRESH]─► FLEXED

Session Persistence
────────────────────
The counter object is stored in Django's session store (serialized to dict).
The `to_dict()` / `from_dict()` methods enable this.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger('posture_coach')


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RepConfig:
    """Configuration for one exercise's repetition counting."""
    exercise: str
    primary_feature: str          # Feature name used to drive state machine
    down_threshold: float         # Angle that means "rep bottom position"
    up_threshold: float           # Angle that means "rep top position"
    direction: str                # "down_to_up" | "up_to_down"
    # direction = "down_to_up": rep counted when going from low to high angle
    # direction = "up_to_down": rep counted when going from high to low angle
    ema_alpha: float = 0.25       # EMA smoothing factor [0=no smooth, 1=no memory]
    hold_frames: int = 3          # Frames to hold in state before transitioning


# Thresholds are based on biomechanics literature and common practice
REP_CONFIGS: dict[str, RepConfig] = {
    "squat": RepConfig(
        exercise="squat",
        primary_feature="left_knee_angle",
        down_threshold=105.0,    # Knee < 105° → at bottom of squat
        up_threshold=155.0,      # Knee > 155° → standing (rep complete)
        direction="up_to_down",  # rep counted at bottom-to-top transition
    ),
    "pushup": RepConfig(
        exercise="pushup",
        primary_feature="left_elbow_angle",
        down_threshold=100.0,    # Elbow < 100° → chest near floor (bottom)
        up_threshold=150.0,      # Elbow > 150° → arms extended (rep complete)
        direction="up_to_down",
    ),
    "barbell_curl": RepConfig(
        exercise="barbell_curl",
        primary_feature="left_elbow_angle",
        down_threshold=60.0,     # Elbow < 60° → bar near shoulder (top)
        up_threshold=140.0,      # Elbow > 140° → arm extended (bottom)
        direction="down_to_up",  # rep counted top-to-bottom (full cycle)
        ema_alpha=0.3,
    ),
    "hammer_curl": RepConfig(
        exercise="hammer_curl",
        primary_feature="left_elbow_angle",
        down_threshold=60.0,
        up_threshold=140.0,
        direction="down_to_up",
        ema_alpha=0.3,
    ),
    "shoulder_press": RepConfig(
        exercise="shoulder_press",
        primary_feature="left_elbow_angle",
        down_threshold=105.0,    # Start position — elbows ~90°
        up_threshold=155.0,      # Arms extended overhead
        direction="up_to_down",  # rep counted when arms go up and return
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────

class RepState(str, Enum):
    IDLE        = "idle"
    AT_BOTTOM   = "at_bottom"   # "down" phase
    AT_TOP      = "at_top"      # "up" phase
    COMPLETED   = "completed"   # transitional — immediately returns to IDLE


class RepCounter:
    """
    Session-persistent exercise repetition counter.

    Usage
    ─────
    counter = RepCounter("squat")
    result = counter.update(features_dict)
    print(result["reps"])    # total reps this session
    """

    def __init__(self, exercise_name: str):
        self.exercise = exercise_name
        self.config = REP_CONFIGS.get(exercise_name)
        if not self.config:
            raise ValueError(f"No rep config for exercise: {exercise_name}")

        self.reps: int = 0
        self.state: RepState = RepState.IDLE
        self._ema_angle: Optional[float] = None
        self._state_hold_count: int = 0
        self._angle_history: list[float] = []  # For debugging / analytics

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def update(self, features: dict[str, float]) -> dict:
        """
        Process a new frame's features and update rep count.

        Returns
        ───────
        {
          "reps": int,
          "state": str,
          "primary_angle": float,
          "rep_just_counted": bool,
        }
        """
        angle = features.get(self.config.primary_feature, None)
        if angle is None:
            return self._make_response(rep_counted=False)

        # EMA smoothing
        smoothed = self._apply_ema(angle)
        self._angle_history.append(smoothed)
        if len(self._angle_history) > 300:
            self._angle_history.pop(0)

        rep_counted = self._advance_state(smoothed)
        return self._make_response(
            rep_counted=rep_counted,
            primary_angle=smoothed,
        )

    def reset(self):
        """Reset rep count and state (new session)."""
        self.reps = 0
        self.state = RepState.IDLE
        self._ema_angle = None
        self._state_hold_count = 0
        self._angle_history.clear()

    def to_dict(self) -> dict:
        """Serialize for Django session storage."""
        return {
            "exercise": self.exercise,
            "reps": self.reps,
            "state": self.state.value,
            "_ema_angle": self._ema_angle,
            "_state_hold_count": self._state_hold_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RepCounter":
        """Deserialize from Django session storage."""
        counter = cls(data["exercise"])
        counter.reps = data.get("reps", 0)
        counter.state = RepState(data.get("state", "idle"))
        counter._ema_angle = data.get("_ema_angle")
        counter._state_hold_count = data.get("_state_hold_count", 0)
        return counter

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE STATE MACHINE LOGIC
    # ──────────────────────────────────────────────────────────────────────

    def _advance_state(self, angle: float) -> bool:
        """
        Advance the state machine given the current smoothed angle.
        Returns True if a new rep was counted on this frame.
        """
        cfg = self.config
        rep_counted = False

        if cfg.direction == "up_to_down":
            # e.g. Squat: UP (standing) → DOWN (squatting) → UP (rep)
            if self.state == RepState.IDLE:
                if angle > cfg.up_threshold:
                    self._transition_to(RepState.AT_TOP)
                elif angle < cfg.down_threshold:
                    # Started in mid-rep position (partially squatted)
                    self._transition_to(RepState.AT_BOTTOM)

            elif self.state == RepState.AT_TOP:
                if angle < cfg.down_threshold:
                    self._state_hold_count += 1
                    if self._state_hold_count >= cfg.hold_frames:
                        self._transition_to(RepState.AT_BOTTOM)
                else:
                    self._state_hold_count = 0

            elif self.state == RepState.AT_BOTTOM:
                if angle > cfg.up_threshold:
                    self._state_hold_count += 1
                    if self._state_hold_count >= cfg.hold_frames:
                        self.reps += 1
                        rep_counted = True
                        self._transition_to(RepState.AT_TOP)
                        logger.debug("Rep counted: %s total=%d", self.exercise, self.reps)
                else:
                    self._state_hold_count = 0

        elif cfg.direction == "down_to_up":
            # e.g. Curl: BOTTOM (extended) → TOP (curled) → BOTTOM (rep)
            if self.state == RepState.IDLE:
                if angle > cfg.up_threshold:
                    self._transition_to(RepState.AT_BOTTOM)
                elif angle < cfg.down_threshold:
                    self._transition_to(RepState.AT_TOP)

            elif self.state == RepState.AT_BOTTOM:
                if angle < cfg.down_threshold:
                    self._state_hold_count += 1
                    if self._state_hold_count >= cfg.hold_frames:
                        self._transition_to(RepState.AT_TOP)
                else:
                    self._state_hold_count = 0

            elif self.state == RepState.AT_TOP:
                if angle > cfg.up_threshold:
                    self._state_hold_count += 1
                    if self._state_hold_count >= cfg.hold_frames:
                        self.reps += 1
                        rep_counted = True
                        self._transition_to(RepState.AT_BOTTOM)
                        logger.debug("Rep counted: %s total=%d", self.exercise, self.reps)
                else:
                    self._state_hold_count = 0

        return rep_counted

    def _transition_to(self, new_state: RepState):
        self.state = new_state
        self._state_hold_count = 0

    def _apply_ema(self, angle: float) -> float:
        if self._ema_angle is None:
            self._ema_angle = angle
        else:
            alpha = self.config.ema_alpha
            self._ema_angle = alpha * angle + (1.0 - alpha) * self._ema_angle
        return self._ema_angle

    def _make_response(
        self,
        rep_counted: bool = False,
        primary_angle: Optional[float] = None,
    ) -> dict:
        return {
            "reps": self.reps,
            "state": self.state.value,
            "primary_angle": round(primary_angle, 1) if primary_angle else None,
            "rep_just_counted": rep_counted,
        }
