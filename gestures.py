"""Rule-based gesture recognition on MediaPipe hand landmarks.

This module is pure logic: no camera, no OpenCV, no pyautogui. You feed it
21 landmarks + a timestamp each frame and it hands back a list of Events.
That keeps it unit-testable (see test_gestures.py) and means the same engine
can later run inside Docker or on a Pi with only the I/O layer swapped out.

Pipeline per frame:
    landmarks -> classify_pose() -> debounce (pose must hold N frames)
              -> per-pose handler (swipe / scroll / cursor / click / close)
              -> Events

The pose classifier here is hand-written geometry. It is deliberately the
piece you'll want to replace with a trained model later.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

# --- MediaPipe hand landmark indices ----------------------------------------
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

FINGERS = {  # finger -> (PIP joint, tip)
    "index": (INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_PIP, RING_TIP),
    "pinky": (PINKY_PIP, PINKY_TIP),
}
PALM = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


class Pose(str, Enum):
    NONE = "none"  # no hand in frame
    OPEN_PALM = "open_palm"
    FIST = "fist"
    POINT = "point"  # index finger only
    TWO_FINGERS = "two_fingers"  # index + middle
    PINCH = "pinch"  # thumb tip touching index tip
    OTHER = "other"


@dataclass(frozen=True)
class Event:
    kind: str  # swipe_left | swipe_right | scroll | move | click | close_tab
    value: object = None  # scroll: int clicks (+ = up); move: (x, y) in 0..1


@dataclass
class Config:
    """Every tunable number in one place.

    Unless noted, distances are in units of *frame height* (x is
    aspect-corrected so a unit is the same size horizontally and vertically).
    """

    # Pose classification
    finger_extended_ratio: float = 1.15  # tip must be this much farther from wrist than the PIP joint
    pinch_on: float = 0.30  # thumb-index gap / palm size below which a pinch starts
    pinch_off: float = 0.45  # ...and above which it ends (hysteresis stops flicker)
    pinch_index_reach: float = 1.25  # index tip must reach past its knuckle, so a fist isn't a pinch
    pose_stable_frames: int = 3  # frames a pose must hold before it counts (~100 ms at 30 fps)

    # Swipe (open palm moving sideways fast)
    swipe_window_s: float = 0.30
    swipe_min_dist: float = 0.30
    swipe_axis_ratio: float = 2.0  # horizontal travel must be 2x the vertical travel
    swipe_cooldown_s: float = 0.8  # also swallows the "return stroke" of your hand

    # Scroll (two fingers, joystick style: offset from where the pose started)
    scroll_deadzone: float = 0.04
    scroll_gain: float = 60.0  # scroll clicks per second, per frame-height past the deadzone

    # Cursor (index finger). Region of the *camera* frame (normalized x0, y0, x1, y1)
    # that maps onto the whole screen, so you can reach screen edges comfortably.
    cursor_region: tuple[float, float, float, float] = (0.2, 0.15, 0.8, 0.75)
    one_euro_min_cutoff: float = 1.5  # lower = smoother when still, but laggier
    one_euro_beta: float = 8.0  # higher = less lag when moving fast

    # Click / close
    click_cooldown_s: float = 0.35
    fist_hold_s: float = 0.8


# --- Smoothing ----------------------------------------------------------------


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012): heavy smoothing when the
    signal is slow (kills jitter), light smoothing when it's fast (kills lag)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t = x, t
            return x
        dt = max(t - self._t, 1e-6)
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * (x - self._x) / dt + (1 - a_d) * self._dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self._dx), dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x


# --- Pose classification --------------------------------------------------------


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))  # 2D only; MediaPipe's z is noisy


def palm_size(pts: np.ndarray) -> float:
    return max(_dist(pts[WRIST], pts[MIDDLE_MCP]), 1e-6)


def finger_extended(pts: np.ndarray, finger: str, cfg: Config) -> bool:
    """Orientation-independent: a straight finger's tip is much farther from the
    wrist than its middle joint; a curled one's tip is closer."""
    pip, tip = FINGERS[finger]
    return _dist(pts[WRIST], pts[tip]) > cfg.finger_extended_ratio * _dist(pts[WRIST], pts[pip])


def classify_pose(pts: np.ndarray, cfg: Config, pinching: bool = False) -> Pose:
    """pts: (21, 2+) aspect-corrected landmarks. `pinching` enables hysteresis."""
    gap = _dist(pts[THUMB_TIP], pts[INDEX_TIP]) / palm_size(pts)
    index_reaching = _dist(pts[WRIST], pts[INDEX_TIP]) > cfg.pinch_index_reach * _dist(
        pts[WRIST], pts[INDEX_MCP]
    )
    if gap < (cfg.pinch_off if pinching else cfg.pinch_on) and index_reaching:
        return Pose.PINCH

    ext = {f: finger_extended(pts, f, cfg) for f in FINGERS}
    n = sum(ext.values())
    if n == 4:
        return Pose.OPEN_PALM
    if n == 0:
        return Pose.FIST
    if ext["index"] and n == 1:
        return Pose.POINT
    if ext["index"] and ext["middle"] and n == 2:
        return Pose.TWO_FINGERS
    return Pose.OTHER


# --- Gesture engine ---------------------------------------------------------------


class GestureEngine:
    """Stateful: call update() once per frame, in time order."""

    def __init__(self, config: Config | None = None, aspect: float = 16 / 9):
        self.cfg = config or Config()
        self.aspect = aspect  # frame width / height
        self.pose = Pose.NONE  # debounced pose (what drives actions)
        self.raw_pose = Pose.NONE  # this frame's pose (handy for the HUD)
        self._candidate, self._candidate_count = Pose.NONE, 0
        self._last_t: float | None = None

        self._swipe_hist: deque[tuple[float, np.ndarray]] = deque()
        self._swipe_cooldown_until = 0.0
        self._scroll_anchor: float | None = None
        self._scroll_accum = 0.0
        self._click_cooldown_until = 0.0
        self._fist_start: float | None = None
        self._fist_fired = False
        self._fx = OneEuroFilter(self.cfg.one_euro_min_cutoff, self.cfg.one_euro_beta)
        self._fy = OneEuroFilter(self.cfg.one_euro_min_cutoff, self.cfg.one_euro_beta)

    def update(self, landmarks: Sequence[Sequence[float]] | None, t: float) -> list[Event]:
        """landmarks: 21 (x, y[, z]) points normalized to the frame (MediaPipe's
        output), or None if no hand. t: time in seconds (monotonic)."""
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t

        raw = pts = None
        if landmarks is None:
            self.raw_pose = Pose.NONE
        else:
            raw = np.asarray(landmarks, dtype=float)
            pts = raw.copy()
            pts[:, 0] *= self.aspect
            self.raw_pose = classify_pose(pts, self.cfg, pinching=self.raw_pose is Pose.PINCH)

        # Debounce: a pose (including "no hand") must hold for N frames before it
        # takes over. Kills single-frame misclassifications and brief dropouts
        # from motion blur mid-swipe.
        if self.raw_pose is self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = self.raw_pose, 1

        events: list[Event] = []
        if self._candidate is not self.pose and self._candidate_count >= self.cfg.pose_stable_frames:
            events += self._transition(self._candidate, t)

        if pts is not None and self.raw_pose is self.pose:
            handler = {
                Pose.OPEN_PALM: self._on_open_palm,
                Pose.TWO_FINGERS: self._on_two_fingers,
                Pose.POINT: self._on_point,
                Pose.FIST: self._on_fist,
            }.get(self.pose)
            if handler:
                events += handler(pts, raw, t, dt)
        return events

    # -- pose enter/exit --

    def _transition(self, new: Pose, t: float) -> list[Event]:
        old, self.pose = self.pose, new
        if old is Pose.OPEN_PALM:
            self._swipe_hist.clear()
        if old is Pose.TWO_FINGERS:
            self._scroll_anchor, self._scroll_accum = None, 0.0

        if new is Pose.POINT:
            self._fx.reset()
            self._fy.reset()
        if new is Pose.FIST:
            self._fist_start, self._fist_fired = t, False
        if new is Pose.PINCH and t >= self._click_cooldown_until:
            self._click_cooldown_until = t + self.cfg.click_cooldown_s
            return [Event("click")]
        return []

    # -- per-pose handlers --

    def _on_open_palm(self, pts, raw, t, dt) -> list[Event]:
        cfg, hist = self.cfg, self._swipe_hist
        center = pts[PALM, :2].mean(axis=0)
        if t < self._swipe_cooldown_until:
            hist.clear()
            return []
        hist.append((t, center))
        while t - hist[0][0] > cfg.swipe_window_s:
            hist.popleft()
        dx, dy = center - hist[0][1]
        if abs(dx) >= cfg.swipe_min_dist and abs(dx) >= cfg.swipe_axis_ratio * abs(dy):
            hist.clear()
            self._swipe_cooldown_until = t + cfg.swipe_cooldown_s
            # Frame is mirrored, so +x is the user's right.
            return [Event("swipe_right" if dx > 0 else "swipe_left")]
        return []

    def _on_two_fingers(self, pts, raw, t, dt) -> list[Event]:
        cfg = self.cfg
        y = (pts[INDEX_TIP, 1] + pts[MIDDLE_TIP, 1]) / 2
        if self._scroll_anchor is None:
            self._scroll_anchor = y
            return []
        offset = self._scroll_anchor - y  # > 0 when the hand moved up
        excess = abs(offset) - cfg.scroll_deadzone
        if excess <= 0:
            return []
        self._scroll_accum += math.copysign(excess, offset) * cfg.scroll_gain * dt
        clicks = int(self._scroll_accum)  # emit whole clicks, carry the remainder
        if clicks:
            self._scroll_accum -= clicks
            return [Event("scroll", clicks)]
        return []

    def _on_point(self, pts, raw, t, dt) -> list[Event]:
        x0, y0, x1, y1 = self.cfg.cursor_region
        nx = float(np.clip((raw[INDEX_TIP, 0] - x0) / (x1 - x0), 0.0, 1.0))
        ny = float(np.clip((raw[INDEX_TIP, 1] - y0) / (y1 - y0), 0.0, 1.0))
        return [Event("move", (self._fx(nx, t), self._fy(ny, t)))]

    def _on_fist(self, pts, raw, t, dt) -> list[Event]:
        if (
            not self._fist_fired
            and self._fist_start is not None
            and t - self._fist_start >= self.cfg.fist_hold_s
        ):
            self._fist_fired = True  # once per fist; open your hand to re-arm
            return [Event("close_tab")]
        return []
