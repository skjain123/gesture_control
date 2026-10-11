"""Tests for gestures.py using synthetic hands (no camera needed).

Run: uv run pytest
"""

import numpy as np
import pytest

from gestures import Config, GestureEngine, OneEuroFilter, Pose, classify_pose

FPS = 30
DT = 1 / FPS


def make_hand(extended=(), cx=0.5, cy=0.5, pinch=False, thumb_out=True):
    """Upright synthetic hand (fingers pointing up the image), normalized coords.

    extended: subset of {"index", "middle", "ring", "pinky"} that are straight.
    """
    pts = np.zeros((21, 3))
    pts[0] = (cx, cy + 0.15, 0)  # wrist
    for i, (name, dx) in enumerate([("index", -0.03), ("middle", -0.01), ("ring", 0.01), ("pinky", 0.03)]):
        mcp = 5 + 4 * i
        base = np.array([cx + dx, cy, 0])
        pts[mcp] = base
        if name == "index" and pinch:
            pts[mcp + 1] = base + (0, -0.05, 0)
            pts[mcp + 2] = base + (-0.02, -0.07, 0)
            pts[mcp + 3] = base + (-0.03, -0.07, 0)
        elif name in extended:
            pts[mcp + 1] = base + (0, -0.05, 0)
            pts[mcp + 2] = base + (0, -0.08, 0)
            pts[mcp + 3] = base + (0, -0.11, 0)
        else:  # curled back toward the palm
            pts[mcp + 1] = base + (0, -0.04, 0)
            pts[mcp + 2] = base + (0, -0.02, 0)
            pts[mcp + 3] = base + (0, 0.01, 0)
    pts[1] = (cx - 0.04, cy + 0.12, 0)
    pts[2] = (cx - 0.06, cy + 0.09, 0)
    pts[3] = (cx - 0.08, cy + 0.06, 0)
    if pinch:
        pts[4] = pts[8] + (0.005, 0.005, 0)
    elif thumb_out:
        pts[4] = (cx - 0.12, cy + 0.03, 0)
    else:
        pts[4] = (cx - 0.01, cy + 0.04, 0)  # tucked across the fingers, as in a fist
    return pts


ALL = ("index", "middle", "ring", "pinky")
CFG = Config()


def feed(engine, frames, t0=0.0):
    """frames: list of landmark arrays (or None). Returns all events."""
    events = []
    for i, f in enumerate(frames):
        events += engine.update(f, t0 + i * DT)
    return events


def kinds(events):
    return [e.kind for e in events if e.kind != "move"]


# --- classification -----------------------------------------------------------


@pytest.mark.parametrize(
    "hand,expected",
    [
        (make_hand(ALL), Pose.OPEN_PALM),
        (make_hand((), thumb_out=False), Pose.FIST),
        (make_hand(("index",)), Pose.POINT),
        (make_hand(("index", "middle")), Pose.TWO_FINGERS),
        (make_hand(("middle", "ring", "pinky"), pinch=True), Pose.PINCH),
        (make_hand((), pinch=True), Pose.PINCH),
        (make_hand(("pinky",)), Pose.OTHER),
    ],
)
def test_classify(hand, expected):
    assert classify_pose(hand, CFG) is expected


def test_fist_with_tucked_thumb_is_not_a_pinch():
    # Thumb tip ends up near the curled index tip in a fist; must not click.
    assert classify_pose(make_hand((), thumb_out=False), CFG) is Pose.FIST


# --- debounce -------------------------------------------------------------------


def test_single_frame_glitch_does_not_change_pose():
    e = GestureEngine(aspect=1.0)
    frames = [make_hand(ALL)] * 5 + [make_hand(())] + [make_hand(ALL)] * 5
    feed(e, frames)
    assert e.pose is Pose.OPEN_PALM


# --- swipe ------------------------------------------------------------------------


def test_fast_swipe_right_fires_once():
    e = GestureEngine(aspect=1.0)
    still = [make_hand(ALL, cx=0.3)] * 5
    swipe = [make_hand(ALL, cx=0.3 + 0.4 * i / 6) for i in range(1, 7)]  # 0.4 in 0.2 s
    hold = [make_hand(ALL, cx=0.7)] * 10
    assert kinds(feed(e, still + swipe + hold)) == ["swipe_right"]


def test_fast_swipe_left():
    e = GestureEngine(aspect=1.0)
    frames = [make_hand(ALL, cx=0.7)] * 5 + [make_hand(ALL, cx=0.7 - 0.4 * i / 6) for i in range(1, 7)]
    assert kinds(feed(e, frames)) == ["swipe_left"]


def test_slow_drift_is_not_a_swipe():
    e = GestureEngine(aspect=1.0)
    frames = [make_hand(ALL, cx=0.3 + 0.4 * i / 60) for i in range(61)]  # 0.4 over 2 s
    assert kinds(feed(e, frames)) == []


def test_diagonal_motion_is_not_a_swipe():
    e = GestureEngine(aspect=1.0)
    frames = [make_hand(ALL, cx=0.3 + 0.06 * i, cy=0.3 + 0.06 * i) for i in range(7)]
    frames = [make_hand(ALL, cx=0.3, cy=0.3)] * 5 + frames
    assert kinds(feed(e, frames)) == []


def test_swipe_needs_open_palm():
    e = GestureEngine(aspect=1.0)
    frames = [make_hand(("index", "middle"), cx=0.3)] * 5
    frames += [make_hand(("index", "middle"), cx=0.3 + 0.4 * i / 6) for i in range(1, 7)]
    assert "swipe_right" not in kinds(feed(e, frames))


# --- scroll ------------------------------------------------------------------------


def test_scroll_up_when_hand_moves_up():
    e = GestureEngine(aspect=1.0)
    frames = [make_hand(("index", "middle"), cy=0.5)] * 5 + [make_hand(("index", "middle"), cy=0.35)] * 30
    scrolls = [ev.value for ev in feed(e, frames) if ev.kind == "scroll"]
    assert scrolls and all(v > 0 for v in scrolls)
    # 0.15 offset - 0.04 deadzone = 0.11 * 60 gain ~ 6.6 clicks/s, for ~1 s
    assert 4 <= sum(scrolls) <= 8


def test_scroll_down():
    e = GestureEngine(aspect=1.0)
    frames = [make_hand(("index", "middle"), cy=0.4)] * 5 + [make_hand(("index", "middle"), cy=0.55)] * 30
    assert sum(ev.value for ev in feed(e, frames) if ev.kind == "scroll") < 0


def test_scroll_deadzone():
    e = GestureEngine(aspect=1.0)
    frames = [make_hand(("index", "middle"), cy=0.5)] * 5 + [make_hand(("index", "middle"), cy=0.48)] * 30
    assert kinds(feed(e, frames)) == []


# --- cursor & click ------------------------------------------------------------------


def test_point_moves_cursor_within_screen():
    e = GestureEngine(aspect=1.0)
    moves = [ev.value for ev in feed(e, [make_hand(("index",))] * 10) if ev.kind == "move"]
    assert moves
    assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in moves)


def test_pinch_clicks_once_per_pinch():
    e = GestureEngine(aspect=1.0)
    point = [make_hand(("index",))] * 5
    pinch = [make_hand((), pinch=True)] * 10
    events = feed(e, point + pinch + point + pinch)
    assert kinds(events) == ["click", "click"]


# --- close tab ------------------------------------------------------------------------


def test_fist_must_be_held_to_close_tab():
    e = GestureEngine(aspect=1.0)
    fist = make_hand((), thumb_out=False)
    assert kinds(feed(e, [fist] * 15)) == []  # 0.5 s: too short
    e = GestureEngine(aspect=1.0)
    assert kinds(feed(e, [fist] * 60)) == ["close_tab"]  # 2 s: fires exactly once


def test_hand_leaving_frame_resets_pose():
    e = GestureEngine(aspect=1.0)
    feed(e, [make_hand(ALL)] * 5 + [None] * 5)
    assert e.pose is Pose.NONE


# --- smoothing ----------------------------------------------------------------------------


def test_one_euro_reduces_jitter():
    rng = np.random.default_rng(0)
    noisy = 0.5 + rng.normal(0, 0.01, 300)
    f = OneEuroFilter(min_cutoff=1.5, beta=8.0)
    smooth = np.array([f(x, i * DT) for i, x in enumerate(noisy)])
    assert smooth[30:].std() < noisy[30:].std() / 2
