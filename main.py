"""Webcam hand-gesture control.

Gestures (hold the pose for a split second so it registers):
  open palm, swipe sideways quickly   -> next / previous tab
  index + middle finger up            -> scroll (move up/down from where you started, like a joystick)
  index finger only                   -> move the mouse cursor
  pinch thumb + index                 -> click
  fist, held ~1 second                -> close tab

Run:
  uv run python main.py              # live: controls your mouse and keyboard
  uv run python main.py --dry-run    # prints what it would do, touches nothing

In the video window: q quits, p pauses/resumes actions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gestures import Config, Event, GestureEngine, Pose

HERE = Path(__file__).resolve().parent

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]  # fmt: skip

POSE_COLORS = {  # BGR
    Pose.OPEN_PALM: (255, 200, 0),
    Pose.TWO_FINGERS: (0, 200, 255),
    Pose.POINT: (0, 255, 0),
    Pose.PINCH: (255, 0, 255),
    Pose.FIST: (0, 0, 255),
}


class ActionRunner:
    """Turns gesture Events into real mouse/keyboard input."""

    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.mod = "command" if sys.platform == "darwin" else "ctrl"
        if dry_run:
            return
        import pyautogui  # imported lazily so --dry-run works without a desktop session

        pyautogui.PAUSE = 0  # default sleeps 0.1 s after every call, which stalls the camera loop
        pyautogui.FAILSAFE = True  # shove the real mouse into a screen corner to abort
        self.gui = pyautogui
        self.screen_w, self.screen_h = pyautogui.size()

    def run(self, ev: Event) -> None:
        if self.dry_run:
            if ev.kind != "move":  # moves fire every frame; too noisy to print
                print(f"[dry-run] {ev.kind}" + (f" {ev.value}" if ev.value is not None else ""))
            return
        g = self.gui
        if ev.kind == "move":
            x, y = ev.value
            # Stay a couple of pixels off the edges so we never trip the corner failsafe ourselves.
            g.moveTo(2 + x * (self.screen_w - 5), 2 + y * (self.screen_h - 5))
        elif ev.kind == "scroll":
            g.scroll(ev.value)
        elif ev.kind == "click":
            g.click()
        elif ev.kind == "swipe_left":  # like flicking a page away: go forward
            g.hotkey("ctrl", "tab")  # Ctrl+Tab switches tabs on macOS browsers too
        elif ev.kind == "swipe_right":
            g.hotkey("ctrl", "shift", "tab")
        elif ev.kind == "close_tab":
            g.hotkey(self.mod, "w")


def draw_hand(frame, landmarks, color) -> None:
    h, w = frame.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (200, 200, 200), 2)
    for p in px:
        cv2.circle(frame, p, 4, color, cv2.FILLED)
    cv2.circle(frame, px[8], 10, color, 2)  # index tip


def draw_cursor_region(frame, cfg: Config) -> None:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = cfg.cursor_region
    cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (80, 80, 80), 1)


def put_text(frame, text, y, color=(255, 255, 255), scale=0.7) -> None:
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Control your computer with hand gestures.")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--model", type=Path, default=HERE / "hand_landmarker.task")
    p.add_argument("--dry-run", action="store_true", help="print actions instead of performing them")
    p.add_argument("--no-cursor", action="store_true", help="disable point-to-move and pinch-to-click")
    p.add_argument("--scroll-gain", type=float, default=Config.scroll_gain,
                   help="scroll speed; raise it on macOS, where one scroll click is tiny")  # fmt: skip
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}")

    cfg = Config(scroll_gain=args.scroll_gain)
    actions = ActionRunner(dry_run=args.dry_run)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Could not open the webcam. On macOS, check System Settings > Privacy > Camera.")

    engine: GestureEngine | None = None
    paused = False
    flash, flash_until = "", 0.0
    fps, t_prev = 0.0, time.monotonic()
    t0, last_ts = t_prev, -1

    print("q = quit, p = pause/resume" + ("  (dry run)" if args.dry_run else ""))
    misses = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Cameras (especially on macOS) often return empty frames while
                # warming up or while the permission prompt is showing, so give
                # them a few seconds before giving up.
                misses += 1
                if misses > 100:
                    sys.exit(
                        "No frames from the webcam. Check that your terminal app is allowed in "
                        "System Settings > Privacy & Security > Camera, that no other app (Zoom, "
                        "FaceTime, Photo Booth) is using it, or try --camera 1."
                    )
                time.sleep(0.05)
                continue
            misses = 0

            frame = cv2.flip(frame, 1)  # mirror, so moving right looks like moving right
            h, w = frame.shape[:2]
            if engine is None:
                engine = GestureEngine(cfg, aspect=w / h)

            now = time.monotonic()
            # VIDEO mode needs real, strictly increasing timestamps (a fixed +33 ms
            # drifts whenever the camera isn't exactly 30 fps).
            ts = max(int((now - t0) * 1000), last_ts + 1)
            last_ts = ts

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            landmarks = None
            if result.hand_landmarks:
                landmarks = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]

            events = engine.update(landmarks, now)
            if args.no_cursor:
                events = [e for e in events if e.kind not in ("move", "click")]

            for ev in events:
                if ev.kind != "move":
                    flash, flash_until = ev.kind + (f" {ev.value}" if ev.value is not None else ""), now + 1.0
                if paused:
                    continue
                try:
                    actions.run(ev)
                except Exception as exc:  # incl. pyautogui.FailSafeException
                    paused = True
                    print(f"Paused after error: {exc!r}")

            # --- HUD ---
            if not args.no_cursor:
                draw_cursor_region(frame, cfg)
            if landmarks:
                draw_hand(frame, landmarks, POSE_COLORS.get(engine.pose, (255, 255, 255)))
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1 / dt) if fps else 1 / dt
            put_text(frame, f"pose: {engine.pose.value}   fps: {fps:.0f}", 30)
            if now < flash_until:
                put_text(frame, flash, 62, (0, 255, 255), 0.9)
            if paused:
                put_text(frame, "PAUSED (p to resume)", h - 20, (0, 0, 255))

            cv2.imshow("Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
