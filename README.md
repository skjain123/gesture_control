# hand-tracking

Control your computer with hand gestures from a webcam, using
[MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).

## Gestures

| Gesture | Action |
|---|---|
| Open palm, quick swipe left / right | Next / previous tab (Ctrl+Tab / Ctrl+Shift+Tab) |
| Index + middle finger up, move hand up/down | Scroll (joystick style, relative to where you started) |
| Index finger only | Move the mouse cursor |
| Pinch thumb and index | Click |
| Fist, held ~1 s | Close tab (Cmd/Ctrl+W) |

Hold a pose for a split second (~100 ms) before it registers.

## Run

```bash
uv sync
uv run python main.py --dry-run   # prints actions, doesn't touch your mouse/keyboard
uv run python main.py             # live
uv run python main.py --no-cursor # gestures only, no cursor/click
```

In the video window: `q` quits, `p` pauses. Moving your real mouse into a screen
corner also stops it (pyautogui failsafe).

**macOS:** grant your terminal both Camera and Accessibility permission
(System Settings > Privacy & Security), or mouse/keyboard control silently does nothing.
Scrolling is slow on macOS; try `--scroll-gain 300`.

## Layout

- `gestures.py`: pose classification and gesture logic. Pure Python + NumPy, no camera; all thresholds are in `Config`.
- `main.py`: camera, MediaPipe, on-screen HUD and turning events into mouse/keyboard input.
- `test_gestures.py`: tests with synthetic hands (`uv add --dev pytest`, then `uv run pytest`).
