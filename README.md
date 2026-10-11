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

Hold a pose for a split second (~100 ms) before it registers. The camera window
shows `pose` (what it's acting on) and `seeing` (its guess for this frame).

**Scrolling:** raise index + middle and hold still. A yellow line marks your
starting point, with a shaded dead zone around it. Move your hand above or below the band and
hold there; the farther out, the faster it scrolls. Come back to the line to stop.

**Swiping:** open palm, then one quick, sideways flick across about a third of the
camera view. A blue trail shows the motion being tracked. Bring your hand back
slowly, or wait a second, so the return isn't read as a swipe the other way.

## Important: which window gets the keys

Keyboard shortcuts (tab switching, close tab) go to whichever window is in front.
After starting, **click on your browser** so it's in front, and keep the small camera
window off to the side. Otherwise Ctrl+Tab goes to the camera window, and a fist
could close your Terminal window instead of a tab. Scrolling goes to whatever is
under the mouse pointer.

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
Scroll speed: `--scroll-gain 120` for faster, `--scroll-gain 40` for slower (default 80).

## Layout

- `gestures.py`: pose classification and gesture logic. Pure Python + NumPy, no camera; all thresholds are in `Config`.
- `main.py`: camera, MediaPipe, on-screen HUD and turning events into mouse/keyboard input.
- `test_gestures.py`: tests with synthetic hands (`uv add --dev pytest`, then `uv run pytest`).
