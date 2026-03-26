"""

Gestures (either hand unless noted):
  • Index finger up       → move cursor
  • Index + Thumb pinch   → right-click  (once per pinch)
  • Index + Middle up     → scroll  (move hand up/down)
  • All 5 fingers closed  → close / quit the program
  • 3 fingers swiped L/R  → switch virtual desktop (Win: Ctrl+Win+Arrow)
 
"""
 
import argparse
import platform
import time
from collections import deque
 
import cv2 as cv
import mediapipe as mp
import numpy as np
import pyautogui
 
# ── safety: pyautogui will NOT raise FailSafe on every edge pixel
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # no artificial delay between calls
 
 
# ──────────────────────────────────────────────
#  - Command-line arguments to communicate with the os of the computer
# ──────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="Gesture Mouse Controller")
    parser.add_argument("--device",  type=int,   default=0)
    parser.add_argument("--width",   type=int,   default=960)
    parser.add_argument("--height",  type=int,   default=540)
    parser.add_argument("--smooth",  type=float, default=0.25,
                        help="Cursor smoothing 0=raw 1=frozen (default 0.25)")
    return parser.parse_args()
 
 
# ──────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────
def fingertip_distance(lm, a: int, b: int) -> float:
    """Euclidean distance between two landmark indices (normalised coords)."""
    ax, ay = lm[a].x, lm[a].y
    bx, by = lm[b].x, lm[b].y
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
 
 
def finger_up(lm, tip: int, pip: int) -> bool:
    """True when a finger is extended (tip above pip in image-y)."""
    return lm[tip].y < lm[pip].y
 
 
def fingers_state(lm) -> list:
    """
    Return [thumb, index, middle, ring, pinky] as 1=up / 0=down.
    MediaPipe landmark indices:
        Thumb tip=4 ip=3 | Index tip=8 pip=6 | Middle tip=12 pip=10
        Ring tip=16 pip=14 | Pinky tip=20 pip=18
    """
    thumb  = 1 if lm[4].x < lm[3].x else 0   # works for right hand (mirrored)
    index  = 1 if finger_up(lm, 8,  6)  else 0
    middle = 1 if finger_up(lm, 12, 10) else 0
    ring   = 1 if finger_up(lm, 16, 14) else 0
    pinky  = 1 if finger_up(lm, 20, 18) else 0
    return [thumb, index, middle, ring, pinky]
 
 
# ──────────────────────────────────────────────
# OS-level desktop switching
# ──────────────────────────────────────────────
def switch_desktop(direction: str):
    """
    direction: 'left' or 'right'
    Windows  → Ctrl + Win + Left/Right
    macOS    → Ctrl + Left/Right  (Mission Control)
    Linux    → Ctrl + Alt + Left/Right  (common for GNOME/KDE)
    """
    os_name = platform.system()
    if os_name == "Windows":
        pyautogui.hotkey("ctrl", "win", direction)
    elif os_name == "Darwin":
        pyautogui.hotkey("ctrl", direction)
    else:
        pyautogui.hotkey("ctrl", "alt", direction)
 
 
# ──────────────────────────────────────────────
# Gesture state machine
# ──────────────────────────────────────────────
class GestureController:
    # Tune these thresholds to your liking
    PINCH_THRESHOLD     = 0.06   # index-thumb distance → right-click
    ALL_CLOSED_THRESH   = 0.10   # all fingertips close to wrist → quit
    SWIPE_FRAMES        = 8      # consecutive frames with 3-finger gesture for swipe
    SWIPE_DELTA         = 0.12   # normalised-x shift to count as a swipe
    SCROLL_SCALE        = 15     # pixels scrolled per unit of vertical movement
 
    def __init__(self, smooth: float = 0.25):
        self.smooth   = smooth
        self.cursor_x, self.cursor_y = pyautogui.position()
 
        # State flags
        self._pinch_active    = False
        self._scroll_ref_y    = None
        self._three_start_x   = None
        self._three_frames    = 0
        self._quit_flag       = False
 
    @property
    def should_quit(self):
        return self._quit_flag
 
    def process(self, lm, frame_w: int, frame_h: int):
        """
        Called once per detected hand per frame.
        lm = list of 21 MediaPipe NormalizedLandmark objects.
        """
        state = fingers_state(lm)
        thumb, index, middle, ring, pinky = state
        up_count = sum(state)
 
        # ── 1. ALL FINGERS CLOSED → quit ─────────────────────────────────
        # Check that all fingertips are close to the wrist (landmark 0)
        all_closed = all(
            fingertip_distance(lm, tip, 0) < self.ALL_CLOSED_THRESH
            for tip in [4, 8, 12, 16, 20]
        )
        if all_closed:
            self._quit_flag = True
            return
 
        # ── 2. THREE-FINGER SWIPE → switch desktop ────────────────────────
        # Detect: index + middle + ring up, thumb + pinky down
        if state == [0, 1, 1, 1, 0]:
            cx = lm[9].x   # middle-finger MCP as reference x
            if self._three_start_x is None:
                self._three_start_x = cx
                self._three_frames  = 1
            else:
                self._three_frames += 1
                if self._three_frames >= self.SWIPE_FRAMES:
                    delta = cx - self._three_start_x
                    if abs(delta) > self.SWIPE_DELTA:
                        direction = "right" if delta > 0 else "left"
                        switch_desktop(direction)
                        time.sleep(0.4)   # debounce
                    self._three_start_x = None
                    self._three_frames  = 0
            return  # don't move cursor during swipe gesture
        else:
            self._three_start_x = None
            self._three_frames  = 0
 
        # ── 3. INDEX + MIDDLE UP → scroll ────────────────────────────────
        if index == 1 and middle == 1 and ring == 0 and pinky == 0:
            ref_y = lm[8].y   # index fingertip y (normalised)
            if self._scroll_ref_y is None:
                self._scroll_ref_y = ref_y
            else:
                dy = (self._scroll_ref_y - ref_y) * self.SCROLL_SCALE
                if abs(dy) > 0.5:
                    pyautogui.scroll(int(dy))
                self._scroll_ref_y = ref_y
            self._pinch_active = False
            return
        else:
            self._scroll_ref_y = None
 
        # ── 4. INDEX + THUMB PINCH → right-click ─────────────────────────
        pinch_dist = fingertip_distance(lm, 8, 4)
        if pinch_dist < self.PINCH_THRESHOLD:
            if not self._pinch_active:
                pyautogui.rightClick()
                self._pinch_active = True
        else:
            self._pinch_active = False
 
        # ── 5. INDEX FINGER UP → move cursor ─────────────────────────────
        if index == 1:
            # Map index fingertip (landmark 8) from camera frame to screen
            screen_w, screen_h = pyautogui.size()
 
            raw_x = lm[8].x * screen_w
            raw_y = lm[8].y * screen_h
 
            # Exponential smoothing to reduce jitter
            self.cursor_x = self.cursor_x + self.smooth * (raw_x - self.cursor_x)
            self.cursor_y = self.cursor_y + self.smooth * (raw_y - self.cursor_y)
 
            pyautogui.moveTo(int(self.cursor_x), int(self.cursor_y))
 
 
# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────
def main():
    args   = get_args()
    ctrl   = GestureController(smooth=args.smooth)
 
    # Camera
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
 
    # MediaPipe Hands
    mp_hands    = mp.solutions.hands
    mp_drawing  = mp.solutions.drawing_utils
    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
 
    print("Gesture Mouse running. Show 'fist' (all fingers closed) to quit.")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — exiting.")
            break
 
        # Mirror so left/right feels natural
        frame = cv.flip(frame, 1)
 
        # MediaPipe expects RGB
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False 
        results = hands_model.process(rgb)
        rgb.flags.writeable = True
 
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                lm = hand_lm.landmark   # list of 21 NormalizedLandmark
 
                # Draw skeleton on preview
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS)
 
                # Run gesture logic
                ctrl.process(lm, args.width, args.height)
 
                if ctrl.should_quit:
                    print("Quit gesture detected — closing.")
                    cap.release()
                    cv.destroyAllWindows()
                    return
 
        # Overlay hint text
        cv.putText(frame, "ESC = quit  |  Fist = quit gesture",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2, cv.LINE_AA)
 
        cv.imshow("Gesture Mouse  (press ESC to exit)", frame)
 
        if cv.waitKey(1) & 0xFF == 27:   # ESC key
            break
 
    cap.release()
    cv.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 
