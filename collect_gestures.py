"""
Guided hand gesture dataset collector.

You sit in front of the camera, and the script walks you through every gesture,
take by take, showing on-screen instructions. Each frame where a hand is found is
saved as one row in a CSV file (landmarks, not images).

Install:
    pip install mediapipe opencv-python numpy

Run (each group member uses their own --subject name):
    python collect_gestures.py --subject ali

Quick test (about a minute):
    python collect_gestures.py --subject test --takes 1 --seconds 3 --out test.csv

Re-record only one gesture:
    python collect_gestures.py --subject ali --only fist

Controls:
    SPACE / ENTER : start a take / accept a take
    R             : redo the take you just recorded
    S             : skip this take
    ESC           : abort the take being recorded
    Q             : quit (data from accepted takes is already saved)

CSV layout (one row per frame with a detected hand):
    subject_id, session_id, label, take, variation, frame_idx, t_sec,
    handedness, handedness_score,
    r0_x r0_y r0_z ... r20_z     raw landmarks (MediaPipe image coordinates)
    f0_x f0_y f0_z ... f20_z     normalized features (wrist removed, scaled by hand
                                 size, left hands mirrored). The f0_* columns are
                                 always 0 and can be dropped when training.

IMPORTANT for the later live demo: the camera image is flipped horizontally
(selfie view) BEFORE it goes to the detector, here and in the demo. Keep this
consistent, otherwise left/right handedness and gestures will not match training.

Tips for good data:
  - Good, even lighting. Avoid strong backlight (a window behind you).
  - Keep the whole hand inside the frame, including fingertips.
  - Run the script several times (different lighting, background, time of day,
    clothing). Each run is saved as a new session.
  - Every group member should record their own data.
"""

import argparse
import csv
import os
import textwrap
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gesture_features import csv_header, normalize_landmarks

# Verify this URL against the MediaPipe docs if the download fails.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Edit this list to change the gesture set. The "command" is only a reminder here.
GESTURES = [
    {
        "label": "open_palm",
        "title": "OPEN PALM  (command: pause)",
        "how": "All five fingers extended and slightly spread, palm facing the camera. "
               "Keep fingers straight and relaxed, with the thumb out to the side.",
    },
    {
        "label": "fist",
        "title": "FIST  (command: play)",
        "how": "All four fingers curled tightly into the palm, thumb folded across them. "
               "Keep the fist facing the camera.",
    },
    {
        "label": "thumbs_up",
        "title": "THUMBS UP  (command: volume up)",
        "how": "Fist with the thumb pointing straight up. Other fingers curled. "
               "Keep the thumb clearly vertical.",
    },
    {
        "label": "thumbs_down",
        "title": "THUMBS DOWN  (command: volume down)",
        "how": "Fist with the thumb pointing straight down. Other fingers curled. "
               "Keep the thumb clearly pointing down.",
    },
    {
        "label": "point",
        "title": "POINT  (command: next slide)",
        "how": "Index finger extended straight up, other fingers curled, thumb tucked in. "
               "Keep the index finger clearly visible.",
    },
    {
        "label": "peace",
        "title": "PEACE SIGN  (command: previous slide)",
        "how": "Index and middle fingers extended in a V, ring and little fingers curled "
               "and held down by the thumb.",
    },
    {
        "label": "none",
        "title": "NONE  (NOT a gesture: must trigger no command)",
        "how": "Natural hand movements that should NEVER trigger a command. "
               "Follow the instruction of each take. Do not hold any of the real gestures.",
        "variations": [
            ("rest", "Relax your hand and let it rest naturally in view, fingers loose."),
            ("random", "Slowly move your hand around and keep changing your finger positions randomly."),
            ("halfway", "Make half-formed, in-between shapes: a half-closed fist, loosely curled fingers, "
                        "two fingers half raised."),
            ("daily", "Act naturally: wave, adjust your hair or glasses, scratch your face, "
                      "with your hand moving in and out of view."),
            ("transitions", "Slowly move between the real gestures (open palm, fist, thumbs up...) "
                            "without holding any of them."),
            ("other_hand", "Switch to your other hand and repeat relaxed, random hand poses."),
        ],
    },
]

# Variations used for every real gesture. Takes cycle through this list.
VARIATIONS = [
    ("normal", "Hand about an arm's length from the camera, centered, facing the camera."),
    ("near", "Bring your hand closer, about half an arm's length, but keep the whole hand in view."),
    ("far", "Move your hand farther away (lean back or extend your arm) and keep the whole hand in view."),
    ("tilt", "Hold the gesture and slowly tilt and rotate your wrist left and right."),
    ("move", "Hold the gesture and slowly move your hand around the frame: left, right, up, down."),
    ("other_hand", "Switch to your other hand and make the same gesture."),
]

TRIM_SEC = 0.4        # discard this much at the start and end of each take
MIN_FRAMES = 20       # warn if a take has fewer saved frames than this
COUNTDOWN_SEC = 3
WHITE, GREEN, YELLOW, RED, GRAY = (255, 255, 255), (0, 220, 0), (0, 220, 255), (60, 60, 255), (180, 180, 180)


class Quit(Exception):
    pass


# --------------------------------------------------------------------------
class HandDetector:
    def __init__(self, swap_handedness: bool):
        self.swap = swap_handedness
        self._ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    @staticmethod
    def _ensure_model():
        if os.path.exists(MODEL_PATH):
            return
        print(f"Downloading model to {MODEL_PATH} ...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            raise SystemExit(
                f"Could not download the model ({e}).\n"
                f"Download a Hand Landmarker .task file from the MediaPipe docs "
                f"and save it as {MODEL_PATH}."
            )

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self.landmarker.detect_for_video(mp_image, ts)
        if not result.hand_landmarks:
            return None
        lms = result.hand_landmarks[0]
        pts = np.array([[l.x, l.y, l.z] for l in lms], dtype=np.float32)
        cat = result.handedness[0][0]
        label = cat.category_name
        if self.swap:
            label = "Left" if label == "Right" else "Right"
        return {"pts": pts, "handed": label, "score": float(cat.score)}


class CameraSource:
    def __init__(self, cap, detector):
        self.cap, self.detector = cap, detector

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit("Camera read failed.")
        frame = cv2.flip(frame, 1)               # selfie view, BEFORE detection
        return frame, self.detector.detect(frame)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw_hand(frame, hand):
    if hand is None:
        return
    h, w = frame.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y, _ in hand["pts"]]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (0, 200, 0), 2)
    for p in px:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)
    cv2.putText(frame, f'{hand["handed"]} hand', (px[0][0] - 30, px[0][1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)


def make_panel(width, lines):
    rendered = []
    for text, color, scale in lines:
        chars = max(20, (width - 20) // max(1, int(18 * scale)))
        for part in textwrap.wrap(text, chars) or [""]:
            rendered.append((part, color, scale))
    height = 10 + sum(int(40 * s) for _, _, s in rendered) + 6
    panel = np.full((height, width, 3), 30, np.uint8)
    y = 10
    for part, color, scale in rendered:
        y += int(40 * scale)
        cv2.putText(panel, part, (10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return panel


def show(frame, lines):
    panel = make_panel(frame.shape[1], lines)
    cv2.imshow("Gesture dataset collector", np.vstack([frame, panel]))


def big_text(frame, text, color=WHITE):
    h, w = frame.shape[:2]
    scale = 4.0
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 8)
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 8, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Interaction steps
# --------------------------------------------------------------------------
def wait_for_key(src, lines, keys):
    """Show a live preview with instructions until one of `keys` is pressed."""
    while True:
        frame, hand = src.read()
        draw_hand(frame, hand)
        status = ("Hand detected" if hand else "NO HAND DETECTED: move your hand into view")
        show(frame, lines + [(status, GREEN if hand else RED, 0.6)])
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("Q")):
            raise Quit()
        if k in keys:
            return k


def countdown(src, header_lines):
    start = time.monotonic()
    while True:
        left = COUNTDOWN_SEC - (time.monotonic() - start)
        if left <= 0:
            return
        frame, hand = src.read()
        draw_hand(frame, hand)
        big_text(frame, str(int(np.ceil(left))), YELLOW)
        show(frame, header_lines + [("Get your hand into the gesture NOW.", YELLOW, 0.6)])
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("Q")):
            raise Quit()


def record_take(src, duration, header_lines):
    """Returns (rows, frames_seen) or None if aborted with ESC.
    rows is a list of (t, hand, feat)."""
    rows, frames_seen = [], 0
    start = time.monotonic()
    while True:
        t = time.monotonic() - start
        if t >= duration:
            break
        frame, hand = src.read()
        frames_seen += 1
        feat = None
        if hand is not None:
            feat = normalize_landmarks(hand["pts"], hand["handed"])
            if feat is not None:
                rows.append((t, hand, feat))
        draw_hand(frame, hand)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (int(w * t / duration), 12), RED, -1)
        cv2.circle(frame, (w - 25, 35), 10, RED, -1)
        cv2.putText(frame, "REC", (w - 75, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
        status = f"Recording... {t:0.1f}/{duration:0.0f}s   frames saved so far: {len(rows)}"
        show(frame, header_lines + [(status, GREEN if hand else RED, 0.6)])
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return None
        if k in (ord("q"), ord("Q")):
            raise Quit()

    kept = [r for r in rows if TRIM_SEC <= r[0] <= duration - TRIM_SEC]
    return kept, frames_seen, len(rows)


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
def append_rows(path, subject, session, label, take_no, var_key, rows):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(csv_header())
        for idx, (t, hand, feat) in enumerate(rows):
            raw = hand["pts"].flatten()
            w.writerow(
                [subject, session, label, take_no, var_key, idx, f"{t:.3f}",
                 hand["handed"], f'{hand["score"]:.3f}']
                + [f"{v:.5f}" for v in raw]
                + [f"{v:.5f}" for v in feat]
            )


def print_summary(path, subject, session):
    if not os.path.exists(path):
        return
    this_run, total = {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            total[row["label"]] = total.get(row["label"], 0) + 1
            if row["subject_id"] == subject and row["session_id"] == session:
                this_run[row["label"]] = this_run.get(row["label"], 0) + 1
    print("\nFrames per label")
    print(f'{"label":<14}{"this run":>10}{"whole file":>12}')
    for label in sorted(total):
        print(f"{label:<14}{this_run.get(label, 0):>10}{total[label]:>12}")
    print(f"\nDataset file: {os.path.abspath(path)}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="your name or ID (use the same every time)")
    ap.add_argument("--session", default=None, help="session ID (default: current date and time)")
    ap.add_argument("--out", default="gesture_dataset.csv")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seconds", type=float, default=6.0, help="recording length per take")
    ap.add_argument("--takes", type=int, default=6, help="takes per gesture")
    ap.add_argument("--only", type=str, default=None, help="record only this gesture label")
    ap.add_argument("--swap-handedness", action="store_true",
                    help="use if your right hand shows up as 'Left'")
    args = ap.parse_args()

    subject = args.subject.strip().replace(" ", "_")
    session = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")

    gestures = GESTURES
    if args.only:
        gestures = [g for g in GESTURES if g["label"] == args.only]
        if not gestures:
            raise SystemExit(f"Unknown gesture '{args.only}'. Options: " +
                             ", ".join(g["label"] for g in GESTURES))

    backend = cv2.CAP_V4L2 if os.name == "posix" and os.uname().sysname == "Linux" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    src = CameraSource(cap, HandDetector(args.swap_handedness))
    saved_takes = 0

    try:
        wait_for_key(src, [
            (f"Subject: {subject}   Session: {session}", WHITE, 0.6),
            ("SETUP: sit facing the camera with good light on your hands.", YELLOW, 0.6),
            ("Check that your RIGHT hand is labeled 'Right' on the preview "
             "(if not, restart with --swap-handedness).", WHITE, 0.55),
            ("Keep your whole hand in view, fingertips included.", WHITE, 0.55),
            ("Press SPACE to begin. Press Q any time to quit.", GREEN, 0.6),
        ], (32, 13))

        for gesture in gestures:
            variations = gesture.get("variations", VARIATIONS)

            wait_for_key(src, [
                (gesture["title"], YELLOW, 0.8),
                (gesture["how"], WHITE, 0.55),
                (f'{args.takes} takes of {args.seconds:.0f} s. Press SPACE when ready.', GREEN, 0.6),
            ], (32, 13))

            take = 0
            while take < args.takes:
                var_key, var_text = variations[take % len(variations)]
                header = [
                    (f'{gesture["title"]}   take {take + 1}/{args.takes}', YELLOW, 0.65),
                    (gesture["how"], GRAY, 0.5),
                    (f"This take: {var_text}", WHITE, 0.55),
                ]

                k = wait_for_key(src, header + [("SPACE = start   S = skip this take", GREEN, 0.55)],
                                 (32, 13, ord("s"), ord("S")))
                if k in (ord("s"), ord("S")):
                    take += 1
                    continue

                countdown(src, header)
                result = record_take(src, args.seconds, header)
                if result is None:
                    continue  # aborted with ESC, redo the same take
                kept, frames_seen, detected = result

                rate = 100.0 * detected / max(frames_seen, 1)
                msgs = [
                    (f"Take done. Saved {len(kept)} frames "
                     f"(hand found in {rate:.0f}% of frames).", WHITE, 0.6),
                ]
                if len(kept) < MIN_FRAMES or rate < 60:
                    msgs.append(("WARNING: the hand was often lost. Check lighting and keep the whole hand in view. "
                                 "Redo recommended.", RED, 0.55))
                msgs.append(("SPACE = accept   R = redo this take", GREEN, 0.6))

                k = wait_for_key(src, msgs, (32, 13, ord("r"), ord("R")))
                if k in (ord("r"), ord("R")):
                    continue

                append_rows(args.out, subject, session, gesture["label"], take + 1, var_key, kept)
                saved_takes += 1
                take += 1

        wait_for_key(src, [("All gestures recorded. Thank you! Press SPACE to finish.", GREEN, 0.7)], (32, 13))

    except Quit:
        print("Quit by user. Accepted takes were already saved.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Accepted takes this run: {saved_takes}")
    print_summary(args.out, subject, session)


if __name__ == "__main__":
    main()
