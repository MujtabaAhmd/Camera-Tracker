#!/usr/bin/env python3
"""
Live gesture control: camera -> hand landmarks -> your trained classifier ->
hold-to-confirm logic -> keyboard / media commands.

Safe first run (prints commands, presses nothing):
    python gesture_runtime.py --model models/svm.joblib

Actually press keys (click on your slides or video player first!):
    python gesture_runtime.py --model models/svm.joblib --live

Why a gesture does not fire instantly
  A command is issued only if the same gesture is held for --hold seconds, with
  high confidence on most of the frames in that window (--min-conf, --agree). After
  firing, the gesture must be released before it can fire again, except thumbs
  up / down, which repeat while held (like holding a volume button).
  Optionally, --arm-label X requires you to hold gesture X first to "arm" the system
  for --arm-timeout seconds, so nothing fires unless you armed it on purpose.

Keys in the preview window: q = quit.

pyautogui note: it presses keys on the computer that runs this script. If you run
this on a Jetson, it controls the Jetson, not your laptop.
"""

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import joblib
import numpy as np

import gesture_features  # noqa: F401  (needed so saved NumpyMLP models can be loaded)
from collect_gestures import HandDetector, draw_hand
from gesture_features import landmarks_to_features

NONE_LABEL = "none"

# gesture label -> key name for pyautogui. Edit freely. Key names are those of pyautogui
# (see pyautogui.KEYBOARD_KEYS). Media keys such as 'playpause' and 'volumeup' may
# behave differently, or not work, on some operating systems or desktop setups.
COMMANDS = {
    "open_palm": "playpause",     # toggles play / pause
    "fist": "volumemute",
    "thumbs_up": "volumeup",
    "thumbs_down": "volumedown",
    "point": "right",             # next slide
    "peace": "left",              # previous slide
}
REPEAT_LABELS = ("thumbs_up", "thumbs_down")   # fire repeatedly while held


# --------------------------------------------------------------------------
class GestureClassifier:
    """Loads a bundle saved by train_gestures.py and predicts from 60 features."""

    def __init__(self, path):
        bundle = joblib.load(path)
        self.estimator = bundle["estimator"]
        self.classes = bundle["classes"]            # names, indexed by encoded label
        self.name = bundle.get("name", "model")
        self.meta = bundle.get("meta", {})
        self._est_classes = np.asarray(self.estimator.classes_, dtype=int)

    def predict(self, feat):
        p = self.estimator.predict_proba(np.asarray(feat, dtype=np.float32).reshape(1, -1))[0]
        i = int(np.argmax(p))
        return self.classes[int(self._est_classes[i])], float(p[i])


@dataclass
class Event:
    label: str
    kind: str   # "command" or "armed"


class GestureEngine:
    """Turns a stream of per-frame predictions into deliberate, one-off commands."""

    def __init__(self, hold_sec=1.0, min_conf=0.8, min_agree=0.85, cooldown_sec=1.0,
                 repeat_labels=REPEAT_LABELS, repeat_sec=0.4,
                 arm_label: Optional[str] = None, arm_timeout=6.0):
        self.hold_sec, self.min_conf, self.min_agree = hold_sec, min_conf, min_agree
        self.cooldown_sec = cooldown_sec
        self.repeat_labels, self.repeat_sec = tuple(repeat_labels), repeat_sec
        self.arm_label, self.arm_timeout = arm_label, arm_timeout

        self.buf = deque()            # (time, effective_label)
        self.locked_label = None      # fired and not yet released
        self.blocked_until = 0.0
        self.armed_until = 0.0

        # for the on-screen display
        self.current = None           # gesture currently being held steadily
        self.progress = 0.0           # 0..1 towards firing

    @property
    def armed(self):
        return self.arm_label is None or time.monotonic() < self.armed_until

    def update(self, label, conf, now) -> Optional[Event]:
        """Call once per frame. label is None when no hand is visible."""
        eff = NONE_LABEL if (label is None or conf < self.min_conf) else label
        self.buf.append((now, eff))
        while self.buf and now - self.buf[0][0] > self.hold_sec:
            self.buf.popleft()

        span = now - self.buf[0][0]
        labels = [l for _, l in self.buf]
        top = max(set(labels), key=labels.count)
        agree = labels.count(top) / len(labels)
        stable = (top != NONE_LABEL) and agree >= self.min_agree

        self.current = top if stable else None
        self.progress = min(1.0, span / self.hold_sec) if stable else 0.0

        # released: the gesture that fired is no longer being held
        if self.locked_label is not None and not (stable and top == self.locked_label):
            self.locked_label = None

        if not stable or span < 0.9 * self.hold_sec or now < self.blocked_until:
            return None
        if top == self.locked_label:
            return None

        if self.arm_label is not None:
            if top == self.arm_label:
                self.armed_until = now + self.arm_timeout
                self.locked_label = top
                self.blocked_until = now + self.cooldown_sec
                return Event(top, "armed")
            if now >= self.armed_until:
                return None            # not armed: ignore
            self.armed_until = now + self.arm_timeout

        if top in self.repeat_labels:
            self.blocked_until = now + self.repeat_sec
        else:
            self.locked_label = top
            self.blocked_until = now + self.cooldown_sec
        return Event(top, "command")


class ActionMapper:
    """Maps gesture labels to key presses. Dry-run unless live=True."""

    def __init__(self, live=False, commands=None):
        self.live = live
        self.commands = dict(COMMANDS if commands is None else commands)
        self._pg = None
        if live:
            try:
                import pyautogui
            except ImportError:
                raise SystemExit("pip install pyautogui   (needed for --live)")
            pyautogui.PAUSE = 0
            self._pg = pyautogui

    def execute(self, label) -> Optional[str]:
        key = self.commands.get(label)
        if key is None:
            return None
        if self._pg is not None:
            self._pg.press(key)
        return key


# --------------------------------------------------------------------------
def draw_gesture_hud(frame, label, conf, engine: GestureEngine, last_msg, live, arm_label=None):
    """Draws the prediction, hold progress, arm state and last command at the bottom."""
    h, w = frame.shape[:2]
    y0 = h - 78
    cv2.rectangle(frame, (0, y0), (w, h), (30, 30, 30), -1)

    pred = f"{label} {conf:.2f}" if label else "no hand"
    cv2.putText(frame, f"Prediction: {pred}", (10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # hold progress bar
    bar_w = int(w * 0.45)
    cv2.rectangle(frame, (10, y0 + 32), (10 + bar_w, y0 + 46), (90, 90, 90), 1)
    cv2.rectangle(frame, (10, y0 + 32), (10 + int(bar_w * engine.progress), y0 + 46), (0, 200, 255), -1)
    holding = engine.current or "-"
    cv2.putText(frame, f"holding: {holding}", (20 + bar_w, y0 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    mode = "LIVE (keys pressed)" if live else "DRY RUN (no keys pressed)"
    cv2.putText(frame, mode, (10, y0 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255) if live else (180, 180, 180), 1, cv2.LINE_AA)
    if arm_label is not None:
        armed = engine.armed
        cv2.putText(frame, "ARMED" if armed else f"not armed (hold '{arm_label}')",
                    (w // 2, y0 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 220, 0) if armed else (0, 140, 255), 1, cv2.LINE_AA)
    if last_msg:
        cv2.putText(frame, last_msg, (10, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)


def classify_frame(frame, hand, classifier):
    """Returns (label, confidence) or (None, 0.0) when no usable hand."""
    if hand is None:
        return None, 0.0
    feat = landmarks_to_features(hand["pts"], hand["handed"])
    if feat is None:
        return None, 0.0
    return classifier.predict(feat)


def add_gesture_args(ap):
    ap.add_argument("--model", required=True, help="a .joblib file from train_gestures.py")
    ap.add_argument("--live", action="store_true", help="really press keys (default: dry run)")
    ap.add_argument("--hold", type=float, default=1.0, help="seconds a gesture must be held")
    ap.add_argument("--min-conf", type=float, default=0.8, help="minimum model confidence")
    ap.add_argument("--agree", type=float, default=0.85,
                    help="share of frames in the hold window that must agree")
    ap.add_argument("--cooldown", type=float, default=1.0)
    ap.add_argument("--repeat", type=float, default=0.4,
                    help="seconds between repeats for thumbs up/down")
    ap.add_argument("--arm-label", default=None,
                    help="gesture that must be held first to arm the system (e.g. open_palm)")
    ap.add_argument("--arm-timeout", type=float, default=6.0)
    ap.add_argument("--swap-handedness", action="store_true",
                    help="use the same setting you used when collecting data")


def build_engine(args):
    return GestureEngine(hold_sec=args.hold, min_conf=args.min_conf, min_agree=args.agree,
                         cooldown_sec=args.cooldown, repeat_sec=args.repeat,
                         arm_label=args.arm_label, arm_timeout=args.arm_timeout)


def handle_event(event, mapper, last_msg_holder):
    if event is None:
        return
    if event.kind == "armed":
        last_msg_holder[0] = f"ARMED by '{event.label}'"
        print(last_msg_holder[0])
        return
    key = mapper.execute(event.label)
    prefix = "" if mapper.live else "[dry run] "
    if key:
        last_msg_holder[0] = f"{prefix}{event.label} -> {key}"
    else:
        last_msg_holder[0] = f"{prefix}{event.label} (no command mapped)"
    print(last_msg_holder[0])


def main():
    ap = argparse.ArgumentParser()
    add_gesture_args(ap)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    classifier = GestureClassifier(args.model)
    print(f"Loaded {classifier.name} model. Classes: {classifier.classes}")
    unmapped = [c for c in classifier.classes if c != NONE_LABEL and c not in COMMANDS]
    if unmapped:
        print(f"Note: no command mapped for {unmapped} (edit COMMANDS at the top of this file).")
    if args.arm_label and args.arm_label not in classifier.classes:
        raise SystemExit(f"--arm-label '{args.arm_label}' is not a class of this model.")

    engine = build_engine(args)
    mapper = ActionMapper(live=args.live)
    detector = HandDetector(args.swap_handedness)

    backend = cv2.CAP_V4L2 if os.name == "posix" and os.uname().sysname == "Linux" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    last_msg = [""]
    last_msg_time = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break
            frame = cv2.flip(frame, 1)           # selfie view, same as when collecting data
            hand = detector.detect(frame)
            label, conf = classify_frame(frame, hand, classifier)

            now = time.monotonic()
            event = engine.update(label, conf, now)
            if event is not None:
                handle_event(event, mapper, last_msg)
                last_msg_time = now
            if now - last_msg_time > 2.0:
                last_msg[0] = ""

            draw_hand(frame, hand)
            draw_gesture_hud(frame, label, conf, engine, last_msg[0], args.live, args.arm_label)
            cv2.imshow("Gesture control", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
