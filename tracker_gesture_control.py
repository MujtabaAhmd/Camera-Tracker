#!/usr/bin/env python3
"""
Combined demo: the camera follows your torso with a pan servo (person_tracker.py parts)
AND recognizes hand gestures to control your computer (gesture_runtime.py parts),
using the same camera frames.

Laptop test with no hardware (virtual pan angle, gesture commands only printed):
    python tracker_gesture_control.py --model models/svm.joblib

With the Arduino pan servo and real key presses:
    python tracker_gesture_control.py --model models/svm.joblib --serial /dev/ttyACM0 --live

How the two parts share the camera
  - The frame is flipped (selfie view) once, then given to both detectors.
  - While a hand is visible, the pan angle is FROZEN (disable with --no-freeze-on-hand),
    so the camera does not swing around while you gesture, and your raised hand does not
    disturb the torso-based aiming.
  - The pose model can run every N frames (--pose-every) to save CPU. Measure your FPS.

Needs person_tracker.py, gesture_runtime.py, collect_gestures.py and gesture_features.py
in the same folder.
"""

import argparse
import os
import time

import cv2

from collect_gestures import HandDetector, draw_hand
from gesture_runtime import (ActionMapper, GestureClassifier, NONE_LABEL, add_gesture_args,
                             build_engine, classify_frame, draw_gesture_hud, handle_event,
                             COMMANDS)
from person_tracker import MediaPipePoseDetector, PanController, ServoLink, draw_overlay


def main():
    ap = argparse.ArgumentParser()
    add_gesture_args(ap)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--serial", default=None, help="Arduino serial port (omit for no servo)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--kp", type=float, default=12.0)
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--deadband", type=float, default=0.04)
    ap.add_argument("--invert", action="store_true", help="flip pan direction")
    ap.add_argument("--pose-every", type=int, default=1,
                    help="run the pose model every N frames (reuse the last result in between)")
    ap.add_argument("--no-freeze-on-hand", action="store_true",
                    help="keep tracking even while a hand is visible")
    args = ap.parse_args()

    classifier = GestureClassifier(args.model)
    print(f"Loaded {classifier.name} model. Classes: {classifier.classes}")
    unmapped = [c for c in classifier.classes if c != NONE_LABEL and c not in COMMANDS]
    if unmapped:
        print(f"Note: no command mapped for {unmapped} (edit COMMANDS in gesture_runtime.py).")
    if args.arm_label and args.arm_label not in classifier.classes:
        raise SystemExit(f"--arm-label '{args.arm_label}' is not a class of this model.")

    engine = build_engine(args)
    mapper = ActionMapper(live=args.live)
    hand_detector = HandDetector(args.swap_handedness)
    pose_detector = MediaPipePoseDetector()
    controller = PanController(kp=args.kp, kd=args.kd, deadband=args.deadband, invert=args.invert)
    link = ServoLink(args.serial, args.baud)

    backend = cv2.CAP_V4L2 if os.name == "posix" and os.uname().sysname == "Linux" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    freeze = not args.no_freeze_on_hand
    last_msg, last_msg_time = [""], 0.0
    target, frame_i = None, 0
    fps, t_prev = 0.0, time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break
            frame = cv2.flip(frame, 1)   # selfie view for BOTH detectors

            if frame_i % max(args.pose_every, 1) == 0:
                target = pose_detector.detect(frame)
            frame_i += 1

            hand = hand_detector.detect(frame)
            label, conf = classify_frame(frame, hand, classifier)

            now = time.monotonic()
            event = engine.update(label, conf, now)
            if event is not None:
                handle_event(event, mapper, last_msg)
                last_msg_time = now
            if now - last_msg_time > 2.0:
                last_msg[0] = ""

            frozen = freeze and hand is not None
            angle = controller.update(None if frozen else target)
            link.send_pan(angle)

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            draw_overlay(frame, target, angle, fps, args.deadband)
            draw_hand(frame, hand)
            if frozen:
                cv2.putText(frame, "PAN FROZEN (hand visible)", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)
            draw_gesture_hud(frame, label, conf, engine, last_msg[0], args.live, args.arm_label)
            cv2.imshow("Tracker + gesture control", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q")):
                break
            if k == ord("c"):
                controller.recentre()
    finally:
        cap.release()
        link.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
