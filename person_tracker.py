"""
Person tracker using MediaPipe Pose Landmarker (Tasks API).

Pipeline: camera -> detector -> aim point -> controller -> (optional) serial to Arduino.

Install:
    pip install mediapipe opencv-python numpy pyserial

Run on laptop (no servo, just a virtual pan angle on screen):
    python person_tracker.py

Run with Arduino on a serial port:
    python person_tracker.py --serial /dev/ttyACM0        (Linux / Jetson)
    python person_tracker.py --serial COM3                (Windows)

Keys: q = quit, c = recentre virtual pan angle.

Serial protocol sent to the Arduino (one line per frame):
    P,<pan_angle_int>\n        e.g. "P,92\n"
The Arduino sketch should clamp the angle and hold position if no
command arrives for a short time.

Note: MediaPipe's API changes between releases. If something fails, check
the version you installed against the current MediaPipe docs.
"""

import argparse
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Model file for the Pose Landmarker. Verify this URL against the MediaPipe docs
# if the download fails.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = "pose_landmarker_lite.task"

# Pose landmark indices (BlazePose 33-point layout)
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24
TORSO_IDS = [L_SHOULDER, R_SHOULDER, R_HIP, L_HIP]


# --------------------------------------------------------------------------
# Detector interface. Anything that returns a Target (or None) can be swapped
# in later, for example a YOLO + TensorRT detector on the Jetson.
# --------------------------------------------------------------------------
@dataclass
class Target:
    cx: float  # normalized 0..1, left to right
    cy: float  # normalized 0..1, top to bottom
    landmarks: Optional[np.ndarray] = None  # (N, 3) normalized x, y, visibility


class MediaPipePoseDetector:
    def __init__(self, min_visibility: float = 0.5):
        self.min_visibility = min_visibility
        self._ensure_model()
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = -1

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
                f"Download a Pose Landmarker .task file manually from the MediaPipe "
                f"docs and save it as {MODEL_PATH}."
            )

    def detect(self, frame_bgr: np.ndarray) -> Optional[Target]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamps must be strictly increasing in VIDEO mode.
        ts_ms = int(time.monotonic() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        result = self.landmarker.detect_for_video(mp_image, ts_ms)
        if not result.pose_landmarks:
            return None

        lms = result.pose_landmarks[0]
        pts = np.array([[l.x, l.y, l.visibility] for l in lms], dtype=np.float32)

        # Aim point: mean of visible torso landmarks (steadier than a box center).
        torso = pts[TORSO_IDS]
        visible = torso[torso[:, 2] >= self.min_visibility]
        if len(visible) == 0:
            # Fall back to any reasonably visible landmark.
            visible = pts[pts[:, 2] >= self.min_visibility]
            if len(visible) == 0:
                return None
        cx, cy = float(visible[:, 0].mean()), float(visible[:, 1].mean())
        return Target(cx=cx, cy=cy, landmarks=pts)


# --------------------------------------------------------------------------
# Controller: turns horizontal pixel error into a pan angle.
# --------------------------------------------------------------------------
class PanController:
    def __init__(
        self,
        kp: float = 12.0,
        kd: float = 2.0,
        deadband: float = 0.04,
        min_angle: float = 0.0,
        max_angle: float = 180.0,
        invert: bool = False,
        max_step: float = 4.0,
    ):
        self.kp, self.kd = kp, kd
        self.deadband = deadband
        self.min_angle, self.max_angle = min_angle, max_angle
        self.sign = -1.0 if invert else 1.0
        self.max_step = max_step  # max degrees change per frame
        self.angle = (min_angle + max_angle) / 2.0
        self._prev_err = 0.0

    def recentre(self):
        self.angle = (self.min_angle + self.max_angle) / 2.0
        self._prev_err = 0.0

    def update(self, target: Optional[Target]) -> float:
        if target is None:
            self._prev_err = 0.0
            return self.angle  # hold position when target is lost

        err = target.cx - 0.5  # positive when target is right of centre
        if abs(err) < self.deadband:
            self._prev_err = err
            return self.angle

        d_err = err - self._prev_err
        self._prev_err = err
        step = self.sign * (self.kp * err + self.kd * d_err)
        step = float(np.clip(step, -self.max_step, self.max_step))
        self.angle = float(np.clip(self.angle + step, self.min_angle, self.max_angle))
        return self.angle


# --------------------------------------------------------------------------
# Optional serial link to the Arduino.
# --------------------------------------------------------------------------
class ServoLink:
    def __init__(self, port: Optional[str], baud: int = 115200):
        self.ser = None
        if port:
            import serial  # pyserial

            self.ser = serial.Serial(port, baud, timeout=0.01)
            time.sleep(2.0)  # Uno resets when the port opens
            print(f"Serial open on {port} @ {baud}")

    def send_pan(self, angle: float):
        if self.ser:
            self.ser.write(f"P,{int(round(angle))}\n".encode())

    def close(self):
        if self.ser:
            self.ser.close()


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def draw_overlay(frame, target: Optional[Target], angle: float, fps: float, deadband: float):
    h, w = frame.shape[:2]

    # Centre line and deadband
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    x1, x2 = int((0.5 - deadband) * w), int((0.5 + deadband) * w)
    cv2.rectangle(frame, (x1, 0), (x2, h), (80, 80, 80), 1)

    if target is not None:
        if target.landmarks is not None:
            for x, y, v in target.landmarks:
                if v >= 0.5:
                    cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)
            torso = target.landmarks[TORSO_IDS]
            if (torso[:, 2] >= 0.5).all():
                poly = np.array([[int(x * w), int(y * h)] for x, y, _ in torso], np.int32)
                cv2.polylines(frame, [poly], True, (0, 200, 255), 2)
        cx_px, cy_px = int(target.cx * w), int(target.cy * h)
        cv2.circle(frame, (cx_px, cy_px), 8, (0, 0, 255), 2)
        cv2.line(frame, (w // 2, h // 2), (cx_px, cy_px), (0, 0, 255), 1)
        status = "TRACKING"
    else:
        status = "NO TARGET (holding)"

    cv2.putText(frame, f"{status}  pan={angle:.1f} deg  fps={fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--serial", type=str, default=None, help="serial port for Arduino")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--kp", type=float, default=12.0)
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--deadband", type=float, default=0.04)
    ap.add_argument("--invert", action="store_true", help="flip pan direction")
    ap.add_argument("--no-mirror", action="store_true", help="disable mirrored preview")
    ap.add_argument("--no-display", action="store_true", help="headless (no window)")
    args = ap.parse_args()

    # CAP_V4L2 is the usual backend on Linux/Jetson; default elsewhere.
    backend = cv2.CAP_V4L2 if os.name == "posix" and os.uname().sysname == "Linux" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    detector = MediaPipePoseDetector()
    controller = PanController(kp=args.kp, kd=args.kd, deadband=args.deadband, invert=args.invert)
    link = ServoLink(args.serial, args.baud)

    fps, t_prev = 0.0, time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            # Mirror the preview only. If you mirror the image the detector sees,
            # the control direction flips; use --invert to correct either way.
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            target = detector.detect(frame)
            angle = controller.update(target)
            link.send_pan(angle)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            if not args.no_display:
                draw_overlay(frame, target, angle, fps, args.deadband)
                cv2.imshow("Person tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    controller.recentre()
    finally:
        cap.release()
        link.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
