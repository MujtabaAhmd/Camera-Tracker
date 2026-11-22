# Person Tracking Pan Camera

A camera on a servo that follows a person. A computer detects the person in a USB camera feed using MediaPipe Pose, computes how far the person is from the center of the image, and sends a pan angle to an Arduino Uno, which drives the servo.


## How it works

```
USB camera
    |
    v
Detector (MediaPipe Pose Landmarker)  ->  aim point (torso center)
    |
    v
Controller (P + D on horizontal error, deadband, step limit)
    |
    v
Serial (USB) "P,<angle>\n"  ->  Arduino Uno  ->  Servo (pan)
```

- **Aim point**: the average of the visible shoulder and hip landmarks. This is steadier than a bounding box center, which moves when limbs move.
- **Controller**: converts horizontal pixel error into a pan angle change, with a deadband around the center to prevent jitter, and a limit on the change per frame.
- **Target lost**: the pan angle holds. The Arduino also holds if commands stop arriving.
- **Detector is swappable**: any class with a `detect(frame)` method that returns a `Target` (or `None`) can replace the MediaPipe detector, for example a YOLO + TensorRT detector on the Jetson.

## Files

| File | Purpose |
|------|---------|
| `person_tracker.py` | Python program: camera, detection, control, serial output |
| `pan_servo.ino` | Arduino Uno sketch: reads serial commands, drives the servo |
| `README.md` | This file |

## Hardware

- Laptop or Raspberry Pi or Jetson Orin
- USB camera
- Arduino Uno
- Hobby servo
- Pan bracket to mount the camera on the servo
- USB cable between the computer and the Uno

### Wiring

| Servo wire | Connect to |
|------------|-----------|
| Signal (orange/yellow/white) | Uno pin D9 |
| Power (red) | External 5V supply (or Uno 5V for a micro servo only) |
| Ground (brown/black) | External supply GND **and** Uno GND (grounds must be common) |

Servo current spikes can reset the Uno if it powers the servo directly, so use an external supply for anything larger than a micro servo.

## Software setup

### 1. Arduino

1. Open `pan_servo.ino` in the Arduino IDE.
2. Set `ANGLE_MIN` and `ANGLE_MAX` to your bracket's safe range so the servo never pushes against a hard stop.
3. Upload to the Uno.
4. Test without Python: open the Serial Monitor at **115200 baud**, set the line ending to **Newline**, and send `P,45` then `P,135`. The servo should move smoothly. If you stop sending, it should hold still after about half a second.

### 2. Python (laptop)

```
pip install mediapipe opencv-python numpy pyserial
python person_tracker.py
```

On the first run the program downloads the Pose Landmarker model (`pose_landmarker_lite.task`) next to the script. If the download fails, get a Pose Landmarker `.task` file from the MediaPipe documentation and save it with that name.

Without `--serial`, the program runs in simulation mode: it shows a virtual pan angle on screen and sends nothing. Use this to check detection and control behavior before connecting hardware.

### 3. Run with the Arduino connected

```
python person_tracker.py --serial /dev/ttyACM0     # Linux / Jetson
python person_tracker.py --serial COM3             # Windows
```

The Uno resets when the serial port opens, so the program waits 2 seconds before sending.

## Command line options

| Option | Default | Meaning |
|--------|---------|---------|
| `--camera N` | 0 | Camera index |
| `--width`, `--height` | 640, 480 | Capture resolution |
| `--serial PORT` | none | Serial port of the Arduino (omit for simulation) |
| `--baud` | 115200 | Serial baud rate (must match the sketch) |
| `--kp` | 12.0 | Proportional gain |
| `--kd` | 2.0 | Derivative gain |
| `--deadband` | 0.04 | Fraction of image width around the center where no movement happens |
| `--invert` | off | Flip the pan direction |
| `--no-mirror` | off | Disable the mirrored preview |
| `--no-display` | off | Run without a window (for SSH) |

Keys in the preview window: `q` quits, `c` recenters the virtual pan angle.

## Serial protocol

One command per line, 115200 baud, newline terminated:

```
P,<angle>\n        e.g. P,92
```

`<angle>` is an integer in degrees. The Arduino clamps it to its configured range. Malformed lines are ignored.

## Tuning

The default gains are starting guesses, not tuned values.

- **Servo moves the wrong way**: run with `--invert`.
- **Oscillates or hunts back and forth**: lower `--kp` first. Latency in the camera and servo loop is the usual cause.
- **Sluggish**: raise `--kp`, or raise `MAX_STEP_DEG` in the Arduino sketch.
- **Jitter when the person stands still**: raise `--deadband`.
- **Laggy image**: use a lower resolution, and a camera mode with low latency (MJPEG if supported). The program already requests a minimal capture buffer.

## Moving to the Jetson Orin

- **Installing MediaPipe may be the hardest step.** Whether a compatible wheel exists for your JetPack and Python version is something to check early with `pip install mediapipe`, and against the MediaPipe documentation and issue tracker if it fails.
- MediaPipe may run on the CPU rather than the Orin's GPU, so measure the frame rate. If it is too slow or will not install, write a YOLO + TensorRT detector with the same `detect(frame)` interface and leave the controller and serial code unchanged.
- Serial access may need your user to be in the `dialout` group on Linux if you get a permission error on `/dev/ttyACM0`.
- Use `--no-display` when working over SSH.

## Known limitations

- Tracks a single person, with no persistent identity. If several people are in view, the pose model may switch between them.
- Pan only (no tilt yet).
- Pose detection works best when the person is reasonably close and mostly visible, and degrades with distance, occlusion, and poor lighting.
- A hobby servo has limited range and is not perfectly smooth.
- No hardware has been tested yet.

## Roadmap

- Add tilt (second servo, `T,<angle>` command).
- Train a model to recognize and follow a specific person (re-identification), or to recognize gesture commands from pose landmarks.
- Fine-tune a person detector on data from the actual camera and environment, and deploy it with TensorRT on the Orin.
- Compare detectors on accuracy and frame rate, using MediaPipe as the baseline.
