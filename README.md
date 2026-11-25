# Hand Gesture Control + Person Tracking Camera

A camera system with two features that can run separately or together:

1. **Gesture control.** Hand gestures (open palm, fist, thumbs up, thumbs down, point, peace sign) control a computer: play/pause, volume, next/previous slide. The gesture classifiers (kNN, SVM, MLP) are **trained by you on a dataset recorded by your group**.
2. **Person tracking.** A servo pans the camera to follow your torso (MediaPipe Pose, PID-style control, Arduino Uno driving the servo).

```
Camera frame (flipped, selfie view)
   |-- Hand Landmarker (pretrained feature extractor) -> 60 features -> YOUR classifier
   |        -> hold-to-confirm logic -> key presses (pyautogui)
   |-- Pose Landmarker (pretrained) -> torso aim point -> controller -> serial -> Arduino -> servo
```

MediaPipe is used only to turn an image into landmark coordinates. The gesture classifiers are trained from scratch on your own data.

## Status: read this first

| Part | Tested? |
|------|---------|
| `train_gestures.py` (kNN, SVM, scikit-learn MLP, plots, saved models) | Run end to end on **synthetic** data only. Real data will give very different, much lower numbers. |
| Hold-to-confirm logic in `gesture_runtime.py` | Checked with simulated frame streams (accidental taps ignored, holds fire once, thumbs repeat, arm mode). |
| Saved models loading and predicting through the runtime path | Checked on synthetic data. |
| MediaPipe API names and imports | Checked to exist in MediaPipe 0.10.x. |
| PyTorch MLP path in `train_gestures.py` | **Not run** (PyTorch could not be installed in my test environment). The numpy inference code was checked against scikit-learn. |
| Real camera, hand/pose model download and detection | **Not run.** |
| `pyautogui` key presses, Arduino servo, Jetson Orin | **Not run.** |

Expect to fix small things when you first run it on real hardware, and verify library versions against their current documentation.

## Files

| File | Purpose |
|------|---------|
| `collect_gestures.py` | Guided data collection. Writes landmarks to a CSV. |
| `gesture_features.py` | Shared feature code (normalization, CSV layout, numpy MLP). Training and live use both import it, so features always match. |
| `train_gestures.py` | Trains and compares kNN, SVM and MLP, evaluates on held-out people, saves models and plots. |
| `gesture_runtime.py` | Live gesture control demo (classifier wrapper, hold-to-confirm engine, key mapping). |
| `person_tracker.py` | Torso tracking with a virtual or real pan servo. |
| `pan_servo.ino` | Arduino Uno sketch for the servo. |
| `tracker_gesture_control.py` | Tracking and gestures together in one camera loop. |
| `requirements.txt` | Python packages. |

## Install

```
pip install -r requirements.txt
```

The first run of each camera script downloads a MediaPipe model file (`hand_landmarker.task`, `pose_landmarker_lite.task`). If a download fails, get the matching Landmarker `.task` file from the MediaPipe documentation and save it next to the scripts under that name.

## Workflow

### Step 1: Record the dataset (each group member)

```
python collect_gestures.py --subject ali
```

- At setup, check that your **right hand is labeled "Right"** in the preview. If not, restart with `--swap-handedness`, and use the same flag in every later script.
- For each gesture you get on-screen instructions and 6 takes (different distances, tilt, movement, other hand). SPACE starts a take, R redoes it, S skips it, Q quits. Accepted takes are saved immediately.
- The `none` class is **not a gesture**. It teaches the model what ordinary hand movement looks like, which is what prevents accidental commands. Do it properly.
- Run it several times: different lighting, background, clothes, time of day. Each run is a new session.
- Every member uses their own `--subject` name. Merge later by passing several CSV files to the training script, or concatenating them (same header).
- Quick test run: `python collect_gestures.py --subject test --takes 1 --seconds 3 --out test.csv`
- Re-record one gesture: `--only fist`

**CSV format**: one row per frame in which a hand was detected: `subject_id, session_id, label, take, variation, frame_idx, t_sec, handedness, handedness_score`, then 63 raw landmark columns (`r0_x` ... `r20_z`) and 63 normalized columns (`f0_x` ... `f20_z`). Training recomputes features from the raw columns, so the `f*` columns are for inspection only.

Only landmark coordinates are saved, not images. Get consent from everyone you record.

### Step 2: Check how much data you have

```python
import pandas as pd
df = pd.read_csv("gesture_dataset.csv")
print(df.groupby("label").size())
print(df.groupby("subject_id").size())
print(df.groupby(["subject_id", "session_id"]).size())
print(pd.crosstab(df.label, df.subject_id))
```

Row count is misleading, because frames from one take are near-duplicates. The number of **people, sessions and takes** matters more. Training prints a summary and warnings too.

### Step 3: Train and evaluate

```
python train_gestures.py --data gesture_dataset.csv --quick            # fast check
python train_gestures.py --data gesture_dataset.csv --curves --loso    # full analysis
python train_gestures.py --data ali.csv sara.csv omar.csv --test-subjects omar
```

What it does:
- **Splits by person.** The test set is one or more held-out people (random by default, or `--test-subjects`). A random frame split would give inflated accuracy.
- **Tunes with grouped cross-validation** on the training people only. The test set is never used to choose hyperparameters.
- **Models:** kNN (k, weights), SVM (linear and RBF, C, gamma), MLP (PyTorch, or scikit-learn if PyTorch is missing). All use identical features and splits.
- Falls back to session-level or take-level groups if you have few people, and prints a warning, because those results are optimistic.

Outputs in `models/`:

| File | Meaning |
|------|---------|
| `knn.joblib`, `svm.joblib`, `mlp.joblib` | Trained models for `gesture_runtime.py` |
| `results.csv`, `metrics.json` | Accuracy, macro-F1, per-class precision/recall/F1, prediction time, best parameters |
| `confusion_<model>.png` | Normalized confusion matrices on the test people |
| `threshold_sweep_<model>.csv` | Effect of the confidence threshold, including how many `none` frames would wrongly trigger a command |
| `cv_results_<model>.csv` | Full cross-validation grid results |
| `pca.png` | PCA plot of the gesture clusters |
| `learning_curve_<model>.png`, `mlp_loss_curves.png` | With `--curves` |
| `loso.csv` | With `--loso`: leave-one-person-out scores |

Reading the results honestly:
- With only a few people, **`--loso` is the best summary**: mean and standard deviation of macro-F1 across people.
- Accuracy alone can hide a bad `none` class. Look at the confusion matrix and the false-command rate in the threshold sweep.
- Picking a threshold from the sweep tunes on the test set. Say so in your report, or keep a separate validation person.
- A simple model (kNN or SVM) beating the MLP is a legitimate result. Explain why.

### Step 4: Try gesture control (dry run first)

```
python gesture_runtime.py --model models/svm.joblib
```

The window shows the prediction, a hold-progress bar, and the last command. In dry-run mode commands are only printed. When it behaves, click on your slides or video player and add `--live` to press real keys:

```
python gesture_runtime.py --model models/svm.joblib --live
```

Default command mapping (edit `COMMANDS` at the top of `gesture_runtime.py`):

| Gesture | Key |
|---------|-----|
| open_palm | `playpause` (toggle) |
| fist | `volumemute` |
| thumbs_up | `volumeup` (repeats while held) |
| thumbs_down | `volumedown` (repeats while held) |
| point | `right` (next slide) |
| peace | `left` (previous slide) |

Key names are `pyautogui` key names. Media keys may not work on every operating system or desktop setup, so test them.

### Step 5: Tracker and gestures together

```
python tracker_gesture_control.py --model models/svm.joblib                       # no hardware
python tracker_gesture_control.py --model models/svm.joblib --serial /dev/ttyACM0 --live
```

While a hand is visible the pan angle is frozen (`--no-freeze-on-hand` disables this). `--pose-every 2` runs the pose model on every second frame to save CPU.

Servo setup (Arduino): upload `pan_servo.ino`, servo signal on D9, servo powered from an external 5V supply (not the Uno's 5V pin unless it is a micro servo), grounds connected. Test from the Serial Monitor at 115200 baud (Newline ending): send `P,45` then `P,135`. The protocol is one line per command, `P,<angle>`. The sketch clamps the angle, limits speed, and holds position if commands stop. Set `ANGLE_MIN`/`ANGLE_MAX` to your bracket's safe range.

## How a gesture becomes a command

A single frame never fires a command. The engine looks at the last `--hold` seconds:

1. Frames below `--min-conf` count as `none`.
2. The most common label in the window must make up at least `--agree` of the frames, and the window must be full.
3. When that holds, the command fires once, then the gesture must be **released** before it can fire again (thumbs up/down repeat every `--repeat` seconds instead).
4. Optional: `--arm-label open_palm` means nothing fires until you hold that gesture to arm the system for `--arm-timeout` seconds.

| Option | Default | Effect |
|--------|---------|--------|
| `--hold` | 1.0 | Seconds a gesture must be held. Longer is safer but feels slower. |
| `--min-conf` | 0.8 | Minimum model confidence |
| `--agree` | 0.85 | Fraction of frames that must agree |
| `--cooldown` | 1.0 | Pause after a command |
| `--repeat` | 0.4 | Repeat interval for thumbs up/down |
| `--arm-label` | none | Gesture that arms the system |

These defaults are starting guesses. **Measure false triggers per minute** while you talk and move naturally, and missed gestures per attempt, using a person who was not in the training data. Those two numbers are what you should tune against and report.

## Troubleshooting

- **Right hand shows as "Left"**: use `--swap-handedness` everywhere, and re-record if you already collected data with the wrong labels.
- **Hand often not detected**: improve front lighting, avoid a bright window behind you, keep the whole hand in frame, and lower the resolution only if the frame rate is poor.
- **Great test scores, poor live behavior**: usually too few people or too little variation in the data, or the camera was not flipped the same way. Every script flips the frame (selfie view) before detection.
- **A gesture confuses with another**: look at the confusion matrix. Replace one of the two gestures with something that looks different from the camera.
- **No `none` class results are bad**: record more varied `none` data (resting, half-formed shapes, scratching your face, transitions between gestures).
- **Servo moves the wrong way**: `--invert`. **Oscillates**: lower `--kp`. **Jitters**: raise `--deadband`.
- **Permission error on `/dev/ttyACM0` (Linux)**: your user may need to be in the `dialout` group.
- **Newer scikit-learn warns that `SVC(probability=True)` is deprecated**: it still worked in the version I tested. If a future version removes it, wrap the SVC in `CalibratedClassifierCV`.
- **Model loading errors**: models are pickled with scikit-learn, so train and run with the same scikit-learn version, or retrain on the machine that runs them.

## Moving to the Jetson Orin

- Installing MediaPipe on JetPack may be the hardest step (compatible wheels, Python version). Try `pip install mediapipe` early and check the MediaPipe documentation if it fails.
- Train on the laptop. The MLP is saved as a plain numpy model, so PyTorch is not needed on the Jetson. kNN and SVM models need scikit-learn there, and its install on the Jetson is something I have not verified.
- `pyautogui` presses keys on the machine running the script, so on the Jetson it controls the Jetson. To control another computer, run the gesture part on that computer, or add a network link (not included). `pyautogui` may also not work under Wayland desktops; I am not sure.
- Measure FPS with both detectors running, and use `--pose-every` if needed.

## Limitations

- Static gestures only. Swipes need a window-based model and are not included.
- One hand, one person. The pose tracker has no persistent identity, so with several people in view it may switch between them.
- Landmark quality drops with motion blur, bad lighting, and hands partly out of frame.
- The scale normalization uses the 2D wrist-to-knuckle distance, which shrinks when the hand points toward the camera. Check whether this hurts any gesture.
- Frame rate and accuracy on your hardware are unknown until measured.

## Mapping to the course

Nearest neighbour (kNN), SVM, neural networks with forward/backward propagation (MLP), regularization and bias-variance (grid search, learning curves, loss curves), PCA (`pca.png`), precision/recall/F1 and confusion matrices, and rejection of non-gestures through the `none` class and confidence thresholds.
