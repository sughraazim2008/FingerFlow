# Hand Gesture Recognition — MediaPipe

AI-powered real-time hand sign and finger gesture recognition using MediaPipe landmarks and a TFLite MLP classifier.

Point a webcam at your hand. Get live gesture classification. See every landmark tracked in real time.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8.1-green) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red) ![TFLite](https://img.shields.io/badge/TFLite-2.3+-orange)

---

## What It Does

- **Point your webcam at your hand** — MediaPipe detects 21 landmarks per hand in real time
- **Static hand sign recognition** — classifies open hand, closed fist, pointing, and custom signs using a trained TFLite MLP
- **Dynamic finger gesture recognition** — classifies movement trajectories (clockwise, counterclockwise, stationary, moving) using a second TFLite LSTM-capable model
- **Live overlay** — landmark skeleton, FPS counter, and classification label rendered directly on the webcam feed
- **Custom training** — record your own gesture data and retrain both models without touching the architecture

---

## How It Works — The Models

### Hand Sign Recognition (`keypoint_classifier`)

Classifies static hand poses from a single frame:

- MediaPipe extracts 21 3D landmarks (x, y, z) per hand — 63 raw values
- Landmarks are normalised relative to the wrist position and scaled by hand size, making the classifier invariant to hand distance from camera
- A 3-layer MLP (Dense → Dropout → Dense → Dropout → Softmax) is trained on the normalised keypoints
- Exported to TFLite for fast inference — runs in real time with no GPU required
- Default classes: open hand (0), closed fist (1), pointing (2) — fully extensible

### Finger Gesture Recognition (`point_history_classifier`)

Classifies movement trajectories over time:

- Tracks the index fingertip (landmark 8) position across 16 consecutive frames
- Normalised coordinate history forms a 32-feature input vector (16 × x,y pairs)
- MLP trained on this temporal sequence — optional LSTM variant available for richer temporal modelling
- Default classes: stationary (0), clockwise (1), counterclockwise (2), moving (3)

---

## Quickstart

```bash
git clone https://github.com/catgirlsughra123/hand-gesture-recognition-mediapipe-main.git
cd hand-gesture-recognition-mediapipe-main
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

---

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--device` | Camera device number | `0` |
| `--width` | Capture width (px) | `960` |
| `--height` | Capture height (px) | `540` |
| `--use_static_image_mode` | MediaPipe static image mode | off |
| `--min_detection_confidence` | Detection confidence threshold | `0.5` |
| `--min_tracking_confidence` | Tracking confidence threshold | `0.5` |

---

## Training Your Own Gestures

### Hand Sign Recognition

1. Run `app.py` and press `k` to enter key point logging mode
2. Press `0`–`9` to assign a class ID — landmarks save to `model/keypoint_classifier/keypoint.csv`
3. Open `keypoint_classification.ipynb` in Jupyter and run all cells
4. Update `keypoint_classifier_label.csv` with your class names
5. Set `NUM_CLASSES` to match your total class count

### Finger Gesture Recognition

1. Run `app.py` and press `h` to enter point history logging mode
2. Press `0`–`9` to assign a class ID — trajectory saves to `model/point_history_classifier/point_history.csv`
3. Open `point_history_classification.ipynb` and run all cells
4. Update `point_history_classifier_label.csv` with your class names

---

## File Structure

```
│  app.py                               ← main inference + webcam loop
│  keypoint_classification.ipynb        ← hand sign model training
│  point_history_classification.ipynb   ← finger gesture model training
│
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv                       ← recorded training data
│  │  │  keypoint_classifier.hdf5           ← full Keras model
│  │  │  keypoint_classifier.tflite         ← exported inference model
│  │  │  keypoint_classifier_label.csv      ← class label names
│  │  └─ keypoint_classifier.py             ← inference wrapper
│  │
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.tflite
│      │  point_history_classifier_label.csv
│      └─ point_history_classifier.py
│
└─utils
    └─cvfpscalc.py                      ← FPS measurement utility
```

---

## Requirements

```
mediapipe==0.8.1
opencv-python>=3.4.2
tensorflow>=2.3.0
scikit-learn>=0.23.2
matplotlib>=3.3.2
```

---

## Roadmap

- [x] Get base repo running locally
- [x] Understand landmark normalisation pipeline
- [ ] Gesture mouse — control cursor with index finger, pinch to click
- [ ] Confidence score overlay — show live class probabilities on screen
- [ ] Session stats + JSON export
- [ ] Feed into [gestureOS](https://github.com/catgirlsughra123/gestureOS) as the capture layer

---

## Credits

This repository is a fork of the original work by:

- **Kazuhito Takahashi** — original author · [github.com/Kazuhito00](https://github.com/Kazuhito00) · [@KzhtTkhs](https://twitter.com/KzhtTkhs)
- **Nikita Kiselov** — English translation · [github.com/kinivi](https://github.com/kinivi)

Original repository: [hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)

---

## License

Inherits [Apache 2.0 License](LICENSE) from the original repository.