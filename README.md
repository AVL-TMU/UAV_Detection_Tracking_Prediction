# UAV Detection → Tracking → Trajectory Prediction (TensorRT + ByteTrack + KF)

End-to-end pipeline for **real-time air-to-air object detection**, **multi-object tracking**, and **short-horizon trajectory prediction**.  
Detections come from a **TensorRT** engine (RF-DETR-style), tracks from **ByteTrack**, and predictions from a lightweight **Kalman filter** with per-frame and per-track evaluation (ADE/FDE, success@τ, heading/speed error, etc.).

> Output = annotated video + CSVs + plots + a summary file, all in a timestamped `evaluation_<video>_<time>` folder.

---

## Table of contents

- [Project structure](#project-structure)  
- [What each module does](#what-each-module-does)  
- [Requirements](#requirements)  
- [Setup](#setup)  
- [Quick start](#quick-start)  
- [Command-line options](#command-line-options)  
- [Outputs & file formats](#outputs--file-formats)  
- [Tuning & key parameters](#tuning--key-parameters)  
- [Programmatic API](#programmatic-api)  
- [Troubleshooting](#troubleshooting)  
- [Notes & assumptions](#notes--assumptions)  
- [License](#license)

---

## Project structure

```
.
├── main_traj.py         # Orchestrates detection → tracking → prediction → logging/plots
├── detection.py         # TensorRT inference wrapper for RF-DETR; returns [x1,y1,x2,y2,score]
├── tracking.py          # ByteTrack wrapper; returns active tracks with IDs + bboxes
└── trajectory.py        # Kalman predictor, metrics, drawings, CSV aggregation
```

---

## What each module does

### `detection.py`
- Loads a **TensorRT** engine and runs batched inference on frames.
- Preprocessing via `rfdetr.datasets.transforms` (square resize to `CUSTOM_RESOLUTION`).
- Returns `np.ndarray` of detections shaped `(N, 5)` with columns `[x1, y1, x2, y2, score]` in **pixel** coordinates, plus an inference time.
- Key constants:
  - `DETECTION_THRESHOLD` (default `0.5`)
  - `CUSTOM_RESOLUTION` (default `560`) — **must match the TRT engine input size**.

### `tracking.py`
- Thin wrapper around **ByteTrack**:
  - `ObjectTracker.update(dets, (H,W))` → list of active targets for the current frame.
- Each target provides: `track_id`, `bbox (x1,y1,x2,y2)`, `center (cx,cy)`, and `score`.
- Tunables: `track_thresh`, `match_thresh`, `track_buffer`, `frame_rate`.

### `trajectory.py`
- **TrajectoryPredictor** keeps per-track state with a Kalman filter (constant-velocity model), maintains position history, and generates **future** position predictions (`future_frames` ahead).
- Evaluates past predictions when their ground truth positions become available:
  - ADE/FDE (pixels), **success rate** @ `SUCCESS_THRESHOLD` (pixels),
  - heading error (radians) & speed error (px/frame).
- Aggregates metrics per frame & per track, exposes helpers to write CSVs and draw overlays.
- Key constants:
  - `SMOOTH_WINDOW` (moving average for measurements),
  - `SUCCESS_THRESHOLD` (default `50` px),
  - `DEBUG_DRAW_HEADINGS` (draw heading vectors on frames).

### `main_traj.py`
- Wires everything together:
  1. Loads the TRT engine via `Detector(ENGINE_PATH)`.
  2. Feeds frame detections into `ObjectTracker`.
  3. Predicts `future_frames` ahead with `TrajectoryPredictor`.
  4. Writes annotated video, per-frame & per-track CSVs, plots, and a text summary.
- CLI entry point; see [Command-line options](#command-line-options).

---

## Requirements

- **Python** 3.9–3.11
- **CUDA** + **TensorRT** compatible with your GPU/driver
- **PyTorch** (CPU is fine; TRT does the heavy lifting)
- **pycuda**, **opencv-python**, **numpy**, **Pillow**, **matplotlib**, **pandas**
- **ByteTrack** (Python tracker; `yolox`-style module path available)
- **rfdetr** (for the compose/resize transforms used in preprocessing)

Install a typical stack:

```bash
pip install opencv-python numpy pillow matplotlib pandas pycuda
# PyTorch: pick the wheel matching your CUDA (see pytorch.org)
# TensorRT & CUDA: install via NVIDIA’s instructions for your OS.
# ByteTrack: ensure `ByteTrack.yolox.tracker.byte_tracker` is importable.
# rfdetr: ensure `rfdetr.datasets.transforms` is importable.
```

---

## Setup

1. **Place or build your TensorRT engine** for the detector (RF-DETR style).  
   - Input size must equal `CUSTOM_RESOLUTION` in `detection.py` (default `560×560`).
2. **Set the engine path**:
   - In `main_traj.py`, the detector is created as `Detector(ENGINE_PATH)`.  
     Define `ENGINE_PATH` there (absolute path recommended), **or** modify `main_traj.py` to read it from an env var (e.g., `ENGINE_PATH=os.environ["RFDETR_ENGINE"]`).
3. Confirm **ByteTrack** is importable at:
   ```
   from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
   ```
   If not, add the ByteTrack repo/root to `PYTHONPATH`.

---

## Quick start

```bash
python main_traj.py   --video /path/to/video.mp4   --future-frames 10   --threshold 0.5   --frame-limit 5000
```

- On start, a folder like `evaluation_DJI_4_20250902_133012/` is created with:
  - `tracked_video.mp4`
  - `csv/frame_metrics.csv`, `csv/track_metrics.csv`
  - `plots/` (latency/FPS, success rate, error histograms, etc.)
  - `summary.txt` (key metrics and top tracks)

---

## Command-line options

`main_traj.py` exposes:

- `--video` (str): Path to the input video.  
  **Default** is set by `DEFAULT_VIDEO_PATH` inside `main_traj.py`.
- `--frame-limit` (int): Max frames to process (useful for quick runs).  
  **Default:** `5000`.
- `--future-frames` (int): How many frames ahead to predict.  
  **Default:** `10`.
- `--threshold` (float): Detection confidence threshold (passed to detector & tracker).  
  **Default:** `0.5` (mirrors `DETECTION_THRESHOLD`).

> You can also edit the module-level constants in `detection.py`, `tracking.py`, and `trajectory.py` for defaults that better fit your setup.

---

## Outputs & file formats

### Annotated video
`evaluation_<name>_<time>/tracked_video.mp4`  
- Shows detected boxes, track IDs, current centers, predicted future point(s), and (optionally) heading vectors.

### CSVs

#### `csv/frame_metrics.csv` (one row per processed frame)
Columns typically include:
- `frame_id`
- `detections` (count)
- `tracks` (count)
- `fps_trt` (instantaneous FPS for TRT)
- `infer_time_ms` (per-frame TRT time)
- `predictions_total`, `predictions_successful`
- `frame_success_rate` (successful / total)
- `frame_displacement_error_mean` (px)
- (optionally) `frame_heading_error_mean` (rad), `frame_speed_error_mean` (px/frame)

#### `csv/track_metrics.csv` (one row per track ID)
- `track_id`
- `predictions` (count)
- `successful_predictions` (count)
- `success_rate` (0–1)
- `avg_displacement_error` (px)
- `avg_heading_error` (rad)
- `avg_speed_error` (px/frame)
- `track_lifespan` (frames with observations)

### Plots
Saved under `plots/`:
- **Latency/FPS over time** (per-frame TRT time, FPS)
- **Frame-level success rate over time**
- **Displacement error histogram** with ADE/FDE markers

### Summary
`summary.txt` includes overall metrics, e.g.:
- **ADE** (Average Displacement Error, px)  
- **FDE** (Final Displacement Error, px)  
- **Success Rate** @ `SUCCESS_THRESHOLD` (e.g., 50 px)  
- **Average Heading Error** (radians)  
- **Average Speed Error** (px/frame)  
- **Std dev** of displacement errors  
- Top 5 tracks by prediction count

---

## Tuning & key parameters

- **Detector**
  - `CUSTOM_RESOLUTION` (default `560`) must match engine input size.
  - `DETECTION_THRESHOLD` affects both which detections are passed to tracking and overall false-positive rate.
- **Tracker (ByteTrack)**
  - `track_thresh` (default ~`0.5`), `match_thresh` (default ~`0.8`), `track_buffer` (default `30`), `frame_rate` (default `30`).
- **Predictor**
  - `future_frames` (CLI) controls prediction horizon (in frames).
  - `SMOOTH_WINDOW` (default `10`) smooths raw centers before KF updates.
  - `SUCCESS_THRESHOLD` (default `50` px) used for success@τ.
  - `DEBUG_DRAW_HEADINGS` to visualize heading vectors.
- **Runtime**
  - `FRAME_LIMIT` in `main_traj.py` caps processed frames for quick evals.
  - Adjust Matplotlib figure sizes if plotting on headless servers.

---

## Programmatic API

Minimal examples (inside your own script):

```python
from detection import Detector
from tracking import ObjectTracker
from trajectory import TrajectoryPredictor

det = Detector("/abs/path/to/your.engine")
tracker = ObjectTracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30)
pred = TrajectoryPredictor(future_frames=10, warmup_frames=10)

# Per frame:
dets_xyxy, det_time = det.detect(frame_bgr)          # (N,5): x1,y1,x2,y2,score
online_targets = tracker.update(dets_xyxy, frame_bgr.shape[:2][::-1])  # (H,W)

for t in online_targets:
    info = tracker.get_track_info(t)                  # id, bbox, center, score
    out = pred.predict(info['id'], info['center'], frame_id=cur_id)
    # out = {'current':(cx,cy),'predicted':(px,py),'velocity':(vx,vy)}

# Later when the future frame arrives, evaluate past predictions:
pred.evaluate(track_id, current_position, frame_id=cur_id)

# At the end:
overall = pred.compute_overall_metrics()  # ADE, FDE, success_rate, heading/speed errs, std
pertrk  = pred.get_track_data()           # list of per-track dicts
```

Example for Webcam Usage:

```python
import cv2
from detection import Detector
from tracking import ObjectTracker
from trajectory import TrajectoryPredictor

# Initialize modules
det = Detector("/abs/path/to/your.engine")
tracker = ObjectTracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30)
pred = TrajectoryPredictor(future_frames=10, warmup_frames=10)

cap = cv2.VideoCapture(0)  # 0 = default webcam
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    dets_xyxy, det_time = det.detect(frame)

    # Run tracking
    online_targets = tracker.update(dets_xyxy, frame.shape[:2][::-1])

    # Prediction + evaluation
    for t in online_targets:
        info = tracker.get_track_info(t)
        out = pred.predict(info['id'], info['center'], frame_id=frame_id)
        pred.evaluate(info['id'], info['center'], frame_id=frame_id)

    # Draw predictions if you want
    # cv2.imshow("Live Tracking", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()

# At the end: export metrics
overall = pred.compute_overall_metrics()
pertrk  = pred.get_track_data()
print(overall)
```

---

## Troubleshooting

- **`ModuleNotFoundError: ByteTrack...`**  
  Add ByteTrack’s repo root to `PYTHONPATH` or install it so  
  `from ByteTrack.yolox.tracker.byte_tracker import BYTETracker` works.

- **`No module named rfdetr.datasets.transforms`**  
  Install/point to your RF-DETR codebase. Replace the transforms with an equivalent if you don’t use RF-DETR.

- **TensorRT resolution mismatch / garbage outputs**  
  Ensure `CUSTOM_RESOLUTION` in `detection.py` **matches the TRT engine** input. Rebuild the engine if needed.

- **`pycuda._driver.LogicError: explicit_context_dependent failed`**  
  Make sure `pycuda.autoinit` is imported **once** (done in `main_traj.py`).  
  Conflicting CUDA contexts or missing driver setup can also cause this.

- **Performance is low**  
  - Verify GPU is actually used (nvidia-smi, TRT FP16/INT8 build).  
  - Avoid unnecessary copies; keep frames in reasonable resolutions.  
  - Increase `track_buffer` carefully; it helps association but may bloat memory.

---

## Notes & assumptions

- Coordinates are **pixels** in the original frame size (not normalized), after detector postprocessing.  
- The motion model is **constant-velocity**; it’s chosen for stability under noisy measurements and small horizons.  
  If your future work needs jerkier maneuvers, consider a CA (constant-acceleration) model and retune process noise.
- Heading error is computed from the vector between successive positions; speed in **px/frame**.

---

## License

Add your preferred license here (MIT/BSD/GPL/Proprietary). If you’re publishing datasets or pretrained engines, state their licenses and any third-party attributions (RF-DETR, ByteTrack, etc.).

---

### Credit
RF-DETR for the detector architecture and transforms; ByteTrack for robust MOT; your Kalman implementation & evaluation glue.
