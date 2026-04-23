# GitHub Copilot Instructions — Roboflow Trackers

Roboflow Trackers is a Python library for multi-object tracking (MOT). It provides clean-room
implementations of SORT, ByteTrack, and OC-SORT that plug into any detection model via the
`supervision` library.

## Installation

```bash
pip install trackers
```

## Core API

All trackers share the same interface:

```python
from trackers import SORTTracker, ByteTrackTracker, OCSORTTracker

tracker = ByteTrackTracker()
tracked = tracker.update(detections)  # detections: supervision.Detections
# tracked.tracker_id contains assigned track IDs
```

## Tracker Classes

- `SORTTracker` — Kalman filter + Hungarian matching. Fastest, no appearance features.
- `ByteTrackTracker` — Two-stage matching (high + low confidence detections). Best for crowded scenes.
- `OCSORTTracker` — Observation-centric SORT with direction consistency. Best for non-linear motion.

## Key Parameters (all trackers)

- `lost_track_buffer` — frames to keep a lost track alive (default varies per tracker)
- `track_activation_threshold` — minimum detection confidence to start a new track
- `minimum_consecutive_frames` — frames before a new track is confirmed
- `minimum_iou_threshold` — minimum IoU for track-detection association

ByteTrack additional: `high_conf_det_threshold` — threshold separating high vs low confidence detections.

OC-SORT additional: `direction_consistency_weight`, `delta_t`.

## File Structure

```
trackers/
├── core/          # SORTTracker, ByteTrackTracker, OCSORTTracker
│   ├── base.py    # BaseTracker ABC — defines update(detections, image) interface
│   ├── sort/      # SORT implementation (kalman.py, tracker.py)
│   ├── bytetrack/ # ByteTrack implementation
│   └── ocsort/    # OC-SORT implementation
├── motion/        # MotionEstimator, homography compensation
├── utils/         # Kalman filter, coordinate converters (xcycsr ↔ xyxy)
├── annotators/    # MotionAwareTraceAnnotator
├── io/            # Video/webcam/RTSP frame reader, MOT format I/O
├── datasets/      # Dataset manifests and download helpers
├── eval/          # CLEAR / HOTA / Identity metrics
└── scripts/       # CLI entry points (trackers track, trackers eval, trackers download)
```

## Detection Format

Input: `supervision.Detections` with `.xyxy` bounding boxes.
Output: same `supervision.Detections` with `.tracker_id` populated.

Internal state uses `xcycsr` (center-x, center-y, area, aspect ratio) for Kalman filter.

## CLI

```bash
# Track a video
trackers track --source video.mp4 --output output.mp4 --model rfdetr-medium --tracker bytetrack

# Evaluate against MOT ground truth
trackers eval --gt-dir ./data/mot17/val --tracker-dir results --metrics CLEAR HOTA Identity

# Download benchmark datasets
trackers download mot17 --split val --asset annotations,detections
```

## Benchmark Results (HOTA, default parameters)

| Tracker   | MOT17 | SportsMOT | SoccerNet | DanceTrack |
| --------- | ----- | --------- | --------- | ---------- |
| SORT      | 58.4  | 70.9      | 81.6      | 45.0       |
| ByteTrack | 60.1  | 73.0      | 84.0      | 50.2       |
| OC-SORT   | 61.9  | 71.7      | 78.4      | 51.8       |

## Documentation

Full documentation: https://trackers.roboflow.com

- [Quickstart](https://trackers.roboflow.com/latest/)
- [Tracker Comparison](https://trackers.roboflow.com/latest/trackers/comparison/)
- [API Reference](https://trackers.roboflow.com/latest/api/trackers/)
- [Track Objects Guide](https://trackers.roboflow.com/latest/learn/track/)
- [Evaluation Guide](https://trackers.roboflow.com/latest/learn/evaluate/)

## License

Apache 2.0 — https://github.com/roboflow/trackers
