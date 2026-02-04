# Python

This page gives a quick overview of using Trackers from Python. For evaluation-specific usage, see the [Evaluate guide](learn/evaluate.md).

## Install

```bash
pip install trackers
```

## Track objects with a tracker

```python
from trackers import ByteTrackTracker
import supervision as sv

tracker = ByteTrackTracker()

# detections is a supervision.Detections object from your detector
detections = tracker.update(detections)

# Each detection carries a tracker_id assigned by the tracker
print(detections.tracker_id[:5])
```

## Evaluate tracking results

```python
from trackers.eval import evaluate_mot_sequence

result = evaluate_mot_sequence(
    gt_path="data/gt/MOT17-02.txt",
    tracker_path="data/trackers/MOT17-02.txt",
    metrics=["CLEAR", "HOTA", "Identity"],
)
print(result.table(columns=["MOTA", "HOTA", "IDF1", "IDSW"]))
```

See also: [Quickstart](index.md), [Evaluate guide](learn/evaluate.md), [Evals API](api-evals.md)
