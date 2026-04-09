# IoU API

IoU variants are pluggable similarity metrics used during detection-to-track
association. You pass one of these classes to a tracker via the `iou=` argument.

## Quick Start

```python
from trackers import SORTTracker
from trackers.utils.iou import IoU

tracker = SORTTracker(
    iou=IoU(),
    minimum_iou_threshold=0.3,
)
```

## Choosing a Metric

- `IoU`: strongest baseline, simplest behavior, score range `[0, 1]`
- `GIoU`: adds distance signal for non-overlapping boxes, score range `[-1, 1]`
- `DIoU`: IoU minus center-distance penalty, often smoother for motion-heavy scenes
- `CIoU`: DIoU plus aspect-ratio consistency, score can go below `-1`
- `BIoU`: buffered IoU (dilates boxes), useful when detections are slightly misaligned

**Formula Summary** (`A, B` boxes, `C` enclosing box, `d` center distance, `c` enclosing diagonal):

- \( \mathrm{GIoU} = \mathrm{IoU} - \frac{|C \setminus (A \cup B)|}{|C|} \)
- \( \mathrm{DIoU} = \mathrm{IoU} - \frac{d^2}{c^2 + \epsilon} \)
- \( \mathrm{CIoU} = \mathrm{DIoU} - \alpha v \), where
    \( v = \frac{4}{\pi^2}\left(\arctan\frac{w_A}{h_A} - \arctan\frac{w_B}{h_B}\right)^2 \)
    and \( \alpha = \frac{v}{1 - \mathrm{IoU} + v + \epsilon} \)

## Tracker Examples

```python
from trackers import OCSORTTracker, SORTTracker
from trackers.utils.iou import BIoU, CIoU, GIoU, IoU

# Standard IoU in SORT
sort_iou = SORTTracker(iou=IoU(), minimum_iou_threshold=0.3)

# GIoU in OC-SORT (negative thresholds are valid)
ocsort_giou = OCSORTTracker(iou=GIoU(), minimum_iou_threshold=-0.3)

# CIoU in OC-SORT
ocsort_ciou = OCSORTTracker(iou=CIoU(), minimum_iou_threshold=-0.3)

# Buffered IoU in SORT
sort_biou = SORTTracker(
    iou=BIoU(buffer_ratio=0.1),
    minimum_iou_threshold=0.3,
)
```

## Threshold Notes

- For `IoU` and `BIoU`, thresholds are typically non-negative.
- For `GIoU`, `DIoU`, and `CIoU`, negative thresholds are often useful.
- Start with your current IoU threshold and sweep downward/upward by metric.

## API Reference

## BaseIoU

::: trackers.utils.iou.BaseIoU

## IoU

::: trackers.utils.iou.IoU

## GIoU

::: trackers.utils.iou.GIoU

## DIoU

::: trackers.utils.iou.DIoU

## CIoU

::: trackers.utils.iou.CIoU

## BIoU

::: trackers.utils.iou.BIoU
