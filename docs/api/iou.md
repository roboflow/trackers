# IoU API

IoU variants are pluggable similarity metrics used during detection to track
association. You just pass one of these classes to a tracker via the `iou=` argument.

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

- `IoU`: strongest baseline, simplest behavior, score range `[0, 1]` (default)
- `GIoU`: adds distance signal for non-overlapping boxes, score range `[-1, 1]`
- `DIoU`: IoU minus distance between centers penalty, often smoother for motion-heavy scenes
- `CIoU`: DIoU minus the paper aspect-ratio term, score can go below `-1`
- `BIoU`: buffered IoU (dilates boxes), useful when objects move fast, or are so small that its difficult to find an overlap

**Formula Summary** (`A, B` boxes, `C` enclosing box, `d` center distance, `c` enclosing diagonal, `r` BIoU `buffer_ratio`):

- Standard \( \mathrm{IoU}(A, B) = |A \cap B| / |A \cup B| \)
- **BIoU:** dilate each box by a relative margin \( r \) around its sides (\( w = x_2 - x_1 \), \( h = y_2 - y_1 \)):  
    \( (x_1', y_1', x_2', y_2') = (x_1 - r w,\, y_1 - r h,\, x_2 + r w,\, y_2 + r h) \), then  
    \( \mathrm{BIoU}_r(A, B) = \mathrm{IoU}(A^{r}, B^{r}) \). For \( r = 0 \) this is plain IoU.
- \( \mathrm{GIoU} = \mathrm{IoU} - \frac{|C \setminus (A \cup B)|}{|C|} \)
- \( \mathrm{DIoU} = \mathrm{IoU} - \frac{d^2}{c^2 + \epsilon} \)
- \( \mathrm{CIoU} = \mathrm{DIoU} - \alpha v \), where
    \( v = \frac{4}{\pi^2}\left(\arctan\frac{w_A}{h_A} - \arctan\frac{w_B}{h_B}\right)^2 \)
    and \( \alpha = \frac{v}{1 - \mathrm{IoU} + v + \epsilon} \)

## Tracker Examples

```python
from trackers import OCSORTTracker, SORTTracker
from trackers.utils.iou import BIoU, CIoU, DIoU, GIoU, IoU

# Standard IoU in SORT
sort_iou = SORTTracker(iou=IoU(), minimum_iou_threshold=0.3)

# GIoU in OC-SORT (negative thresholds are valid)
ocsort_giou = OCSORTTracker(iou=GIoU(), minimum_iou_threshold=-0.3)

# DIoU in OC-SORT
ocsort_diou = OCSORTTracker(iou=DIoU(), minimum_iou_threshold=-0.3)

# CIoU in OC-SORT
ocsort_ciou = OCSORTTracker(iou=CIoU(), minimum_iou_threshold=-0.3)

# Buffered IoU in SORT
sort_biou = SORTTracker(
    iou=BIoU(buffer_ratio=0.1),
    minimum_iou_threshold=0.3,
)
```

## Threshold Notes

- For `IoU` and `BIoU`, thresholds are non-negative.
- For `GIoU`, `DIoU`, and `CIoU`, negative thresholds are possible, they now give a signal for non-overlapping boxes (what IoU doesn't do).

## Empirical HOTA deltas 

The following numbers come from running **OC-SORT** on **MOT17 train (FRCNN)** and **SportsMOT val**: for each sequence we take the **best HOTA** among **IoU** trials and the **best HOTA** among trials for **GIoU**, **DIoU**, **CIoU**, and **BIoU**, then **Δ HOTA = HOTA(variant) − HOTA(IoU)**. On SportsMOT, **detections are derived from GT boxes** (oracle feed); on MOT17 we use **FRCNN public detections** when available. This matches `notebooks/iou_variant_hota_sweep.py`, so deltas reflect **association metric** choice under that detection setting.

The following OC-SORT settings are held fixed (`lost_track_buffer=30`, `minimum_consecutive_frames=3`, `direction_consistency_weight=0.2`, `high_conf_det_threshold=0.25`, `delta_t=3`). The grid only varies **which IoU class** is passed to the tracker, **`minimum_iou_threshold`** (for IoU/BIoU: 0.0, 0.1, 0.2, 0.3, 0.4; for GIoU/DIoU/CIoU: −0.5, −0.3, −0.1, 0.0, 0.1, 0.3), and for **BIoU** only **`buffer_ratio`** (0.15 and 0.25).

HOTA in tables below is shown as **percentage** (0–100 scale). **Δ** is **percentage points** on that scale.

### Per-dataset mean Δ HOTA

| Dataset | Sequences | GIoU mean Δ | DIoU mean Δ | CIoU mean Δ | BIoU mean Δ |
| :------ | --------: | ----------: | ----------: | ----------: | ----------: |
| MOT17 train (FRCNN) | 7 | +0.35 | −0.10 | −0.10 | +0.66 |
| SportsMOT val | 45 | +1.15 | +0.89 | +0.89 | +0.74 |


### Side-by-side examples (videos)

Each of the following clips compares and illustrates what difference each IoU variant can make.

#### GIoU — Sequence: `v_0kUtTtmLaJA_c006`

| | HOTA (%) | Δ (pts) |
| :- | ------: | ------: |
| Best IoU | 73.07 | — |
| Best GIoU | 89.31 | **+16.24** |

<video width="100%" controls muted loop>
  <source src="../../assets/iou_vs_GIoU_v_0kUtTtmLaJA_c006.mp4" type="video/mp4">
</video>

#### DIoU — Sequence: `v_0kUtTtmLaJA_c006`

| | HOTA (%) | Δ (pts) |
| :- | ------: | ------: |
| Best IoU | 73.07 | — |
| Best DIoU | 86.53 | **+13.46** |

<video width="100%" controls muted loop>
  <source src="../../assets/iou_vs_DIoU_v_0kUtTtmLaJA_c006.mp4" type="video/mp4">
</video>

> Note: DIoU and CIoU are very close overall, but no perfectly identical across every sequence.

#### CIoU — Sequence: `v_0kUtTtmLaJA_c006`

| | HOTA (%) | Δ (pts) |
| :- | ------: | ------: |
| Best IoU | 73.07 | — |
| Best CIoU | 86.53 | **+13.46** |

<video width="100%" controls muted loop>
  <source src="../../assets/iou_vs_CIoU_v_0kUtTtmLaJA_c006.mp4" type="video/mp4">
</video>

#### BIoU — Sequence: `v_9MHDmAMxO5I_c004`

| | HOTA (%) | Δ (pts) |
| :- | ------: | ------: |
| Best IoU | 80.54 | — |
| Best BIoU | 88.00 | **+7.46** |

<video width="100%" controls muted loop>
  <source src="../../assets/iou_vs_BIoU_v_9MHDmAMxO5I_c004.mp4" type="video/mp4">
</video>

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
