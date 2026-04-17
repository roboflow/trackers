# IoU variants

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

## Overview

| Variant | Score range | When to use                                                            |
| :------ | :---------- | :--------------------------------------------------------------------- |
| `IoU`   | `[0, 1]`    | Default — strong baseline for most scenes                              |
| `GIoU`  | `[-1, 1]`   | Scenes where boxes frequently lose overlap (occlusion, re-entry)       |
| `DIoU`  | `[-1, 1]`   | Fast-moving objects; centre-distance signal without aspect sensitivity |
| `CIoU`  | `(−∞, 1]`   | Same as DIoU plus aspect-ratio consistency                             |
| `BIoU`  | `[0, 1]`    | Very small or very fast objects where raw boxes rarely overlap         |

Negative thresholds are meaningful for `GIoU`, `DIoU`, and `CIoU` because they extend their range to give a signal even when there is no pixel overlap. For `IoU` and `BIoU` thresholds
must be non-negative.

---

## IoU

**Standard Intersection over Union** — the classic baseline.

\[
\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
\]

Scores are `0` (no overlap) to `1` (perfect overlap). Because it returns `0` whenever
boxes do not intersect, the tracker gets no gradient to recover a lost track; a
variant from the list below can help in those cases.

```python
from trackers import SORTTracker
from trackers.utils.iou import IoU

tracker = SORTTracker(iou=IoU(), minimum_iou_threshold=0.3)
```

---

## GIoU

**Generalised IoU** (Rezatofighi et al., 2019) — penalises the gap inside the
smallest enclosing box `C` that neither `A` nor `B` fills.

\[
\mathrm{GIoU}(A, B) = \mathrm{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}
\]

When boxes do not overlap at all, IoU is flat at `0`, but the penalty term still
changes as boxes move closer or farther apart — giving the tracker a meaningful
signal to bridge short gaps.

```python
from trackers import OCSORTTracker
from trackers.utils.iou import GIoU

# Negative thresholds are valid and often optimal for GIoU
tracker = OCSORTTracker(iou=GIoU(), minimum_iou_threshold=-0.3)
```

**Example — SportsMOT `v_0kUtTtmLaJA_c006`**

|           | HOTA (%) |    Δ (pts) |
| :-------- | -------: | ---------: |
| Best IoU  |    73.07 |          — |
| Best GIoU |    89.31 | **+16.24** |

Left: IoU. Right: GIoU. Camera movements confuses IoU by introducing an external movement, producing ID switches when this lands in other object. GIoU gives a partial solution to this by considering how similar the boxes are, which keeps most of the tracks that with IoU are confused or lost due direction changes and non linear motion. E.g: tracks 5, 12/13.

<video width="100%" controls muted loop>
  <source src="../../assets/iou_vs_GIoU_v_0kUtTtmLaJA_c006.mp4" type="video/mp4">
</video>

---

## DIoU

**Distance IoU** (Zheng et al., 2019) — adds a centre-distance penalty to IoU,
normalised by the enclosing box diagonal.

\[
\mathrm{DIoU}(A, B) = \mathrm{IoU} - \frac{d^2}{c^2 + \epsilon}
\]

where `d` is the Euclidean distance between box centres and `c` is the diagonal of
the smallest enclosing rectangle. This encourages centre alignment independently of
aspect ratio and tends to produce smoother associations in fast-motion sequences.

```python
from trackers import OCSORTTracker
from trackers.utils.iou import DIoU

tracker = OCSORTTracker(iou=DIoU(), minimum_iou_threshold=-0.3)
```

**Example — SportsMOT `v_0kUtTtmLaJA_c006`**

|           | HOTA (%) |    Δ (pts) |
| :-------- | -------: | ---------: |
| Best IoU  |    73.07 |          — |
| Best DIoU |    86.53 | **+13.46** |

Left: IoU. Right: DIoU. Camera movements confuses IoU by introducing an external movement, producing ID switches when this prediction runs into another object. Watch how the centre-distance term keeps track IDs stable
when camera accelerates quickly, so that objects geometrically close would lose the track with IoU. E.g: tracks 3, 4 and 5.

<video width="100%" controls muted loop>
  <source src="../../assets/iou_vs_DIoU_v_0kUtTtmLaJA_c006.mp4" type="video/mp4">
</video>

---

## CIoU

**Complete IoU** (Zheng et al., 2019) — extends DIoU with a penalty for aspect-ratio
mismatch between the two boxes.

\[
\mathrm{CIoU}(A, B) = \mathrm{DIoU} - \alpha v
\]

\[
v = \frac{4}{\pi^2}\!\left(\arctan\frac{w_A}{h_A} - \arctan\frac{w_B}{h_B}\right)^2, \quad
\alpha = \frac{v}{1 - \mathrm{IoU} + v + \epsilon}
\]

`v` measures aspect-ratio divergence; `α` scales it so the penalty is low when IoU
is already high. On tracking benchmarks CIoU and DIoU behave nearly identically —
the aspect term rarely changes which assignment wins.

```python
from trackers import OCSORTTracker
from trackers.utils.iou import CIoU

tracker = OCSORTTracker(iou=CIoU(), minimum_iou_threshold=-0.3)
```

**Example — Soccernet `SNMOT-122`**

|           | HOTA (%) |   Δ (pts) |
| :-------- | -------: | --------: |
| Best IoU  |    77.36 |         — |
| Best CIoU |    85.58 | **+8.22** |

Left: IoU. Right: CIoU. In this example, CIoU is the only IoU variant that is capable of perfectly keeping the track of the ball, which is explained by the fact that the ball is a small and fast moving object, with constant aspect ratio, so it helps a lot to math considering the distance between the detections tracks and the aspect ratio.

<video width="100%" controls muted loop>
  <source src="../../assets/snmot_122_botsort_iou_vs_CIoU_web.mp4" type="video/mp4">
</video>

---

## BIoU

**Buffered IoU** (Yang et al., 2022) — dilates each box by a relative margin `r`
before computing standard IoU. Let `w = x2 − x1`, `h = y2 − y1`:

\[
A^r = (x_1 - rw,\; y_1 - rh,\; x_2 + rw,\; y_2 + rh)
\]

\[
\mathrm{BIoU}_r(A, B) = \mathrm{IoU}(A^r, B^r)
\]

`r = 0` recovers plain IoU exactly. Enlarging boxes creates artificial overlap for
objects that are geometrically close, which is useful when detections are very small
or objects move fast enough so that consecutive boxes miss each other entirely.

```python
from trackers import SORTTracker
from trackers.utils.iou import BIoU

tracker = SORTTracker(iou=BIoU(buffer_ratio=0.15), minimum_iou_threshold=0.3)
```

**Example — SportsMOT `v_9MHDmAMxO5I_c004`**

|           | HOTA (%) |   Δ (pts) |
| :-------- | -------: | --------: |
| Best IoU  |    80.54 |         — |
| Best BIoU |    88.00 | **+7.46** |

Left: IoU. Right: BIoU. Notice how ID switches happen when fast players
temporarily produce non-overlapping boxes between frames. The buffer closes
that gap and keeps the same ID. E.g: tracks 7 and 8.

<video width="100%" controls muted loop>
  <source src="../../assets/iou_vs_BIoU_v_9MHDmAMxO5I_c004.mp4" type="video/mp4">
</video>

---

## IoU Variant Performance Across Benchmarks

Now that we understand each variant, we evaluate how much each one changes performance over different datasets.
For each `(dataset, tracker)` pair, we first keep the state estimator
(`xyxy` or `xcycsr`) with the best **IoU HOTA**, then compute mean
`ΔHOTA = HOTA(variant) − HOTA(IoU)` across those retained tracker rows. In parenthesis, the mean of HOTAs achieved by different trackers using the corresponding variant.

For more information on the datasets see: [dataset comparison](../trackers/comparison.md)

<!-- Positive value  = green, negative = red,  |delta| < 0.1 = yellow  -->

<style>
  .delta {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 6px;
    font-variant-numeric: tabular-nums;
  }
  .delta.pos { background: rgba(46, 125, 50, 0.16); }
  .delta.neg { background: rgba(198, 40, 40, 0.16); }
  .delta.neutral { background: rgba(251, 192, 45, 0.22); }
</style>

| Dataset        |                                      GIoU mean Δ |                                      DIoU mean Δ |                                      CIoU mean Δ |                                  BIoU mean Δ | IoU mean HOTA |
| :------------- | -----------------------------------------------: | -----------------------------------------------: | -----------------------------------------------: | -------------------------------------------: | ------------: |
| MOT17 val      | <span class="delta neutral">−0.09 (38.00)</span> | <span class="delta neutral">−0.04 (38.05)</span> | <span class="delta neutral">−0.04 (38.05)</span> | <span class="delta neg">−0.28 (37.80)</span> |         38.09 |
| SportsMOT val  |     <span class="delta pos">+0.65 (80.85)</span> |     <span class="delta pos">+0.95 (81.16)</span> |     <span class="delta pos">+0.88 (81.09)</span> | <span class="delta pos">+0.36 (80.57)</span> |         80.21 |
| DanceTrack val |     <span class="delta neg">−0.80 (49.47)</span> |     <span class="delta neg">−0.34 (49.94)</span> | <span class="delta neutral">+0.05 (50.32)</span> | <span class="delta pos">+0.15 (50.42)</span> |         50.27 |
| SoccerNet test |     <span class="delta pos">+1.57 (84.77)</span> |     <span class="delta pos">+2.82 (86.03)</span> |     <span class="delta pos">+2.76 (85.96)</span> | <span class="delta pos">+1.41 (84.62)</span> |         83.21 |

Over SportsMOT and Soccernet all IoU variants perform better than standard IoU, with DIoU and CIoU strongest on SoccerNet and DIoU slightly ahead of CIoU on SportsMOT. In MOT17, standard IoU is the best one by a small margin, which is almost negligible (DIoU and CIoU match each other here). On DanceTrack, GIoU and DIoU underperform IoU at this aggregation, while CIoU and BIoU are slightly ahead.

What we find from this is that IoU variants seems to give better performance depending on the task, but does detection quality matter? Soccernet gives perfect detections, and SportsMOT has detections from a really accurate detector, whereas MOT17 and DanceTrack are know for having noisy ground truth detection.

We tried SportsMOT and MOT17 with its ground truth tracks as detections, and we found that the ΔHOTA was even bigger in SportsMOT (making IoU variants advantage bigger) and in MOT17 the difference becomes smaller, where BIoU gives even a positive performance always. This makes sense, a better detection makes the Kalman Filter estimate a better track location and then associating using additional information other than the intersection and union will give better matches

---

## API Reference

### BaseIoU

::: trackers.utils.iou.BaseIoU

### IoU

::: trackers.utils.iou.IoU

### GIoU

::: trackers.utils.iou.GIoU

### DIoU

::: trackers.utils.iou.DIoU

### CIoU

::: trackers.utils.iou.CIoU

### BIoU

::: trackers.utils.iou.BIoU
