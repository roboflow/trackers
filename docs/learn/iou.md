---
description: Learn IoU, GIoU, DIoU, CIoU, and BIoU for object tracking association, with practical guidance on choosing metrics and thresholds.
---

# IoU API

IoU variants are pluggable similarity metrics used during detection-to-track
association. You pass one of these classes to a tracker via the `iou` argument.

**What you'll learn:**

- What each IoU variant measures and when to use it
- How score ranges affect `minimum_iou_threshold` tuning
- How to configure IoU variants in any tracker
- How IoU variants perform across common tracking benchmarks

---

## Install

Get started by installing `trackers`.

```text
pip install trackers
```

For more options, see the [install guide](install.md).

---

## Quickstart

```python
from trackers import SORTTracker
from trackers.utils.iou import IoU

tracker = SORTTracker(
    iou=IoU(),
    minimum_iou_threshold=0.3,
)
```

## Choosing a Metric

| Variant | Score range | When to use                                                            |
| :------ | :---------- | :--------------------------------------------------------------------- |
| `IoU`   | `[0, 1]`    | Default — strong baseline for most scenes                              |
| `GIoU`  | `[-1, 1]`   | Scenes where boxes frequently lose overlap (occlusion, re-entry)       |
| `DIoU`  | `[-1, 1]`   | Fast-moving objects; centre-distance signal without aspect sensitivity |
| `CIoU`  | `[-1, 1]`   | Same as DIoU plus aspect-ratio consistency                             |
| `BIoU`  | `[0, 1]`    | Very small or very fast objects where raw boxes rarely overlap         |

**Formula Summary** (`A, B` boxes, `C` enclosing box, `d` center distance, `c` enclosing diagonal):

- \( \mathrm{GIoU} = \mathrm{IoU} - \frac{|C \setminus (A \cup B)|}{|C|} \)
- \( \mathrm{DIoU} = \mathrm{IoU} - \frac{d^2}{c^2 + \epsilon} \)
- \( \mathrm{CIoU} = \mathrm{DIoU} - \alpha v \), where
    \( v = \frac{4}{\pi^2}\left(\arctan\frac{w_A}{h_A} - \arctan\frac{w_B}{h_B}\right)^2 \)
    and \( \alpha = \frac{v}{1 - \mathrm{IoU} + v + \epsilon} \)

## IoU

**Standard Intersection over Union** — the classic baseline.

\[
\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
\]

<figure class="iou-variant-figure">
  <img src="../../assets/IoU%20variants/IoU%20visualization.png" alt="IoU visualization" loading="lazy" decoding="async"/>
</figure>
Scores are `0` (no overlap) to `1` (perfect overlap). Because it returns `0` whenever
boxes do not intersect, the tracker gets no gradient to recover a lost track; a
variant from the list below can help in those cases.

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

Set `minimum_iou_threshold` based on the score range of your chosen metric.

- `IoU` and `BIoU` usually work with non-negative thresholds (for example, `0.2` to `0.5`).
- `GIoU`, `DIoU`, and `CIoU` can produce negative scores, so negative thresholds are valid.
- Tune thresholds per dataset and tracker; there is no universal best value.

## GIoU

**Generalised IoU** ([Rezatofighi et al., 2019](https://arxiv.org/abs/1902.09630)) — penalises the gap inside the
smallest enclosing box `C` that neither `A` nor `B` fills.

\[
\mathrm{GIoU}(A, B) = \mathrm{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}
\]

<figure class="iou-variant-figure">
  <img src="../../assets/IoU%20variants/GIoU%20visualization.png" alt="GIoU visualization" loading="lazy" decoding="async"/>
</figure>

When boxes do not overlap at all, IoU is flat at `0`, but the penalty term still
changes as boxes move closer or farther apart, giving the tracker a meaningful
signal based on distances, sizes and shapes.

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

Left: IoU. Right: GIoU. Camera movements can introduce unexpected displacement,
producing ID switches with IoU-based association. GIoU still provides a signal when
there is no overlap by considering enclosing-box geometry, which helps preserve
tracks that IoU would otherwise confuse or lose due to direction changes and
non-linear motion (for example, tracks `5`, `12` on the left vs `13` on the right).

<video width="100%" controls muted loop>
  <source src="https://github.com/user-attachments/assets/dd38120d-ebbe-4705-8140-fcf24bc8ce99" type="video/mp4">
</video>

---

## DIoU

**Distance IoU** ([Zheng et al., 2019](https://arxiv.org/abs/1911.08287)) — adds a centre-distance penalty to IoU,
normalised by the enclosing box diagonal.

\[
\mathrm{DIoU}(A, B) = \mathrm{IoU} - \frac{d^2}{c^2 + \epsilon}
\]

<figure class="iou-variant-figure">
  <img src="../../assets/IoU%20variants/DIoU%20visualization.png" alt="DIoU visualization" loading="lazy" decoding="async"/>
</figure>

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

Left: IoU. Right: DIoU. Highly non-linear motion can make IoU drop to zero,
causing the Kalman prediction to attach to another object and produce an ID switch.
The centre-distance term keeps the score smoother and preserves IDs more often
(for example, tracks `3–5`).

<video width="100%" controls muted loop>
  <source src="https://github.com/user-attachments/assets/011f6cfa-a2be-4109-8326-a98bcae4ed93" type="video/mp4">
</video>

---

## CIoU

**Complete IoU** ([Zheng et al., 2019](https://arxiv.org/abs/1911.08287)) — extends DIoU with a penalty for aspect-ratio
mismatch between the two boxes.

\[
\mathrm{CIoU}(A, B) = \mathrm{DIoU} - \alpha v
\]

\[
v = \frac{4}{\pi^2}\!\left(\arctan\frac{w_A}{h_A} - \arctan\frac{w_B}{h_B}\right)^2, \quad
\alpha = \frac{v}{1 - \mathrm{IoU} + v + \epsilon}
\]

<figure class="iou-variant-figure">
  <img src="../../assets/IoU%20variants/CIoU%20visualization.png" alt="CIoU visualization" loading="lazy" decoding="async"/>
</figure>

`v` measures aspect-ratio divergence; `α` scales it so the penalty is low when IoU
is already high. On tracking benchmarks CIoU and DIoU behave similarly.

```python
from trackers import OCSORTTracker
from trackers.utils.iou import CIoU

tracker = OCSORTTracker(iou=CIoU(), minimum_iou_threshold=-0.3)
```

**Example — SoccerNet `SNMOT-122`**

|           | HOTA (%) |   Δ (pts) |
| :-------- | -------: | --------: |
| Best IoU  |    77.36 |         — |
| Best CIoU |    85.58 | **+8.22** |

Left: IoU. Right: CIoU. In this example, CIoU is capable of perfectly keeping the track of the ball, which is explained by the fact that the ball is a small and fast moving object, with roughly constant aspect ratio, where CIoU’s distance + aspect terms help more than overlap alone.

<video width="100%" controls muted loop>
  <source src="https://github.com/user-attachments/assets/48cb3d28-7cbf-4551-96da-4a8f9b43306c" type="video/mp4">
</video>

---

## BIoU

**Buffered IoU** ([Yang et al., 2022](https://arxiv.org/abs/2211.14317)) — expands each box by a relative margin `r`
before computing standard IoU. Let `w = x2 − x1`, `h = y2 − y1`:

\[
A^r = (x_1 - rw,\; y_1 - rh,\; x_2 + rw,\; y_2 + rh)
\]

\[
\mathrm{BIoU}_r(A, B) = \mathrm{IoU}(A^r, B^r)
\]

<figure class="iou-variant-figure">
  <img src="../../assets/IoU%20variants/BIoU%20visualization.png" alt="BIoU visualization" loading="lazy" decoding="async"/>
</figure>

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
that gap and keeps the same ID. (e.g. tracks 7 and 8).

<video width="100%" controls muted loop>
  <source src="https://github.com/user-attachments/assets/9a74a27b-0470-4cd8-b545-0507a0d2b053" type="video/mp4">
</video>

---

## IoU Variant Performance Across Benchmarks

We evaluate how much each variant changes performance across datasets.
For each `(dataset, tracker)` pair, we keep the `state_estimator` with the highest **IoU HOTA** on the evaluation split, then report mean
`ΔHOTA = HOTA(variant) − HOTA(IoU)` over trackers (same split; thresholds tuned per experiment).

For more information on the datasets, see: [dataset comparison](../trackers/comparison.md).

<style>
  .iou-variant-figure {
    margin: 0.75rem auto 1.1rem;
    max-width: min(22rem, min(100%, 92vw));
    width: 100%;
    text-align: center;
  }
  .iou-variant-figure img {
    width: 100%;
    max-width: min(22rem, 92vw);
    height: auto;
    display: block;
    margin-inline: auto;
  }
</style>

| Dataset        | IoU mean HOTA | GIoU mean Δ | DIoU mean Δ | CIoU mean Δ | BIoU mean Δ |
| :------------- | ------------: | ----------: | ----------: | ----------: | ----------: |
| MOT17 val      |         38.09 |   **−0.09** |   **−0.04** |   **−0.04** |   **−0.28** |
| SportsMOT val  |         80.21 |   **+0.65** |   **+0.95** |   **+0.88** |   **+0.36** |
| DanceTrack val |         50.27 |   **−0.80** |   **−0.34** |   **+0.05** |   **+0.15** |
| SoccerNet test |         83.21 |   **+1.57** |   **+2.82** |   **+2.76** |   **+1.41** |

Over SportsMOT and SoccerNet, all IoU variants outperform standard IoU, with DIoU
and CIoU strongest on SoccerNet and DIoU slightly ahead of CIoU on SportsMOT. In
MOT17, standard IoU is best by a small margin (DIoU and CIoU are similar). On
DanceTrack, GIoU and DIoU underperform IoU, while CIoU and BIoU perform slightly better.

These experiments suggest IoU variants provide task-dependent gains, with larger
improvements on sports datasets. We hypothesize detection quality plays a major role:
SoccerNet uses perfect detections, and SportsMOT detections come from a strong
detector, and both show the largest improvements. To test this, we run an additional
experiment using ground-truth boxes from MOT17 and SportsMOT as tracker detections.

| Dataset (GT-as-det) | IoU mean HOTA | GIoU mean Δ | DIoU mean Δ | CIoU mean Δ | BIoU mean Δ |
| :------------------ | ------------: | ----------: | ----------: | ----------: | ----------: |
| MOT17 val           |         97.17 |   **−0.05** |   **−0.07** |   **−0.05** |   **+0.31** |
| SportsMOT val       |         87.18 |   **+0.47** |   **+1.09** |   **+1.06** |   **+0.46** |

With ground-truth detections, mean ΔHOTA increases for three of four variants on
SportsMOT compared to YOLOX detections. On MOT17, gaps narrow overall: GIoU moves
closer to IoU, DIoU and CIoU remain slightly below IoU, and BIoU becomes positive on
average. This is consistent with cleaner inputs: Kalman predictions align better
with detections, so richer association signals can help more.

---

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
