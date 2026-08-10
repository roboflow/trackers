---
title: ReID Appearance — BoT-SORT Appearance Association | Trackers
description: Use ReID appearance association with BoT-SORT in Roboflow Trackers, from model loading to appearance threshold selection, with MOT17 and SoccerNet results.
---

# ReID Appearance

BoT-SORT can fuse appearance embeddings with IoU during association. Embeddings
come from a model in the standalone [`reid`](https://github.com/roboflow/re-ID)
package. See the [ReID API](../api/reid.md) for the association helpers.

**What you'll learn:**

- How to enable appearance association on BoT-SORT
- Which parameters control the appearance gate
- How to pick `appearance_threshold` for your encoder and domain
- What ReID changes on MOT17 and SoccerNet

---

## Install

```bash
pip install "trackers[reid]"
```

For extra contents and other options, see the [install guide](install.md).

---

## Quickstart

```python
from reid import ReIDModel

from trackers import BoTSORTTracker

reid_model = ReIDModel.from_pretrained("fastreid_mot17_sbs50")
tracker = BoTSORTTracker(reid_model=reid_model, appearance_threshold=0.2)
```

!!! warning "A frame is required when ReID is enabled"

    Pass the current video frame as `tracker.update(detections, frame=frame_bgr)`.
    When `reid_model` is set, `update()` raises if `frame` is omitted.

For the model catalog and fine-tuning, see the
[`reid` training guide](https://github.com/roboflow/re-ID/blob/main/docs/learn/train.md).

---

## Key Parameters

| Parameter              | Purpose                                                                                                       | Tuning guidance                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `reid_model`           | Appearance encoder queried during association.                                                                | Leave unset for IoU and CMC only. Pick a checkpoint trained on your object domain where possible.                |
| `reid_ema_alpha`       | EMA momentum for a track's appearance feature.                                                                | Default 0.9. Higher keeps a stable long-term identity; lower adapts faster to appearance change but drifts more. |
| `appearance_threshold` | Maximum appearance distance `d_app` for appearance to lower a pair's matching cost.                           | BoT-SORT paper default 0.25. Calibrate per encoder and domain, see below.                                        |
| `proximity_threshold`  | IoU gate applied before appearance (`IoU ≥ 1 - proximity_threshold`), from true IoU even with GIoU/DIoU/CIoU. | Default 0.5. Lower restricts how far apart a pair may be before appearance stops contributing.                   |

---

## Choosing an appearance threshold

BoT-SORT fuses costs as `min(d_iou, d_app)` with
`d_app = 0.5 * (1 - cos_sim)`, and discards the appearance term when `d_app`
exceeds `appearance_threshold` (paper default 0.25) or when the pair fails the
`proximity_threshold` IoU gate. Appearance can therefore only lower a pair's
cost, never veto a geometric match. Pick θ on a labeled split with the encoder
you will track with:

1. Embed GT crops.
2. Histogram `d_app` for association-local pairs: same video only, with
    frame gap bounded by the lost-track horizon (default 30 frames). Positives
    are same-ID; negatives are different-ID that could co-compete. Sample both
    classes with the same per-sequence quota, otherwise one crowded sequence
    decides the answer.
3. Choose θ so most same-ID pairs fall below it and most different-ID pairs fall
    above it.

Regenerate these figures with
[`notebooks/plot_reid_appearance_distances.py`](https://github.com/roboflow/trackers/blob/develop/notebooks/plot_reid_appearance_distances.py).

**MOT17 val, `fastreid_mot17_sbs50`.** Same-ID distances peak near 0 and
different-ID near 0.4. On association-local GT crop pairs (5000 same-ID, 10000
different-ID, frame gap 1 to 30), θ=0.2 keeps 68% of same-ID pairs while passing
1.1% of different-ID pairs. Raising θ to the BoT-SORT default 0.25 recovers
same-ID pairs (79%) but nearly triples the different-ID pairs it admits (2.9%),
which is why 0.2 is the better operating point here
([MOT17 re-ID study](https://www-sop.inria.fr/members/Francois.Bremond/Postscript/Tomasz__SCCAI_2025.pdf)
Table 8 uses the same threshold).

![FastReID MOT17 SBS on MOT17 val GT](../assets/reid/mot17-fastreid-appearance-distances.png)

**SoccerNet test, `osnet_x1_0_msmt17_combineall`.** A pedestrian encoder on soccer
footage squeezes every distance into a narrow range: same-ID pairs peak near 0.05
and different-ID pairs near 0.20 (similar kits). The two shapes still separate, but
the scale no longer matches the thresholds BoT-SORT was tuned with. On
association-local GT crop pairs (5000 same-ID, 10000 different-ID, frame gap 1 to
30), θ=0.2 admits 96% of same-ID pairs but also 49% of different-ID pairs, and
tracking stays flat against CMC-only. θ=0.1 holds different-ID pairs to 9%, yet
appearance still assists a mix of correct and same-kit pairs and costs HOTA and
IDF1 (see the SoccerNet table below). Calibrate θ on your own domain rather than
carrying 0.2 or 0.25 across.

![OSNet MSMT17 on SoccerNet test GT](../assets/reid/soccernet-osnet-appearance-distances.png)

---

## How far the threshold carries

A histogram fixes one frame gap, so it only describes re-association over that
horizon. Sweeping the gap shows how long a track can stay lost before appearance
stops helping to re-find it.

On MOT17 val, different-ID distances barely move with the gap: the median stays
near 0.41 and the 5th percentile near 0.27 from a 1-frame gap out to 240 frames.
Same-ID distances spread steadily, from a median of 0.04 at a 1-frame gap to 0.20
across the 16 to 30 band and 0.28 beyond 120 frames.

ROC AUC below is the chance that a random same-ID pair scores closer than a random
different-ID pair. 1.0 means the two distributions never overlap, 0.5 means
appearance carries no information at all. It is reported because it needs no
operating point, so the column does not depend on a target true-positive or
false-positive rate. The two rates beside it evaluate the thresholds Trackers
actually ships rather than deriving a new one.

| Frame gap  | ROC AUC | same-ID below 0.2 | different-ID below 0.2 |
| :--------- | :-----: | :---------------: | :--------------------: |
| 1          |  0.998  |       98.0%       |          1.7%          |
| 2 to 5     |  0.987  |       87.6%       |          1.5%          |
| 6 to 15    |  0.957  |       67.4%       |          1.1%          |
| 16 to 30   |  0.929  |       51.4%       |          1.1%          |
| 31 to 60   |  0.899  |       39.8%       |          0.9%          |
| 61 to 120  |  0.865  |       31.7%       |          0.8%          |
| 121 to 240 |  0.854  |       28.6%       |          0.8%          |

![FastReID MOT17 SBS separability vs frame gap](../assets/reid/mot17-fastreid-appearance-distances-vs-gap.png)

Two things follow. First, a threshold validated on adjacent frames says little
about re-association: at θ=0.2 appearance helps 98% of same-ID pairs one frame
apart but only 51% across the default 30-frame lost-track buffer. Second, the
price of a tight θ over long gaps is missed re-associations rather than extra
wrong ones, because the different-ID rate stays near 1% throughout. If you raise
`lost_track_buffer` to recover tracks after long occlusions, raise
`appearance_threshold` with it and re-check the different-ID column.

The cross-domain encoder fails differently. On SoccerNet the different-ID rate at
θ=0.2 is flat near 49% at every gap, so the frame gap is not what limits it; the
encoder simply cannot separate players in matching kits at any horizon. Widening
the gap costs same-ID pairs (99.6% down to 87.0%) without ever making the
different-ID side usable, which is why θ has to come down to about 0.1 on this
domain instead of being traded against the gap.

![OSNet MSMT17 separability vs frame gap](../assets/reid/soccernet-osnet-appearance-distances-vs-gap.png)

---

## BoT-SORT with and without ReID

### MOT17 test

YOLOX detections, CMC on, Codabench MOT17 test (same protocol as the
[tracker comparison](../trackers/comparison.md) default table). ReID:
`fastreid_mot17_sbs50`, `appearance_threshold=0.2`
([MOT17 re-ID study](https://www-sop.inria.fr/members/Francois.Bremond/Postscript/Tomasz__SCCAI_2025.pdf)
Table 8).

| Config          |   HOTA   |   IDF1   |   MOTA   |
| :-------------- | :------: | :------: | :------: |
| BoT-SORT        |   63.7   |   78.7   | **79.2** |
| BoT-SORT + ReID | **63.9** | **79.2** | **79.2** |

### MOT17 val-half

YOLOX detections, CMC on, MOT17 val-half split, same encoder and threshold,
scored with `trackers eval`.

| Config          |   HOTA   |   IDF1   |   MOTA   |
| :-------------- | :------: | :------: | :------: |
| BoT-SORT        |   68.9   |   81.2   |   78.3   |
| BoT-SORT + ReID | **69.1** | **81.9** | **78.4** |

The MOT17 re-ID study reports 68.43 HOTA / 80.92 IDF1 without ReID and
68.95 / 81.98 with, on the same split at `appearance_threshold=0.2`
(Table 8 and Table 13; MOTA is not reported for that YOLOX setup).

### SoccerNet test (OSNet MSMT17)

Oracle detections, CMC on, SoccerNet-tracking test (same protocol as the
[tracker comparison](../trackers/comparison.md) default table). ReID:
`osnet_x1_0_msmt17_combineall` (MSMT17 pretrained), so this is a cross-domain
encoder on soccer footage.

| Config                          |   HOTA   |   IDF1   |   MOTA   |
| :------------------------------ | :------: | :------: | :------: |
| BoT-SORT                        |   84.5   |   79.3   | **96.6** |
| BoT-SORT + OSNet MSMT17 (θ=0.2) | **84.6** | **79.4** | **96.6** |
| BoT-SORT + OSNet MSMT17 (θ=0.1) |   82.9   |   77.7   |   96.5   |
