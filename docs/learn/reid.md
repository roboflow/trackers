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
    are same-ID; negatives are different-ID that could co-compete.
3. Choose θ so most same-ID pairs fall below it and most different-ID pairs fall
    above it.

**MOT17 val, `fastreid_mot17_sbs50`.** Same-ID distances peak near 0 and
different-ID near 0.4. On association-local GT crop pairs (5000 same-ID, 10000
different-ID), θ=0.2 keeps 77% of same-ID pairs while passing 1% of different-ID
pairs, which is why it beats the BoT-SORT default 0.25 here
([MOT17 re-ID study](https://www-sop.inria.fr/members/Francois.Bremond/Postscript/Tomasz__SCCAI_2025.pdf)
Table 8 uses the same threshold).

![FastReID MOT17 SBS on MOT17 val GT](../assets/reid/mot17-fastreid-appearance-distances.png)

**SoccerNet test, `osnet_x1_0_msmt17_combineall`.** Same-ID and different-ID
distances overlap heavily (similar kits). On association-local GT crop pairs
(5000 same-ID, 10000 different-ID), θ=0.2 admits 97% of same-ID pairs but also
52% of different-ID pairs, and tracking stays flat against CMC-only. θ=0.1 holds
different-ID pairs to 6%, yet appearance still assists a mix of correct and
same-kit pairs and costs HOTA and IDF1 (see the SoccerNet table below).

![OSNet MSMT17 on SoccerNet test GT](../assets/reid/soccernet-osnet-appearance-distances.png)

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
