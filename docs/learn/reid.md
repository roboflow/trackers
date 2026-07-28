---
description: Use ReID appearance association with BoT-SORT in Roboflow Trackers, from model loading to appearance threshold selection, with MOT17 and SoccerNet results.
---

# ReID Appearance

BoT-SORT can fuse appearance embeddings with IoU during association. Embeddings
come from a model in the standalone [`reid`](https://github.com/roboflow/re-ID)
package; the association helpers are in `trackers.core.reid` (see the
[ReID API](../api/reid.md)).

**What you'll learn:**

- How to enable appearance association on BoT-SORT
- Which parameters control the appearance gate
- How to pick `appearance_threshold` for your encoder and domain
- What ReID changes on MOT17 and SoccerNet

---

## Install

The `reid` extra installs PyTorch, timm, Hugging Face Hub, safetensors, Pillow,
and gdown for ReID model loading.

=== "pip"
    ```bash
    pip install "trackers[reid]"
    ```

=== "uv"
    ```bash
    uv pip install "trackers[reid]"
    ```

---

## Quickstart

```python
from reid import ReIDModel

from trackers import BoTSORTTracker

reid_model = ReIDModel.from_pretrained("fastreid_mot17_sbs50")
tracker = BoTSORTTracker(reid_model=reid_model, appearance_threshold=0.2)
```

For the model catalog and fine-tuning, see the
[`reid` training guide](https://github.com/roboflow/re-ID/blob/main/docs/learn/train.md).

---

## Parameters

| Parameter              | Default | Purpose                                                                                              |
| ---------------------- | ------- | ---------------------------------------------------------------------------------------------------- |
| `reid_ema_alpha`       | 0.9     | EMA momentum for a track's appearance feature.                                                       |
| `appearance_threshold` | 0.25    | Max `d_app` for an appearance match (BoT-SORT paper default; the MOT17 setup below uses 0.2).         |
| `proximity_threshold`  | 0.5     | IoU gate before appearance (`IoU ≥ 1 - proximity_threshold`), from true IoU even with GIoU/DIoU/CIoU. |

---

## Choosing an appearance threshold

BoT-SORT rejects an appearance match when `d_app = 0.5 * (1 - cos_sim)` exceeds
`appearance_threshold` (paper default 0.25). Pick θ on a labeled split with the
encoder you will track with:

1. Embed GT crops.
2. Histogram `d_app` for same-ID vs different-ID pairs.
3. Choose θ so most same-ID pairs fall below it and most different-ID pairs fall
   above it.

**MOT17 val, `fastreid_mot17_sbs50`.** Same-ID peaks near 0, different-ID near
0.4. θ=0.2 keeps ~88% of same-ID pairs and ~1% of different-ID pairs
([MOT17 re-ID study](https://www-sop.inria.fr/members/Francois.Bremond/Postscript/Tomasz__SCCAI_2025.pdf)
Table 8; stricter than the paper default 0.25).

![FastReID MOT17 SBS on MOT17 val GT](../assets/reid/mot17-fastreid-appearance-distances.png)

**SoccerNet test, `osnet_x1_0_msmt17_combineall`.** Same-ID and different-ID
overlap heavily (similar kits). θ=0.2 passes ~50% of different-ID pairs and
stays flat vs CMC-only; θ=0.1 rejects too many same-ID pairs and costs HOTA and
IDF1 (see the SoccerNet table below).

![OSNet MSMT17 on SoccerNet test GT](../assets/reid/soccernet-osnet-appearance-distances.png)

---

## BoT-SORT with and without ReID

### MOT17 test

YOLOX detections, CMC on, Codabench MOT17 test (same protocol as the
[tracker comparison](../trackers/comparison.md) default table). ReID:
`fastreid_mot17_sbs50`, `appearance_threshold=0.2`
([MOT17 re-ID study](https://www-sop.inria.fr/members/Francois.Bremond/Postscript/Tomasz__SCCAI_2025.pdf)
Table 8).

| Config          |  HOTA  |  IDF1  |  MOTA  |
| :-------------- | :----: | :----: | :----: |
| BoT-SORT        |  63.7  |  78.7  | **79.2** |
| BoT-SORT + ReID | **63.9** | **79.2** | **79.2** |

### MOT17 val-half

YOLOX detections, CMC on, MOT17 val-half split, same encoder and threshold.

| Config          |  HOTA  |  MOTA  |  IDF1  |
| :-------------- | :----: | :----: | :----: |
| BoT-SORT        |  68.9  |  78.3  |  81.2  |
| BoT-SORT + ReID | **69.1** | **78.4** | **81.9** |

### SoccerNet test (OSNet MSMT17)

Oracle detections, CMC on, SoccerNet-tracking test (same protocol as the
[tracker comparison](../trackers/comparison.md) default table). ReID:
`osnet_x1_0_msmt17_combineall` (MSMT17 pretrained).

| Config                              |  HOTA  |  IDF1  |  MOTA  |
| :---------------------------------- | :----: | :----: | :----: |
| BoT-SORT                            |  84.5  |  79.3  | **96.6** |
| BoT-SORT + OSNet MSMT17 (θ=0.2)     | **84.6** | **79.4** | **96.6** |
| BoT-SORT + OSNet MSMT17 (θ=0.1)     |  82.9  |  77.7  |  96.5  |
