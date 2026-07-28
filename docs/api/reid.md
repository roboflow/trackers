---
description: ReID encoder protocol, feature bank, and appearance association utilities in Roboflow Trackers.
---

# ReID API

Requires the optional extra:

```bash
pip install 'trackers[reid]'
```

This page covers the `ReIDEncoder` protocol, `FeatureBank`, and appearance
association helpers in `trackers.core.reid`. ReID model loading, gallery
evaluation, and MOT fine-tuning are documented in the standalone
[`reid`](https://github.com/roboflow/re-ID) package. BoT-SORT usage is on the
[BoT-SORT](../trackers/botsort.md) page.

## BoT-SORT with and without ReID

### MOT17 test (Codabench)

Same YOLOX detections and Codabench MOT17 test protocol as the
[tracker comparison](../trackers/comparison.md) default table. BoT-SORT runs
with CMC enabled. The ReID row uses `fastreid_mot17_sbs50` and
`appearance_threshold=0.2` (MOT17 re-ID study Table 8).

| Config          |  HOTA  |  IDF1  |  MOTA  |
| :-------------- | :----: | :----: | :----: |
| BoT-SORT        |  63.7  |  78.7  | **79.2** |
| BoT-SORT + ReID | **63.9** | **79.2** | **79.2** |

### MOT17 val-half

Scores from
[`eval_trackers_reid.ipynb`](../../notebooks/eval_trackers_reid.ipynb) on MOT17
val-half with YOLOX detections, CMC on, `fastreid_mot17_sbs50`, and
`appearance_threshold=0.2`.

| Config          |  HOTA  |  MOTA  |  IDF1  |
| :-------------- | :----: | :----: | :----: |
| BoT-SORT        |  68.9  |  78.3  |  81.2  |
| BoT-SORT + ReID | **69.1** | **78.4** | **81.9** |

For published reference points on the same split, see the MOT17 re-ID study
([*Does Re-ID Really Help in Multi-Object Tracking?*](https://www-sop.inria.fr/members/Francois.Bremond/Postscript/Tomasz__SCCAI_2025.pdf),
Table 8 + Table 13 combined row, app th=0.2): HOTA 68.43 / IDF1 80.92 without
ReID and 68.95 / 81.98 with MOT17 FastReID (MOTA not reported for that YOLOX
setup), and the [BoT-SORT paper](https://arxiv.org/abs/2206.14651) Table 1.

## Use with BoT-SORT

Import the encoder from `reid` and pass any object that implements
`ReIDEncoder` to `BoTSORTTracker`. For MOT17-style replication, load the
official FastReID SBS weights and match the study appearance threshold:

```python
from reid import ReIDModel

from trackers import BoTSORTTracker

reid_model = ReIDModel.from_pretrained("fastreid_mot17_sbs50")
tracker = BoTSORTTracker(reid_model=reid_model, appearance_threshold=0.2)
```

| Parameter              | Default | Purpose                                                                                                                                                                           |
| ---------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reid_ema_alpha`       | 0.9     | EMA momentum for a track's appearance feature; higher retains more history.                                                                                                       |
| `appearance_threshold` | 0.25    | Appearance-distance gate (BoT-SORT paper default). Rejects matches when `0.5 * (1 - cos_sim)` exceeds this value. MOT17 Codabench / eval uses `0.2` per the re-ID study Table 8. |
| `proximity_threshold`  | 0.5     | Standard-IoU gate applied before appearance is used (requires IoU ≥ `1 - proximity_threshold`), computed from true IoU even when `iou` is GIoU/DIoU/CIoU.                         |

See the [`reid` training guide](https://github.com/roboflow/re-ID/blob/main/docs/learn/train.md)
for crop generation, `train_reid`, and Colab tips; the package README covers the
model catalog, `from_pretrained` sources, and gallery evaluation.

## Encoder protocol

`ReIDEncoder` is the interface BoT-SORT expects: a single `extract_features`
method. `reid.ReIDModel` satisfies it; custom encoders can implement the
protocol without the model stack.

::: trackers.core.reid.encoder.ReIDEncoder

## Feature bank

Per-track EMA of appearance embeddings. L2-normalize before and after the EMA
blend, following BoT-SORT
[`STrack.update_features`](https://github.com/NirAharon/BoT-SORT/blob/main/tracker/bot_sort.py).

::: trackers.core.reid.feature_bank.FeatureBank

## Appearance

::: trackers.core.reid.appearance.appearance_similarity

::: trackers.core.reid.appearance.extract_detection_embeddings
