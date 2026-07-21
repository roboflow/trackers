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

Reference scores on MOT17 val-half with YOLOX detections from the
[BoT-SORT paper](https://arxiv.org/abs/2206.14651) Table 1. The
[`eval_trackers_reid.ipynb`](../../notebooks/eval_trackers_reid.ipynb) notebook
uses the same split and detector with `fastreid_mot17_sbs50` and
`appearance_threshold=0.2` (MOT17 re-ID study Table 8); run it for
trackers-local results.

| Config | HOTA | MOTA | IDF1 |
| :----- | :--: | :--: | :--: |
| BoT-SORT | 69.11 | 78.39 | 81.53 |
| BoT-SORT + ReID | 69.17 | 78.46 | 82.07 |

The MOT17 re-ID study
([*Does Re-ID Really Help in Multi-Object Tracking?*](https://www-sop.inria.fr/members/Francois.Bremond/Postscript/Tomasz__SCCAI_2025.pdf),
Table 8 + Table 13 combined row, app th=0.2) reports HOTA 68.43 / IDF1 80.92
without ReID and 68.95 / 81.98 with MOT17 FastReID; MOTA is not reported for
that YOLOX setup.

## Use with BoT-SORT

Import the encoder from `reid` and pass any object that implements
`ReIDEncoder` to `BoTSORTTracker`:

```python
from reid import ReIDModel

from trackers import BoTSORTTracker

reid_model = ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall", device="cpu")
tracker = BoTSORTTracker(reid_model=reid_model)
```

See the [`reid` package documentation](https://github.com/roboflow/re-ID) for
the model catalog, `from_pretrained` sources, gallery evaluation, and MOT
fine-tuning.

## Encoder protocol

`ReIDEncoder` is the interface BoT-SORT expects: a single `extract_features`
method. `reid.ReIDModel` satisfies it; custom encoders can implement the
protocol without the model stack.

::: trackers.core.reid.encoder.ReIDEncoder

## Feature bank

::: trackers.core.reid.feature_bank.FeatureBank

## Appearance

::: trackers.core.reid.appearance.appearance_similarity

::: trackers.core.reid.appearance.extract_detection_embeddings
