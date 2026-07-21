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
