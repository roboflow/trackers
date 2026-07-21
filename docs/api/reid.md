---
description: Appearance-ReID association utilities in Roboflow Trackers.
---

# ReID API

Appearance-based re-identification (ReID) lets BoT-SORT match tracks across
frames using visual appearance in addition to motion. It requires the optional
extra:

```bash
pip install 'trackers[reid]'
```

Trackers ships only the numpy-only association glue documented below. The
appearance encoder, pretrained weights, preprocessing, model catalog, and
gallery evaluation live in the standalone [`reid`](https://github.com/roboflow/re-ID)
package (`roboflow-reid`), which the `trackers[reid]` extra installs for you.

## Loading a model

Import the encoder from `reid` and pass it to BoT-SORT:

```python
from reid import ReIDModel

from trackers import BoTSORTTracker

reid_model = ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall", device="cpu")
tracker = BoTSORTTracker(reid_model=reid_model)
```

See the [`reid` package documentation](https://github.com/roboflow/re-ID) for
the full model catalog, `from_pretrained` sources (curated aliases, `hf://`
repos, local checkpoints, architecture-only init), gallery evaluation
(`ReIDEvaluator`, `load_market1501`, `load_msmt17`), and how to add
architectures.

## Encoder protocol

`ReIDEncoder` is the minimal interface BoT-SORT depends on: a single
`extract_features` method. `reid.ReIDModel` satisfies it, and you can implement
it yourself for a custom encoder without depending on the model stack.

::: trackers.core.reid.encoder.ReIDEncoder

## Feature bank

::: trackers.core.reid.feature_bank.FeatureBank

## Appearance similarity

::: trackers.core.reid.appearance.appearance_similarity

::: trackers.core.reid.appearance.extract_detection_embeddings
