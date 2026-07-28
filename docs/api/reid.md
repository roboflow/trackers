---
description: Python API reference for the ReID encoder protocol, feature bank, and appearance association utilities in Roboflow Trackers.
---

# ReID API

Requires the optional extra:

```bash
pip install 'trackers[reid]'
```

This page covers the `ReIDEncoder` protocol, `FeatureBank`, and appearance
association helpers in `trackers.core.reid`. For enabling appearance on
BoT-SORT, threshold selection, and benchmark results, see the
[ReID appearance guide](../learn/reid.md). Model loading and gallery evaluation
are in the standalone [`reid`](https://github.com/roboflow/re-ID) package.

## ReIDEncoder

::: trackers.core.reid.encoder.ReIDEncoder

## FeatureBank

::: trackers.core.reid.feature_bank.FeatureBank

## appearance_similarity

::: trackers.core.reid.appearance.appearance_similarity

## extract_detection_embeddings

::: trackers.core.reid.appearance.extract_detection_embeddings
