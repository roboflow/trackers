---
description: Python API reference for the ReID encoder protocol, feature bank, appearance association utilities, and threshold-selection plots in Roboflow Trackers.
---

# ReID API

Requires the `reid` extra (`pip install "trackers[reid]"`, see the [install guide](../guides/install.md)).

This page covers the `ReIDEncoder` protocol, `FeatureBank`, appearance association helpers, and the threshold-selection plots in `trackers.core.reid`. For enabling appearance on BoT-SORT and for benchmark results, see the [ReID appearance guide](../guides/reid.md). Model loading and gallery evaluation are in the standalone [`reid`](https://github.com/roboflow/re-ID) package.

## ReIDEncoder

::: trackers.core.reid.encoder.ReIDEncoder

## FeatureBank

::: trackers.core.reid.feature_bank.FeatureBank

## appearance_similarity

::: trackers.core.reid.appearance.appearance_similarity

## extract_detection_embeddings

::: trackers.core.reid.appearance.extract_detection_embeddings

## Choosing a threshold

Measure your own encoder on your own footage instead of inheriting a threshold from a paper. These helpers embed a labeled dataset, sample the distances a tracker actually sees, plot them, and report separability. Plotting needs `matplotlib`, which ships with the `reid` extra.

Both plot functions take their reference lines as `ThresholdLines`, either a sequence of values or a mapping from value to annotation.

```python
from trackers.core.reid import (
    extract_ground_truth_embeddings,
    plot_appearance_distances,
    plot_frame_gap_sweep,
    sample_appearance_distances,
    sweep_frame_gap,
)

embeddings, ids, frame_ids, sequence_ids = extract_ground_truth_embeddings(model, "mot17/val", keep_classes=(1,))
distances = sample_appearance_distances(embeddings, ids, frame_ids, sequence_ids)
same_id_rate, different_id_rate = distances.rates_at(0.25)
plot_appearance_distances(distances, thresholds={0.20: "selected", 0.25: "default"})
plot_frame_gap_sweep(sweep_frame_gap(embeddings, ids, frame_ids, sequence_ids))
```

### extract_ground_truth_embeddings

::: trackers.core.reid.appearance.extract_ground_truth_embeddings

### AppearanceDistances

::: trackers.core.reid.thresholds.AppearanceDistances

### sample_appearance_distances

::: trackers.core.reid.thresholds.sample_appearance_distances

### sweep_frame_gap

::: trackers.core.reid.thresholds.sweep_frame_gap

### roc_auc

::: trackers.core.reid.thresholds.roc_auc

### plot_appearance_distances

::: trackers.core.reid.thresholds.plot_appearance_distances

### plot_frame_gap_sweep

::: trackers.core.reid.thresholds.plot_frame_gap_sweep
