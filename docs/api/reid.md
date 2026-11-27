---
description: Python API reference for ReID encoders, appearance association, long-term identity joining, model preparation, and threshold-selection tools in Roboflow Trackers.
---

# ReID API

The encoder protocol, feature bank, and NumPy helpers are included in the base package. For model loading and plotting, install the `reid` extra (`pip install "trackers[reid]"`; see the [install guide](../guides/install.md)).

This page covers `ReIDConfig`, the `ReIDEncoder` protocol, `FeatureBank`, appearance association helpers, and the threshold-selection plots in `trackers.reid`. It also covers `LongTermReIDConfig`, `LongTermReID`, and `prepare_sports_reid_model` for optional long-term identity joining. For enabling appearance on BoT-SORT and for benchmark results, see the [ReID appearance guide](../guides/reid.md). For joining and historical output in McByte++, see the [McByte++ guide](../trackers/mcbyteplusplus.md#optional-long-term-re-identification). Model loading and gallery evaluation are in the standalone [`reid`](https://github.com/roboflow/re-ID) package.

## ReIDConfig

::: trackers.reid.config.ReIDConfig

## ReIDEncoder

::: trackers.reid.encoder.ReIDEncoder

## LongTermReIDConfig

Settings for optional trajectory joining, distinct from BoT-SORT's association configuration. Subject-occlusion filtering of new re-ID features, pausing the affected pending identity decision deadline and historical buffering default to enabled. McByte++ separately enables the camera-motion-triggered re-ID pause; the reusable joining component does not interpret CMC estimates. See [McByte++ long-term tracking](../trackers/mcbyteplusplus.md#optional-long-term-re-identification) for output buffering and examples.

::: trackers.reid.long_term_config.LongTermReIDConfig

## LongTermReID

::: trackers.reid.long_term.LongTermReID

## prepare_sports_reid_model

::: trackers.reid.sports_model.prepare_sports_reid_model

## FeatureBank

::: trackers.reid.feature_bank.FeatureBank

## appearance_similarity

::: trackers.reid.appearance.appearance_similarity

## extract_detection_embeddings

::: trackers.reid.appearance.extract_detection_embeddings

## Choosing a threshold

Measure your own encoder on your own footage instead of inheriting a threshold from a paper. These helpers embed a labeled dataset, sample the distances a tracker actually sees, plot them, and report separability.

Both plot functions take their reference lines as `ThresholdLines`, either a sequence of values or a mapping from value to annotation.

```python
from trackers.reid import (
    extract_ground_truth_embeddings,
    plot_appearance_distances,
    plot_frame_gap_sweep,
    sample_appearance_distances,
    sweep_frame_gap,
)

embeddings, ids, frame_ids, sequence_ids = extract_ground_truth_embeddings(
    "fastreid_mot17_sbs50", "mot17/val", keep_classes=(1,)
)
distances = sample_appearance_distances(embeddings, ids, frame_ids, sequence_ids)
same_id_rate, different_id_rate = distances.rates_at(0.25)
plot_appearance_distances(distances, thresholds={0.20: "selected", 0.25: "default"})
plot_frame_gap_sweep(sweep_frame_gap(embeddings, ids, frame_ids, sequence_ids))
```

### extract_ground_truth_embeddings

::: trackers.reid.appearance.extract_ground_truth_embeddings

### AppearanceDistances

::: trackers.reid.threshold_selection.AppearanceDistances

### sample_appearance_distances

::: trackers.reid.threshold_selection.sample_appearance_distances

### sweep_frame_gap

::: trackers.reid.threshold_selection.sweep_frame_gap

### roc_auc

::: trackers.reid.threshold_selection.roc_auc

### plot_appearance_distances

::: trackers.reid.plots.plot_appearance_distances

### plot_frame_gap_sweep

::: trackers.reid.plots.plot_frame_gap_sweep
