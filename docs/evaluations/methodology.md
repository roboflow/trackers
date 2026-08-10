---
title: Benchmark Methodology | Trackers
description: How Trackers' benchmark results are produced — detection sources, tuning procedure, and train/validation/test split usage.
---

# Methodology

### Detections

Each dataset uses one of two detection sources: oracle detections (ground-truth bounding boxes provided by the dataset) or model detections (produced by a YOLOX detector following the ByteTrack procedure). The source is noted per dataset on the [Results](results.md) page.

### Tuning

Best parameters per tracker and dataset were found via grid search (SORT, ByteTrack, OC-SORT, BoT-SORT) or Optuna (`n_trials=100`, objective HOTA, trial 0 = defaults for C-BIoU), selecting the configuration with the highest HOTA on the tune split. McByte is not tuned here: defaults with mask-conditioned association enabled are reported, matching the [McByte](../trackers/mcbyte.md) page (source: [PR #513](https://github.com/roboflow/trackers/pull/513)). Tuning and evaluation always use separate data splits to reflect real-world usage:

- Train + validation + test: tune on validation, report on test.
- Train + validation: tune on train, report on validation.
- Train + test: tune on train, report on test.
