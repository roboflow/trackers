---
description: ReID model loading, evaluation, and appearance helpers in Roboflow Trackers.
---

# ReID API

Requires the optional extra:

```bash
pip install 'trackers[reid]'
```

This page covers the standalone ReID stack (model loading, gallery evaluation,
and appearance helpers). Tracker association wiring is documented with BoT-SORT.

## Encoders

:class:`~trackers.core.reid.eval.evaluator.ReIDEvaluator` and custom encoders
can use these lightweight interfaces instead of the full
:class:`~trackers.core.reid.model.ReIDModel` stack:

::: trackers.core.reid.encoder.ReIDEncoder

::: trackers.core.reid.encoder.ReIDPathEncoder

## Model

::: trackers.core.reid.model.ReIDModel

## Evaluation

::: trackers.core.reid.eval.evaluator.ReIDEvaluator

::: trackers.core.reid.eval.metrics.ReIDMetrics

::: trackers.core.reid.eval.metrics.compute_reid_metrics

## Datasets

::: trackers.core.reid.eval.datasets.load_market1501

::: trackers.core.reid.eval.datasets.load_msmt17

## Feature bank

::: trackers.core.reid.feature_bank.FeatureBank
