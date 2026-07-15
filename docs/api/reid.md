---
description: ReID model loading, evaluation, and appearance helpers in Roboflow Trackers.
---

# ReID API

Requires the optional extra:

```bash
pip install 'trackers[reid]'
```

## Protocols

Trackers and :class:`~trackers.core.reid.eval.evaluator.ReIDEvaluator` accept lightweight
protocol types so tests and custom encoders do not need the full
:class:`~trackers.core.reid.model.ReIDModel` stack:

::: trackers.core.reid.protocols.ReIDEncoder

::: trackers.core.reid.protocols.ReIDPathEncoder

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
