---
description: ReID model loading, evaluation, and tracker utilities in Roboflow Trackers.
---

# ReID API

Requires the optional extra:

```bash
pip install 'trackers[reid]'
```

This page covers ReID model loading, gallery evaluation, and shared tracker
utilities. Tracker association is documented with BoT-SORT.

## Ways to load a model

Use `ReIDModel.from_pretrained(...)`. Pick the form that matches what you have:

1. **Default / curated alias** — `ReIDModel.from_pretrained()` or
    `ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")`.
    Use when you want a known library recipe. The alias resolves through the
    registry to architecture, weights URL, and preprocessing.

2. **Saved directory or HF repo with `reid_config.json`** —
    `ReIDModel.from_pretrained("/path/to/export")` or
    `ReIDModel.from_pretrained("hf://org/repo")`.
    Use after `save_pretrained` (or an equivalent Hub upload). The config names
    the architecture and preprocessing; weights come from
    `weights.safetensors` next to the config.

3. **Bare checkpoint file** —
    `ReIDModel.from_pretrained("weights.pth", architecture="osnet_x1_0")`
    (also works for `hf://.../file.pth` and `gd://...`).
    Use when you only have weights and already know the backbone. Architecture
    is required; preprocessing defaults from the architecture unless you pass
    `preprocessing=`.

4. **Architecture only (random init)** —
    `ReIDModel.from_pretrained(architecture="osnet_x1_0")` with no weights
    source.
    Use for tests, scaffolding, or before you train / attach a checkpoint.

## Adding a new architecture

1. Implement the module under `architectures/` (or use `timm:<name>`).
2. Register it in `architectures/__init__.py` via `build_architecture` /
    `list_architectures`.
3. If bare `.pth` loads need non-default crop behaviour, add an entry in
    `ARCHITECTURE_DEFAULT_PREPROCESSING` in `models/registry.py`.
4. Optionally add a curated alias in `ALIASES`.

## Encoder

`ReIDEncoder` is the appearance-encoder interface: `extract_features` for
tracker association and `extract_features_from_paths` for gallery evaluation.
`ReIDModel` is the concrete encoder we ship; it also handles loading, saving,
and preprocessing.

::: trackers.core.reid.encoder.ReIDEncoder

## Model

::: trackers.core.reid.model.ReIDModel

## Registry and preprocessing

::: trackers.core.reid.models.registry.DEFAULT_MODEL

::: trackers.core.reid.models.registry.ModelCard

::: trackers.core.reid.models.registry.resolve_model_card

::: trackers.core.reid.models.preprocessing.ReIDPreprocessing

## Evaluation

::: trackers.core.reid.eval.evaluator.ReIDEvaluator

::: trackers.core.reid.eval.metrics.ReIDMetrics

::: trackers.core.reid.eval.metrics.compute_reid_metrics

## Datasets

::: trackers.core.reid.eval.datasets.load_market1501

::: trackers.core.reid.eval.datasets.load_msmt17

## Feature bank

::: trackers.core.reid.feature_bank.FeatureBank

## Appearance

::: trackers.core.reid.appearance.appearance_similarity
