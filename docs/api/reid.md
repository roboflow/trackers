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

## Package layout

| Area                                | Role                                                                                                           |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `architectures/`                    | Build a backbone (`osnet_*`, `timm:<name>`, or a raw `nn.Module`).                                             |
| `models/registry.py`                | Curated aliases and `reid_config.json` → `ModelCard` (architecture, weights, preprocessing, optional warning). |
| `models/loaders.py`                 | Fetch and load checkpoint bytes (`hf://`, `gd://`, local) into a module.                                       |
| `models/preprocessing.py`           | Crop resize / colour; optional embedding L2.                                                                   |
| `model.py`                          | Public facade: `ReIDModel.from_pretrained` / `save_pretrained` / `extract_features`.                           |
| `appearance.py` / `feature_bank.py` | Association helpers (cosine similarity, per-track EMA).                                                        |
| `eval/`                             | Gallery metrics and Market-1501 / MSMT17 loaders.                                                              |
| `encoder.py`                        | Lightweight protocols for custom encoders.                                                                     |

### Ways to load a model

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

### Adding a new architecture

To support a new backbone topology (not just a new weight file):

1. Implement the module under `architectures/` (or rely on `timm:` if timm
    already provides it).
2. Register it in `architectures/__init__.py`: teach `build_architecture` the
    name, and include the name in `list_architectures()`.
3. If bare `.pth` loads need non-default crop size or resize behaviour, add an
    entry in `ARCHITECTURE_DEFAULT_PREPROCESSING` in `models/registry.py`.
4. Optionally add a curated alias in `ALIASES` when you want a short name that
    pins architecture + weights + preprocessing together.

A new weight file for an architecture that already exists only needs step 4
(or callers can use the bare-checkpoint form above with `architecture=`).

## Encoders

``ReIDEvaluator`` and custom encoders can use these lightweight interfaces
instead of the full ``ReIDModel`` stack:

::: trackers.core.reid.encoder.ReIDEncoder

::: trackers.core.reid.encoder.ReIDPathEncoder

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
