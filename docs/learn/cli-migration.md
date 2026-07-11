---
title: CLI Migration Guide | Trackers
description: Migrate Trackers CLI commands from the legacy argparse interface to the jsonargparse CLI.
---

# CLI Migration Guide

The CLI now groups `track` options by responsibility. The new dotted arguments
map directly to the option dataclasses used by `track`, and are also available
in YAML configuration files.

Semantic legacy spellings remain available during the transition. Each use
emits a `FutureWarning` with the replacement. Do not combine a legacy argument
and its replacement in one command; the CLI rejects that ambiguity. The
negative `--no-boxes`, `--no-show.boxes`, `--no-ids`, and `--no-show.ids`
aliases are removed and do not participate in this transition. A removal
version for the remaining semantic aliases has not been scheduled.

## Track command

Use dotted paths for grouped detection, filtering, output, visualization, and
tracker parameters. `--display` remains a flat action.

```text
trackers track \
    --source source.mp4 \
    --detection.model rfdetr-base \
    --detection.confidence 0.3 \
    --filters.classes person,car \
    --tracker_params.lost_track_buffer 40 \
    --output.video tracked.mp4 \
    --show.boxes false
```

Boolean tracker parameters now take an explicit value. For example, the legacy
`--tracker.enable_cmc` toggle becomes `--tracker_params.enable_cmc false`.
As in develop, specify either a model or a precomputed MOT file, not both.

| Legacy argument | Current argument |
| --- | --- |
| `--model` | `--detection.model` |
| `--detections` | `--detection.mot_file` |
| `--model.confidence` | `--detection.confidence` |
| `--model.device` | `--detection.device` |
| `--model.api_key` | `--detection.api_key` |
| `--classes` | `--filters.classes` |
| `--track_ids` | `--filters.track_ids` |
| `--tracker.<name>` | `--tracker_params.<name>` |
| `-o`, `--output` | `--output.video` |
| `--mot-output` | `--output.mot_results` |
| `--overwrite` | `--output.overwrite` |
| `--display` | `--display` (unchanged) |
| `--show-boxes` | `--show.boxes true` |
| `--no-boxes`, `--no-show.boxes` | Removed; use `--show.boxes false` |
| `--show-masks` | `--show.masks` |
| `--show-labels` | `--show.labels` |
| `--show-ids` | `--show.ids true` |
| `--no-ids`, `--no-show.ids` | Removed; use `--show.ids false` |
| `--show-confidence` | `--show.confidence` |
| `--show-trajectories` | `--show.trajectories` |

## Other commands

Hyphens and underscores are interchangeable in current option names, without a
warning. The canonical documentation spelling uses underscores.

| Command | Hyphenated spelling | Canonical underscore spelling |
| --- | --- | --- |
| `eval` | `--gt-dir` | `--gt_dir` |
| `eval` | `--tracker-dir` | `--tracker_dir` |
| `tune` | `--gt-dir` | `--gt_dir` |
| `tune` | `--detections-dir` | `--detections_dir` |
| `tune` | `--n-trials` | `--n_trials` |
| `tune` | `--fixed-params` | `--fixed_params` |
| `tune` | `--images-dir` | `--images_dir` |
| `tune` | `--no-enqueue-defaults` | `--no-enqueue_defaults` |
| `download` | `--cache-dir` | `--cache_dir` |

The remaining deprecated transitions are:

| Command | Legacy argument | Current argument |
| --- | --- | --- |
| `eval`, `tune`, `download` | `-o` | `--output` |
| `eval` | `--metrics CLEAR HOTA` | `--metrics '["CLEAR", "HOTA"]'` |
| `eval` | `--columns MOTA HOTA` | `--columns '["MOTA", "HOTA"]'` |
| `tune` | `--metrics CLEAR HOTA` | `--metrics '["CLEAR", "HOTA"]'` |
| `download` | positional `DATASET` | `--dataset DATASET` |
| `download` | `--list` | `--list_available` |

Use underscores in canonical names. Hyphenated spellings are accepted as
equivalents, so `--detection.mot-file` and `--detection.mot_file` mean the
same thing.

## Pre-merge PR aliases

These paths came from the earlier jsonargparse implementation on this branch,
not from the develop argparse CLI. They remain warning-emitting aliases while
the semantic names transition.

| Earlier PR path | Intended CLI path |
| --- | --- |
| `--detection.detections` | `--detection.mot_file` |
| `--out.output` | `--output.video` |
| `--out.mot_results` | `--output.mot_results` |
| `--out.overwrite` | `--output.overwrite` |
| `--vis.display` | `--display` |

For example, replace:

```text
trackers download mot17 --cache-dir .cache
```

with:

```text
trackers download --dataset mot17 --cache_dir .cache
```

## YAML configuration

The same nesting is used in `--config` files. Command-line values override
configuration values.

```yaml
source: source.mp4
detection:
  model: rfdetr-base
  confidence: 0.3
tracker_params:
  lost_track_buffer: 40
output:
  video: tracked.mp4
  overwrite: true
display: false
show:
  boxes: false
```

Run it with:

```text
trackers track --config tracking.yaml
```
