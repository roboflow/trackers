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
`--no-show.boxes` and `--no-show.ids` aliases are removed; develop's
`--no-boxes` and `--no-ids` still work and map to an explicit `false`. Each
warning states its scheduled removal release: the current version plus 0.3.

## Track command

Use dotted paths for grouped detection, filtering, output, visualization, and
tracker options. `--display` remains a flat action.

```text
trackers track \
    --source source.mp4 \
    --detection.model rfdetr-base \
    --detection.confidence 0.3 \
    --filters.classes [person,car] \
    --tracker sort \
    --tracker.lost_track_buffer 40 \
    --output.video tracked.mp4 \
    --show.boxes false
```

The algorithm and its parameters share one group, mirroring `--detection.model`
and the rest of the detection options. `--tracker sort` is shorthand for
`--tracker.name sort`; both spellings are supported and neither warns.

Boolean tracker parameters now take an explicit value: `--tracker.enable_cmc true` or `--tracker.enable_cmc false`. Develop's bare `--tracker.enable_cmc`
flag turned camera motion compensation **off**, since the parameter defaults to
`True`, so the bare spelling still maps to `false` and warns. Prefer the
explicit value; it says what it does. The same applies to
`--tracker.instant_first_frame_activation`.

As in develop, specify either a model or a precomputed MOT file, not both.

| Legacy argument       | Current argument                  |
| --------------------- | --------------------------------- |
| `--model`             | `--detection.model`               |
| `--detections`        | `--detection.mot_file`            |
| `--model.confidence`  | `--detection.confidence`          |
| `--model.device`      | `--detection.device`              |
| `--model.api_key`     | `--detection.api_key`             |
| `--classes`           | `--filters.classes`               |
| `--track_ids`         | `--filters.track_ids`             |
| `--tracker`           | `--tracker` (unchanged)           |
| `--tracker.<name>`    | `--tracker.<name>`                |
| `-o`, `--output`      | `--output.video`                  |
| `--mot-output`        | `--output.mot_results`            |
| `--overwrite`         | `--output.overwrite`              |
| `--display`           | `--display` (unchanged)           |
| `--show-boxes`        | `--show.boxes true`               |
| `--no-boxes`          | `--show.boxes false`              |
| `--no-show.boxes`     | Removed; use `--show.boxes false` |
| `--show-masks`        | `--show.masks`                    |
| `--show-labels`       | `--show.labels`                   |
| `--show-ids`          | `--show.ids true`                 |
| `--no-ids`            | `--show.ids false`                |
| `--no-show.ids`       | Removed; use `--show.ids false`   |
| `--show-confidence`   | `--show.confidence`               |
| `--show-trajectories` | `--show.trajectories`             |

### List-valued filters

`--filters.classes` and `--filters.track_ids` take lists, matching the
list-valued `--metrics` and `--columns` options of `eval` and `tune`. Bracket
shorthand needs no quoting, so `--filters.classes [person,car]`,
`--filters.classes [0,2]`, and the mixed `--filters.classes [person,2]` all
work. Comma-separated strings remain available as a warning-emitting alias.

| Legacy value form              | Current value form               |
| ------------------------------ | -------------------------------- |
| `--filters.classes person,car` | `--filters.classes [person,car]` |
| `--filters.track_ids 1,3,5`    | `--filters.track_ids [1,3,5]`    |

### Abbreviated tracker parameters

Tracker parameter names abbreviate their standard leading token on the command
line: `minimum_` becomes `min_` and `maximum_` becomes `max_`. Domain words such
as `threshold` stay spelled out. Every develop parameter path whose spelling did
not change keeps working as-is, without a warning; the unabbreviated paths remain
as warning-emitting aliases.

| Legacy argument                                     | Current argument                                |
| --------------------------------------------------- | ----------------------------------------------- |
| `--tracker.minimum_consecutive_frames`              | `--tracker.min_consecutive_frames`              |
| `--tracker.minimum_iou_threshold`                   | `--tracker.min_iou_threshold`                   |
| `--tracker.minimum_iou_threshold_first_assoc`       | `--tracker.min_iou_threshold_first_assoc`       |
| `--tracker.minimum_iou_threshold_second_assoc`      | `--tracker.min_iou_threshold_second_assoc`      |
| `--tracker.minimum_iou_threshold_unconfirmed_assoc` | `--tracker.min_iou_threshold_unconfirmed_assoc` |
| `--tracker.iou`                                     | `--tracker.iou_variant`                         |

These short forms are **CLI aliases only**. The Python constructor keywords are
unchanged, so `ByteTrackTracker(minimum_iou_threshold=0.3)` stays correct, and
so do `tune --fixed_params`, each tracker's `search_space` keys, and the
"Valid parameters" list printed on a `search_space` error. A consequence worth
knowing: `tune` reports the long parameter names, so its output cannot be
pasted verbatim into a `track` command — abbreviate the leading `minimum_` or
`maximum_` token first.

## Other commands

Hyphens and underscores are interchangeable in current option names, without a
warning. The canonical documentation spelling uses underscores.

| Command    | Hyphenated spelling     | Canonical underscore spelling |
| ---------- | ----------------------- | ----------------------------- |
| `eval`     | `--gt-dir`              | `--gt_dir`                    |
| `eval`     | `--tracker-dir`         | `--tracker_dir`               |
| `tune`     | `--gt-dir`              | `--gt_dir`                    |
| `tune`     | `--detections-dir`      | `--detections_dir`            |
| `tune`     | `--n-trials`            | `--n_trials`                  |
| `tune`     | `--fixed-params`        | `--fixed_params`              |
| `tune`     | `--images-dir`          | `--images_dir`                |
| `tune`     | `--no-enqueue-defaults` | `--no-enqueue_defaults`       |
| `download` | `--cache-dir`           | `--cache_dir`                 |

The remaining deprecated transitions are:

| Command                    | Legacy argument        | Current argument                |
| -------------------------- | ---------------------- | ------------------------------- |
| `eval`, `tune`, `download` | `-o`                   | `--output`                      |
| `eval`                     | `--metrics CLEAR HOTA` | `--metrics '["CLEAR", "HOTA"]'` |
| `eval`                     | `--columns MOTA HOTA`  | `--columns '["MOTA", "HOTA"]'`  |
| `tune`                     | `--metrics CLEAR HOTA` | `--metrics '["CLEAR", "HOTA"]'` |
| `download`                 | positional `DATASET`   | `--dataset DATASET`             |
| `download`                 | `--list`               | `--list_available`              |

Use underscores in canonical names. Hyphenated spellings are accepted as
equivalents, so `--detection.mot-file` and `--detection.mot_file` mean the
same thing.

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
tracker:
  name: bytetrack
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
