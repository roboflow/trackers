---
title: CLI Migration Guide
description: Migrate Trackers CLI commands from the legacy argparse interface to the jsonargparse CLI.
---

# CLI Migration Guide

The CLI now groups `track` options by responsibility. The new dotted arguments map directly to the option dataclasses used by `track`, and are also available in YAML configuration files.

Semantic legacy spellings remain available during the transition. Each use emits a `FutureWarning` with the replacement. Do not combine a legacy argument and its replacement in one command; the CLI rejects that ambiguity. Develop's `--no-boxes` and `--no-ids` still work and map to `--show.no_boxes` and `--show.no_ids`. Each warning names the release that removes it: 2.10.0.

## Track command

Use dotted paths for grouped detection, filtering, output, visualization, and tracker options. `--display` remains ungrouped.

```text
trackers track \
    --source source.mp4 \
    --detection.model rfdetr-base \
    --detection.confidence 0.3 \
    --filters.classes [person,car] \
    --tracker sort \
    --tracker.lost_track_buffer 40 \
    --output.video tracked.mp4 \
    --show.no_boxes
```

Every boolean option is a pair: `--show.boxes` turns it on, `--show.no_boxes` turns it off, and `--show.boxes false` or `--show.boxes=false` spell the same thing explicitly. The negation sits on the field rather than the group, so it stays readable for any group name — `--detection.no_fast`, never `--no_detection.fast`. Repeating both halves is allowed; the last one wins. A `--config` file keeps one plain boolean key per field, and a command-line flag always overrides it.

This covers ungrouped options too — `--display` / `--no_display`, `--no_enqueue_defaults` on `tune`, `--no_list_available` on `download`.

The algorithm and its parameters share one group, mirroring `--detection.model` and the rest of the detection options. `--tracker sort` is shorthand for `--tracker.name sort`; both spellings are supported and neither warns. `tune` accepts the same two spellings for its own `--tracker.name` group — `tune --tracker sort` and `tune --tracker.name sort` are identical, though `tune` has no `--tracker.<param>` group of its own (its per-run overrides are `--fixed_params` and `search_space`). This also changes the `--config` YAML shape for `tune`: a flat `tracker: sort` key no longer parses — use `tracker: {name: sort}`, matching `track`'s config shape.

Tracker parameters are the exception: they default to `None`, meaning "leave the tracker's own default alone", so they take an explicit value and have no negative half. Develop's bare `--tracker.enable_cmc` flag turned camera motion compensation **off**, since the parameter itself defaults to `True`, so the bare spelling still maps to `--tracker.enable_cmc=false` and warns. Prefer `--tracker.enable_cmc true` or `--tracker.enable_cmc false`; they say what they do. The same applies to `--tracker.instant_first_frame_activation`.

As in develop, specify either a model or a precomputed MOT file, not both.

| Legacy argument       | Current argument               |
| --------------------- | ------------------------------ |
| `--model`             | `--detection.model`            |
| `--detections`        | `--detection.mot_file`         |
| `--model.confidence`  | `--detection.confidence`       |
| `--model.device`      | `--detection.device`           |
| `--model.api_key`     | `--detection.api_key`          |
| `--classes`           | `--filters.classes`            |
| `--track_ids`         | `--filters.track_ids`          |
| `--tracker`           | `--tracker` (unchanged)        |
| `--tracker.<name>`    | `--tracker.<name>`             |
| `-o`, `--output`      | `--output.video`               |
| `--mot-output`        | `--output.mot_results`         |
| `--overwrite`         | `--output.overwrite`           |
| `--display`           | `--display` (unchanged)        |
| `--show-boxes`        | `--show.boxes`                 |
| `--no-boxes`          | `--show.no_boxes`              |
| `--no-show.boxes`     | Removed; use `--show.no_boxes` |
| `--show-masks`        | `--show.masks`                 |
| `--show-labels`       | `--show.labels`                |
| `--show-ids`          | `--show.ids`                   |
| `--no-ids`            | `--show.no_ids`                |
| `--no-show.ids`       | Removed; use `--show.no_ids`   |
| `--show-confidence`   | `--show.confidence`            |
| `--show-trajectories` | `--show.trajectories`          |

### List-valued filters

`--filters.classes` and `--filters.track_ids` take lists, matching the list-valued `--metrics` and `--columns` options of `eval` and `tune`. Bracket shorthand needs no quoting, so `--filters.classes [person,car]`, `--filters.classes [0,2]`, and the mixed `--filters.classes [person,2]` all work. Comma-separated strings remain available as a warning-emitting alias.

| Legacy value form              | Current value form               |
| ------------------------------ | -------------------------------- |
| `--filters.classes person,car` | `--filters.classes [person,car]` |
| `--filters.track_ids 1,3,5`    | `--filters.track_ids [1,3,5]`    |

### Abbreviated tracker parameters

Tracker parameter names abbreviate their standard leading token on the command line: `minimum_` becomes `min_` and `maximum_` becomes `max_`. Domain words such as `threshold` stay spelled out. Every develop parameter path whose spelling did not change keeps working as-is, without a warning; the unabbreviated paths remain as warning-emitting aliases.

| Legacy argument                                     | Current argument                                |
| --------------------------------------------------- | ----------------------------------------------- |
| `--tracker.minimum_consecutive_frames`              | `--tracker.min_consecutive_frames`              |
| `--tracker.minimum_iou_threshold`                   | `--tracker.min_iou_threshold`                   |
| `--tracker.minimum_iou_threshold_first_assoc`       | `--tracker.min_iou_threshold_first_assoc`       |
| `--tracker.minimum_iou_threshold_second_assoc`      | `--tracker.min_iou_threshold_second_assoc`      |
| `--tracker.minimum_iou_threshold_unconfirmed_assoc` | `--tracker.min_iou_threshold_unconfirmed_assoc` |
| `--tracker.iou`                                     | `--tracker.iou_variant`                         |

These short forms are **CLI aliases only**. The Python constructor keywords are unchanged, so `ByteTrackTracker(minimum_iou_threshold=0.3)` stays correct, and so do `tune --fixed_params`, each tracker's `search_space` keys, and the "Valid parameters" list printed on a `search_space` error. A consequence worth knowing: `tune` reports the long parameter names, so its output cannot be pasted verbatim into a `track` command — abbreviate the leading `minimum_` or `maximum_` token first.

### mcbyte mask settings

`mcbyte`'s mask pipeline configuration (`McByteMaskConfig` — SAM/Cutie device, checkpoints, Hydra config) is one more exact-name rename: it appears on the command line as `--tracker.mask.*`, not `--tracker.mask_config.*`. Unlike the `minimum_`/`maximum_` prefixes above, this is not a deprecation alias — `mask_config` never shipped under its Python name on any CLI release, so there is nothing to warn about and no legacy spelling to migrate from. Run `trackers track --tracker.mask.help` for the full sub-option list.

Note this differs from `trackers inspect mcbyte`, whose own comparison-only `--mask.*` group is a peer of `--sequence.*` and `--cmc.*` rather than nested under a tracker selector — see the [Inspect the Mask Pipeline guide](inspect.md).

## Hyphens and underscores

Interchangeable in every option name, on every command, without a warning. Only the name is rewritten: each `-` after the leading `--` becomes `_`, and values are left alone, so `--detection.model rfdetr-base` and `--source my-dir/clip.mp4` keep their hyphens. This covers dotted paths and negations alike — `--show.no-ids`, `--tracker.min-iou-threshold` and `--no-display` all reach the parser as their underscore spellings. The canonical documentation spelling uses underscores.

Anything after a bare `--` is passed through untouched.

This holds for the deprecated spellings in the tables below too, so a develop command ports without also having to guess which separator each option wanted: `--no-boxes` and `--no_boxes` resolve alike, as do `--mot-output` and `--mot_output`, `--track-ids` and `--track_ids`. `--help` lists the underscore spelling.

## `eval` prediction inputs

`--tracker` names the tracking algorithm in `track` and `tune`. In `eval` it meant something else entirely — a file of results that algorithm had already produced. Both prediction inputs are renamed so one option name no longer carries two meanings:

| Legacy argument | Current argument    |
| --------------- | ------------------- |
| `--tracker`     | `--predictions`     |
| `--tracker_dir` | `--predictions_dir` |

The old spellings still parse and warn.

## Other commands

| Command    | Hyphenated spelling     | Canonical underscore spelling |
| ---------- | ----------------------- | ----------------------------- |
| `eval`     | `--gt-dir`              | `--gt_dir`                    |
| `eval`     | `--predictions-dir`     | `--predictions_dir`           |
| `tune`     | `--gt-dir`              | `--gt_dir`                    |
| `tune`     | `--detections-dir`      | `--detections_dir`            |
| `tune`     | `--n-trials`            | `--n_trials`                  |
| `tune`     | `--fixed-params`        | `--fixed_params`              |
| `tune`     | `--images-dir`          | `--images_dir`                |
| `tune`     | `--no-enqueue-defaults` | `--no_enqueue_defaults`       |
| `download` | `--cache-dir`           | `--cache_dir`                 |

The remaining deprecated transitions are:

| Command                    | Legacy argument        | Current argument                |
| -------------------------- | ---------------------- | ------------------------------- |
| `eval`, `tune`, `download` | `-o`                   | `--output`                      |
| `eval`                     | `--metrics CLEAR HOTA` | `--metrics '["CLEAR", "HOTA"]'` |
| `eval`                     | `--columns MOTA HOTA`  | `--columns '["MOTA", "HOTA"]'`  |
| `tune`                     | `--metrics CLEAR HOTA` | `--metrics '["CLEAR", "HOTA"]'` |
| `download`                 | positional `DATASET`   | `--name DATASET`                |
| `download`                 | `--dataset`            | `--name`                        |
| `download`                 | `--list`               | `--list_available`              |

For example, replace:

```text
trackers download mot17 --cache-dir .cache
```

with:

```text
trackers download --name mot17 --cache_dir .cache
```

## YAML configuration

The same nesting is used in `--config` files. Command-line values override configuration values.

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
