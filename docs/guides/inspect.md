---
title: Inspect the Mask Pipeline — SAM, Cutie, EdgeTAM, MaskManager
description: Render what each stage of the mask pipeline produced, frame by frame, with the trackers inspect command.
---

# Inspect the Mask Pipeline

`trackers inspect` renders what one stage of the mask pipeline actually produced, frame by frame, so you can look at it instead of guessing from metrics.

**What you'll learn:**

- Which component each subcommand inspects
- How to replay the same boxes and lifecycle events with Cutie and EdgeTAM, including EdgeTAM session rollover
- How to drive `mask-manager` from hand-written boxes or from ground truth
- How to compare McByte's two association modes on one sequence

## Components

Each name is the thing being inspected, not the tracker that happens to use it:

| Command                         | Inspects                                 | Answers                                                                |
| ------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------- |
| `trackers inspect sam`          | `SAMBoxMaskGenerator`                    | Did SAM turn these boxes into the masks I expected?                    |
| `trackers inspect cutie`        | `CutieMaskPropagator`                    | Did the masks survive propagation across frames?                       |
| `trackers inspect edgetam`      | `EdgeTAMMaskBackend`                     | Do creation, propagation and bounded sessions preserve mask ownership? |
| `trackers inspect mask-manager` | `MaskManager`                            | Were masks created, added, and removed on the right frames?            |
| `trackers inspect mcbyte`       | [`McByteTracker`](../trackers/mcbyte.md) | What did mask conditioning change versus locked IoU?                   |

The mask components live in `trackers.core.masks` and are tracker-agnostic. Only `mcbyte` inspects a tracker.

!!! warning "Install the component's optional dependencies"

    For SAM, Cutie, MaskManager and McByte inspection:

    ```bash
    pip install "trackers[mask]"
    ```

    For EdgeTAM creation and propagation:

    ```bash
    pip install "trackers[edgetam]"
    ```

    Without the extra the command exits with an install hint rather than a traceback.

    For more options, see the [install guide](install.md).

## Inspect SAM

Give it an image and one or more boxes; it saves the image with masks and boxes overlaid.

```bash
trackers inspect sam \
    --image_path frame.jpg \
    --box='[[10,20,110,220],[30,40,130,240]]'
```

Boxes are one list, not a repeated option. `--box+='[[300,400,430,540]]'` appends to boxes already given.

## Inspect Cutie

SAM initializes masks on the first selected frame, then Cutie propagates them across the rest.

```bash
trackers inspect cutie \
    --image_dir frames \
    --start_file 000001.jpg --end_file 000010.jpg \
    --box='[[10,20,110,220]]'
```

Masks can be added and removed mid-range to see how propagation reacts:

```bash
trackers inspect cutie \
    --image_dir frames \
    --start_file 000001.jpg --end_file 000010.jpg \
    --box='[[10,20,110,220]]' \
    --add_at+ 000004.jpg:300,400,430,540 \
    --remove_at+ 000007.jpg:1
```

An add event names the frame where the box is valid; the mask is applied there and propagated to the next frame, matching McByte timing. Add events therefore cannot sit on the last selected frame, and remove events cannot sit on the first.

## Inspect EdgeTAM

EdgeTAM creates masks from boxes and propagates them using one loaded model. The command follows the same manual ID and event timing as `cutie`:

```bash
trackers inspect edgetam \
    --image_dir frames \
    --start_file 000001.jpg --end_file 000010.jpg \
    --box='[[10,20,110,220],[30,40,130,240]]' \
    --add_at+ 000004.jpg:300,400,430,540 \
    --remove_at+ 000007.jpg:1 \
    --max_session_frames 3
```

This short session limit deliberately exercises rollover. The default is **200** stored images, including the seed frame, for every input source. Use `--max_session_frames 1000` for a longer history or `--max_session_frames null` to disable rollover. Finite limits must be at least two. Filename gaps do not count as processed images. A session limit bounds retained history; total memory also depends on image size and object count.

Masks may overlap. Additions submit the existing latest masks and new masks together, retaining earlier history. Removal reseeds the surviving objects; rollover transfers their latest masks, including empty masks. This command does not repair an existing object's empty mask or recover its identity. Each addition creates a new manual ID.

Outputs go under `outputs/inspect/edgetam/<timestamp>/`:

- Annotated frames show mask colours, active IDs, session position and rollover events.
- `config.json` records replay options, including the resolved model revision when available. Repeat with `trackers inspect edgetam --config path/to/config.json`.
- `run.json` records completion or failure, dependency versions, input hashes, added/removed IDs and session indices for each saved frame.

The default model, `yonigozlan/EdgeTAM-hf`, is pinned to tested commit `c266ce53b3fc00f0f495b583f6a116c4e57f53bb`. Override it with `--revision`, for example `--revision main` to follow the model's main branch. A custom `--model_id` does not inherit this commit; specify its own revision for reproducible comparisons. The resolved commit and dependency versions are saved for each run. Inference defaults to fp32 and `--device auto` selects CUDA when available, otherwise CPU. `--mask_threshold` is a logit threshold (default zero), not a mask confidence score.

This is **manual mask replay**, suitable for fixed boxes and lifecycle events extracted from a labelled sequence. It does not run a tracker or compute HOTA, IDF1 or MOTA. When evaluating tracking on SoccerNet oracle detections, use ground-truth identities only for scoring; a ground-truth identity replay is a separate inspection experiment.

Configuration files are optional: every setting can also be passed directly as a command-line argument, as above. For a longer replay, the shared [SNMOT-116 example config](../assets/inspect/snmot-116.json) supplies identical frames, initial boxes, additions and removals to both `edgetam` and `cutie`. It selects frames `000548.jpg` through `000650.jpg`, with seven initial objects, four additions and four removals. IDs follow the order of initial boxes and then additions; these are manual inspection IDs, not SoccerNet ground-truth identities.

After installing the `edgetam` and `mask` extras, run these commands from the repository root, replacing the image directory with your dataset location. Arguments after `--config` override the file's values, so the generic path in the example does not need editing:

```bash
trackers inspect edgetam \
    --config docs/assets/inspect/snmot-116.json \
    --image_dir /path/to/tracking/test/SNMOT-116/img1 \
    --max_session_frames 50

trackers inspect cutie \
    --config docs/assets/inspect/snmot-116.json \
    --image_dir /path/to/tracking/test/SNMOT-116/img1
```

The config uses CPU for both backends. EdgeTAM's session limit stays on its command line because it is not a Cutie option; 50 stored images exercises rollover within this sequence. Both commands save annotated images in their own default output directories under `outputs/inspect/edgetam/` or `outputs/inspect/cutie/`. EdgeTAM also saves the effective configuration in `config.json` for replay and a `run.json` manifest.

The backend also implements the existing mask interfaces:

```python
from trackers.core.masks.edgetam import EdgeTAMMaskBackend
from trackers.core.masks.manager import MaskManager

backend = EdgeTAMMaskBackend(max_session_frames=200)
manager = MaskManager(mask_generator=backend, mask_propagator=backend)
```

These mask interfaces take **RGB** uint8 images. Generation uses a temporary session, preserving temporal state. Outputs contain original-resolution boolean masks and `mask_avg_prob_dict=None`. Tracker association must explicitly support masks without confidence before using this backend for tracking; current McByte confidence gating is unchanged. `backend.reset()` releases all temporal frames and features while keeping the model loaded.

## Inspect MaskManager

This one calls `MaskManager.get_updated_masks()` directly, so it exercises the real orchestration: initialize, propagate, add, remove. `--mode` is required and selects where tracklets come from.

### Manual mode

Tracklets come from boxes you write, with lifecycle events you schedule.

```bash
trackers inspect mask-manager --mode manual \
    --image_dir frames \
    --start_file 000001.jpg --end_file 000010.jpg \
    --box='[[10,20,110,220]]' \
    --add_at+ 000004.jpg:300,400,430,540 \
    --remove_at+ 000007.jpg:2
```

### Ground-truth mode

Tracklets are replayed from a MOT-format ground-truth file, which is the better way to watch delayed mask creation on crowded real data.

```bash
trackers inspect mask-manager --mode gt \
    --image_dir frames \
    --gt_file gt.txt \
    --start_frame 1 --end_frame 100 \
    --tracklet_id='[3,7]'
```

Omit `--tracklet_id`, or pass `--tracklet_id='[all]'`, to replay every tracklet.

Ground-truth mode colors each box by its lifecycle state:

| Color  | Meaning                                  |
| ------ | ---------------------------------------- |
| Blue   | Newly visible tracklet                   |
| Yellow | Pending mask creation                    |
| Green  | Tracklet already has a mask              |
| Purple | Visible, but not masked, pending, or new |

Options belong to one mode. Passing `--gt_file` with `--mode manual` is an error rather than a silently ignored option, and the check runs before any model loads:

```console
$ trackers inspect mask-manager --mode manual --image_dir frames --gt_file gt.txt
Error: --gt_file (or --gt-file) is a --mode gt option and cannot be used with --mode manual.
```

## Compare McByte association modes

Runs the same detections through two McByte configurations on one sequence: `locked_iou` (clear-match locking, no MaskManager) and `mask_conditioned` (the full mask pipeline).

```bash
trackers inspect mcbyte \
    --sequence.image_dir frames \
    --sequence.det_file det.txt \
    --sequence.det_format mot_tlwh \
    --sequence.start_frame 1 --sequence.end_frame 200 \
    --mask.device cuda
```

Both runs save per-frame visualizations and MOTChallenge-format results, so the two directories can be diffed or scored against each other. `--modes=[locked_iou]` runs just one of them.

Detection files are read as either `mot_tlwh` (`frame,id,left,top,width,height,confidence,...`, identity column ignored) or `xyxy` (`frame,x1,y1,x2,y2,confidence`).

## Shared conventions

Every `inspect` command follows the same rules as the rest of the CLI:

- `--image-dir` and `--image_dir` are the same option.
- `--config run.yaml` supplies the same keys from a file.
- Boolean options have a negative half, such as `--cmc.no_enable`.
- List options take bracket syntax (`--modes=[locked_iou]`) and append with `+`.

Outputs go to `outputs/inspect/<component>/` under the current working directory, one timestamped directory per run. `mask-manager` groups its runs by mode first, so they land in `outputs/inspect/mask-manager/<mode>/`, and a manual run is never mixed in with a ground-truth one. The resolved path is printed when the run finishes.
