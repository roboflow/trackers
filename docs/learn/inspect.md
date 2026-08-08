---
title: Inspect the Mask Pipeline — SAM, Cutie, MaskManager | Trackers
description: Render what each stage of the mask pipeline produced, frame by frame, with the trackers inspect command.
---

# Inspect the Mask Pipeline

`trackers inspect` renders what one stage of the mask pipeline actually produced, frame by frame, so you can look at it instead of guessing from metrics.

**What you'll learn:**

- Which component each subcommand inspects
- How to drive `mask-manager` from hand-written boxes or from ground truth
- How to compare McByte's two association modes on one sequence

## Components

Each name is the thing being inspected, not the tracker that happens to use it:

| Command                         | Inspects                                 | Answers                                                     |
| ------------------------------- | ---------------------------------------- | ----------------------------------------------------------- |
| `trackers inspect sam`          | `SAMBoxMaskGenerator`                    | Did SAM turn these boxes into the masks I expected?         |
| `trackers inspect cutie`        | `CutieMaskPropagator`                    | Did the masks survive propagation across frames?            |
| `trackers inspect mask-manager` | `MaskManager`                            | Were masks created, added, and removed on the right frames? |
| `trackers inspect mcbyte`       | [`McByteTracker`](../trackers/mcbyte.md) | What did mask conditioning change versus locked IoU?        |

The first three live in `trackers.core.masks` and are tracker-agnostic: nothing in them depends on McByte. Only the last one inspects a tracker.

!!! warning "Requires the mask extra"

    Every `inspect` command builds SAM and Cutie, so install them first:

    ```bash
    pip install "trackers[mask]"
    ```

    Without the extra the command exits with an install hint rather than a traceback.

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
