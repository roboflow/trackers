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

| Command                           | Inspects                                                 | Answers                                                                         |
| --------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `trackers inspect sam`            | `SAMBoxMaskGenerator`                                    | Did SAM turn these boxes into the masks I expected?                             |
| `trackers inspect cutie`          | `CutieMaskPropagator`                                    | Did the masks survive propagation across frames?                                |
| `trackers inspect edgetam`        | `EdgeTAMMaskBackend`                                     | Do creation, propagation and bounded sessions preserve mask ownership?          |
| `trackers inspect mask-manager`   | `MaskManager`                                            | Were masks created, added, and removed on the right frames?                     |
| `trackers inspect mcbyte`         | [`McByteTracker`](../trackers/mcbyte.md)                 | What did mask conditioning change versus locked IoU?                            |
| `trackers inspect mcbyteplusplus` | [`McBytePlusPlusTracker`](../trackers/mcbyteplusplus.md) | How do EdgeTAM masks interact with detector association and occlusion recovery? |

The mask components live in `trackers.core.masks` and are tracker-agnostic. `mcbyte` and `mcbyteplusplus` inspect complete trackers.

!!! warning "Install the component's optional dependencies"

    For SAM, Cutie, MaskManager and McByte inspection:

    ```bash
    pip install "trackers[mask]"
    ```

    For EdgeTAM creation, propagation and McByte++ inspection:

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

Configuration files are optional: every setting can also be passed directly as a command-line argument, as above. For a longer replay, the shared [SNMOT-116 example config](https://github.com/roboflow/trackers/blob/develop/src/trackers/cli/inspect/configs/snmot-116.json) ([download JSON](https://raw.githubusercontent.com/roboflow/trackers/develop/src/trackers/cli/inspect/configs/snmot-116.json)) supplies identical frames, initial boxes, additions and removals to both `edgetam` and `cutie`. It selects frames `000548.jpg` through `000650.jpg`, with seven initial objects, four additions and four removals. IDs follow the order of initial boxes and then additions; these are manual inspection IDs, not SoccerNet ground-truth identities.

The example configs live in `src/trackers/cli/inspect/configs/` in the repository and are not bundled with the installed package. To use them without cloning the repository, save the linked JSON file locally and pass its path to `--config`, for example `--config ~/Downloads/snmot-116.json`. The links use the `develop` branch; select the matching release tag on GitHub when using a released version.

After installing the `edgetam` and `mask` extras, run these commands from the repository root, replacing the image directory with your dataset location. Arguments after `--config` override the file's values, so the generic path in the example does not need editing:

```bash
trackers inspect edgetam \
    --config src/trackers/cli/inspect/configs/snmot-116.json \
    --image_dir /path/to/tracking/test/SNMOT-116/img1 \
    --max_session_frames 50

trackers inspect cutie \
    --config src/trackers/cli/inspect/configs/snmot-116.json \
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

These mask interfaces take **RGB** uint8 images. Generation uses a temporary session, preserving temporal state. Outputs contain original-resolution boolean masks and `mask_avg_prob_dict=None`. McByte++ explicitly accepts masks without confidence; existing McByte confidence gating is unchanged. `backend.reset()` releases all temporal frames and features while keeping the model loaded.

## Inspect McByte++ with oracle detections

Run a SoccerNet test sequence using its oracle `det/det.txt` boxes:

```bash
trackers inspect mcbyteplusplus \
    --sequence.image_dir /path/to/SoccerNet/tracking/test/SNMOT-116/img1 \
    --sequence.det_file /path/to/SoccerNet/tracking/test/SNMOT-116/det/det.txt \
    --sequence.start_frame 1 --sequence.end_frame 50 \
    --sequence.frame_rate 25 --sequence.confidence_override 1.0 \
    --edgetam.device cpu --edgetam.max_session_frames 200
```

This inspector always enables EdgeTAM masks. CMC is enabled by default with ORB at downscale 4 and can be disabled with `--cmc.enable false`. The mask switch is needed only when enabling the default backend through the Python API or the generic `trackers track` command.

The confidence override is explicit because some oracle files use placeholder scores. Identity columns are ignored. The inspector converts loaded RGB images to the tracker's BGR input; the tracker converts them for EdgeTAM internally. Only the inspection command enumerates selected filenames: the tracker and backend receive one image at a time and need no future frames or final sequence length.

A portable [example config](https://github.com/roboflow/trackers/blob/develop/src/trackers/cli/inspect/configs/mcbyteplusplus-soccernet.json) ([download JSON](https://raw.githubusercontent.com/roboflow/trackers/develop/src/trackers/cli/inspect/configs/mcbyteplusplus-soccernet.json)) provides the same settings. From a repository checkout, override its placeholder paths:

```bash
trackers inspect mcbyteplusplus \
    --config src/trackers/cli/inspect/configs/mcbyteplusplus-soccernet.json \
    --sequence.image_dir /your/sequence/img1 \
    --sequence.det_file /your/sequence/det/det.txt
```

With an installed package and no repository checkout, save the JSON from the download link and replace the `--config` argument above with its local path, for example `--config ~/Downloads/mcbyteplusplus-soccernet.json`. Keep the dataset path overrides.

Outputs are saved under `outputs/inspect/mcbyteplusplus/<timestamp>/`: annotated `frames/`, MOT-format `results.txt`, reusable `config.json`, and `run.json` with requested/resolved model revisions, dependency versions, input hashes, per-frame session counters and output IDs. Replay with `--config <run>/config.json`.

The overlay shows all raw masks, including excluded masks retained in EdgeTAM memory. The `Excluded mask IDs` status line and per-frame `excluded_mask_ids` in `run.json` report internal IDs excluded from association after that update. Empty masks are excluded before association; newly detected displacement is assessed against matched detector boxes afterwards and takes effect on the next image. A validated refresh is required to restore association use.

After model loading, a progress bar shows completed frames, percentage, elapsed time and estimated time remaining. Runtime warnings and the Hugging Face authentication notice are saved in the `warnings` field of `run.json` instead of printed to the terminal. A message gives the file path if any warnings were captured. The CLI beta notice, errors and other logged diagnostics remain visible; warning capture does not change tracking calculations.

Use `--tracker.enable_mask_refresh false` to disable recovery, or change `--tracker.mask_refresh_coverage_threshold` (default 0.5, strict comparison). The independent `--tracker.enable_mask_trust_filter false` control restores the previous policy for comparison; disabling only refresh leaves suspect masks excluded. The occlusion veto is `--edgetam.mask_creation_bbox_overlap_threshold` (default 0.1). See the [McByte++ guide](../trackers/mcbyteplusplus.md) for lifecycle details. For a quick rollover check, set `--edgetam.max_session_frames 3`; ordinary runs retain the default 200.

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
