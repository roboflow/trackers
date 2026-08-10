---
title: CLI Command Reference — trackers track, eval, tune, inspect | Trackers
description: One-page reference for every trackers CLI command and subcommand, including the full flag tables for trackers inspect and a link to each command's detailed guide.
---

# CLI Reference

Every command the `trackers` CLI exposes, in one table, plus the full flag reference for `trackers inspect`'s four subcommands — the one part of the CLI with no flag table anywhere else in the docs.

**What you'll learn:**

- Every top-level command and subcommand the CLI exposes
- Where each command's detailed guide and flag table lives
- The full flag reference for `trackers inspect sam` / `cutie` / `mask-manager` / `mcbyte`

---

## Install

Get started by installing the package.

```text
pip install trackers
```

`inspect` subcommands additionally need the `mask` extra — see [Requires the mask extra](#requires-the-mask-extra) below. For more options, see the [install guide](install.md).

---

## Commands

<table>
  <colgroup>
    <col style="width: 30%">
    <col style="width: 45%">
    <col style="width: 25%">
  </colgroup>
  <thead>
    <tr>
      <th>Command</th>
      <th>Purpose</th>
      <th>Reference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>trackers track</code></td>
      <td>Run a tracker on a video, webcam, RTSP stream, or image directory.</td>
      <td><a href="track.md#cli-reference">Track guide</a></td>
    </tr>
    <tr>
      <td><code>trackers eval</code></td>
      <td>Score tracker output against ground truth with CLEAR, HOTA, and Identity metrics.</td>
      <td><a href="evaluate.md#cli-reference">Evaluate guide</a></td>
    </tr>
    <tr>
      <td><code>trackers download</code></td>
      <td>Download MOT benchmark datasets — MOT17 and SportsMOT.</td>
      <td><a href="download.md#cli-reference">Download guide</a></td>
    </tr>
    <tr>
      <td><code>trackers tune</code></td>
      <td>Search tracker hyperparameters with Optuna.</td>
      <td><a href="tune.md#cli-reference">Tune guide</a></td>
    </tr>
    <tr>
      <td><code>trackers inspect sam</code></td>
      <td>Run SAM mask generation on a single image and save a visualization.</td>
      <td><a href="#inspect-sam">Below</a></td>
    </tr>
    <tr>
      <td><code>trackers inspect cutie</code></td>
      <td>Seed masks with SAM on the first frame, then propagate them with Cutie across a frame range.</td>
      <td><a href="#inspect-cutie">Below</a></td>
    </tr>
    <tr>
      <td><code>trackers inspect mask-manager</code></td>
      <td>Run the real <code>MaskManager</code> orchestration — initialize, propagate, add, remove — in <code>manual</code> or <code>gt</code> mode.</td>
      <td><a href="#inspect-mask-manager">Below</a></td>
    </tr>
    <tr>
      <td><code>trackers inspect mcbyte</code></td>
      <td>Compare locked-IoU vs. mask-conditioned McByte on one sequence.</td>
      <td><a href="#inspect-mcbyte">Below</a></td>
    </tr>
    <tr>
      <td><code>trackers benchmark mcbyte</code></td>
      <td>Run McByte over complete benchmark test sets and write MOTChallenge-format results.</td>
      <td><a href="mcbyte-benchmark.md#cli-reference">McByte Benchmarks guide</a></td>
    </tr>
  </tbody>
</table>

For a narrative walkthrough of `track`, `eval`, `download`, or `tune`, see their own guides linked above. For `inspect`, see the [Inspect the Mask Pipeline guide](inspect.md) for worked examples — the flag tables below cover the full surface of each subcommand.

Every `trackers` subcommand accepts `--config <file>` to load arguments from a YAML file, and hyphens and underscores are interchangeable in flag names (`--cmc-downscale` and `--cmc_downscale` are the same option). Boolean flags gain a negated `--no_<name>` counterpart.

---

## Requires the mask extra

All four `inspect` subcommands build SAM and Cutie, so install them first:

```bash
pip install "trackers[mask]"
```

Without the extra, the command exits with an install hint rather than a traceback. For more options, see the [install guide](install.md).

---

## Inspect SAM

All arguments accepted by `trackers inspect sam`.

<table>
  <colgroup>
    <col style="width: 40%">
    <col style="width: 40%">
    <col style="width: 20%">
  </colgroup>
  <thead>
    <tr>
      <th>Argument</th>
      <th>Description</th>
      <th>Default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--image_path</code></td>
      <td>Path to the input image.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--box</code></td>
      <td>Bounding boxes in <code>xyxy</code> format, given as one list: <code>--box='[[x1,y1,x2,y2]]'</code>. Append with <code>--box+=...</code>.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--output_path</code></td>
      <td>Path to save the visualization.</td>
      <td><code>outputs/inspect/sam/sam_masks.jpg</code></td>
    </tr>
    <tr>
      <td><code>--device</code></td>
      <td>Device used by SAM, e.g. <code>cpu</code> or <code>cuda</code>. <code>auto</code> resolves to CUDA when available, otherwise CPU.</td>
      <td><code>auto</code></td>
    </tr>
    <tr>
      <td><code>--model_type</code></td>
      <td>SAM model type.</td>
      <td><code>vit_b</code></td>
    </tr>
  </tbody>
</table>

---

## Inspect Cutie

All arguments accepted by `trackers inspect cutie`.

<table>
  <colgroup>
    <col style="width: 40%">
    <col style="width: 40%">
    <col style="width: 20%">
  </colgroup>
  <thead>
    <tr>
      <th>Argument</th>
      <th>Description</th>
      <th>Default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--image_dir</code></td>
      <td>Directory containing input frames.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--start_file</code></td>
      <td>First frame filename, included in the selected frame range.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--end_file</code></td>
      <td>Last frame filename, included in the selected frame range.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--box</code></td>
      <td>Bounding boxes on the first selected frame, <code>xyxy</code>, as one list. Append with <code>--box+=...</code>.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--add_at</code></td>
      <td>Masks to add via a box on a given frame, each <code>filename:x1,y1,x2,y2</code>. Applied on that frame then propagated to the next, matching McByte timing; each event is treated as a new object.</td>
      <td>none</td>
    </tr>
    <tr>
      <td><code>--remove_at</code></td>
      <td>Masks to remove before propagating to the given frame, each <code>filename:manual_mask_id</code>.</td>
      <td>none</td>
    </tr>
    <tr>
      <td><code>--output_root</code></td>
      <td>Root directory for timestamped outputs.</td>
      <td><code>outputs/inspect/cutie</code></td>
    </tr>
    <tr>
      <td><code>--device</code></td>
      <td>Device used by SAM and Cutie, e.g. <code>cpu</code>/<code>cuda</code>. <code>auto</code> resolves to CUDA when available, otherwise CPU.</td>
      <td><code>auto</code></td>
    </tr>
    <tr>
      <td><code>--sam_model_type</code></td>
      <td>SAM model type.</td>
      <td><code>vit_b</code></td>
    </tr>
    <tr>
      <td><code>--cutie_model_type</code></td>
      <td>Cutie model type.</td>
      <td><code>base-mega</code></td>
    </tr>
    <tr>
      <td><code>--cutie_config_path</code></td>
      <td>Optional path to Cutie's Hydra config directory.</td>
      <td>none</td>
    </tr>
    <tr>
      <td><code>--cutie_config_name</code></td>
      <td>Cutie Hydra config name.</td>
      <td><code>eval_config</code></td>
    </tr>
  </tbody>
</table>

---

## Inspect MaskManager

All arguments accepted by `trackers inspect mask-manager`. `--mode` selects `manual` or `gt`; passing an option that belongs to the other mode is an error.

<table>
  <colgroup>
    <col style="width: 40%">
    <col style="width: 40%">
    <col style="width: 20%">
  </colgroup>
  <thead>
    <tr>
      <th>Argument</th>
      <th>Description</th>
      <th>Default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--image_dir</code></td>
      <td>Directory holding the frame images.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--mode</code></td>
      <td><code>manual</code> (boxes and lifecycle events from the CLI) or <code>gt</code> (replay from a MOT-format ground-truth file).</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--start_file</code></td>
      <td><em>Manual mode.</em> Filename of the first frame to process.</td>
      <td>none</td>
    </tr>
    <tr>
      <td><code>--end_file</code></td>
      <td><em>Manual mode.</em> Filename of the last frame to process.</td>
      <td>none</td>
    </tr>
    <tr>
      <td><code>--box</code></td>
      <td><em>Manual mode.</em> Initial tracklet boxes on the first selected frame, <code>xyxy</code>, as one list. Append with <code>+</code>.</td>
      <td>none</td>
    </tr>
    <tr>
      <td><code>--add_at</code></td>
      <td><em>Manual mode.</em> New tracklets to add from given frame boxes, each <code>filename:x1,y1,x2,y2</code>.</td>
      <td>none</td>
    </tr>
    <tr>
      <td><code>--remove_at</code></td>
      <td><em>Manual mode.</em> Tracklets to remove before propagating to the given frame, each <code>filename:tracklet_id</code>.</td>
      <td>none</td>
    </tr>
    <tr>
      <td><code>--gt_file</code></td>
      <td><em>GT mode.</em> MOT-format ground-truth file to replay.</td>
      <td>none</td>
    </tr>
    <tr>
      <td><code>--start_frame</code></td>
      <td><em>GT mode.</em> First frame number to replay.</td>
      <td>none</td>
    </tr>
    <tr>
      <td><code>--end_frame</code></td>
      <td><em>GT mode.</em> Last frame number to replay.</td>
      <td>none</td>
    </tr>
    <tr>
      <td><code>--tracklet_id</code></td>
      <td><em>GT mode.</em> Tracklet IDs to replay, e.g. <code>--tracklet_id='[3,7]'</code>. Omit, or pass <code>[all]</code>, to replay every tracklet.</td>
      <td>all</td>
    </tr>
    <tr>
      <td><code>--output_root</code></td>
      <td>Directory holding run directories, grouped by mode (<code>&lt;output_root&gt;/&lt;mode&gt;/&lt;timestamp&gt;/</code>).</td>
      <td><code>outputs/inspect/mask-manager</code></td>
    </tr>
    <tr>
      <td><code>--device</code></td>
      <td>Device used by SAM and Cutie, e.g. <code>cuda</code>/<code>cpu</code>. <code>auto</code> resolves to CUDA when available, otherwise CPU.</td>
      <td><code>auto</code></td>
    </tr>
    <tr>
      <td><code>--sam_model_type</code></td>
      <td>SAM model type.</td>
      <td><code>vit_b</code></td>
    </tr>
    <tr>
      <td><code>--cutie_model_type</code></td>
      <td>Cutie model type.</td>
      <td><code>base-mega</code></td>
    </tr>
    <tr>
      <td><code>--cutie_config_path</code></td>
      <td>Directory holding the Cutie config. Omit to use the config shipped with the installed Cutie package.</td>
      <td>none</td>
    </tr>
    <tr>
      <td><code>--cutie_config_name</code></td>
      <td>Name of the Cutie config to load.</td>
      <td><code>eval_config</code></td>
    </tr>
    <tr>
      <td><code>--mask_creation_bbox_overlap_threshold</code></td>
      <td><em>GT mode.</em> Overlap threshold above which mask creation is delayed.</td>
      <td><code>0.6</code></td>
    </tr>
  </tbody>
</table>

---

## Inspect McByte

All arguments accepted by `trackers inspect mcbyte`. Options are grouped into nested dataclasses: `--sequence.*`, `--cmc.*`, `--mask.*`.

<table>
  <colgroup>
    <col style="width: 40%">
    <col style="width: 40%">
    <col style="width: 20%">
  </colgroup>
  <thead>
    <tr>
      <th>Argument</th>
      <th>Description</th>
      <th>Default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--sequence.image_dir</code></td>
      <td>Directory containing sequence frames.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--sequence.det_file</code></td>
      <td>Path to the detection file.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--sequence.start_frame</code></td>
      <td>First frame number to process, inclusive.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--sequence.end_frame</code></td>
      <td>Last frame number to process, inclusive.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--sequence.det_format</code></td>
      <td>Detection-file column format: <code>mot_tlwh</code> (<code>frame,id,left,top,width,height,confidence,...</code>) or <code>xyxy</code> (<code>frame,x1,y1,x2,y2,confidence</code>).</td>
      <td><code>mot_tlwh</code></td>
    </tr>
    <tr>
      <td><code>--sequence.frame_rate</code></td>
      <td>Sequence frame rate used to scale the lost-track buffer.</td>
      <td><code>30.0</code></td>
    </tr>
    <tr>
      <td><code>--cmc.enable</code></td>
      <td>Enable camera motion compensation in both runs.</td>
      <td><code>false</code></td>
    </tr>
    <tr>
      <td><code>--cmc.method</code></td>
      <td>Camera-motion compensation method. Options: <code>orb</code>, <code>sift</code>, <code>sparseOptFlow</code>, <code>ecc</code>.</td>
      <td><code>sparseOptFlow</code></td>
    </tr>
    <tr>
      <td><code>--cmc.downscale</code></td>
      <td>Image downscale factor used by CMC.</td>
      <td><code>6</code></td>
    </tr>
    <tr>
      <td><code>--mask.device</code></td>
      <td>Device used by SAM and Cutie in the mask-conditioned run. <code>auto</code> resolves to CUDA when available, otherwise CPU.</td>
      <td><code>auto</code></td>
    </tr>
    <tr>
      <td><code>--mask.enable_isolated_matching</code></td>
      <td>Allow mask evidence to rescue isolated positive-IoU pairs below the normal association threshold.</td>
      <td><code>false</code></td>
    </tr>
    <tr>
      <td><code>--modes</code></td>
      <td>Tracker configurations to run: <code>--modes=[locked_iou]</code>.</td>
      <td>both <code>locked_iou</code> and <code>mask_conditioned</code></td>
    </tr>
    <tr>
      <td><code>--output_dir</code></td>
      <td>Directory holding one timestamped run directory per invocation; both comparison runs write inside it.</td>
      <td><code>outputs/inspect/mcbyte</code></td>
    </tr>
  </tbody>
</table>
</content>
