---
title: McByte Benchmark Runner — MOT17, SportsMOT, DanceTrack, SoccerNet | Trackers
description: Run McByte over complete MOT benchmark test sets and write MOTChallenge-format results per sequence with the trackers mcbyte command.
---

# McByte Benchmarks

Run McByte over a complete benchmark test set — MOT17, DanceTrack, SportsMOT, or SoccerNet-tracking — and write one MOTChallenge-format result file per sequence.

**What you'll learn:**

- Point the command at your detection and frame directories
- Select one or more datasets to run
- Read the output layout, including the MOT17 submission files

---

## Install

Get started by installing the package.

```text
pip install trackers
```

`trackers mcbyte` always builds McByte's full SAM + Cutie mask pipeline (`enable_mask_manager=True`), so — unlike the default `McByteTracker()` construction described on the [McByte page](../trackers/mcbyte.md) — both SAM and Cutie must be installed before running this command. See the [optional heavyweight dependencies note](../trackers/mcbyte.md#overview) for install steps. For more general install options, see the [install guide](install.md).

---

## Supported Datasets

| Dataset | Detection format | Layout note |
| ------------ | ---------------- | ----------------------------------------------------------- |
| `mot17` | `xyxy` | Frame directories use the `<sequence>-FRCNN` suffix. |
| `dancetrack` | `xyxy` | — |
| `sportsmot` | `xyxy` | — |
| `soccernet` | `mot` | Detection filenames follow the SoccerNet naming convention. |

Each sequence is processed independently with a fresh McByte tracker. If a sequence fails, the error is logged and the run continues with the remaining sequences.

---

## Configure Dataset Roots

Every dataset needs a `detection_root` (one detection file per sequence) and an `image_root` (one frame directory per sequence). Neither has a built-in value — supply both per run, either through a `--config` file or inline as JSON with `--dataset_roots`.

=== "CLI"

    Supply roots as JSON on the command line.

    ```text
    trackers mcbyte --dataset_roots='{"mot17": {"detection_root": "/data/dets", "image_root": "/data/frames"}}'
    ```

=== "Config file"

    A config file is the readable spelling.

    ```yaml
    # run.yaml
    dataset: [mot17]
    dataset_roots:
      mot17:
        detection_root: /data/detections/MOT17/test
        image_root: /data/datasets/MOT17/test
    ```

    ```text
    trackers mcbyte --config run.yaml
    ```

---

## Select Datasets

Pass `--dataset` as a list to choose which datasets to run. Omit it to run every dataset in the table above.

```text
trackers mcbyte --dataset=[mot17,soccernet] --device=cuda
```

A bare repeated `--dataset` overwrites the previous value — use `--dataset+` to append instead:

```text
trackers mcbyte --dataset+ mot17 --dataset+ soccernet
```

As with every `trackers` subcommand, hyphens and underscores are interchangeable (`--cmc-downscale` and `--cmc_downscale` are the same option), and every boolean flag has a `--no_` negation, e.g. `--no_enable_cmc`.

---

## Expected Directory Layout

`detection_root` holds one detection `.txt` file per sequence:

```text
detections/MOT17/test/MOT17-01.txt
detections/dancetrack/test/dancetrack0003.txt
detections/sportsmot/test/v_-9kabh1K8UA_c008.txt
detections/SoccerNet_tracking_2022_test_set_dets/SNMOT-116__det.txt
```

`image_root` holds one directory per sequence, each with an `img1` subdirectory of frames, for every dataset:

```text
datasets/dancetrack/test/
    dancetrack0003/img1/
    dancetrack0009/img1/
    ...
```

MOT17, DanceTrack and SportsMOT detections use `frame,x1,y1,x2,y2,confidence` (XYXY). SoccerNet-tracking detections use the original ground-truth MOT layout, `frame,id,left,top,width,height,confidence,...` — the identity column is ignored, since tracker identities are produced by McByte.

---

## Output

Results are written under `--output_root`, in one timestamped directory per run (`<timestamp>__isolation` or `<timestamp>__no_isolation`, reflecting `--enable_isolated_mask_matching`), with one subdirectory per dataset and a `run.log` capturing progress and failures.

```text
outputs/mcbyte_benchmarks/
└── 20260807_120000__no_isolation/
    ├── run.log
    └── mot17/
        ├── raw/
        │   ├── MOT17-01.txt
        │   └── ...
        └── submission/
            ├── MOT17-01-FRCNN.txt
            ├── MOT17-01-SDP.txt
            ├── MOT17-01-DPM.txt
            └── ...
```

Each sequence result is first written to a `.partial` file and only replaces the final file on success; use `--skip_existing` to skip a sequence whose result file is already present, and `--keep_partial_results` to keep a failed sequence's `.partial` file instead of deleting it.

### MOT17 submission files

The MOT17 evaluation server expects one result file per detector name (`FRCNN`, `SDP`, `DPM`). Since McByte is detector-agnostic, `trackers mcbyte` duplicates each completed sequence's result across all three suffixes under `mot17/submission/`. The remaining MOT17 sequence numbers this run produces no result for are written as empty placeholder files for all three suffixes, so the submission directory always contains the complete set of names.

---

## CLI Reference

All arguments accepted by `trackers mcbyte`.

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
      <td><code>--config</code></td>
      <td>Path to a configuration file.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--print_config</code></td>
      <td>Print the configuration after applying all other arguments, then exit.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--dataset</code></td>
      <td>Datasets to run, as a list: <code>--dataset=[mot17,soccernet]</code>. Repeat as <code>--dataset+</code> to append instead of overwrite.</td>
      <td>all datasets</td>
    </tr>
    <tr>
      <td><code>--dataset_roots</code></td>
      <td>Where each dataset's files live, keyed by the same names <code>--dataset</code> selects. Each entry holds a <code>detection_root</code> and an <code>image_root</code>; neither has a built-in value.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--device</code></td>
      <td>Device for SAM + Cutie, e.g. <code>cuda</code>, <code>cpu</code>, or <code>mps</code>. <code>auto</code> resolves to CUDA when available, otherwise CPU; MPS is never auto-selected and must be requested explicitly.</td>
      <td><code>auto</code></td>
    </tr>
    <tr>
      <td><code>--enable_isolated_mask_matching</code></td>
      <td>Match masks in isolation. Negate with <code>--no_enable_isolated_mask_matching</code>.</td>
      <td><code>false</code></td>
    </tr>
    <tr>
      <td><code>--output_root</code></td>
      <td>Directory holding one timestamped run directory per run.</td>
      <td><code>outputs/mcbyte_benchmarks</code></td>
    </tr>
    <tr>
      <td><code>--skip_existing</code></td>
      <td>Skip a sequence whose result file is already present. Negate with <code>--no_skip_existing</code>.</td>
      <td><code>false</code></td>
    </tr>
    <tr>
      <td><code>--enable_cmc</code></td>
      <td>Compensate for camera motion. Negate with <code>--no_enable_cmc</code>.</td>
      <td><code>true</code></td>
    </tr>
    <tr>
      <td><code>--cmc_method</code></td>
      <td>Camera-motion compensation method. Options: <code>orb</code>, <code>sift</code>, <code>sparseOptFlow</code>, <code>ecc</code>.</td>
      <td><code>sparseOptFlow</code></td>
    </tr>
    <tr>
      <td><code>--cmc_downscale</code></td>
      <td>Frame downscale factor applied before compensation.</td>
      <td><code>6</code></td>
    </tr>
    <tr>
      <td><code>--keep_partial_results</code></td>
      <td>Keep the <code>.partial</code> file a failed sequence leaves behind instead of deleting it. Negate with <code>--no_keep_partial_results</code>.</td>
      <td><code>false</code></td>
    </tr>
  </tbody>
</table>
