---
title: SORT vs ByteTrack vs OC-SORT vs BoT-SORT vs C-BIoU — MOT Benchmark Comparison | Trackers
description: Side-by-side benchmark comparison of SORT, ByteTrack, OC-SORT, BoT-SORT, and C-BIoU on MOT17, DanceTrack, SportsMOT, and SoccerNet — HOTA, IDF1, MOTA with default and tuned parameters.
---

# Tracker Comparison

This page shows head-to-head performance of SORT, ByteTrack, OC-SORT, BoT-SORT, and C-BIoU on standard MOT benchmarks. Results are shown with default parameters and with parameter-tuned configurations found via grid search.

!!! info "Benchmark version"

    Results use **trackers v2.3.0** (released 2026-03-16). Detections are from YOLOX (MOT17, SportsMOT, DanceTrack) or ground-truth oracle boxes (SoccerNet). Parameters were tuned via grid search on held-out splits. See [Methodology](#methodology) for details.

!!! note "Benchmark methodology"

    Results measured using YOLOX detections (MOT17, SportsMOT, DanceTrack) or oracle ground-truth boxes (SoccerNet) with default and grid-searched parameters. Performance varies across detectors — see [Detection Quality Matters](../learn/detection-quality.md) for the impact of detector quality on tracking metrics.

## [MOT17](https://arxiv.org/abs/1603.00831)

Pedestrian tracking with crowded scenes and frequent occlusions. Strongly tests re-identification and identity stability.

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/datasets/MOT17_MOT17-04-DPM-1280x720.mp4" type="video/mp4">
</video>
<p align="center" style="margin-top: -0.4em;"><small>Visualization of ground-truth annotations for MOT17.</small></p>

!!! info

    Parameters were tuned on the validation set. Results are reported on the
    test set via Codabench submission. Detections come from a YOLOX model.

=== "Default"

    Results using default tracker parameters.

    |  Tracker  |   HOTA   |   IDF1   |   MOTA   |
    | :-------: | :------: | :------: | :------: |
    |   SORT    |   58.4   |   69.9   |   67.2   |
    | ByteTrack |   60.1   |   73.2   |   74.1   |
    |  OC-SORT  |   61.9   |   76.4   |   76.0   |
    | BoT-SORT  | **63.7** |   78.7   | **79.2** |
    |  C-BIoU   |   63.0   | **79.1** |   77.4   |

=== "Tuned"

    Results after grid search over tracker parameters.

    |  Tracker  |   HOTA   |   IDF1   |   MOTA   |
    | :-------: | :------: | :------: | :------: |
    |   SORT    |   60.4   |   72.5   |   75.8   |
    | ByteTrack |   60.5   |   72.7   |   76.1   |
    |  OC-SORT  |   62.0   |   76.5   |   77.3   |
    | BoT-SORT  | **63.8** |   78.7   | **79.4** |
    |  C-BIoU   |   63.0   | **79.1** |   77.4   |

    Tuned configuration for each tracker.

    ```yaml
    SORT:
      lost_track_buffer: 10
      track_activation_threshold: 0.75
      minimum_consecutive_frames: 2
      minimum_iou_threshold: 0.3

    ByteTrack:
      lost_track_buffer: 10
      track_activation_threshold: 0.7
      minimum_consecutive_frames: 1
      minimum_iou_threshold: 0.3
      high_conf_det_threshold: 0.5

    OC-SORT:
      lost_track_buffer: 30
      minimum_iou_threshold: 0.3
      minimum_consecutive_frames: 3
      direction_consistency_weight: 0.2
      high_conf_det_threshold: 0.4
      delta_t: 1

    BoT-SORT:
      lost_track_buffer: 30
      minimum_consecutive_frames: 2
      minimum_iou_threshold_first_assoc: 0.2
      minimum_iou_threshold_second_assoc: 0.5
      minimum_iou_threshold_unconfirmed_assoc: 0.2
      high_conf_det_threshold: 0.5
      track_activation_threshold: 0.6
      enable_cmc: true
      cmc_method: sparseOptFlow

    C-BIoU:
      lost_track_buffer: 30
      minimum_consecutive_frames: 2
      minimum_iou_threshold_first_assoc: 0.2
      minimum_iou_threshold_second_assoc: 0.5
      minimum_iou_threshold_unconfirmed_assoc: 0.3
      high_conf_det_threshold: 0.6
      track_activation_threshold: 0.7
      buffer_ratio_first: 0.3
      buffer_ratio_second: 0.5
    ```

## [SportsMOT](https://arxiv.org/abs/2304.05170)

Sports broadcast tracking with fast motion, camera pans, and similar-looking targets. Tests association under speed and appearance ambiguity.

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/datasets/SportsMOT_v_-6Os86HzwCs_c001-1280x720.mp4" type="video/mp4">
</video>
<p align="center" style="margin-top: -0.4em;"><small>Visualization of ground-truth annotations for SportsMOT.</small></p>

!!! info

    Parameters were tuned on the validation set. Results are reported on the
    test set via Codabench submission. Detections come from a YOLOX model.

=== "Default"

    Results using default tracker parameters.

    |  Tracker  |   HOTA   |   IDF1   |   MOTA   |
    | :-------: | :------: | :------: | :------: |
    |   SORT    |   70.8   |   68.9   |   95.5   |
    | ByteTrack |   73.0   |   72.5   |   96.4   |
    |  OC-SORT  |   71.7   |   71.4   |   95.0   |
    | BoT-SORT  | **73.8** | **73.4** | **96.9** |
    |  C-BIoU   |   73.1   |   72.6   |   96.7   |

=== "Tuned"

    Results after grid search over tracker parameters.

    |  Tracker  |   HOTA   |   IDF1   |   MOTA   |
    | :-------: | :------: | :------: | :------: |
    |   SORT    |   72.9   |   73.0   |   95.8   |
    | ByteTrack |   73.3   |   73.5   |   95.9   |
    |  OC-SORT  |   74.0   | **75.4** |   95.6   |
    | BoT-SORT  | **74.1** |   74.0   | **96.9** |
    |  C-BIoU   |   73.1   |   72.6   |   96.7   |

    Tuned configuration for each tracker.

    ```yaml
    SORT:
      lost_track_buffer: 60
      track_activation_threshold: 0.9
      minimum_consecutive_frames: 2
      minimum_iou_threshold: 0.05

    ByteTrack:
      lost_track_buffer: 10
      track_activation_threshold: 0.9
      minimum_consecutive_frames: 1
      minimum_iou_threshold: 0.05
      high_conf_det_threshold: 0.7

    OC-SORT:
      lost_track_buffer: 60
      minimum_iou_threshold: 0.1
      minimum_consecutive_frames: 3
      direction_consistency_weight: 0.2
      high_conf_det_threshold: 0.6
      delta_t: 3

    BoT-SORT:
      lost_track_buffer: 30
      minimum_consecutive_frames: 2
      minimum_iou_threshold_first_assoc: 0.1
      minimum_iou_threshold_second_assoc: 0.5
      minimum_iou_threshold_unconfirmed_assoc: 0.3
      high_conf_det_threshold: 0.7
      track_activation_threshold: 0.8
      enable_cmc: true
      cmc_method: sparseOptFlow

    C-BIoU:
      lost_track_buffer: 30
      minimum_consecutive_frames: 2
      minimum_iou_threshold_first_assoc: 0.2
      minimum_iou_threshold_second_assoc: 0.5
      minimum_iou_threshold_unconfirmed_assoc: 0.3
      high_conf_det_threshold: 0.6
      track_activation_threshold: 0.7
      buffer_ratio_first: 0.3
      buffer_ratio_second: 0.5
    ```

## [SoccerNet-tracking](https://arxiv.org/abs/2204.06918)

Long sequences with dense interactions and partial occlusions. Tests long-term ID consistency.

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/datasets/SoccerNet-tracking_SNMOT-060-1280x720.mp4" type="video/mp4">
</video>
<p align="center" style="margin-top: -0.4em;"><small>Visualization of ground-truth annotations for SoccerNet.</small></p>

!!! info

    Parameters were tuned on the train set. Results are reported on the test
    set. SoccerNet-tracking has no validation split. This dataset provides
    oracle (ground-truth) detections.

=== "Default"

    Results using default tracker parameters.

    |  Tracker  |   HOTA   |   IDF1   |   MOTA   |
    | :-------: | :------: | :------: | :------: |
    |   SORT    |   81.6   |   76.2   |   95.1   |
    | ByteTrack |   84.0   |   78.1   | **97.8** |
    |  OC-SORT  |   78.4   |   72.6   |   94.1   |
    | BoT-SORT  | **84.5** | **79.3** |   96.6   |
    |  C-BIoU   |   82.6   |   76.6   |   97.0   |

=== "Tuned"

    Results after grid search over tracker parameters.

    |  Tracker  |   HOTA   |   IDF1   |   MOTA   |
    | :-------: | :------: | :------: | :------: |
    |   SORT    |   84.2   |   78.2   |   98.2   |
    | ByteTrack |   84.0   |   78.1   |   98.2   |
    |  OC-SORT  |   82.9   |   77.9   |   96.8   |
    | BoT-SORT  |   85.0   |   79.7   |   97.2   |
    |  C-BIoU   | **85.7** | **80.0** | **99.3** |

    Tuned configuration for each tracker.

    ```yaml
    SORT:
      lost_track_buffer: 30
      track_activation_threshold: 0.25
      minimum_consecutive_frames: 2
      minimum_iou_threshold: 0.05

    ByteTrack:
      lost_track_buffer: 30
      track_activation_threshold: 0.2
      minimum_consecutive_frames: 1
      minimum_iou_threshold: 0.05
      high_conf_det_threshold: 0.5

    OC-SORT:
      lost_track_buffer: 60
      minimum_iou_threshold: 0.1
      minimum_consecutive_frames: 3
      direction_consistency_weight: 0.2
      high_conf_det_threshold: 0.4
      delta_t: 1

    BoT-SORT:
      lost_track_buffer: 60
      minimum_consecutive_frames: 2
      minimum_iou_threshold_first_assoc: 0.1
      minimum_iou_threshold_second_assoc: 0.6
      minimum_iou_threshold_unconfirmed_assoc: 0.2
      high_conf_det_threshold: 0.6
      track_activation_threshold: 0.7
      enable_cmc: true
      cmc_method: sparseOptFlow

    C-BIoU:
      lost_track_buffer: 43
      minimum_consecutive_frames: 2
      minimum_iou_threshold_first_assoc: 0.05
      minimum_iou_threshold_second_assoc: 0.46
      minimum_iou_threshold_unconfirmed_assoc: 0.27
      high_conf_det_threshold: 0.40
      track_activation_threshold: 0.48
      buffer_ratio_first: 0.68
      buffer_ratio_second: 0.50
    ```

!!! note "SoccerNet buffer ordering exception"

    This config uses `buffer_ratio_first: 0.68 > buffer_ratio_second: 0.50`, which reverses
    the general `b1 < b2` recommendation in the [C-BIoU docs](cbiou.md#buffer-ordering).
    Optuna found this ordering yields higher HOTA on SoccerNet's dense, long-sequence scenarios.
    On most other datasets the `b1 < b2` default applies.

## [DanceTrack](https://arxiv.org/abs/2111.14690)

Group dancing tracking with uniform appearance, diverse motions, and extreme articulation. Tests motion-based association without relying on visual discrimination.

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/datasets/DanceTrack_dancetrack0052-1280x720.mp4" type="video/mp4">
</video>
<p align="center" style="margin-top: -0.4em;"><small>Visualization of ground-truth annotations for DanceTrack.</small></p>

!!! info

    Parameters were tuned on the validation set. Results are reported on the
    test set via [Codabench](https://www.codabench.org/competitions/14885/) submission.
    Detections come from a YOLOX model.

=== "Default"

    Results using default tracker parameters.

    |  Tracker  |   HOTA   |   IDF1   |   MOTA   |
    | :-------: | :------: | :------: | :------: |
    |   SORT    |   47.2   |   41.0   |   86.5   |
    | ByteTrack |   53.3   |   53.6   |   90.3   |
    |  OC-SORT  |   54.1   |   53.3   |   89.3   |
    | BoT-SORT  | **57.8** | **57.9** | **92.2** |
    |  C-BIoU   |   56.7   |   56.7   | **92.2** |

=== "Tuned"

    Hyperparameter tuning, reporting the best tuned configuration per
    tracker evaluated on the test set (tuning performed on the valid split;
    if tuning did not outperform registry defaults, defaults are shown).

    |  Tracker  |   HOTA   |   IDF1   |   MOTA   |
    | :-------: | :------: | :------: | :------: |
    |   SORT    |   54.3   |   53.4   |   89.5   |
    | ByteTrack |   55.3   |   55.2   |   89.9   |
    |  OC-SORT  |   54.1   |   53.3   |   89.3   |
    | BoT-SORT  | **57.8** |   57.9   |   92.2   |
    |  C-BIoU   |   57.7   | **58.7** | **92.4** |

    Best configuration for each tracker.

    ```yaml
    SORT:
      lost_track_buffer: 91
      track_activation_threshold: 0.89
      minimum_consecutive_frames: 3
      minimum_iou_threshold: 0.21

    ByteTrack:
      lost_track_buffer: 76
      track_activation_threshold: 0.9
      minimum_consecutive_frames: 4
      minimum_iou_threshold: 0.33
      high_conf_det_threshold: 0.52

    OC-SORT:
      lost_track_buffer: 30
      minimum_iou_threshold: 0.3
      minimum_consecutive_frames: 3
      direction_consistency_weight: 0.2
      high_conf_det_threshold: 0.6
      delta_t: 3

    BoT-SORT:
      lost_track_buffer: 30
      minimum_consecutive_frames: 2
      minimum_iou_threshold_first_assoc: 0.2
      minimum_iou_threshold_second_assoc: 0.5
      minimum_iou_threshold_unconfirmed_assoc: 0.3
      high_conf_det_threshold: 0.6
      track_activation_threshold: 0.7
      enable_cmc: true
      cmc_method: sparseOptFlow

    C-BIoU:
      lost_track_buffer: 37
      track_activation_threshold: 0.71
      minimum_consecutive_frames: 3
      minimum_iou_threshold_first_assoc: 0.22
      minimum_iou_threshold_second_assoc: 0.70
      minimum_iou_threshold_unconfirmed_assoc: 0.26
      high_conf_det_threshold: 0.34
      buffer_ratio_first: 0.12
      buffer_ratio_second: 0.10
    ```

!!! note "DanceTrack buffer ordering exception"

    This config uses `buffer_ratio_first: 0.12 > buffer_ratio_second: 0.10`, which reverses
    the general `b1 < b2` recommendation in the [C-BIoU docs](cbiou.md#buffer-ordering).
    Optuna found this ordering on DanceTrack's validation split; the margin (0.02) is small
    and the `b1 < b2` default applies on most other datasets.

## Methodology

### Detections

Each dataset uses one of two detection sources: oracle detections (ground-truth
bounding boxes provided by the dataset) or model detections (produced by a YOLOX
detector following the ByteTrack procedure). The source is noted per dataset above.

### Tuning

Best parameters per tracker and dataset were found via grid search (SORT, ByteTrack,
OC-SORT, BoT-SORT) or Optuna (`n_trials=100`, objective HOTA, trial 0 = defaults for
C-BIoU), selecting the configuration with the highest HOTA on the tune split. Tuning and
evaluation always use separate data splits to reflect real-world usage:

- Train + validation + test: tune on validation, report on test.
- Train + validation: tune on train, report on validation.
- Train + test: tune on train, report on test.

## When to Use Each Tracker

**SORT** is the right choice when speed is the primary constraint and scenes are not heavily
occluded. Its Kalman filter plus Hungarian matching runs at hundreds of frames per second and
produces clean, easy-to-debug results. Use SORT as a baseline before adding more complex
trackers, or when deploying on edge devices with tight compute budgets.

**ByteTrack** is the default recommendation for most applications. It outperforms SORT on all
four benchmarks by recovering low-confidence detections that SORT discards. The two-stage
association adds almost no extra compute and consistently reduces missed tracks and identity
switches. Use ByteTrack when your detector produces noisy or variable-confidence outputs —
sports video, aerial footage, and crowded retail scenes all benefit.

**OC-SORT** is best when camera motion is significant or objects follow non-linear paths. Its
observation-centric re-update mechanism and direction consistency cost reduce drift from the
linear motion assumption. Use OC-SORT when SORT or ByteTrack loses tracks on fast turns,
camera pans, or erratic motion — the benchmark edge on MOT17 reflects exactly
these conditions.

**BoT-SORT** is the choice when camera ego-motion is strong and you need the most stable
identities. It extends ByteTrack with camera motion compensation (CMC) and confidence-aware
association, which reduces ID switches on panning or handheld footage. Use BoT-SORT for sports
broadcasts, drone video, or any scene where the camera moves frequently. The CMC overhead is
small relative to the detector, so the trade-off favors identity stability over raw speed.

**C-BIoU** targets fast or irregular motion when you want buffered, cascaded geometric matching without camera motion compensation. In these benchmarks it leads on SoccerNet, reaches the highest tuned IDF1 and MOTA on DanceTrack, and achieves the highest IDF1 on MOT17 among the trackers listed here. Use C-BIoU when BoT-SORT-style association is a good fit but CMC is unavailable or harmful, or when plain IoU matching is too strict. See [C-BIoU](cbiou.md) for buffer scales **b1** and **b2**.

## Metric Definitions

**HOTA** (Higher Order Tracking Accuracy) — the primary benchmark metric. HOTA decomposes
tracking quality into detection accuracy (DetA) and association accuracy (AssA), then takes
their geometric mean. It weights identity consistency equally with detection recall and
precision, unlike older metrics that under-penalize fragmented tracks. Higher HOTA indicates
both good detection and stable long-term identity.

**IDF1** (Identity F1) — measures how long the system correctly identifies each ground-truth
object over its lifetime. IDF1 is the harmonic mean of identification precision and
identification recall. High IDF1 means tracks stay on the correct identity; low IDF1 means
frequent identity switches.

**MOTA** (Multiple Object Tracking Accuracy) — combines the count of false positives, missed
detections, and identity switches into a single score relative to the total number of
ground-truth objects. MOTA is dominated by detection recall and precision; a detector with
near-perfect recall produces high MOTA even when identity switches are frequent.
