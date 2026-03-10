# Tracker Comparison

This page shows head-to-head performance of SORT, ByteTrack, and OC-SORT on standard MOT benchmarks. Results are shown with default parameters and with parameter-tuned configurations found via grid search.

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
    |  OC-SORT  | **61.9** | **76.4** | **76.0** |

=== "Tuned"

    Results after grid search over tracker parameters.

    |  Tracker  |   HOTA   |   IDF1   |   MOTA   |
    | :-------: | :------: | :------: | :------: |
    |   SORT    |   60.4   |   72.5   |   75.8   |
    | ByteTrack |   60.5   |   72.7   |   76.1   |
    |  OC-SORT  | **62.0** | **76.5** | **77.3** |

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
    |   SORT    |   70.9   |   68.9   |   95.7   |
    | ByteTrack | **73.0** | **72.5** | **96.4** |
    |  OC-SORT  |   71.7   |   71.4   |   95.0   |

=== "Tuned"

    Results after grid search over tracker parameters.

    |  Tracker  |   HOTA   |   IDF1   |   MOTA   |
    | :-------: | :------: | :------: | :------: |
    |   SORT    |   72.9   |   73.0   |   95.8   |
    | ByteTrack |   73.3   |   73.5   | **95.9** |
    |  OC-SORT  | **74.0** | **75.4** |   95.6   |

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
    | ByteTrack | **84.0** | **78.1** | **97.8** |
    |  OC-SORT  |   78.4   |   72.6   |   94.1   |

=== "Tuned"

    Results after grid search over tracker parameters.

    |  Tracker  |   HOTA   |   IDF1   |   MOTA   |
    | :-------: | :------: | :------: | :------: |
    |   SORT    | **84.2** | **78.2** | **98.2** |
    | ByteTrack |   84.0   |   78.1   |   97.8   |
    |  OC-SORT  |   82.9   |   77.9   |   96.8   |

    Tuned configuration for each tracker.

    ```yaml
    SORT:
      lost_track_buffer: 30
      track_activation_threshold: 0.25
      minimum_consecutive_frames: 2
      minimum_iou_threshold: 0.05

    ByteTrack:
      lost_track_buffer: 30
      track_activation_threshold: 0.5
      minimum_consecutive_frames: 2
      minimum_iou_threshold: 0.1
      high_conf_det_threshold: 0.5

    OC-SORT:
      lost_track_buffer: 60
      minimum_iou_threshold: 0.1
      minimum_consecutive_frames: 3
      direction_consistency_weight: 0.2
      high_conf_det_threshold: 0.4
      delta_t: 1
    ```

## [DanceTrack](https://arxiv.org/abs/2111.14690)

Group dancing tracking with uniform appearance, diverse motions, and extreme articulation. Tests motion-based association without relying on visual discrimination.

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/datasets/DanceTrack_dancetrack0052-1280x720.mp4" type="video/mp4">
</video>
<p align="center" style="margin-top: -0.4em;"><small>Visualization of ground-truth annotations for DanceTrack.</small></p>

!!! warning

    DanceTrack test set evaluation is currently unavailable because CodaLab, which hosted
    the benchmark, has been [discontinued](https://docs.codabench.org/dev/Newsletters_Archive/CodaLab-in-2025/).
    Migration to Codabench is [in progress](https://github.com/DanceTrack/DanceTrack/issues/42).
    Results below use the validation set instead.

!!! info

    Parameters were tuned on the train set. Results are reported on the
    validation set. This dataset provides oracle (ground-truth) detections.

=== "Default"

    Results using default tracker parameters.

    |  Tracker  |   HOTA   |   IDF1   |   MOTA   |
    | :-------: | :------: | :------: | :------: |
    |   SORT    |   45.0   |   39.0   |   80.6   |
    | ByteTrack |   50.2   |   49.9   |   86.2   |
    |  OC-SORT  | **51.8** | **50.9** | **87.3** |

=== "Tuned"

    Results after grid search over tracker parameters.

    |  Tracker  |   HOTA   |   IDF1   |   MOTA   |
    | :-------: | :------: | :------: | :------: |
    |   SORT    |   50.6   |   49.6   |   84.3   |
    | ByteTrack | **53.2** | **54.6** |   86.8   |
    |  OC-SORT  |   52.0   |   51.8   | **87.2** |

    Tuned configuration for each tracker.

    ```yaml
    SORT:
      lost_track_buffer: 10
      track_activation_threshold: 0.9
      minimum_consecutive_frames: 2
      minimum_iou_threshold: 0.05

    ByteTrack:
      lost_track_buffer: 60
      track_activation_threshold: 0.9
      minimum_consecutive_frames: 1
      minimum_iou_threshold: 0.1
      high_conf_det_threshold: 0.5

    OC-SORT:
      lost_track_buffer: 30
      minimum_iou_threshold: 0.1
      minimum_consecutive_frames: 3
      direction_consistency_weight: 0.2
      high_conf_det_threshold: 0.6
      delta_t: 1
    ```

## Methodology

### Detections

Each dataset uses one of two detection sources: oracle detections (ground-truth
bounding boxes provided by the dataset) or model detections (produced by a YOLOX
detector following the ByteTrack procedure). The source is noted per dataset above.

### Tuning

Best parameters per tracker and dataset were found via grid search, selecting the
configuration with the highest HOTA. Tuning and evaluation always use separate data
splits to reflect real-world usage:

- Train + validation + test: tune on validation, report on test.
- Train + validation: tune on train, report on validation.
- Train + test: tune on train, report on test.
