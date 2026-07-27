---
title: McByte Tracker — Mask-Conditioned Multi-Object Tracking | Trackers
comments: true
description: McByte extends BoT-SORT-style association with temporally propagated SAM/Cutie segmentation masks as an additional matching cue, without per-video tuning.
---

# McByte

## Overview

McByte extends a [BoT-SORT](botsort.md)-style tracking-by-detection pipeline with an optional mask-conditioned association stage. Instead of relying only on IoU and detection confidence, McByte can use a temporally propagated segmentation mask per track as extra evidence when an IoU-based match is ambiguous. Clear, unambiguous matches are locked before mask evidence is considered, so masks only influence genuinely uncertain pairs (and, optionally, isolated low-IoU candidates). Because the mask evidence comes from general-purpose, pre-trained segmentation models, McByte requires no per-video or per-dataset tuning. Mask management is optional and disabled by default: without it, McByte behaves as a clear-match-locking, reduced-assignment variant of the ByteTrack/BoT-SORT association pipeline using IoU alone.

!!! note "Optional heavyweight dependencies"

    The default mask pipeline uses [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) for mask initialization and [Cutie](https://github.com/hkchengrex/Cutie) for temporal mask propagation. Both require `torch`/`torchvision` plus their own installation steps and are **not** installed by `pip install trackers`. `McByteTracker` can be constructed and used without either — set `enable_mask_manager=True` only after installing SAM and Cutie, or supply a custom `mask_manager` (for example a lightweight test double) instead.

## How does McByte compare to other trackers?

For comparisons with other trackers, plus dataset context and evaluation details, see the [tracker comparison](comparison.md) page.

The table below compares McByte (mask-conditioned association enabled) against BoT-SORT without re-identification — the baseline association pipeline McByte builds on — using default parameters for both trackers, with no dataset-specific tuning.

|  Dataset   | Tracker    |   HOTA   |   IDF1   |   MOTA   |
| :--------: | :--------- | :------: | :------: | :------: |
|   MOT17    | BoT-SORT   |   63.7   |   78.7   | **79.2** |
|   MOT17    | **McByte** | **64.1** | **79.7** |   79.1   |
| SportsMOT  | BoT-SORT   |   73.8   |   73.4   |   96.9   |
| SportsMOT  | **McByte** | **76.5** | **76.9** | **97.0** |
| SoccerNet  | BoT-SORT   |   84.5   |   79.3   |   96.6   |
| SoccerNet  | **McByte** | **85.0** | **79.9** | **97.0** |
| DanceTrack | BoT-SORT   |   57.8   |   57.9   |   92.2   |
| DanceTrack | **McByte** | **67.2** | **68.6** | **92.5** |

Source: [PR #513](https://github.com/roboflow/trackers/pull/513), reported by the McByte author against Trackers' BoT-SORT baseline.

## Algorithm

McByte keeps the same tracking-by-detection backbone as [BoT-SORT](botsort.md) — Kalman prediction, optional camera motion compensation (CMC), and multi-stage confidence-aware IoU association — and adds clear-match locking plus optional mask conditioning around the assignment step.

**Processing flow.** Detections → Kalman prediction → optional CMC → multi-stage IoU association → clear-match locking → mask-conditioned ambiguous association (if a mask manager is configured) → reduced linear assignment → tracklet lifecycle update.

**Clear-match locking.** Within each association stage, a track-detection pair whose similarity clears the stage threshold is locked immediately when it is the only eligible candidate in both its row and column. Locked pairs skip the Hungarian solver entirely, so mask evidence never has a chance to disturb matches that are already unambiguous.

**Mask conditioning.** The remaining, non-locked pairs form a reduced similarity matrix. When a `MaskManager` is configured and a propagated mask meets its confidence and overlap requirements (`minimum_mask_average_confidence`, `minimum_mask_coverage`, `minimum_mask_fill_ratio`), the mask evidence is added into that pair's similarity score before assignment. Optionally (`enable_isolated_mask_matching`), an isolated candidate with positive but below-threshold IoU can also be rescued by strong mask evidence. The Hungarian algorithm then solves the reduced assignment problem, and matches are mapped back to the original track and detection indices.

**Mask lifecycle.** With mask management enabled, masks for frame `t` are prepared *before* association on frame `t`, using the tracker state from frame `t-1`: `MaskManager` initializes masks for newly created tracklets with SAM and propagates existing masks forward with Cutie. Temporarily lost (but still alive) tracklets keep their masks; masks are only dropped once a tracklet is pruned. This means a frame is required both for CMC and for mask updates — when no frame is passed to `update()`, both are skipped and McByte behaves as pure clear-match-locking IoU association.

**Track lifecycle.** New tracks are created from unmatched high-confidence detections and confirmed after `minimum_consecutive_frames` consecutive matches. Unmatched unconfirmed tracks are removed; confirmed tracks unmatched for more than `lost_track_buffer` frames are deleted.

## Key Parameters

| Parameter                                 | Purpose                                                                                                                                  | Tuning guidance                                                                                                                                                                |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lost_track_buffer`                       | Frames to keep an unmatched track alive before deletion (specified in 30 FPS units, scaled proportionally by `frame_rate`).              | Higher tolerates longer occlusions/camera shake but can increase false re-association. 10-30 common; up to 60 for long gaps.                                                   |
| `track_activation_threshold`              | Minimum detection confidence required to start a new track.                                                                              | Higher reduces noisy track creation; lower retains harder objects. 0.5-0.9 typical depending on detector quality.                                                              |
| `minimum_consecutive_frames`              | Consecutive matches required before confirming a new track.                                                                              | 1 for immediate activation; 2-3 improves robustness against flicker and false positives.                                                                                       |
| `minimum_iou_threshold_first_assoc`       | Minimum association similarity for the first pass (high-confidence detections vs. confirmed and lost tracks).                            | Default `0.1` is intentionally lower than in BoT-SORT: it only rejects clearly implausible pairs, leaving a broader candidate set for mask-conditioned association to resolve. |
| `minimum_iou_threshold_second_assoc`      | Minimum association similarity for the second pass (low-confidence detections vs. remaining tracked tracks).                             | Usually set lower than the first-pass threshold to recover weak detections without over-matching.                                                                              |
| `minimum_iou_threshold_unconfirmed_assoc` | Minimum association similarity when associating unconfirmed tracks.                                                                      | Higher values make tentative tracks harder to confirm spuriously; lower values help short-lived or noisy objects survive.                                                      |
| `high_conf_det_threshold`                 | Confidence split between stage-1 and stage-2 detections.                                                                                 | 0.5-0.7 common. Higher shifts more detections to the recovery stage; lower gives stage-1 broader coverage.                                                                     |
| `enable_cmc`                              | Enables camera motion compensation before association.                                                                                   | Keep enabled for moving-camera footage (sports, drone, handheld). Disable mainly for static cameras if you need maximal speed.                                                 |
| `cmc_method`                              | Camera motion compensation method (see [BoT-SORT](botsort.md)).                                                                          | Default `sparseOptFlow` is a good general-purpose choice.                                                                                                                      |
| `enable_mask_manager`                     | Whether to construct McByte's default SAM + Cutie mask pipeline.                                                                         | Disabled by default so importing/using `McByteTracker` never requires the optional SAM/Cutie dependencies. Enable only after installing them, or pass a custom `mask_manager`. |
| `mask_config`                             | `McByteMaskConfig` used to build the default mask pipeline. Requires `enable_mask_manager=True`; mutually exclusive with `mask_manager`. | Adjust when you need a non-default SAM/Cutie checkpoint, model variant, or device.                                                                                             |
| `minimum_mask_average_confidence`         | Minimum average confidence a propagated mask must have before it can influence association.                                              | Raise to trust masks only when Cutie is confident; lower to let weaker masks still contribute.                                                                                 |
| `minimum_mask_coverage`                   | Minimum fraction of a tracklet's mask that must fall inside a candidate detection box.                                                   | Raise for stricter geometric consistency between mask and box; lower to tolerate partial occlusion.                                                                            |
| `minimum_mask_fill_ratio`                 | Minimum fraction of a candidate detection box's area that must be covered by the tracklet mask.                                          | Raise to avoid matching a small mask fragment to an oversized box; lower to allow slimmer objects to match larger boxes.                                                       |
| `enable_isolated_mask_matching`           | Whether mask evidence may rescue an isolated candidate with positive IoU below the normal stage threshold.                               | Off by default (conservative). Enable to recover more matches under heavy occlusion at the risk of more false positives.                                                       |

### Mask Pipeline Parameters (`McByteMaskConfig`)

`McByteMaskConfig` is only used when `McByteTracker` builds its default SAM + Cutie pipeline (`enable_mask_manager=True` and no custom `mask_manager` supplied).

| Parameter                                 | Purpose                                                                                                                  | Default                  |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------ |
| `device`                                  | Device shared by SAM and Cutie, e.g. `"cuda"`, `"cuda:0"`, or `"cpu"`.                                                   | `"cpu"`                  |
| `sam_checkpoint_path`                     | Optional SAM checkpoint path. When omitted, the default checkpoint for `sam_model_type` is downloaded automatically.     | `None`                   |
| `sam_model_type`                          | SAM model variant used for box-prompted mask generation.                                                                 | `"vit_b"`                |
| `cutie_weights_path`                      | Optional Cutie checkpoint path. When omitted, the default checkpoint for `cutie_model_type` is downloaded automatically. | `None`                   |
| `cutie_model_type`                        | Cutie model variant used for temporal mask propagation.                                                                  | `"base-mega"`            |
| `cutie_config_path` / `cutie_config_name` | Optional Cutie Hydra configuration directory / name.                                                                     | `None` / `"eval_config"` |
| `cutie_use_amp`                           | Whether Cutie may use automatic mixed precision (only activated on a CUDA device).                                       | `True`                   |
| `mask_creation_bbox_overlap_threshold`    | Bounding-box overlap fraction at or above which mask creation for a new tracklet is delayed.                             | `0.6`                    |

## Run on video

`McByteTracker` is not (yet) exported from the top-level `trackers` package — import it directly from its module. The example below runs McByte in its lightweight, mask-free configuration (clear-match locking with IoU only), which needs no extra dependencies beyond `trackers` itself.

```python
import cv2
import supervision as sv
from rfdetr import RFDETRMedium
from trackers.core.mcbyte.tracker import McByteTracker

tracker = McByteTracker()
model = RFDETRMedium()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

video_capture = cv2.VideoCapture("<SOURCE_VIDEO_PATH>")
if not video_capture.isOpened():
    raise RuntimeError("Failed to open video source")

while True:
    success, frame_bgr = video_capture.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = model.predict(frame_rgb)
    detections = tracker.update(detections, frame=frame_bgr)

    annotated_frame = box_annotator.annotate(frame_bgr, detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame,
        detections,
        labels=detections.tracker_id,
    )

    cv2.imshow("RF-DETR + McByte", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
```

!!! tip "Enabling full mask-conditioned association"

    After installing SAM and Cutie (see the [PR #513 installation notes](https://github.com/roboflow/trackers/pull/513)), construct the tracker with `McByteTracker(enable_mask_manager=True)` (optionally passing a `McByteMaskConfig`) to enable the full SAM/Cutie mask pipeline. Passing `frame=frame_bgr` to `update()` remains required for masks to propagate.

## Reference

Stanczyk, T., Yoon, S., and Bremond, F. (2025). No Train Yet Gain: Towards Generic Multi-Object Tracking in Sports and Beyond. [arXiv:2506.01373](https://arxiv.org/abs/2506.01373). Original implementation: [tstanczyk95/McByte](https://github.com/tstanczyk95/McByte).
