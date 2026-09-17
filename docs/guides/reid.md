---
title: ReID Appearance
description: Use ReID appearance association with BoT-SORT in Roboflow Trackers, from model loading to appearance threshold selection, with MOT17 and SoccerNet results.
---

# ReID Appearance

BoT-SORT can fuse appearance embeddings with IoU during association. Embeddings come from a model in the standalone [`reid`](https://github.com/roboflow/re-ID) package. See the [ReID API](../api/reid.md) for the association helpers.

**What you'll learn:**

- How to enable appearance association on BoT-SORT
- Which parameters control the appearance gate
- How to pick `reid_appearance_threshold` for your encoder and domain
- What ReID changes on MOT17, SportsMOT, DanceTrack and SoccerNet, with a generic and a fine-tuned encoder

---

## Install

```bash
pip install "trackers[reid]"
```

For extra contents and other options, see the [install guide](install.md).

---

## Quickstart

=== "Python"

    ```python
    from reid import ReIDModel

    from trackers import BoTSORTTracker

    reid_model = ReIDModel.from_pretrained("fastreid_mot17_sbs50")
    tracker = BoTSORTTracker(reid_model=reid_model, reid_appearance_threshold=0.2)
    ```

=== "CLI"

    `--reid.model` takes a curated alias, an `hf://` URL, or a local path, and enables appearance association on its own. Tracker parameters keep their Python names, so the appearance gate is `--tracker.reid_appearance_threshold`. See the [CLI reference](cli.md) for every argument.

    ```bash
    trackers track \
        --source <SOURCE_VIDEO_PATH> \
        --tracker botsort \
        --reid.model fastreid_mot17_sbs50 \
        --tracker.reid_appearance_threshold 0.2 \
        --output.video output.mp4
    ```

!!! warning "A frame is required when ReID is enabled"

    Pass the current video frame as `tracker.update(detections, frame=frame_bgr)`. When `reid_model` is set, `update()` raises if `frame` is omitted.

A guide to the model catalog and fine-tuning in `reid` is coming soon.

---

## Key Parameters

|          Parameter          |                                                                        Purpose                                                                        |                                                                                                                                   Tuning guidance                                                                                                                                    |
| :-------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|        `reid_model`         |                                                    Appearance encoder queried during association.                                                     |                                                                                          Leave unset for IoU and CMC only. Pick a checkpoint trained on your object domain where possible.                                                                                           |
|      `reid_ema_alpha`       |                                                    EMA momentum for a track's appearance feature.                                                     |                                                                                   Default 0.9. Higher keeps a stable long-term identity; lower adapts faster to appearance change but drifts more.                                                                                   |
| `reid_appearance_threshold` |                                  Maximum appearance distance `d_app` for appearance to lower a pair's matching cost.                                  |                                                                                                      BoT-SORT paper default 0.25. Calibrate per encoder and domain, see below.                                                                                                       |
| `reid_proximity_threshold`  |                  IoU gate applied before appearance (`IoU ≥ 1 - reid_proximity_threshold`), from true IoU even with GIoU/DIoU/CIoU.                   |                                                                                  Default 0.5. Raise to 1.0 where targets leave the frame and return, see [below](#choosing-a-proximity-threshold).                                                                                   |
|        `reid_fusion`        |         How appearance combines with geometry: `"botsort"` takes the minimum of the two costs, `"adaptive"` adds a weighted appearance term.          |                                                                                           Default `"botsort"`. See [choosing a fusion method](#choosing-a-fusion-method) before switching.                                                                                           |
|  `reid_appearance_weight`   |                                       Base appearance weight when `reid_fusion="adaptive"`. Ignored otherwise.                                        |                                                                                                             Default 0.75. Raise where geometry is unreliable, see below.                                                                                                             |
| `reid_adaptive_weight_cap`  |                                    Ceiling on the adaptive bonus when `reid_fusion="adaptive"`. Ignored otherwise.                                    |                                                                                                              Default 0.5. Raise together with `reid_appearance_weight`.                                                                                                              |
|   `reid_appearance_floor`   | Minimum cosine similarity for appearance to contribute when `reid_fusion="adaptive"`; below it a pair is scored on geometry alone. Ignored otherwise. | Default 0.0 (off, as in the Deep OC-SORT fusion). Calibrate per encoder: 0.7 with `reid_proximity_threshold=1.0` is the value for `osnet_x1_0` fine-tuned on SoccerNet and loses HOTA with the MOT17 and MSMT17 encoders, see [choosing a fusion method](#choosing-a-fusion-method). |

---

## Choosing an encoder

The encoder decides how much appearance can help, and every threshold below depends on it. There are two kinds:

- **A generic encoder**, trained on another dataset, such as `osnet_x1_0_msmt17_combineall` (the default) or `fastreid_mot17_sbs50`. It works out of the box, but on footage unlike its training data appearance helps little, see [results](#results).
- **An encoder fine-tuned on your footage**, where the largest gains in the results come from. A guide to fine-tuning is coming soon.

Load either with `ReIDModel.from_pretrained`, as in the [quickstart](#quickstart). Each encoder has its own distance scale, so choose the thresholds for the encoder you will track with.

---

## Choosing a fusion method

On every frame, BoT-SORT gives each track and detection pair a score and then picks the best matching. With ReID enabled, that score combines two signals: how much the boxes overlap, and how alike the two crops look. `reid_fusion` sets how they are combined:

```python
tracker = BoTSORTTracker(
    reid_model=reid_model,
    reid_fusion="adaptive",  # default: "botsort"
)
```

- `"botsort"`: a pair scores as well as the stronger of the two, `score = 1 - min(box distance, appearance distance)`.
- `"adaptive"`: the two add up, `score = box overlap + weight * appearance similarity`.

In more detail:

**`"botsort"`.** Appearance is only used for a pair when its distance is below `reid_appearance_threshold` and the boxes pass the `reid_proximity_threshold` check. Otherwise the pair is scored on box overlap alone.

**`"adaptive"`.** This is the appearance fusion from Deep OC-SORT. Each pair starts from its box overlap, and appearance is added on top: `score = box overlap + (reid_appearance_weight + bonus) * appearance similarity`. The bonus rewards a clear winner. For a track, it is how much more its most similar detection looks like it than the second most similar one; for a detection, the same against its two most similar tracks. A pair's bonus is the average of the two, capped at `reid_adaptive_weight_cap`. So appearance counts more when one candidate clearly stands out, and less when several look alike. Negative similarities count as zero, so appearance can only raise a score. `reid_appearance_floor`, added by this library, sets the lowest appearance similarity that counts: below it, a pair is scored on box overlap alone. With the proximity gate open, this stops a lost track from taking an unrelated detection that only looks vaguely similar.

!!! note "What `adaptive` includes"

    Only the appearance fusion is taken from Deep OC-SORT; the rest of that tracker is not added to BoT-SORT. `reid_appearance_threshold` has no effect with `reid_fusion="adaptive"`.

Two consequences are worth knowing before switching:

- **The similarity range changes.** `"botsort"` returns values in `[0, 1]`; `"adaptive"` returns `[0, 1 + reid_appearance_weight + reid_adaptive_weight_cap]`, which is `[0, 2.25]` at the defaults. `minimum_iou_threshold_first_assoc` is applied to that fused value, so a threshold tuned for one method is a different gate under the other. Retune it when you switch, or the comparison measures the gate rather than the fusion.
- **The weights are domain-dependent.** The Deep OC-SORT paper uses `reid_appearance_weight=0.75` with `reid_adaptive_weight_cap=0.5` for MOT17 and MOT20, and `1.25` with `1.0` for DanceTrack, where dancers occlude constantly and geometry carries less. The defaults here follow the MOT17/MOT20 pair.

With the gate open, `adaptive` keeps IDs more stable than `"botsort"`, which switches between look-alike candidates from one frame to the next. `"botsort"` stays the default because it is the safer choice: appearance is only used below `reid_appearance_threshold`, and turning it on does not require retuning the other thresholds.

---

## Choosing an appearance threshold

This threshold only applies to `reid_fusion="botsort"`; with `"adaptive"` it has no effect.

When a track and a detection look alike, appearance can make their match cheaper. It never blocks a match that geometry already allows. `reid_appearance_threshold` sets how alike they must look, as a distance of `0.5 * (1 - cosine similarity)`: 0 means identical, and the BoT-SORT paper uses 0.25.

The right value depends on the encoder and the footage, so measure it on labeled data:

1. Embed the ground-truth crops with the encoder you will track with.
2. Compare pairs of crops that could meet during tracking: same video, at most the lost-track buffer apart (30 frames by default). Take the same number of pairs from each sequence, so one crowded video does not decide the result.
3. Pick the value that keeps most same-person pairs below it and lets few different-person pairs through.

Trackers has a helper for each step. `extract_ground_truth_embeddings` reads any dataset in [MOT format](../evaluations/evaluate.md#data-format):

```python
from trackers.core.reid import (
    extract_ground_truth_embeddings,
    plot_appearance_distances,
    sample_appearance_distances,
)

embeddings, ids, frame_ids, sequence_ids = extract_ground_truth_embeddings(model, "mot17/val", keep_classes=(1,))
distances = sample_appearance_distances(embeddings, ids, frame_ids, sequence_ids)
for threshold in (0.10, 0.20, 0.25):
    same_id_rate, different_id_rate = distances.rates_at(threshold)
    print(f"θ={threshold:.2f}: same-ID {same_id_rate:.1%}, different-ID {different_id_rate:.1%}")

plot_appearance_distances(distances, thresholds={0.20: "selected", 0.25: "default"})
```

The [ReID API reference](../api/reid.md#choosing-a-threshold) lists the full signatures, and the [ReID cookbook](https://colab.research.google.com/github/roboflow/trackers/blob/develop/docs/cookbooks/how-to-add-reid-to-trackers.ipynb) runs the whole flow in Colab. The two examples below use 5000 same-person and 10000 different-person pairs, 1 to 30 frames apart.

### MOT17 with a pedestrian encoder

Blue bars are pairs of the same person, red bars are pairs of different people. A good threshold sits where the blue bars end and the red ones start. `fastreid_mot17_sbs50` on MOT17 val keeps the two groups mostly apart, with some overlap between 0.2 and 0.35.

![FastReID MOT17 SBS on MOT17 val GT](../assets/reid/mot17-fastreid-appearance-distances.png)

This is what `rates_at` prints for two thresholds:

| `reid_appearance_threshold` | same-person pairs below | different-person pairs below |
| :-------------------------- | :---------------------: | :--------------------------: |
| 0.20                        |           68%           |             1.1%             |
| 0.25 (paper default)        |           79%           |             2.9%             |

Moving the threshold right accepts more same-person pairs, but it also starts letting different-person pairs through. We use 0.2, the same value as the [MOT17 re-ID study](https://www-sop.inria.fr/members/Francois.Bremond/Postscript/Tomasz__SCCAI_2025.pdf) (Table 8).

### SoccerNet with the same kind of encoder

`osnet_x1_0_msmt17_combineall` was trained on pedestrians. On SoccerNet test, players in the same kit look alike to it, so the blue and red bars overlap.

![OSNet MSMT17 on SoccerNet test GT](../assets/reid/soccernet-osnet-appearance-distances.png)

When the two groups overlap like this, no threshold separates them: any value that keeps most same-person pairs also lets many different-person pairs through. Tuning the threshold will not fix it. Check the [proximity threshold](#choosing-a-proximity-threshold) and [fusion method](#choosing-a-fusion-method) first, and consider an encoder fine-tuned on your footage, see [results](#results).

---

## How far the threshold carries

A histogram fixes one frame gap, so it only describes re-association over that horizon. Sweeping the gap shows how long a track can stay lost before appearance stops helping to re-find it. `sweep_frame_gap` repeats the sampling above across widening bands, and `plot_frame_gap_sweep` draws the result:

```python
from trackers.core.reid import plot_frame_gap_sweep, sweep_frame_gap

sweep = sweep_frame_gap(embeddings, ids, frame_ids, sequence_ids)
plot_frame_gap_sweep(sweep, thresholds={0.20: "selected", 0.25: "default"})
```

On MOT17 val, pairs of different people stay about as far apart at every gap. Pairs of the same person drift apart as the gap grows, because pose, lighting and occlusion change, so a threshold that catches them one frame apart misses more of them after a long absence.

![FastReID MOT17 SBS separability vs frame gap](../assets/reid/mot17-fastreid-appearance-distances-vs-gap.png)

From this we learn:

- **Check the threshold at the gap you care about.** A threshold that works one frame apart can fail after a lost track: at 0.2 it matches 98% of same-person pairs one frame apart, but only 51% across the default 30-frame `lost_track_buffer`.
- **Long gaps make people harder to tell apart.** The same person looks less like themselves after a long absence, so same-person and different-person distances overlap more: ROC AUC, explained below, falls from 0.998 at a 1-frame gap to 0.854 beyond 120 frames. With the threshold at 0.2, appearance stops recognizing most people after a long absence: only 29% of same-person pairs more than 120 frames apart are under 0.2, against 98% one frame apart. Raising the threshold to catch those also lets more different people through, so if you increase `lost_track_buffer`, check both rates at that gap before raising `reid_appearance_threshold`.

??? info "How to read ROC AUC, and the numbers per frame gap"

    ROC AUC answers one question: if you pick one pair of crops of the same person and one pair of different people at random, how often is the same-person pair closer? 1.0 means always, so some threshold separates the two groups perfectly. 0.5 means a coin flip, so appearance tells you nothing. Unlike the two rates in the table, it does not depend on a chosen threshold. It is also not where the shaded bands overlap in the figure: those bands only span the 10th to 90th percentiles, so at a 1-frame gap they never touch even though the AUC is 0.998.

    How it is calculated: each frame-gap band gets its own sample of same-person and different-person pairs. For every threshold from 0 to 1, count the share of same-person pairs below it and the share of different-person pairs below it, and plot the first share against the second. That traces the ROC curve from (0, 0) to (1, 1), and ROC AUC is the area under it. `sweep_frame_gap` gets the same number without drawing the curve, by comparing every same-person distance with every different-person distance and counting how often the same-person one is smaller, with ties counting as half (`trackers.core.reid.roc_auc`). The result is the lower panel of the figure above.

    The last two columns show the two rates at a threshold of 0.2.

    | Frame gap  | ROC AUC | same-ID below 0.2 | different-ID below 0.2 |
    | :--------- | :-----: | :---------------: | :--------------------: |
    | 1          |  0.998  |       98.0%       |          1.7%          |
    | 2 to 5     |  0.987  |       87.6%       |          1.6%          |
    | 6 to 15    |  0.957  |       67.4%       |          1.1%          |
    | 16 to 30   |  0.929  |       51.4%       |          1.1%          |
    | 31 to 60   |  0.899  |       39.8%       |          0.9%          |
    | 61 to 120  |  0.865  |       31.8%       |          0.8%          |
    | 121 to 240 |  0.854  |       28.7%       |          0.8%          |

??? info "The generic encoder on SoccerNet"

    The cross-domain encoder fails differently. On SoccerNet the different-ID rate at θ=0.2 stays between 44% and 51% at every gap, so the frame gap is not what limits it; the encoder simply cannot separate players in matching kits at any horizon. Widening the gap costs same-ID pairs (99.6% down to 87.0%) without ever making the different-ID side usable, so no threshold makes the generic encoder useful here under the minimum fusion; the additive rule with an open gate is what does, see the SoccerNet table under Other encoders. The SoccerNet rows in the tables above go further with an encoder fine-tuned on the dataset's own train split.

    ![OSNet MSMT17 separability vs frame gap](../assets/reid/soccernet-osnet-appearance-distances-vs-gap.png)

---

## Choosing a proximity threshold

`reid_proximity_threshold` decides which track-detection pairs appearance is allowed to score. A pair is dropped before appearance is consulted whenever `1 - IoU` exceeds the threshold, so the 0.5 default limits appearance to pairs that already overlap at `IoU >= 0.5`, and 0.99 still requires `IoU >= 0.01`. Only 1.0 disables the gate.

A lost track is matched through its Kalman prediction. After a short occlusion the prediction has drifted, so it barely overlaps the detection when the target reappears. After the target leaves the frame, the prediction keeps moving out of view and the overlap drops to zero. The 0.5 default blocks appearance in both cases, 0.99 lets it score the first one, and only 1.0 lets it score the second. BoT-SORT has no separate step for targets that leave and return, so opening the gate is the only way to keep their ID.

SoccerNet test, oracle detections, `osnet_x1_0` fine-tuned on SoccerNet train, library-default motion parameters, `reid_fusion="adaptive"` with its default weights:

| `reid_proximity_threshold` | appearance consulted when | HOTA  | IDF1  | ID switches |
| :------------------------- | :------------------------ | :---: | :---: | :---------: |
| none (no ReID)             |                           | 84.56 | 79.35 |    2939     |
| 0.5 (default)              | `IoU >= 0.50`             | 84.52 | 79.27 |    3012     |
| 0.8                        | `IoU >= 0.20`             | 85.02 | 79.71 |    3268     |
| 1.0                        | always                    | 85.46 | 81.18 |    1734     |

<video controls autoplay muted loop style="display: block; margin: 0 auto; max-width: 100%; max-height: 80vh;">
  <source src="https://github.com/user-attachments/assets/9e5a02ea-51b5-4421-8365-25ab09ec73f4" type="video/mp4">
</video>
<p align="center" style="margin-top: -0.4em;"><small><em>SoccerNet SNMOT-147, top to bottom: gate 0.5, 0.8 and 1.0. The player who leaves the frame gets a new ID at 0.5 and 0.8, and keeps it at 1.0.</em></small></p>

HOTA and IDF1 improve as the gate opens, but ID switches do not follow the same curve. At 0.8 they are higher than with the default gate or without ReID: appearance can now match more distant boxes, but not yet the ones that recover a lost track. With the gate fully open they drop below both. If you open the gate, open it all the way.

The right value depends on the footage. Use 1.0 when targets leave the frame or the camera moves a lot. Keep 0.5 when targets stay in view, as in DanceTrack: there an open gate only adds wrong matches between people who look alike.

---

## Tuning the thresholds

The steps above give a good starting point. `trackers tune` then searches `reid_appearance_threshold` and `reid_proximity_threshold` together, keeping the motion parameters you pass fixed, see [Tune ReID Thresholds](tune.md#tune-reid-thresholds). The command tunes the `"botsort"` thresholds; for `"adaptive"` it does not search the weights yet.

---

## Results

BoT-SORT with and without ReID, using the same detections and motion parameters. The first table shows HOTA on the split where the ReID thresholds were tuned, the second shows test results. The better value for each dataset is in bold. Motion parameters are the tuned values from the [tracker comparison](../evaluations/results.md).

<!-- BENCH-XREF copy-of: [docs/evaluations/results.md](../evaluations/results.md) BoT-SORT and BoT-SORT + ReID rows in the mot17/sportsmot/soccernet/dancetrack Tuned tables (test rows only). The encoder, split and parameter columns exist only here. Update results.md first, then mirror the test rows here. -->

=== "Generic encoder"

    Encoders that were not trained on the evaluated dataset. Each dataset uses the fusion method that scored higher on the tuning split, `reid_fusion="botsort"` unless noted.

    Tuning split:

    | Dataset    | Config          |   HOTA    |
    | :--------- | :-------------- | :-------: |
    | MOT17      | BoT-SORT        |   69.05   |
    |            | BoT-SORT + ReID | **69.64** |
    | SportsMOT  | BoT-SORT        |   82.00   |
    |            | BoT-SORT + ReID | **82.69** |
    | DanceTrack | BoT-SORT        |   53.89   |
    |            | BoT-SORT + ReID | **58.34** |
    | SoccerNet¹ | BoT-SORT        |   86.95   |
    |            | BoT-SORT + ReID | **86.98** |

    Test split:

    | Dataset    | Config          |   HOTA    |   IDF1    |   MOTA    |
    | :--------- | :-------------- | :-------: | :-------: | :-------: |
    | MOT17      | BoT-SORT        | **63.86** |   78.74   | **79.42** |
    |            | BoT-SORT + ReID |   63.85   | **78.82** |   79.41   |
    | SportsMOT  | BoT-SORT        |   74.15   |   74.06   |   96.89   |
    |            | BoT-SORT + ReID | **75.62** | **75.66** | **96.90** |
    | DanceTrack | BoT-SORT        | **57.8**  | **57.9**  | **92.2**  |
    |            | BoT-SORT + ReID |   57.6    |   57.6    |   92.1    |
    | SoccerNet  | BoT-SORT        | **85.00** | **79.68** |   97.25   |
    |            | BoT-SORT + ReID |   84.96   |   79.66   |   97.25   |

    Tuned ReID configuration for each dataset. Motion parameters are the BoT-SORT values from the [tracker comparison](../evaluations/results.md).

    ```yaml
    MOT17:
      reid_model: fastreid_mot17_sbs50
      reid_fusion: botsort
      reid_appearance_threshold: 0.10
      reid_proximity_threshold: 0.5

    SportsMOT:
      reid_model: fastreid_mot17_sbs50
      reid_fusion: botsort
      reid_appearance_threshold: 0.30
      reid_proximity_threshold: 0.5

    DanceTrack:
      reid_model: fastreid_mot17_sbs50
      reid_fusion: botsort
      reid_appearance_threshold: 0.4822
      reid_proximity_threshold: 0.5068

    SoccerNet:
      reid_model: fastreid_mot17_sbs50
      reid_fusion: botsort
      reid_appearance_threshold: 0.0467
      reid_proximity_threshold: 0.3511
    ```

    For MOT17, DanceTrack and SoccerNet, both thresholds were tuned together with `trackers tune`. The SportsMOT threshold comes from a sweep at the default proximity threshold. Each configuration was then evaluated once on test.

    ¹ SoccerNet-tracking has no validation split, so its tuning split is train. The other datasets tune on val, MOT17 on val-half.

    A generic encoder improves HOTA on every tuning split, but only SportsMOT keeps the gain on test. On the other datasets, test HOTA stays within 0.2 of BoT-SORT without ReID. On SoccerNet the tuned thresholds are strict enough that appearance rarely changes a match.

    SoccerNet uses ground-truth boxes as detections, so its numbers are not comparable to the YOLOX rows.

=== "Fine-tuned encoder"

    `osnet_x1_0` fine-tuned on each dataset's train split, with `reid_fusion="botsort"` and `reid_fusion="adaptive"`.

    Tuning split:

    | Dataset    | Config                      |   HOTA    |
    | :--------- | :-------------------------- | :-------: |
    | MOT17      | BoT-SORT                    | **69.05** |
    |            | BoT-SORT + ReID             |   69.00   |
    | SoccerNet² | BoT-SORT                    |   85.72   |
    |            | BoT-SORT + ReID             |   88.97   |
    |            | BoT-SORT + ReID, `adaptive` | **89.91** |

    Test split:

    | Dataset   | Config                      |   HOTA    |   IDF1    |   MOTA    |
    | :-------- | :-------------------------- | :-------: | :-------: | :-------: |
    | MOT17     | BoT-SORT                    |   63.8    |   78.7    |   79.4    |
    |           | BoT-SORT + ReID             | **64.12** | **79.16** |   79.36   |
    | SoccerNet | BoT-SORT                    |   85.00   |   79.68   |   97.25   |
    |           | BoT-SORT + ReID             |   87.30   |   83.16   |   98.73   |
    |           | BoT-SORT + ReID, `adaptive` | **88.43** | **84.40** | **99.26** |

    Tuned ReID configuration for each dataset and fusion method. Motion parameters are the BoT-SORT values from the [tracker comparison](../evaluations/results.md), except `minimum_iou_threshold_first_assoc` where listed.

    ```yaml
    MOT17:
      botsort:
        reid_model: osnet_x1_0 fine-tuned on MOT17 train
        reid_appearance_threshold: 0.25
        reid_proximity_threshold: 0.5

    SoccerNet:
      botsort:
        reid_model: osnet_x1_0 fine-tuned on SoccerNet train
        reid_appearance_threshold: 0.075
        reid_proximity_threshold: 1.0
      adaptive:
        reid_model: osnet_x1_0 fine-tuned on SoccerNet train
        reid_appearance_weight: 0.75
        reid_adaptive_weight_cap: 0.5
        reid_proximity_threshold: 1.0
        reid_appearance_floor: 0.8
    ```

    Each configuration was evaluated once on test.

    !!! note "Being tuned"

        The DanceTrack rows and MOT17's `adaptive` row are not here yet: they were never tuned, only run at a few hand-picked values. A `trackers tune` search is under way for all three datasets, and they will be added with what it finds.

    ² SoccerNet-tracking has no validation split. This encoder was trained on the first 45 of the 57 train sequences, so its thresholds were tuned on the other 12 (SNMOT-159 to SNMOT-170). The other datasets tune on val, MOT17 on val-half.

    Fine-tuning makes the biggest difference on SoccerNet: the generic encoder leaves HOTA flat, while the fine-tuned one adds 2.30 on test, and 3.43 with `adaptive`.
