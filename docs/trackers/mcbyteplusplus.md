---
description: Track objects with McByte++ and streaming EdgeTAM masks, including bounded sessions and occlusion-aware mask recovery.
---

# McByte++

!!! note "Documentation in progress"

    This guide covers the implemented mask, conditional CMC and optional long-term re-identification paths. Broader benchmarking and documentation are still in progress. Usage examples and parameter descriptions are kept aligned with the implementation.

`McBytePlusPlusTracker` is an independent tracker alongside `McByteTracker`. It combines three-stage box association with overlapping EdgeTAM masks. EdgeTAM generates and propagates the masks using one model; masks can condition ambiguous associations without confidence scores.

The implementation includes the tracker, mask lifecycle, conditional camera motion compensation and optional long-term re-identification. Part 2 evaluated both mask-refresh/trust-filter variants on all 49 SoccerNet-tracking 2022 test sequences using oracle detections and the original CMC. Those results do not validate conditional CMC or long-term re-ID; see [PR #644](https://github.com/roboflow/trackers/pull/644) for their scores and settings.

## Python usage

Install the optional backend:

```bash
pip install "trackers[edgetam]"
```

```python
from trackers import McBytePlusPlusMaskConfig, McBytePlusPlusTracker

tracker = McBytePlusPlusTracker(
    frame_rate=25,
    enable_mask_manager=True,
    edgetam_config=McBytePlusPlusMaskConfig(device="cpu", max_session_frames=200),
)

# detections is supervision.Detections; frame_bgr is an OpenCV uint8 image.
tracked = tracker.update(detections, frame=frame_bgr)
```

The public API takes **BGR uint8** images. Camera motion compensation receives BGR; the tracker converts to RGB for the mask pipeline. Camera motion compensation defaults to enabled, using ORB with downscale 4, and can be disabled with `enable_cmc=False`.

Conditional CMC is enabled by default for `orb`, `sift`, `sparseOptFlow` and `ecc`. The independent `trackers.utils.conditional_cmc.ConditionalCMC` component follows the existing CMC interface and estimates motion on every accepted image, with no cached-transform interval. ORB and SIFT require `cmc_minimum_matches=12` candidate correspondences before statistical filtering and at least five afterwards. All methods reject non-finite transforms or affine column norms outside `[cmc_minimum_scale, cmc_maximum_scale]`, defaulting to `[0.3, 3.0]`.

For the default centre/width/height state, accepted corrections retain the original McByte++ transformation of centre, dimensions, velocities and covariance. A non-finite candidate or non-positive width/height leaves that track's prediction and covariance unchanged; valid corrections for other tracks still apply. This protects against invalid states while retaining the original McByte++ approximation for positive dimensions. It does not compute the geometric enclosing box under rotation. Other state representations retain their existing transformation rules. Unavailable features return identity; expected OpenCV failures return identity and clear the reference. Reset and image-resolution changes start a fresh reference.

Use `enable_conditional_cmc=False` to select part 2's original shared CMC behavior for any method. The inspector and benchmark expose this as `--cmc.conditional false`, alongside `--cmc.minimum_matches`, `--cmc.minimum_scale` and `--cmc.maximum_scale`. Saved configurations record these settings. Older configurations without the new fields use the new defaults; explicitly disable conditional CMC when reproducing part 2.

`enable_mask_manager=False` is the Python API default. With no custom `mask_manager`, this runs box association without constructing or loading EdgeTAM. Set `enable_mask_manager=True` to construct the default EdgeTAM pipeline. A supplied `mask_manager` is used regardless of the flag and skips automatic EdgeTAM construction. `edgetam_config` configures only automatic construction: it requires `enable_mask_manager=True` and cannot accompany a custom manager.

`enable_mask_refresh` and `enable_mask_trust_filter` both default to `True`, but neither creates a mask manager; they take effect only when one is active. A custom manager's propagator must provide a callable `refresh_masks` method when refresh is enabled. Set `enable_mask_refresh=False` for a propagator without this capability.

`lost_track_buffer` defaults to `30` in both the tracker and the shared CLI options. Trackers expresses this budget in 30 FPS units and scales it by the reference frame rate passed to the tracker. The SoccerNet inspection example uses `36` with the native 25 FPS rate: `25 / 30 * 36 = 30` frames (1.2 seconds). The [shared benchmark configuration](../evaluations/benchmark-runner.md#shared-configuration-for-all-four-datasets), `mcbyteplusplus-common.json`, instead sets `tracker_frame_rate=30` and `tracker.lost_track_buffer=30`, giving the original McByte++'s 30-frame budget on every dataset. Both settings give 30 frames on SoccerNet. The benchmark reference-rate override does not change image selection or video playback FPS. Using the default buffer of `30` with SoccerNet's native 25 FPS would give 25 frames (one second).

## Mask lifecycle

- Confirmed detector matches become eligible for creation after `minimum_mask_creation_frames` real-image observations (default 1). Boxes and images remain paired: prompts are applied to the stored image before propagation to the next image. Consequently, the first update returns tracks before masks become available.
- Creation and refresh are deferred when another box with a greater bottom Y coordinate covers at least `mask_creation_bbox_overlap_threshold` of the target box (default 0.1). The largest individual overlap is used; unmasked and unconfirmed detections also participate in this check.
- With `enable_mask_refresh=True`, a matched track can recover an empty mask or refresh a displaced mask whose foreground fraction inside the detector box is **strictly less than** `mask_refresh_coverage_threshold` (default 0.5). Exactly 0.5 does not trigger refresh. Previously excluded masks also remain eligible for refresh if they spontaneously reappear. `mask_refresh_interval` sets the minimum image interval between refreshes. `enable_mask_refresh=False` disables refreshing.
- Lost or occluded objects retain their propagated masks, including empty masks. They are not box-prompted until a suitable detector match returns. Rollover alone does not trigger refresh.
- Terminating a track removes its mask. New tracks can be inserted without changing existing internal IDs. These internal IDs own masks separately from the public identities returned to callers.

With `enable_mask_trust_filter=True` (the default), mask trust is independent of refresh eligibility:

1. Empty propagated masks are excluded from association immediately, even without a detector match. Displaced masks are excluded after association, using the matched detector box and the strict coverage threshold above. Occlusion and refresh cooldown do not block exclusion.
2. Excluded masks retain their EdgeTAM state but cannot contribute association scores. Reappearing or moving back into a box does not restore trust. Lost tracks, missing images, duplicate timestamps and session rollover preserve this exclusion.
3. A suitable visible detector match permits a fresh box prompt. Both the generated replacement on that stored image and its output on the next processed image must pass coverage and fill checks. The latter also needs a non-occluded detector match. Coverage must meet both `minimum_mask_coverage` (default 0.9) and `mask_refresh_coverage_threshold`; fill must meet `minimum_mask_fill_ratio` (default 0.05). Once validated, the mask can contribute on the following image. Failed validation leaves it excluded and eligible for a later refresh.

Displacement is **not checked against predicted boxes before association**, so a newly displaced mask can still affect its first bad frame. These geometric checks also cannot prove identity: a wrong mask and a wrong detector match can agree. This policy limits suspect mask evidence; it does not guarantee elimination of identity switches or erase EdgeTAM memory.

Use `enable_mask_trust_filter=False` to restore the previous association/refresh policy for comparison. This control is independent of `enable_mask_refresh`: disabling only refresh leaves suspect masks excluded until removal or reset. `tracker.untrusted_mask_ids` reports excluded internal IDs after the latest update; `tracker.last_mask_output` still exposes the raw masks, including excluded ones.

`max_session_frames=200` bounds EdgeTAM history for both files and streams. The count includes the seed image. Masks transfer to a new session at the boundary. Use Python `None` or JSON `null` to disable the bound. The backend needs no future frames or final sequence length.

If `frame=None` or a timestamp is duplicated, box association still runs, but mask inference and CMC are skipped. Both retain propagation memory. A duplicate call with an image preserves queued creation/refresh prompts, their original image and boxes, and pending creation ages; it neither advances those ages nor replaces the accepted observation. Track removals are still honoured. An explicit `frame=None` call discards queued prompts and clears pending creation ages, even at a duplicate timestamp; eligible requests are rebuilt from fresh observations when images return.

Timestamp gaps advance Kalman time, while EdgeTAM consumes only the actual images received. Processing every Nth frame with increasing timestamps does not cancel prompts or require intermediate `frame=None` calls. Backward or non-finite timestamps skip the whole update, following `BaseTracker`.

Call `tracker.reset()` between sequences. This clears IDs, mask sessions and CMC state while retaining loaded weights. An out-of-memory error resets mask memory and logs a fallback to box association, following McByte's policy. Three consecutive failed inference attempts disable masks until `reset()`; calls that defer inference do not clear the failure count. Other inference errors reset mask memory and raise.

## Implementation structure

`core/mcbyteplusplus/` follows McByte's layout: `tracker.py`, `tracklet.py`, `mask_association.py` and `utils.py`. The three association stages are visible in `update()`, with corresponding helpers for IoU, mask conditioning, assignment, spawning, pruning and mask lifecycle events. The association module belongs to McByte++ so its policy can evolve independently.

McByte++ adds explicit refresh helpers and separates actual mask ownership from pending creation. It keeps BGR-to-RGB conversion, confidence-free overlapping masks and internal tracklet IDs separate from public identities. Missing images invalidate queued prompts; duplicate calls with images retain the last accepted mask observation. Timestamp expiry removes dead mask owners before propagation. Deprecated McByte compatibility modules are not duplicated.

Long-term joining lives independently in `reid/long_term.py`, with configuration in `reid/long_term_config.py`. The box-only feature-occlusion helper in `reid/occlusion.py` has no dependency on masks. Other trackers can reuse `LongTermReID` by supplying current eligible internal-ID observations, birth references and true termination events. McByte++ supplies these events itself; applications do not need to implement that adapter.

The McByte and McByte++ inspection commands share sequence options and MOT result writing through `cli/_tracking.py`. The McByte++ benchmark also uses that MOT writer. Each command keeps its own execution flow; McByte++ inspection and benchmarking share CLI settings and overlays through `cli/_mcbyteplusplus.py`. McByte++ inspection extends the shared sequence options with `confidence_override`.

## Reproducibility and inspection

The default Hugging Face model is `yonigozlan/EdgeTAM-hf`, pinned to revision `c266ce53b3fc00f0f495b583f6a116c4e57f53bb`. `McBytePlusPlusMaskConfig` can select another model or revision. The inspector saves the resolved model revision, device, dependencies, configuration, input hashes and tracking results.

Use the [SoccerNet inspection example](../guides/inspect.md#inspect-mcbyte-with-oracle-detections) to run a short CPU sequence. `trackers inspect mcbyteplusplus` always enables EdgeTAM masks and defaults to CMC enabled with ORB at downscale 4. Oracle detections supply boxes; their identity column is never passed into the tracker. Ground-truth identities belong to evaluation.

For full sequences or complete test splits, use [`trackers benchmark mcbyteplusplus`](../evaluations/benchmark-runner.md#mcbyte-on-soccernet). The [shared configuration and dataset commands](../evaluations/benchmark-runner.md#shared-configuration-for-all-four-datasets) cover SoccerNet, MOT17, DanceTrack and SportsMOT. The command saves MOT results, annotated frame images and per-sequence/pooled FPS for later evaluation. Annotated videos can also be enabled. This is an output-generation command; HOTA, IDF1 and MOTA are computed separately.

See the [API reference](../api/trackers.md#mcbyte_1) for all constructor settings.

## Optional long-term re-identification

Re-ID is disabled by default. Supply `reid_model` to enable it; `reid_joining_config` is optional and requires a model. The existing three-stage association, internal tracklet IDs, Kalman state and mask ownership remain independent of the public identity joining. Re-ID can also run with `enable_mask_manager=False`.

Once a re-ID model is selected, **both re-ID feature extraction pause policies are enabled by default**: one for strong estimated camera motion, and one for subject occlusion detected from current bounding boxes. A pause skips new descriptors and extends the affected pending identity decision deadline; existing descriptors can still support a reconnection. Historical buffering is also enabled by default. These policies do not pause tracking, mask propagation or lost-track expiry.

| Setting                                                   | Default | Effect when re-ID is enabled                                                                                                                            |
| --------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enable_reid_camera_motion_pause`                         | `True`  | Skip new re-ID descriptors during strong estimated camera motion and pause pending identity decision ages. Requires conditional CMC when CMC is active. |
| `reid_camera_motion_threshold`                            | `0.02`  | Trigger only above this displacement fraction of the image diagonal.                                                                                    |
| `reid_joining_config.enable_feature_occlusion_filter`     | `True`  | Skip new descriptors for an individually occluded subject.                                                                                              |
| `reid_joining_config.pause_reconnection_on_occlusion`     | `True`  | Also pause that subject's pending identity decision deadline. Requires the feature occlusion filter.                                                    |
| `reid_joining_config.feature_occlusion_overlap_threshold` | `0.6`   | Trigger at or above this maximum individual foreground-box overlap divided by target-box area.                                                          |
| `reid_joining_config.buffer_results`                      | `True`  | Recover earlier pending observations when their public identity is resolved.                                                                            |

Install the additional model dependency:

```bash
pip install "trackers[edgetam,reid]"
```

Select the original McByte++ sports checkpoint explicitly:

```python
from trackers import LongTermReIDConfig, McBytePlusPlusTracker

tracker = McBytePlusPlusTracker(
    enable_mask_manager=True,
    reid_model="mcbyteplusplus_sports",
    reid_joining_config=LongTermReIDConfig(),
)
```

`mcbyteplusplus_sports` is a Trackers model alias. On first use it downloads and verifies the original [GTA-link sports checkpoint](https://github.com/sjc042/gta-link/blob/main/reid_checkpoints/sports_model.pth.tar-60), then creates a cached Roboflow-compatible OSNet x1.0 model. The checkpoint SHA-256 is `8d5b2fd8763db34c2aad69810466adf413f0426d9f8119d322227e0e639c5fbd`. Conversion preserves all 565 inference tensors, excludes only the training classifier, and records provenance and artifact hashes. It uses restricted tensor loading, never an unrestricted pickle fallback. Cached files live under `$XDG_CACHE_HOME/trackers/reid` or `~/.cache/trackers/reid`.

The `reid` dependency is pinned to `0.1.0`. To prepare an already downloaded checkpoint, choose a cache directory, or select an explicit inference device:

```python
from reid import ReIDModel
from trackers import McBytePlusPlusTracker, prepare_sports_reid_model

model_dir = prepare_sports_reid_model(
    checkpoint="/path/to/sports_model.pth.tar-60",
    cache_dir="outputs/reid-models",
)
encoder = ReIDModel.from_pretrained(source=str(model_dir), device="cpu")
tracker = McBytePlusPlusTracker(reid_model=encoder, enable_mask_manager=False)
```

Other Roboflow model sources or injected `ReIDEncoder` implementations are supported. Their quality and thresholds need separate validation. The sports model uses BGR crops converted to RGB, resized to 256 pixels high and 128 pixels wide with bilinear interpolation and ImageNet normalization. The tracker still accepts BGR images; do not convert the whole input to RGB for re-ID.

**Scores obtained with Roboflow-trained re-ID models have only an indicative character.** They illustrate the potential of McByte++ with models adapted to particular data. Custom models may benefit from validation and adjustment of the joining threshold.

### Joining policy

The policy follows the [original McByte++ implementation](https://github.com/tstanczyk95/McBytePlusPlus/blob/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9/yolox/tracker/mcbyteplusplus_tracker__with_reid.py):

- Extract from currently matched, confirmed boxes with confidence at least 0.6, width at least 25, height at least 50 and area at least 1024 pixels, strictly inside a 4-pixel image margin. Sample at least 5 processing steps apart. Missing confidence follows the tracker convention of 1.0.
- Normalize each descriptor, retain the latest 50, average them and normalize the mean. A successful sample counts once; the default minimum is one actual sample on both sides. These controls are configurable through `LongTermReIDConfig`.
- Offer identities to the gallery only after true termination. Active and temporarily lost owners remain reserved. The gallery survives an empty scene and has no automatic expiry.
- A new confirmed tracklet can reconnect while its decision age is at most 10 steps after its original birth, excluding paused steps. The tracklet and gallery identity must be each other's highest-similarity eligible candidate, with cosine similarity at least 0.8. Reject pairs whose observation intervals overlap **before** selecting these mutual best matches. Old last observation must strictly precede new first observation; termination time is not that endpoint.
- First-frame tracklets receive final public IDs once output-eligible. Other pending identities receive fresh IDs when observed beyond their decision window, at true termination, or at `finish()`. Fresh IDs have never been allocated earlier in this sequence; intentional reuse requires a valid reconnection.

This similarity threshold is distinct from BoT-SORT's appearance-distance/fusion configuration. No old motion state, feature history or mask is transferred into a new tracklet. Identity ownership and output uniqueness are checked inside the component before results are exposed.

The default `enable_feature_occlusion_filter=True` defers re-ID feature extraction when a current detector box with a strictly greater bottom Y coordinate covers at least `feature_occlusion_overlap_threshold=0.6` of the target's area. The threshold must be greater than zero and at most one; zero would reject even unoccluded boxes. It uses the largest individual overlap, not IoU or a sum of overlaps. Other current detections, including unconfirmed or feature-ineligible boxes, can be occluders. A skip retains existing descriptors and does not consume a sampling slot. The default `pause_reconnection_on_occlusion=True` also extends the affected pending identity's decision window, as described below. This perspective heuristic is independent of the mask-creation threshold of 0.1 and cannot guarantee that every occluded crop is excluded.

### Re-ID feature extraction pauses for subject occlusion

Both `enable_feature_occlusion_filter=True` and `pause_reconnection_on_occlusion=True` are defaults in `LongTermReIDConfig`. They skip new re-ID feature extraction for an occluded subject and preserve its pending tracklet's joining opportunity. The deadline pause requires the feature filter; enabling the pause without the filter raises a configuration error. To disable the occlusion policy entirely, set both options to `False`. Setting only `pause_reconnection_on_occlusion=False` keeps feature filtering but allows the decision clock to advance. This policy works independently of masks and CMC. The following configuration makes the defaults explicit:

```python
joining_config = LongTermReIDConfig(
    enable_feature_occlusion_filter=True,
    feature_occlusion_overlap_threshold=0.6,
    pause_reconnection_on_occlusion=True,
    buffer_results=True,
)
```

On advancing updates with an image and host-enabled feature extraction, check the current matched box against the other current detections, even when the next feature sample is not due. Pause only the affected pending tracklet's decision age. Pauses before confirmation are retained from the tracklet's original birth. Do not use predicted boxes for temporarily lost tracks. Missing-image and duplicate-timestamp updates do not trigger this pause. Confidence, size and image-border rejection alone do not request a pause; an occlusion can still request one when another rejection reason also applies. Established identities still skip occluded feature samples but have no joining deadline to pause.

Existing descriptors may still reconnect a pending identity during occlusion. Skips do not increase the successful-sample count or move the last successful extraction step; a clear crop can be sampled immediately once the ordinary interval has elapsed. Simultaneous camera-motion and occlusion pauses count as **one** paused step and share `maximum_reconnection_pause_steps`. The default cap is `None` (unlimited); zero prevents deadline extension while retaining extraction filtering. Kalman, mask and lost-track clocks keep their usual behavior. Termination and `finish()` still assign an identity and release buffered records, even when no descriptor was collected.

Keep `buffer_results=True` for complete text output: occlusion pauses can prolong unresolved `-1` periods, and without buffering those observations cannot be recovered. With an unlimited pause cap, pending detection records can grow during prolonged occlusion; images and masks are not retained.

### Re-ID feature extraction pauses for strong camera motion

The default `enable_reid_camera_motion_pause=True` suppresses new re-ID feature extraction during strong estimated camera movement and extends pending identities' decision time. It reuses the existing **conditional CMC** estimate and supports all four methods; it adds no crop-sharpness computation or second CMC pass. With re-ID or CMC disabled there is no motion-triggered pause. To use re-ID with active legacy regular CMC, explicitly set `enable_reid_camera_motion_pause=False`; otherwise construction raises a configuration error. The following example makes the camera-motion defaults explicit; subject-occlusion filtering and deadline pausing also use their enabled defaults:

```python
tracker = McBytePlusPlusTracker(
    reid_model="mcbyteplusplus_sports",
    enable_mask_manager=True,
    enable_reid_camera_motion_pause=True,
    reid_camera_motion_threshold=0.02,
    reid_joining_config=LongTermReIDConfig(
        maximum_reconnection_pause_steps=None,
    ),
)
```

The threshold is the largest displacement of the image corners and centre divided by the image diagonal. A successful accepted estimate **strictly above** `0.02` triggers the pause; initialization, unavailable estimates and missing/duplicate images do not infer a motion flag. The score measures displacement between supplied images, not blur during exposure. Its suitability depends on sampling interval and scene content. Blur caused only by a moving player does not trigger this policy.

On a flagged image, both pending and established tracklets skip new descriptors. Existing descriptors can still reconnect a pending identity. Only pending decision age pauses: Kalman/mask/lost-track clocks and processing-frame references keep advancing. The usual sampling gap also advances, so a successful sample on step 1, a blocked sample on step 6 and a suitable image on step 7 yield samples on steps 1, 7, 12, 17. The separate box-occlusion gate suppresses sampling and, when `pause_reconnection_on_occlusion=True`, can also extend the affected tracklet's decision window. Both reasons share one per-step pause and the same cumulative cap.

`maximum_reconnection_pause_steps=None` permits unlimited additional waiting; an integer such as `1000` caps cumulative paused steps per pending tracklet. Zero disables deadline extension while retaining feature suppression. After the cap, decision age advances despite further motion flags. A currently observed pending track receives a fresh ID once its decision age exceeds `maximum_reconnection_age`; an age equal to that limit still permits joining. Termination and `finish()` still resolve pending identities, and popping ready history remains independent per tracklet. Pending detection records can grow during unlimited pauses; images and masks are not buffered.

**Keep `buffer_results=True` when enabling camera-motion pausing for complete trajectories or accuracy evaluation.** Combining `enable_reid_camera_motion_pause=True` with `buffer_results=False` is supported but not recommended for these uses. Pausing can prolong the period in which observations have unresolved ID `-1`; without buffering, those observations are skipped in MOT text output and cannot be recovered. This can discard valid portions of a trajectory, or every observation of a tracklet that remains pending throughout its visible lifetime. Assigning an ID at termination or `finish()` cannot recover discarded observations. Observations with already established public IDs continue to be returned and written normally, even while new feature extraction is paused. Buffering OFF remains an option for live-only use when losing pending history is acceptable.

For inspection and benchmark commands with re-ID enabled, these optional arguments explicitly restate the camera-motion defaults:

```bash
--tracker.enable_reid_camera_motion_pause true \
    --tracker.reid_camera_motion_threshold 0.02 \
    --tracker.reid_joining_config.maximum_reconnection_pause_steps null
```

These settings are recorded in the saved configuration. The live/historical output contract below is unchanged.

Selecting `--reid.model mcbyteplusplus_sports` enables re-ID with both feature extraction pause policies and historical buffering by default. To make those policies explicit in an inspection or benchmark command, use:

```bash
--tracker.enable_reid_camera_motion_pause true \
    --tracker.reid_camera_motion_threshold 0.02 \
    --tracker.reid_joining_config.enable_feature_occlusion_filter true \
    --tracker.reid_joining_config.feature_occlusion_overlap_threshold 0.6 \
    --tracker.reid_joining_config.pause_reconnection_on_occlusion true \
    --tracker.reid_joining_config.buffer_results true
```

To disable both re-ID feature extraction pause policies and their deadline extensions while keeping re-ID enabled, use:

```bash
--tracker.enable_reid_camera_motion_pause false \
    --tracker.reid_joining_config.enable_feature_occlusion_filter false \
    --tracker.reid_joining_config.pause_reconnection_on_occlusion false
```

To retain only the strong-camera-motion policy, disable the two occlusion options above and leave `enable_reid_camera_motion_pause=True`. With only `enable_reid_camera_motion_pause=False`, subject-occlusion filtering and deadline pausing remain active. Saved configurations with explicit `False` values keep those values when replayed; omitted fields use the current enabled defaults. Neither policy enables re-ID without a model.

### Live and historical output

`update(detections, frame, timestamp)` keeps the ordinary `sv.Detections` return type. With re-ID enabled, its reserved row fields are:

| Field                  | Meaning                                                                   |
| ---------------------- | ------------------------------------------------------------------------- |
| `tracker_id`           | Final public identity, or `-1` while unassigned/pending.                  |
| `data["tracklet_id"]`  | Internal tracklet/mask owner; ordinary unassigned rows have `-1`.         |
| `data["reid_pending"]` | True only for output-eligible observations awaiting an identity decision. |

Visualize current output directly, including `-1` if desired. For MOT text output, write only nonnegative IDs. Historical buffering defaults to enabled when re-ID is enabled. `pop_finalized_results()` adds previously pending observations with their resolved IDs; it does not repeat current observations that `update()` already returned with final IDs.

For an application's frame loop, the collection pattern is:

```python
from collections import defaultdict

records = defaultdict(list)


def collect(frame_number, detections):
    assigned = detections[detections.tracker_id >= 0]
    if len(assigned):
        records[frame_number].append(assigned)


for frame_bgr, detections in observations:
    current = tracker.update(detections, frame=frame_bgr)
    collect(tracker.frame_id, current)
    # Visualize current here. Historical output does not replay images.
    if tracker.reid_joiner is not None and tracker.reid_joiner.config.buffer_results:
        for frame_number, finalized in tracker.pop_finalized_results().items():
            collect(frame_number, finalized)

for frame_number, finalized in tracker.finish().items():
    collect(frame_number, finalized)

# Write each collection in sorted(records) order to obtain frame-sorted output.
```

Here `observations` is the application's iterable of `(BGR frame, sv.Detections)` pairs. The tracker does not write files itself. The built-in commands handle collection, source-frame mapping, EOF closure and final MOT writing automatically.

Each identity releases independently: one unresolved identity cannot block another's history. The same frame key may arrive in several pops, so **append**, rather than replace, that frame's records. Pending snapshots own copies of boxes, confidences, classes and per-detection data; they contain no image, segmentation mask or collection metadata. Queued records remain until popped, finished or reset; for long streams, drain them regularly.

Keys are one-based accepted-update references, not necessarily original video frame numbers. Duplicate timestamps and `frame=None` calls receive references; skipped invalid/backward timestamps do not. A separate processing-step clock controls feature sampling and joining age. It advances once per non-duplicate accepted call, including missing-image updates. Duplicate timestamps advance only the reference, including when the image is missing; other accepted missing-image updates advance both clocks.

The joining calculation is `decision_age = current_step - birth_step - paused_steps`. Joining remains possible through `maximum_reconnection_age`, inclusive; a currently observed pending tracklet receives a fresh ID once this age exceeds the limit. A temporarily lost pending tracklet waits until it is observed again, terminated or finalized by `finish()`. Pausing this decision age does not pause the ordinary feature-sampling clock. No image means no new features, while retained features and pending records remain available. Large timestamp gaps affect Kalman expiry but do not invent re-ID samples. Built-in commands translate processing references to selected source frame numbers automatically; a custom source with its own numbering can map `tracker.frame_id` to that number.

Set `LongTermReIDConfig(buffer_results=False)` to keep only live current output. Pending `-1` rows then have no later historical recovery, including at EOF. Calling `pop_finalized_results()` without re-ID or without buffering raises `RuntimeError`; `finish()` is safe in every mode and returns `{}` when there are no historical rows. It assigns fresh IDs to any remaining eligible pending identities, closes the sequence, and never creates rows for never-confirmed tracks. Repeated calls return `{}`. After closure, call `reset()` before another update. Reset discards all gallery/identity/history state and preserves loaded models.

For complete text output, keep buffering enabled, especially with either of the enabled [re-ID feature extraction pause policies](#re-id-feature-extraction-pauses-for-strong-camera-motion): longer pending periods can otherwise omit valid track fragments or entire tracklets.

### Command-line examples

From a checkout, this runs a short CPU inspection with the original sports encoder:

```bash
trackers inspect mcbyteplusplus \
    --config src/trackers/cli/inspect/configs/mcbyteplusplus-soccernet.json \
    --sequence.image_dir /path/to/SNMOT-116/img1 \
    --sequence.det_file /path/to/SNMOT-116/det/det.txt \
    --sequence.end_frame 30 \
    --edgetam.device cpu \
    --reid.model mcbyteplusplus_sports \
    --reid.device cpu
```

Inspection, benchmarking and `trackers track --tracker.name mcbyteplusplus` accept `--reid.model` and `--reid.device`. McByte++ requires an explicit model selection. Joining settings use `--tracker.reid_joining_config.*`, for example `--tracker.reid_joining_config.buffer_results false` or `--tracker.reid_joining_config.pause_reconnection_on_occlusion false`. Existing configurations remain re-ID-disabled until a model is selected. Inspection and benchmark run metadata record the selected model and joining settings; visualizations show public identity separately from internal mask ownership.
