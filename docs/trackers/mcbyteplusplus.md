---
description: Track objects with McByte++ and streaming EdgeTAM masks, including bounded sessions and occlusion-aware mask recovery.
---

# McByte++

`McBytePlusPlusTracker` is an independent tracker alongside `McByteTracker`. It combines three-stage box association with overlapping EdgeTAM masks. EdgeTAM generates and propagates the masks using one model; masks can condition ambiguous associations without confidence scores.

This part implements the tracker and its mask lifecycle. Long-term re-identification and selective camera motion compensation are planned separately. Full-dataset tracking quality and speed have not yet been benchmarked for this implementation.

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

`enable_mask_manager=False` is the Python API default. With no custom `mask_manager`, this runs box association without constructing or loading EdgeTAM. Set `enable_mask_manager=True` to construct the default EdgeTAM pipeline. A supplied `mask_manager` is used regardless of the flag and skips automatic EdgeTAM construction. `edgetam_config` configures only automatic construction: it requires `enable_mask_manager=True` and cannot accompany a custom manager.

`enable_mask_refresh` and `enable_mask_trust_filter` both default to `True`, but neither creates a mask manager; they take effect only when one is active. A custom manager's propagator must provide a callable `refresh_masks` method when refresh is enabled. Set `enable_mask_refresh=False` for a propagator without this capability.

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

The McByte and McByte++ inspection commands share sequence options and MOT result writing through `cli/_tracking.py`. The McByte++ benchmark also uses that MOT writer. Each command keeps its own execution flow; McByte++ inspection and benchmarking share CLI settings and overlays through `cli/_mcbyteplusplus.py`. McByte++ inspection extends the shared sequence options with `confidence_override`.

## Reproducibility and inspection

The default Hugging Face model is `yonigozlan/EdgeTAM-hf`, pinned to revision `c266ce53b3fc00f0f495b583f6a116c4e57f53bb`. `McBytePlusPlusMaskConfig` can select another model or revision. The inspector saves the resolved model revision, device, dependencies, configuration, input hashes and tracking results.

Use the [SoccerNet inspection example](../guides/inspect.md#inspect-mcbyte-with-oracle-detections) to run a short CPU sequence. `trackers inspect mcbyteplusplus` always enables EdgeTAM masks and defaults to CMC enabled with ORB at downscale 4. Oracle detections supply boxes; their identity column is never passed into the tracker. Ground-truth identities belong to evaluation.

For full sequences or the complete SoccerNet test directory, use [`trackers benchmark mcbyteplusplus`](../evaluations/benchmark-runner.md#mcbyte-on-soccernet). It saves MOT results, annotated frame images and per-sequence/pooled FPS for later evaluation. Annotated videos can also be enabled. This is an output-generation command; HOTA, IDF1 and MOTA are computed separately.

See the [API reference](../api/trackers.md#mcbyte_1) for all constructor settings.
