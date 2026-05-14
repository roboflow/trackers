# Plan: Add CBIoU Tracker (BoTSORT + no CMC + BIoU association)

## Brief

Add `CBIoUTracker` as a thin `BoTSORTTracker` subclass with CMC permanently disabled and `BIoU` as the association metric, registering it as `"cbiou"` in the tracker registry with its own `search_space`, full test coverage, and public export — keeping all changes self-contained in a new `cbiou/` module.

Classification : feature
Complexity     : small
Affected files : 6 files across 2 modules (core + tests)
Key risks      : none — BIoU already exists, BoTSORT accepts `iou=` and `enable_cmc=`
Agent review   : ✓ agents ready (0 corrections incorporated)

| # | Step | What changes | Stop condition |
|---|------|--------------|----------------|
| 1 | Create `src/trackers/core/cbiou/` module | New `__init__.py` + `tracker.py` with `CBIoUTracker` | Class instantiates cleanly |
| 2 | Export from package `__init__.py` | Add import + `__all__` entry for `CBIoUTracker` | `from trackers import CBIoUTracker` works |
| 3 | Register in shared test IDs | `ALL_TRACKER_IDS += ["cbiou"]` | Generic tracker tests include "cbiou" |
| 4 | Update `test_registration.py` explicit list | Add `CBIoUTracker` to `TestSearchSpaceValidation` loop | `search_space` validation passes |
| 5 | Add CBIoU-specific test file | `tests/core/test_cbiou_tracker.py` covering CMC-disabled, BIoU behavior, `buffer_ratio` forwarding | All tests green |

---

## Full Plan

**Classification**: feature
**Complexity**: small
**Date**: 2026-05-14

### Goal

Introduce `CBIoUTracker` — a registered, tunable, fully-tested MOT tracker that is exactly BoT-SORT with CMC disabled and BIoU (Buffered IoU) as the association metric. The BIoU `buffer_ratio` is elevated to a first-class constructor parameter. CMC-related parameters (`enable_cmc`, `cmc_method`, `cmc_downscale`) are hidden from the public signature since they have no effect. The tracker is exported from the top-level `trackers` package and participates in all generic tracker contract tests via `ALL_TRACKER_IDS`.

### Affected files

- `src/trackers/core/cbiou/__init__.py` — new package init (empty re-export)
- `src/trackers/core/cbiou/tracker.py` — `CBIoUTracker(BoTSORTTracker)` subclass
- `src/trackers/__init__.py` — add `CBIoUTracker` import + `__all__` entry
- `tests/core/shared_ids.py` — add `"cbiou"` to `ALL_TRACKER_IDS`
- `tests/core/test_registration.py` — add `CBIoUTracker` to `TestSearchSpaceValidation` explicit tracker loop
- `tests/core/test_cbiou_tracker.py` — new CBIoU-specific tests (new file)

### Design sketch

```python
# src/trackers/core/cbiou/tracker.py

class CBIoUTracker(BoTSORTTracker):
    """BoT-SORT with CMC disabled and BIoU association (Buffered IoU).

    Identical to BoTSORTTracker but permanently disables camera motion
    compensation and uses BIoU (Buffered IoU) for all association steps.
    The buffer_ratio parameter controls how much each bounding box is
    expanded before computing IoU.
    """

    tracker_id = "cbiou"
    search_space: ClassVar[dict[str, dict]] = {
        "lost_track_buffer": {"type": "randint", "range": [10, 91]},
        "track_activation_threshold": {"type": "uniform", "range": [0.1, 0.9]},
        "minimum_iou_threshold_first_assoc": {"type": "uniform", "range": [0.05, 0.7]},
        "minimum_iou_threshold_second_assoc": {"type": "uniform", "range": [0.05, 0.7]},
        "minimum_iou_threshold_unconfirmed_assoc": {"type": "uniform", "range": [0.05, 0.7]},
        "high_conf_det_threshold": {"type": "uniform", "range": [0.3, 0.8]},
        "minimum_consecutive_frames": {"type": "randint", "range": [1, 4]},
        "buffer_ratio": {"type": "uniform", "range": [0.0, 0.5]},
    }

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold_first_assoc: float = 0.2,
        minimum_iou_threshold_second_assoc: float = 0.5,
        minimum_iou_threshold_unconfirmed_assoc: float = 0.3,
        high_conf_det_threshold: float = 0.6,
        instant_first_frame_activation: bool = True,
        state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator,
        buffer_ratio: float = 0.1,
    ) -> None:
        super().__init__(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold_first_assoc=minimum_iou_threshold_first_assoc,
            minimum_iou_threshold_second_assoc=minimum_iou_threshold_second_assoc,
            minimum_iou_threshold_unconfirmed_assoc=minimum_iou_threshold_unconfirmed_assoc,
            high_conf_det_threshold=high_conf_det_threshold,
            enable_cmc=False,
            instant_first_frame_activation=instant_first_frame_activation,
            state_estimator_class=state_estimator_class,
            iou=BIoU(buffer_ratio=buffer_ratio),
        )
        self.buffer_ratio = buffer_ratio
```

### Test coverage plan

`tests/core/test_cbiou_tracker.py` should cover:

1. **CMC is always off**: passing `frame=...` to `update()` triggers `UserWarning` (via `_warn_if_frame_unused` from BoTSORT's CMC-disabled path)
2. **BIoU association tolerance**: a near-miss detection (slightly outside track's predicted region) that would be missed by standard IoU *is* associated by CBIoU (buffer expansion closes the gap)
3. **buffer_ratio forwarded**: `tracker.iou.buffer_ratio == buffer_ratio` after construction
4. **buffer_ratio=0 recovers standard IoU**: behavior identical to BoTSORT(enable_cmc=False) when buffer_ratio=0
5. **Registration**: `"cbiou"` in `BaseTracker._registered_trackers()` (covered generically via shared_ids)

### Risks

- None: the design reuses existing infrastructure (BoTSORT + BIoU), no new algorithms or dependencies needed.

### Follow-up command

/develop feature add CBIoU tracker (BoTSORT with CMC disabled and BIoU association)

---

## Confidence
**Score**: 0.95 — high
**Gaps**:
- BIoU `normalize_for_fusion` returns values in [0,1] (same as IoU), so the score-fusion path in BoTSORT steps 1 & 3 remains numerically valid.
- The `_warn_if_frame_unused` warning path: BoTSORT only calls `_warn_if_frame_unused` if `enable_cmc=False` AND frame is provided; need to verify this in the existing code. (Checked: BoTSORT does NOT call `_warn_if_frame_unused` — it simply skips the CMC block. The warning test should instead assert that `frame` is ignored silently, or we add a `_warn_if_frame_unused` call in CBIoU's `update` override.)

**Refinements**: 1 pass.
- Pass 1: Verified BIoU, BoTSORT, BaseTracker APIs against source. Confirmed no breaking changes.
