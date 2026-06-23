# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared and tracker-specific contracts for tracklet objects.

Tracklets are the internal per-object state managed by a tracker.  The
tests here operate directly on tracklet instances rather than going through
the tracker's update() loop.

Sections
--------
1. Shared contracts: parametrized over all concrete tracklet classes
2. Tracklet-specific contracts: tests tied to a single tracklet class
"""

from __future__ import annotations

import numpy as np
import pytest

from trackers.core.botsort.tracklet import BoTSORTTracklet
from trackers.core.bytetrack.tracklet import ByteTrackTracklet
from trackers.core.ocsort.tracklet import OCSORTTracklet
from trackers.core.sort.tracklet import SORTTracklet
from trackers.utils.base_tracklet import BaseTracklet

# All concrete tracklet classes and their short IDs used as test suffixes.
_TRACKLET_PARAMS = [
    pytest.param(SORTTracklet, id="sort"),
    pytest.param(ByteTrackTracklet, id="bytetrack"),
    pytest.param(OCSORTTracklet, id="ocsort"),
    pytest.param(BoTSORTTracklet, id="botsort"),
]
_SUCCESSFUL_UPDATES_PARAMS = [
    pytest.param(SORTTracklet, id="sort"),
    pytest.param(BoTSORTTracklet, id="botsort"),
]

_BBOX = np.array([10.0, 20.0, 30.0, 40.0])
_BBOX2 = np.array([12.0, 22.0, 32.0, 42.0])


@pytest.fixture
def bbox() -> np.ndarray:
    return _BBOX.copy()


def _new_tracklet(tracklet_class: type[BaseTracklet], bbox: np.ndarray) -> BaseTracklet:
    """Construct a concrete tracklet class with default constructor params."""
    return tracklet_class(bbox)  # type: ignore[call-arg]


# ==========================================================================
# 1. Shared tracklet contracts
# ==========================================================================


@pytest.mark.parametrize("tracklet_class", _TRACKLET_PARAMS)
def test_tracklet_starts_with_zero_time_since_update(
    tracklet_class: type[BaseTracklet],
) -> None:
    """Every tracklet must start with time_since_update == 0."""
    assert _new_tracklet(tracklet_class, _BBOX).time_since_update == 0


@pytest.mark.parametrize("tracklet_class", _TRACKLET_PARAMS)
def test_tracklet_predict_increments_time_since_update(
    tracklet_class: type[BaseTracklet],
) -> None:
    """predict() must increment time_since_update by 1 each call."""
    t = _new_tracklet(tracklet_class, _BBOX)
    t.predict()
    assert t.time_since_update == 1
    t.predict()
    assert t.time_since_update == 2


@pytest.mark.parametrize("tracklet_class", _TRACKLET_PARAMS)
def test_tracklet_predict_increments_age(
    tracklet_class: type[BaseTracklet],
) -> None:
    """predict() must increment age by 1 each call."""
    t = _new_tracklet(tracklet_class, _BBOX)
    initial_age = t.age
    t.predict()
    assert t.age == initial_age + 1


@pytest.mark.parametrize("tracklet_class", _TRACKLET_PARAMS)
def test_tracklet_predict_returns_finite_bbox(
    tracklet_class: type[BaseTracklet],
) -> None:
    """predict() must always return finite bbox coordinates."""
    t = _new_tracklet(tracklet_class, _BBOX)
    for _ in range(10):
        bbox_out = t.predict()
        assert np.all(np.isfinite(bbox_out))


@pytest.mark.parametrize("tracklet_class", _TRACKLET_PARAMS)
def test_tracklet_update_resets_time_since_update(
    tracklet_class: type[BaseTracklet],
) -> None:
    """update(bbox) must reset time_since_update to 0."""
    t = _new_tracklet(tracklet_class, _BBOX)
    t.predict()
    assert t.time_since_update == 1
    t.update(_BBOX2)
    assert t.time_since_update == 0


@pytest.mark.parametrize("tracklet_class", _SUCCESSFUL_UPDATES_PARAMS)
def test_tracklet_starts_with_one_successful_update(
    tracklet_class: type[BaseTracklet],
) -> None:
    """Construction counts as the first successful update for SORT-like counters."""
    tracklet = _new_tracklet(tracklet_class, _BBOX)
    assert getattr(tracklet, "number_of_successful_updates") == 1


@pytest.mark.parametrize("tracklet_class", _SUCCESSFUL_UPDATES_PARAMS)
def test_tracklet_update_increments_successful_updates(
    tracklet_class: type[BaseTracklet],
) -> None:
    """update(bbox) increments number_of_successful_updates for SORT-like counters."""
    tracklet = _new_tracklet(tracklet_class, _BBOX)
    before = getattr(tracklet, "number_of_successful_updates")
    tracklet.update(_BBOX2)
    assert getattr(tracklet, "number_of_successful_updates") == before + 1


# ==========================================================================
# 2. Tracklet-specific contracts
# ==========================================================================


def test_sort_tracklet_predict_does_not_drift_bbox(bbox: np.ndarray) -> None:
    """A single predict() step must not move the bbox of a freshly created tracklet."""
    tracklet = SORTTracklet(bbox)
    initial_bbox = tracklet.get_state_bbox().copy()

    tracklet.predict()

    assert tracklet.time_since_update == 1
    np.testing.assert_allclose(tracklet.get_state_bbox(), initial_bbox, atol=1e-6)


def test_bytetrack_tracklet_starts_with_one_successful_consecutive_update(
    bbox: np.ndarray,
) -> None:
    """Construction counts as the first successful consecutive update."""
    tracklet = ByteTrackTracklet(bbox)
    assert tracklet.number_of_successful_consecutive_updates == 1


def test_ocsort_oru_triggers_on_single_frame_gap(bbox: np.ndarray) -> None:
    """ORU unfreeze fires correctly after exactly one missed frame.

    Copilot raised a concern that ``_observed`` stays ``True`` throughout
    the first missed frame, so ORU would not fire on a 1-frame gap.  The
    freeze is intentionally deferred to the *start* of the next
    ``predict()`` call (the re-match frame), where ``time_since_update > 0
    AND _observed`` is the reliable first-miss signal.  At that point the
    frozen KF state is identical to what an immediate freeze would have
    saved, and ``_unfreeze()`` is called by ``update()`` in the same frame.
    """
    tracklet = OCSORTTracklet(bbox)

    # Provide a second observation so last_observation / velocity are set.
    second_bbox = np.array([15.0, 25.0, 35.0, 45.0])
    tracklet.predict()
    tracklet.update(second_bbox)
    assert tracklet._observed is True
    assert tracklet._frozen_state is None

    # Miss exactly one frame — predict advances the clock but no update follows.
    tracklet.predict()
    # Freeze has NOT fired yet: _observed is still True.
    assert tracklet._observed is True
    assert tracklet._frozen_state is None

    # Re-match frame: predict() fires the freeze at its very start
    # (time_since_update > 0 AND _observed), then update() unfreezes.
    re_match_bbox = np.array([20.0, 30.0, 40.0, 50.0])
    tracklet.predict()
    assert tracklet._observed is False
    assert tracklet._frozen_state is not None

    tracklet.update(re_match_bbox)
    assert tracklet._frozen_state is None  # _unfreeze() cleared it
    assert tracklet._observed is True
