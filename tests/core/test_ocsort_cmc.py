# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""OC-SORT camera motion compensation (opt-in ``enable_cmc``).

CMC estimates a global affine transform between consecutive frames and warps every track into the current frame's
coordinates before association: the live Kalman state (``CMC.apply_batch``) and the geometry OC-SORT keeps beside it --
the observation history used by OCR and velocity estimation, and the frozen filter state ORU restores. These tests
replace ``CMC.estimate`` with scripted transforms so the camera motion is exact.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import supervision as sv

from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.core.ocsort.tracklet import OCSORTTracklet
from trackers.utils.state_representations import XCYCSRStateEstimator, XYXYStateEstimator

_FRAME = np.zeros((8, 8, 3), dtype=np.uint8)
_IDENTITY = np.eye(2, 3)


def _shift(dx: float, dy: float = 0.0) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])


class _ScriptedMotion:
    """Stands in for ``CMC``: returns one scripted affine per ``estimate`` call."""

    def __init__(self, affines: list[np.ndarray]) -> None:
        self.affines = list(affines)
        self.reset_calls = 0

    def estimate(self, frame: np.ndarray, dets_xyxy: np.ndarray | None = None) -> np.ndarray:
        return self.affines.pop(0)

    def reset(self) -> None:
        self.reset_calls += 1


def _detection(x1: float) -> sv.Detections:
    return sv.Detections(xyxy=np.array([[x1, 100.0, x1 + 60.0, 260.0]]), confidence=np.array([0.9]))


def _track_through_pan(tracker: OCSORTTracker) -> list[int]:
    """A static object filmed while the camera pans: it jumps 80 px left per frame after frame 3."""
    ids = []
    for x1 in (300.0, 300.0, 300.0, 220.0, 140.0, 60.0):
        result = tracker.update(_detection(x1), frame=_FRAME)
        assert result.tracker_id is not None
        ids.append(int(result.tracker_id[0]))
    return ids


def test_cmc_is_disabled_by_default_and_frames_are_ignored() -> None:
    tracker = OCSORTTracker()
    assert tracker.cmc is None
    with pytest.warns(UserWarning, match="does not use it"):
        tracker.update(_detection(300.0), frame=_FRAME)


def test_enabled_cmc_consumes_frames_without_warning() -> None:
    tracker = OCSORTTracker(enable_cmc=True, cmc_method="orb", cmc_downscale=3)
    assert tracker.cmc is not None
    assert tracker.cmc.cfg.method == "orb"
    assert tracker.cmc.downscale == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tracker.update(_detection(300.0), frame=_FRAME)


@pytest.mark.parametrize("state_estimator_class", [XCYCSRStateEstimator, XYXYStateEstimator])
def test_camera_pan_keeps_identity_only_with_cmc(state_estimator_class: type) -> None:
    """An 80 px pan leaves no overlap with the predicted box, so plain OC-SORT restarts the track."""
    compensated = OCSORTTracker(enable_cmc=True, state_estimator_class=state_estimator_class)
    compensated.cmc = _ScriptedMotion([_IDENTITY] * 3 + [_shift(-80.0)] * 3)  # type: ignore[assignment]
    with pytest.warns(UserWarning):
        plain_ids = _track_through_pan(OCSORTTracker(state_estimator_class=state_estimator_class))

    compensated_ids = _track_through_pan(compensated)

    assert compensated_ids[2] != -1
    assert compensated_ids[3:] == [compensated_ids[2]] * 3
    assert plain_ids[3] != plain_ids[2]


def test_apply_camera_motion_warps_observations_and_frozen_state() -> None:
    box = np.array([100.0, 100.0, 160.0, 260.0])
    tracklet = OCSORTTracklet(box)
    for _ in range(2):
        tracklet.predict()
        tracklet.update(box)
    tracklet.predict()
    tracklet.predict()  # missed a frame: the ORU state is frozen
    assert tracklet._frozen_state is not None
    frozen_center = tracklet._frozen_state["state"][:2, 0].copy()
    frozen_covariance = tracklet._frozen_state["state_covariance"].copy()

    tracklet.apply_camera_motion(_shift(10.0, -5.0))

    expected = box + np.array([10.0, -5.0, 10.0, -5.0])
    np.testing.assert_allclose(tracklet.last_observation, expected)
    assert all(np.allclose(obs, expected) for obs in tracklet.observations.values())
    np.testing.assert_allclose(tracklet._frozen_state["state"][:2, 0], frozen_center + np.array([10.0, -5.0]))
    np.testing.assert_allclose(tracklet._frozen_state["state_covariance"], frozen_covariance)


def test_lost_track_is_recovered_in_current_frame_coordinates() -> None:
    """The camera pans while a static object is occluded.

    On recovery, ORU re-smooths the filter from the last observation to the new
    detection. With the stored geometry warped along with the camera, the replay
    sees no motion; otherwise it reads the pan as object motion (a velocity of about
    -32 px/frame and a centre 14 px off in this scene).
    """
    tracker = OCSORTTracker(enable_cmc=True)
    tracker.cmc = _ScriptedMotion([_IDENTITY] * 4 + [_shift(-80.0)] * 3)  # type: ignore[assignment]
    for _ in range(4):
        tracker.update(_detection(300.0), frame=_FRAME)
    track_id = tracker.tracks[0].tracker_id
    assert track_id != -1

    tracker.update(sv.Detections.empty(), frame=_FRAME)  # occluded while the camera moves
    tracker.update(sv.Detections.empty(), frame=_FRAME)
    tracker.update(_detection(60.0), frame=_FRAME)

    assert len(tracker.tracks) == 1
    tracklet = tracker.tracks[0]
    assert tracklet.tracker_id == track_id
    state = tracklet.state_estimator.kf.state.ravel()
    assert state[0] == pytest.approx(90.0, abs=1e-6)  # centre of the 60..120 detection
    assert state[4] == pytest.approx(0.0, abs=1e-6)  # the object never moved


def test_reset_resets_camera_motion_state() -> None:
    tracker = OCSORTTracker(enable_cmc=True)
    scripted = _ScriptedMotion([])
    tracker.cmc = scripted  # type: ignore[assignment]

    tracker.reset()

    assert scripted.reset_calls == 1


def test_apply_batch_moves_xcycsr_centre_and_keeps_size() -> None:
    """OC-SORT's ``[xc, yc, s, r, vx, vy, vs]`` state shares the centre/velocity slots CMC transforms."""
    from trackers.utils.cmc import CMC

    tracklet = OCSORTTracklet(np.array([100.0, 100.0, 160.0, 260.0]))
    tracklet.state_estimator.kf.state[4:7, 0] = [2.0, -1.0, 0.5]
    before = tracklet.state_estimator.kf.state.ravel().copy()
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90 degrees

    CMC.apply_batch(np.hstack([rotation, [[5.0], [7.0]]]), [tracklet])

    after = tracklet.state_estimator.kf.state.ravel()
    np.testing.assert_allclose(after[0:2], rotation @ before[0:2] + [5.0, 7.0])
    np.testing.assert_allclose(after[4:6], rotation @ before[4:6])
    np.testing.assert_allclose(after[[2, 3, 6]], before[[2, 3, 6]])
