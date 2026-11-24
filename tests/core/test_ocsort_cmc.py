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
from unittest import mock

import numpy as np
import pytest
import supervision as sv

from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.core.ocsort.tracklet import OCSORTTracklet
from trackers.utils.cmc import CMC, CMCConfig
from trackers.utils.iou import IoU
from trackers.utils.state_representations import XCYCSRStateEstimator, XCYCWHStateEstimator, XYXYStateEstimator

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


class _RecordingIoU(IoU):
    """Public ``iou=`` hook that records the track boxes handed to the first association stage."""

    def __init__(self) -> None:
        self.track_boxes: list[np.ndarray] = []

    def compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        self.track_boxes.append(np.array(boxes_1, dtype=np.float64))
        return super().compute(boxes_1, boxes_2)


def _moving_box(step: int) -> np.ndarray:
    """A box that drifts 10 px right per step, so every update stores a distinct observation."""
    return np.array([100.0 + 10.0 * step, 100.0, 160.0 + 10.0 * step, 260.0])


_ESTIMATORS = [
    pytest.param(XCYCSRStateEstimator, id="xcycsr"),
    pytest.param(XYXYStateEstimator, id="xyxy"),
]


class TestObservationHistoryWarp:
    """Every stored observation moves with the camera, each one by its own age."""

    @pytest.mark.parametrize("state_estimator_class", _ESTIMATORS)
    def test_each_observation_is_warped_under_its_own_age(self, state_estimator_class: type) -> None:
        """Distinct boxes per update, so a reordered or skipped mapping cannot hide behind identical values.

        A tracklet is updated with a box drifting 10 px right per frame; a pure (+7, -3) camera shift must land on
        ``last_observation``, ``previous_to_last_observation`` and each ``observations[age]`` as that age's box plus (7,
        -3).
        """
        tracklet = OCSORTTracklet(_moving_box(0), state_estimator_class=state_estimator_class)
        box_by_age: dict[int, np.ndarray] = {}
        for step in range(4):
            tracklet.predict()
            tracklet.update(_moving_box(step))
            box_by_age[tracklet.age] = _moving_box(step)
        offset = np.array([7.0, -3.0, 7.0, -3.0])
        assert len(tracklet.observations) >= 2
        expected_observations = {age: (box_by_age[age] + offset).tolist() for age in tracklet.observations}

        tracklet.apply_camera_motion(_shift(7.0, -3.0))

        assert {age: obs.tolist() for age, obs in tracklet.observations.items()} == expected_observations
        assert tracklet.last_observation.tolist() == (_moving_box(3) + offset).tolist()
        assert tracklet.previous_to_last_observation is not None
        assert tracklet.previous_to_last_observation.tolist() == (_moving_box(2) + offset).tolist()


class TestPredictionCacheAfterCmc:
    """The boxes OC-SORT associates with must be the warped predictions, not the pre-warp cache."""

    @pytest.mark.parametrize("state_estimator_class", _ESTIMATORS)
    def test_first_stage_association_sees_warped_prediction(self, state_estimator_class: type) -> None:
        """A static object, then an 80 px left pan: stage-1 track box must be the object's new position.

        The recording ``iou=`` hook sees the boxes actually used for association. A stale (pre-warp) cache would still
        read x 300..360, and the later last-observation rescue would hide that from an id-only assertion.
        """
        recorder = _RecordingIoU()
        tracker = OCSORTTracker(enable_cmc=True, state_estimator_class=state_estimator_class, iou=recorder)
        tracker.cmc = _ScriptedMotion([_IDENTITY] * 3 + [_shift(-80.0)])  # type: ignore[assignment]
        for x1 in (300.0, 300.0, 300.0):
            tracker.update(_detection(x1), frame=_FRAME)

        tracker.update(_detection(220.0), frame=_FRAME)

        np.testing.assert_allclose(recorder.track_boxes[-1], [[220.0, 100.0, 280.0, 260.0]], atol=1e-6)
        assert tracker.tracks[0].last_observation.tolist() == [220.0, 100.0, 280.0, 260.0]


def _build_tracker(state_estimator_class: type) -> OCSORTTracker:
    """Three moving objects tracked for 6 frames, then one lost for 2 (so it holds a frozen ORU state)."""
    tracker = OCSORTTracker(state_estimator_class=state_estimator_class)
    for frame in range(8):
        boxes = [
            [100.0 + 8.0 * frame, 100.0, 160.0 + 8.0 * frame, 260.0],
            [400.0, 300.0 + 5.0 * frame, 460.0, 420.0 + 5.0 * frame],
            [600.0 - 6.0 * frame, 50.0, 690.0 - 6.0 * frame, 200.0],
        ]
        tracker.update(
            sv.Detections(xyxy=np.array(boxes[: 3 if frame < 6 else 2]), confidence=np.full(3 if frame < 6 else 2, 0.9))
        )
    return tracker


def _tracklet_snapshot(tracklet: OCSORTTracklet) -> np.ndarray:
    """Flatten everything CMC touches on a tracklet into one vector (NaN marks an absent optional)."""
    absent4 = np.full(4, np.nan)
    frozen = tracklet._frozen_state
    parts = [
        np.array(sorted(tracklet.observations), dtype=np.float64),
        tracklet.last_observation,
        absent4 if tracklet.previous_to_last_observation is None else tracklet.previous_to_last_observation,
        *(tracklet.observations[age] for age in sorted(tracklet.observations)),
        np.full(2, np.nan) if tracklet.velocity is None else tracklet.velocity,
        np.empty(0) if frozen is None else frozen["state"].ravel(),
        np.empty(0) if frozen is None else frozen["state_covariance"].ravel(),
        tracklet.state_estimator.kf.state.ravel(),
        tracklet.state_estimator.kf.state_covariance.ravel(),
    ]
    return np.concatenate([np.asarray(part, dtype=np.float64) for part in parts])


class TestTrackerCompensationMatchesPerTracklet:
    """``OCSORTTracker._compensate_camera_motion`` equals warping each tracklet on its own."""

    @pytest.mark.parametrize("state_estimator_class", _ESTIMATORS)
    def test_tracker_warp_equals_per_tracklet_warp(self, state_estimator_class: type) -> None:
        """Rotation + 1.2x zoom + translation as a float32 estimate (what real CMC returns).

        Reference: for every tracklet of an identically built tracker, ``CMC.apply_batch`` on that tracklet alone
        followed by ``apply_camera_motion``. The tracker must end with the same observation window, last and
        previous observation, velocity, frozen state + covariance and live state + covariance.
        """
        angle = np.deg2rad(15.0)
        affine = (
            1.2 * np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0]])
            + np.array([[0.0, 0.0, 13.0], [0.0, 0.0, -8.0]])
        ).astype(np.float32)
        tracker = _build_tracker(state_estimator_class)
        reference = _build_tracker(state_estimator_class)
        assert any(t._frozen_state is not None for t in tracker.tracks)  # the scene exercises the frozen branch
        assert any(t.velocity is not None for t in tracker.tracks)
        tracker.enable_cmc = True
        tracker.cmc = _ScriptedMotion([affine])  # type: ignore[assignment]

        applied = tracker._compensate_camera_motion(_FRAME, np.empty((0, 4)))

        for tracklet in reference.tracks:
            CMC.apply_batch(affine, [tracklet])
            tracklet.apply_camera_motion(affine)
        assert applied is True
        assert len(tracker.tracks) == len(reference.tracks)
        got = np.concatenate([_tracklet_snapshot(t) for t in tracker.tracks])
        want = np.concatenate([_tracklet_snapshot(t) for t in reference.tracks])
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-9, equal_nan=True)


_ROT_90_ZOOM_1_5 = np.array([[0.0, -1.5, 10.0], [1.5, 0.0, 20.0]])


class TestRotationAndZoomOfObservations:
    """Observations follow the same geometry as the Kalman state under rotation and zoom, not just translation."""

    @pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
    @pytest.mark.parametrize(
        ("state_estimator_class", "expected"),
        [
            # centre (130, 180) -> (-1.5 * 180 + 10, 1.5 * 130 + 20) = (-260, 215); width 60 and height 160 are kept
            pytest.param(XCYCSRStateEstimator, [-290.0, 135.0, -230.0, 295.0], id="xcycsr-centre-only"),
            # corners (100,100) (160,100) (160,260) (100,260) -> (-140,170) (-140,260) (-380,260) (-380,170)
            pytest.param(XYXYStateEstimator, [-380.0, 170.0, -140.0, 260.0], id="xyxy-enclosing-box"),
        ],
    )
    def test_90deg_rotation_with_1_5_zoom_gives_hand_computed_box(
        self, state_estimator_class: type, expected: list[float], dtype: type
    ) -> None:
        """Rotate 90 degrees, zoom 1.5, translate by (10, 20): every stored observation lands on the expected box."""
        box = np.array([100.0, 100.0, 160.0, 260.0])
        tracklet = OCSORTTracklet(box, state_estimator_class=state_estimator_class)
        for _ in range(2):
            tracklet.predict()
            tracklet.update(box)

        tracklet.apply_camera_motion(_ROT_90_ZOOM_1_5.astype(dtype))

        np.testing.assert_allclose(tracklet.last_observation, expected, atol=1e-9)
        assert tracklet.previous_to_last_observation is not None
        np.testing.assert_allclose(tracklet.previous_to_last_observation, expected, atol=1e-9)
        assert all(np.allclose(obs, expected, atol=1e-9) for obs in tracklet.observations.values())

    def test_lost_xcycsr_track_keeps_its_size_under_alternating_roll(self) -> None:
        """30 alternating +-0.2 degree rolls while a track is lost must not inflate its remembered box.

        Warping a box into the enclosing box of its rotated corners grows it a little each time; with the centre-only
        model the 60 x 160 box stays 60 x 160 however many noisy rolls it sees.
        """
        box = np.array([100.0, 100.0, 160.0, 260.0])
        tracklet = OCSORTTracklet(box)
        for _ in range(2):
            tracklet.predict()
            tracklet.update(box)
        tracklet.predict()
        tracklet.predict()  # lost
        for step in range(30):
            angle = np.deg2rad(0.2 if step % 2 == 0 else -0.2)
            roll = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0]])
            tracklet.apply_camera_motion(roll)

        last = tracklet.last_observation
        assert (last[2] - last[0], last[3] - last[1]) == pytest.approx((60.0, 160.0), abs=1e-9)
        assert all(
            (obs[2] - obs[0], obs[3] - obs[1]) == pytest.approx((60.0, 160.0), abs=1e-9)
            for obs in tracklet.observations.values()
        )


def _lost_tracklet(state_estimator_class: type) -> OCSORTTracklet:
    """A tracklet that missed two frames, so it holds a frozen ORU state."""
    box = np.array([100.0, 100.0, 160.0, 260.0])
    tracklet = OCSORTTracklet(box, state_estimator_class=state_estimator_class)
    for _ in range(2):
        tracklet.predict()
        tracklet.update(box)
    tracklet.predict()
    tracklet.predict()
    assert tracklet._frozen_state is not None
    return tracklet


def _dense_spd(dim: int) -> np.ndarray:
    """Fixed dense symmetric positive-definite matrix, anisotropic in every 2x2 block."""
    seed = (np.arange(dim * dim, dtype=np.float64).reshape(dim, dim) * 7.0 % 11.0 - 5.0) / 5.0 + np.eye(dim)
    return seed @ seed.T + np.eye(dim)


class TestFrozenStateWarp:
    """The ORU frozen filter state moves with the camera, uncertainty included."""

    def test_xcycsr_frozen_state_and_covariance_follow_the_rotation(self) -> None:
        """Frozen centre -> R c + t, velocity -> R v, covariance -> A P A.T (A = R on [0:2] and [4:6])."""
        rot = np.array([[1.1, -0.4], [0.4, 1.1]])
        shift = np.array([12.0, -7.0])
        tracklet = _lost_tracklet(XCYCSRStateEstimator)
        assert tracklet._frozen_state is not None
        covariance = _dense_spd(7)
        state = np.array([130.0, 180.0, 9600.0, 0.375, 2.0, -1.0, 0.5])
        tracklet._frozen_state["state"] = state.reshape(-1, 1).copy()
        tracklet._frozen_state["state_covariance"] = covariance.copy()
        oracle = np.eye(7)
        oracle[0:2, 0:2] = rot
        oracle[4:6, 4:6] = rot

        tracklet.apply_camera_motion(np.hstack([rot, shift[:, np.newaxis]]))

        frozen = tracklet._frozen_state
        np.testing.assert_allclose(frozen["state"].ravel()[0:2], rot @ state[0:2] + shift)
        np.testing.assert_allclose(frozen["state"].ravel()[4:6], rot @ state[4:6])
        np.testing.assert_allclose(frozen["state"].ravel()[[2, 3, 6]], state[[2, 3, 6]])
        np.testing.assert_allclose(frozen["state_covariance"], oracle @ covariance @ oracle.T, rtol=1e-12)

    def test_xyxy_axis_aligned_zoom_conjugates_frozen_covariance_per_corner_block(self) -> None:
        """An axis-aligned (2, 0.5) zoom scales every x / y pair of the XYXY frozen covariance."""
        rot = np.array([[2.0, 0.0], [0.0, 0.5]])
        tracklet = _lost_tracklet(XYXYStateEstimator)
        assert tracklet._frozen_state is not None
        covariance = _dense_spd(8)
        tracklet._frozen_state["state_covariance"] = covariance.copy()
        oracle = np.kron(np.eye(4), rot)

        tracklet.apply_camera_motion(np.hstack([rot, [[3.0], [-4.0]]]))

        np.testing.assert_allclose(
            tracklet._frozen_state["state_covariance"], oracle @ covariance @ oracle.T, rtol=1e-12
        )

    def test_xyxy_cross_axis_rotation_leaves_frozen_covariance_untouched(self) -> None:
        """A 90 degree rotation mixes axes; the XYXY covariance is deliberately left as it was (like the live state)."""
        tracklet = _lost_tracklet(XYXYStateEstimator)
        assert tracklet._frozen_state is not None
        covariance = _dense_spd(8)
        tracklet._frozen_state["state_covariance"] = covariance.copy()

        tracklet.apply_camera_motion(np.array([[0.0, -1.0, 5.0], [1.0, 0.0, 6.0]]))

        np.testing.assert_array_equal(tracklet._frozen_state["state_covariance"], covariance)

    @pytest.mark.parametrize(
        ("state_estimator_class", "expected_box"),
        [
            pytest.param(XYXYStateEstimator, [60.0, 100.0, 120.0, 260.0], id="xyxy"),
            pytest.param(XCYCWHStateEstimator, [90.0, 180.0, 60.0, 160.0], id="xcycwh"),
        ],
    )
    def test_lost_track_is_recovered_in_current_frame_coordinates_for_corner_and_wh_states(
        self, state_estimator_class: type, expected_box: list[float]
    ) -> None:
        """XYXY / XCYCWH twin of the XCYCSR lost-track pan: ORU must see a static object, not a 32 px/frame drift."""
        tracker = OCSORTTracker(enable_cmc=True, state_estimator_class=state_estimator_class)
        tracker.cmc = _ScriptedMotion([_IDENTITY] * 4 + [_shift(-80.0)] * 3)  # type: ignore[assignment]
        for _ in range(4):
            tracker.update(_detection(300.0), frame=_FRAME)
        track_id = tracker.tracks[0].tracker_id
        assert track_id != -1

        tracker.update(sv.Detections.empty(), frame=_FRAME)
        tracker.update(sv.Detections.empty(), frame=_FRAME)
        tracker.update(_detection(60.0), frame=_FRAME)

        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].tracker_id == track_id
        state = tracker.tracks[0].state_estimator.kf.state.ravel()
        np.testing.assert_allclose(state[:4], expected_box, atol=1e-6)
        np.testing.assert_allclose(state[4:], 0.0, atol=1e-6)  # the object never moved


class TestVelocityDirectionWarp:
    """The stored unit direction ``[dy, dx]`` is rotated with the camera's linear part."""

    @pytest.mark.parametrize(
        ("linear", "velocity", "expected"),
        [
            pytest.param([[0.0, -1.0], [1.0, 0.0]], [0.0, 1.0], [1.0, 0.0], id="rot90-plus-x-becomes-plus-y"),
            pytest.param([[0.0, -1.0], [1.0, 0.0]], [1.0, 0.0], [0.0, -1.0], id="rot90-plus-y-becomes-minus-x"),
            pytest.param([[2.0, 0.0], [0.0, 2.0]], [0.6, 0.8], [0.6, 0.8], id="zoom-keeps-unit-direction"),
            pytest.param([[0.0, -3.0], [3.0, 0.0]], [0.0, 1.0], [1.0, 0.0], id="rot90-with-zoom-stays-unit"),
        ],
    )
    def test_direction_is_rotated_and_stays_unit_length(
        self, linear: list[list[float]], velocity: list[float], expected: list[float]
    ) -> None:
        """Rotation turns the direction; zoom never changes its length."""
        tracklet = OCSORTTracklet(np.array([100.0, 100.0, 160.0, 260.0]))
        tracklet.velocity = np.array(velocity)

        tracklet.apply_camera_motion(np.hstack([np.array(linear), [[9.0], [-9.0]]]))

        assert tracklet.velocity is not None
        np.testing.assert_allclose(tracklet.velocity, expected, atol=1e-12)
        assert np.linalg.norm(tracklet.velocity) == pytest.approx(np.linalg.norm(velocity), abs=1e-12)

    def test_pure_translation_leaves_direction_bit_identical(self) -> None:
        """A pan changes where things are, not which way they move: the direction array is unchanged exactly."""
        tracklet = OCSORTTracklet(np.array([100.0, 100.0, 160.0, 260.0]))
        tracklet.velocity = np.array([0.6, 0.8])

        tracklet.apply_camera_motion(_shift(30.0, -12.0))

        assert tracklet.velocity is not None
        np.testing.assert_array_equal(tracklet.velocity, np.array([0.6, 0.8]))

    def test_missing_direction_stays_missing(self) -> None:
        """A tracklet without a direction estimate does not acquire one from a camera rotation."""
        tracklet = OCSORTTracklet(np.array([100.0, 100.0, 160.0, 260.0]))

        tracklet.apply_camera_motion(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]))

        assert tracklet.velocity is None


class TestIdentityEstimateShortCircuit:
    """An identity estimate is a no-op: no warp is applied, and results equal a CMC-free tracker."""

    @pytest.mark.parametrize(
        ("estimate", "warp_expected"),
        [
            pytest.param(np.eye(2, 3, dtype=np.float32), False, id="float32-identity-first-frame"),
            pytest.param(np.eye(2, 3), False, id="float64-identity"),
            pytest.param(_shift(1.0), True, id="one-pixel-pan"),
        ],
    )
    def test_warp_is_applied_only_for_a_non_identity_estimate(self, estimate: np.ndarray, warp_expected: bool) -> None:
        """The batch warp runs once for a real pan and not at all for an identity estimate."""
        tracker = OCSORTTracker(enable_cmc=True)
        tracker.cmc = _ScriptedMotion([_IDENTITY, estimate])  # type: ignore[assignment]
        tracker.update(_detection(300.0), frame=_FRAME)

        with mock.patch.object(CMC, "apply_batch") as apply_batch:
            tracker.update(_detection(300.0), frame=_FRAME)

        assert apply_batch.call_count == (1 if warp_expected else 0)

    def test_identity_estimates_reproduce_a_tracker_without_cmc_exactly(self) -> None:
        """A moving object tracked with all-identity estimates ends bit-identical to a tracker with CMC off."""
        plain = OCSORTTracker()
        compensated = OCSORTTracker(enable_cmc=True)
        compensated.cmc = _ScriptedMotion([_IDENTITY] * 6)  # type: ignore[assignment]
        plain_ids = [plain.update(_detection(300.0 + 7.0 * k)).tracker_id for k in range(6)]
        compensated_ids = [compensated.update(_detection(300.0 + 7.0 * k), frame=_FRAME).tracker_id for k in range(6)]

        np.testing.assert_array_equal(np.concatenate(compensated_ids), np.concatenate(plain_ids))
        plain_kf = plain.tracks[0].state_estimator.kf
        compensated_kf = compensated.tracks[0].state_estimator.kf
        np.testing.assert_array_equal(compensated_kf.state, plain_kf.state)
        np.testing.assert_array_equal(compensated_kf.state_covariance, plain_kf.state_covariance)


def _textured_frame(shift_y: int = 0, shift_x: int = 0) -> np.ndarray:
    """Fixed 240 x 320 BGR scene of 8 px blocks with hashed grey levels, circularly shifted by ``(shift_y, shift_x)``.

    Built from integer arithmetic only (no random generator), so every run sees the same pixels. Block corners give the
    feature trackers unambiguous, non-periodic structure; ``np.roll`` is an exact translation of the content.
    """
    by, bx = np.mgrid[0:30, 0:40]
    levels = (((bx * 7919 + by * 104729 + bx * by * 31) * 2654435761) >> 7) % 256
    gray = np.kron(levels, np.ones((8, 8))).astype(np.uint8)
    return np.roll(np.dstack([gray] * 3), (shift_y, shift_x), axis=(0, 1))


class TestCmcRuntimeEdgeCases:
    """Opt-in wiring around ``enable_cmc``: missing frames, runtime toggling, invalid method, every method, reset."""

    @staticmethod
    def _user_warnings(tracker: OCSORTTracker, frames: int) -> list[warnings.WarningMessage]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for step in range(frames):
                tracker.update(_detection(300.0 + step))
        return [w for w in caught if issubclass(w.category, UserWarning)]

    def test_missing_frame_with_cmc_enabled_warns_exactly_once_per_instance(self) -> None:
        """Three updates without a frame on a CMC-enabled tracker produce a single UserWarning, not one per frame."""
        tracker = OCSORTTracker(enable_cmc=True)

        caught = self._user_warnings(tracker, frames=3)

        assert [str(w.message) for w in caught if "frame=None" in str(w.message)] != []
        assert len(caught) == 1

    def test_missing_frame_with_cmc_disabled_does_not_warn(self) -> None:
        """Without CMC, ``frame=None`` is the normal call and must stay silent."""
        tracker = OCSORTTracker()

        caught = self._user_warnings(tracker, frames=3)

        assert caught == []

    def test_disabling_cmc_after_construction_stops_estimation(self) -> None:
        """Switching ``enable_cmc`` off at runtime means no estimate is requested; a passed frame is reported unused."""
        tracker = OCSORTTracker(enable_cmc=True)
        tracker.cmc = _ScriptedMotion([])  # type: ignore[assignment]  # any estimate() call would raise IndexError
        tracker.enable_cmc = False

        with pytest.warns(UserWarning, match="does not use it"):
            tracker.update(_detection(300.0), frame=_FRAME)

    def test_enabling_cmc_after_construction_without_an_estimator_is_ignored(self) -> None:
        """Flipping ``enable_cmc`` on when no CMC instance was built neither crashes nor pretends to compensate."""
        tracker = OCSORTTracker()
        tracker.enable_cmc = True

        with pytest.warns(UserWarning, match="does not use it"):
            result = tracker.update(_detection(300.0), frame=_FRAME)

        assert len(result) == 1

    def test_unknown_cmc_method_raises_when_cmc_is_enabled(self) -> None:
        """An unsupported method name is rejected with the list of valid methods once CMC is actually requested."""
        with pytest.raises(ValueError, match="Unknown CMC method"):
            OCSORTTracker(enable_cmc=True, cmc_method="bogus")  # type: ignore[arg-type]

    @pytest.mark.parametrize("method", ["sparseOptFlow", "orb", "sift", "ecc"])
    def test_every_documented_cmc_method_builds_and_tracks(self, method: str) -> None:
        """Each method constructs and survives two real frames; the tracker still outputs one row per detection."""
        tracker = OCSORTTracker(enable_cmc=True, cmc_method=method)  # type: ignore[arg-type]
        assert tracker.cmc is not None
        assert tracker.cmc.cfg.method == method

        tracker.update(_detection(300.0), frame=_textured_frame())
        result = tracker.update(_detection(300.0), frame=_textured_frame(0, 6))

        assert len(result) == 1
        assert result.tracker_id is not None

    def test_tracker_reset_returns_real_cmc_to_identity(self) -> None:
        """After ``reset()`` the next estimate is the identity again, even though a real pan was being tracked."""
        tracker = OCSORTTracker(enable_cmc=True, cmc_method="sparseOptFlow")
        cmc = tracker.cmc
        assert cmc is not None
        cmc.estimate(_textured_frame())
        assert not np.array_equal(cmc.estimate(_textured_frame(4, -24)), np.eye(2, 3))

        tracker.reset()

        np.testing.assert_array_equal(cmc.estimate(_textured_frame(8, -48)), np.eye(2, 3))


class TestRealCameraMotionEndToEnd:
    """Real ``CMC`` (sparse optical flow) on frames that are exact ``np.roll`` shifts of one fixed texture.

    The scripted-affine tests above fix the transform by hand; these use the estimator itself, so a flipped sign or
    swapped axis in the estimate-to-warp chain shows up. Pixels come from ``_textured_frame`` (no random generator), and
    a 24 px / 4 px shift is well inside the range this method tracks reliably.
    """

    def test_estimate_recovers_the_roll_direction_and_size(self) -> None:
        """Content rolled 24 px left and 4 px down gives a near-identity linear part and translation (-24, +4)."""
        cmc = CMC(CMCConfig(method="sparseOptFlow", downscale=2))
        cmc.estimate(_textured_frame())

        estimate = cmc.estimate(_textured_frame(4, -24))

        np.testing.assert_allclose(estimate[:, :2], np.eye(2), atol=0.02)
        np.testing.assert_allclose(estimate[:, 2], [-24.0, 4.0], atol=1.0)

    def test_static_object_keeps_its_id_and_prediction_through_a_real_pan(self) -> None:
        """A world-fixed object seen while the camera pans: its id never changes and stage 1 predicts its new box.

        The object's detection moves with the rolled texture (24 px left, 4 px down per frame). With the estimate
        applied in the right direction the first-stage track box equals the detection to within a couple of pixels; a
        flipped sign would put it about 48 px away.
        """
        recorder = _RecordingIoU()
        tracker = OCSORTTracker(enable_cmc=True, cmc_method="sparseOptFlow", iou=recorder)
        ids = []
        for step in range(6):
            box = np.array([[200.0 - 24.0 * step, 100.0 + 4.0 * step, 260.0 - 24.0 * step, 200.0 + 4.0 * step]])
            detections = sv.Detections(xyxy=box, confidence=np.array([0.9]))
            result = tracker.update(detections, frame=_textured_frame(4 * step, -24 * step))
            assert result.tracker_id is not None
            ids.append(int(result.tracker_id[0]))

        assert ids[0] == -1
        assert ids[1] != -1
        assert ids[1:] == [ids[1]] * 5
        np.testing.assert_allclose(recorder.track_boxes[-1], box, atol=3.0)
