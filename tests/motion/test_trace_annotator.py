# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import Any

import cv2
import numpy as np
import pytest
import supervision as sv

from trackers import HomographyTransformation, IdentityTransformation, MotionAwareTraceAnnotator


def _detection_at(x: int, y: int) -> sv.Detections:
    """Create one tracked box centered at a fixed point."""
    return sv.Detections(
        xyxy=np.array([[x - 2, y - 2, x + 2, y + 2]], dtype=np.float32),
        tracker_id=np.array([7]),
    )


@pytest.mark.parametrize("shift", [0, 100, -100])
@pytest.mark.parametrize("thickness", [1, 2])
def test_trace_preserves_projected_segments(shift: int, thickness: int) -> None:
    """Clip the original path, including a sample moved out of frame by the camera."""
    scene = np.zeros((64, 64, 3), dtype=np.uint8)
    trace = MotionAwareTraceAnnotator(color=sv.Color.WHITE, thickness=thickness)
    trace.annotate(scene.copy(), _detection_at(10, 10))
    transform = HomographyTransformation(np.array([[1, 0, 0], [0, 1, -shift], [0, 0, 1]], dtype=float))
    trace.annotate(scene.copy(), _detection_at(30, 10), coord_transform=transform)
    actual = trace.annotate(scene.copy(), _detection_at(50, 10))

    path = np.array([[10, 10], [30, 10 + shift], [50, 10]], dtype=np.int32)
    expected = cv2.polylines(scene.copy(), [path], False, (255, 255, 255), thickness)
    np.testing.assert_array_equal(actual, expected)


def test_trace_draws_crossing_segment_with_both_endpoints_outside() -> None:
    """Retain an earlier segment that crosses the frame between two offscreen points."""
    scene = np.zeros((64, 64, 3), dtype=np.uint8)
    trace = MotionAwareTraceAnnotator(color=sv.Color.WHITE, thickness=1)
    for x_shift in (40, -70):
        transform = HomographyTransformation(np.array([[1, 0, x_shift], [0, 1, 0], [0, 0, 1]], dtype=float))
        trace.annotate(scene.copy(), _detection_at(30, 30), coord_transform=transform)
    actual = trace.annotate(scene.copy(), _detection_at(50, 50))

    path = np.array([[-10, 30], [100, 30], [50, 50]], dtype=np.int32)
    expected = cv2.polylines(scene.copy(), [path], False, (255, 255, 255), 1)
    np.testing.assert_array_equal(actual, expected)


class _GapProjection(IdentityTransformation):
    """Return an invalid projection for one interior point of the trace."""

    def __init__(self, invalid_x: float, dtype: type[np.float16] | type[np.float64]) -> None:
        self.invalid_x = invalid_x
        self.dtype = dtype

    def abs_to_rel(self, points: np.ndarray) -> np.ndarray:
        """Project the sample at x=32 outside OpenCV's coordinate domain."""
        projected = points.astype(self.dtype)
        projected[projected[:, 0] == 32, 0] = self.invalid_x
        return projected


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    ("invalid_x", "dtype"),
    [(np.nan, np.float64), (np.inf, np.float64), (float(2**31), np.float64), (-np.inf, np.float16)],
)
def test_trace_does_not_bridge_an_invalid_projection(
    invalid_x: float, dtype: type[np.float16] | type[np.float64]
) -> None:
    """Draw each valid run independently when a point cannot be drawn."""
    scene = np.zeros((64, 64, 3), dtype=np.uint8)
    trace = MotionAwareTraceAnnotator(color=sv.Color.WHITE, thickness=1)
    for x in (8, 16, 32, 48):
        trace.annotate(scene.copy(), _detection_at(x, 10))
    actual = trace.annotate(scene.copy(), _detection_at(56, 10), coord_transform=_GapProjection(invalid_x, dtype))

    paths = [
        np.array([[8, 10], [16, 10]], dtype=np.int32),
        np.array([[48, 10], [56, 10]], dtype=np.int32),
    ]
    expected = cv2.polylines(scene.copy(), paths, False, (255, 255, 255), 1)
    np.testing.assert_array_equal(actual, expected)


def test_trace_draws_far_crossing_with_bounded_endpoints() -> None:
    """Draw the part of a long thick stroke that crosses the frame."""
    scene = np.zeros((64, 64, 3), dtype=np.uint8)
    trace = MotionAwareTraceAnnotator(color=sv.Color.WHITE, thickness=3)
    for y_shift in (1_000_000, -1_000_000):
        transform = HomographyTransformation(np.array([[1, 0, 0], [0, 1, y_shift], [0, 0, 1]], dtype=float))
        trace.annotate(scene.copy(), _detection_at(20, 30), coord_transform=transform)
    actual = trace.annotate(scene.copy(), _detection_at(20, 30))

    expected = cv2.line(scene.copy(), (20, 0), (20, 63), (255, 255, 255), 3)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(0, 64, 3), (64, 0, 3)])
def test_trace_keeps_history_when_scene_is_empty(shape: tuple[int, int, int]) -> None:
    """Return empty canvases unchanged and retain the samples for a later render."""
    scene = np.zeros((64, 64, 3), dtype=np.uint8)
    empty = np.zeros(shape, dtype=np.uint8)
    trace = MotionAwareTraceAnnotator(color=sv.Color.WHITE, thickness=1)
    trace.annotate(scene.copy(), _detection_at(10, 10))
    assert trace.annotate(empty, _detection_at(30, 30)) is empty
    actual = trace.annotate(scene.copy(), _detection_at(50, 10))

    path = np.array([[10, 10], [30, 30], [50, 10]], dtype=np.int32)
    expected = cv2.polylines(scene.copy(), [path], False, (255, 255, 255), 1)
    np.testing.assert_array_equal(actual, expected)


class _FixedProjection(IdentityTransformation):
    """Project trajectory samples to explicitly supplied coordinates."""

    def __init__(self, endpoints: np.ndarray) -> None:
        self.endpoints = endpoints

    def abs_to_rel(self, points: np.ndarray) -> np.ndarray:
        """Return the frame coordinates, retaining their integer dtype."""
        assert len(points) == len(self.endpoints)
        return self.endpoints


@pytest.mark.parametrize(
    ("projected", "visible"),
    [
        ([(20, 20), (2**31 - 1, 20)], [(20, 20), (66, 20)]),
        ([(32, -(2**31)), (32, 32)], [(32, -3), (32, 32)]),
        ([(-(2**31), -1), (2**31 - 1, -1)], [(-3, -1), (66, -1)]),
    ],
)
def test_trace_clips_int32_endpoints(
    projected: list[tuple[int, int]],
    visible: list[tuple[int, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the visible parts and stroke halo without overflow at int32 limits."""
    polylines = cv2.polylines

    def draw_bounded(scene: np.ndarray, paths: list[np.ndarray], **kwargs: Any) -> np.ndarray:
        """Check the draw boundary before using the actual OpenCV rasterizer."""
        for path in paths:
            assert ((path >= -3) & (path <= 66)).all()
        return polylines(scene, paths, **kwargs)

    monkeypatch.setattr(cv2, "polylines", draw_bounded)
    scene = np.zeros((64, 64, 3), dtype=np.uint8)
    trace = MotionAwareTraceAnnotator(color=sv.Color.WHITE, thickness=3)
    trace.annotate(scene.copy(), _detection_at(20, 20))
    actual = trace.annotate(
        scene.copy(),
        _detection_at(32, 32),
        coord_transform=_FixedProjection(np.array(projected, dtype=np.int32)),
    )

    expected = polylines(scene.copy(), [np.array(visible, dtype=np.int32)], False, (255, 255, 255), 3)
    np.testing.assert_array_equal(actual, expected)


def test_far_sample_does_not_change_a_nearby_segment() -> None:
    """Keep native pixel rounding for an ordinary segment next to a far segment."""
    scene = np.zeros((64, 64, 3), dtype=np.uint8)
    trace = MotionAwareTraceAnnotator(color=sv.Color.WHITE, thickness=3)
    trace.annotate(scene.copy(), _detection_at(10, 10))
    trace.annotate(scene.copy(), _detection_at(20, 20))
    projected = np.array([[10, 10], [45, 82], [1_000_000, 1_000_000]], dtype=np.int32)
    actual = trace.annotate(scene.copy(), _detection_at(30, 30), coord_transform=_FixedProjection(projected))

    expected = cv2.polylines(scene.copy(), [projected[:2]], False, (255, 255, 255), 3)
    np.testing.assert_array_equal(actual, expected)
