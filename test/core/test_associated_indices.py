# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for deterministic unmatched-index ordering in _get_associated_indices.

All three trackers (SORT, ByteTrack, OC-SORT) must return sorted lists for
unmatched tracks and unmatched detections so that tracker-ID assignment is
stable across CPython versions and implementations.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from trackers.core.bytetrack.tracker import ByteTrackTracker
from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.core.sort.tracker import SORTTracker


def _call_sort(
    n_tracks: int, n_detections: int, iou_matrix: np.ndarray
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Call SORTTracker._get_associated_indices with n_tracks dummy tracks."""
    tracker = SORTTracker()
    tracker.tracks = [None] * n_tracks  # type: ignore[list-item]
    detection_boxes = np.zeros((n_detections, 4))
    return tracker._get_associated_indices(iou_matrix, detection_boxes)


def _call_bytetrack(
    n_tracks: int, n_detections: int, iou_matrix: np.ndarray
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Call ByteTrackTracker._get_associated_indices with the given matrix."""
    tracker = ByteTrackTracker()
    return tracker._get_associated_indices(iou_matrix, tracker.minimum_iou_threshold)


def _call_ocsort(
    n_tracks: int, n_detections: int, iou_matrix: np.ndarray
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Call OCSORTTracker._get_associated_indices with zeros for direction matrix."""
    tracker = OCSORTTracker()
    direction_matrix = np.zeros_like(iou_matrix)
    return tracker._get_associated_indices(iou_matrix, direction_matrix)


_TRACKER_CALL_FNS: list[
    pytest.param  # type: ignore[type-arg]
] = [
    pytest.param(_call_sort, id="sort"),
    pytest.param(_call_bytetrack, id="bytetrack"),
    pytest.param(_call_ocsort, id="ocsort"),
]

CallFn = Callable[
    [int, int, np.ndarray], tuple[list[tuple[int, int]], list[int], list[int]]
]


class TestGetAssociatedIndicesSortedOutput:
    """_get_associated_indices returns sorted unmatched lists across all trackers."""

    @pytest.mark.parametrize("call_fn", _TRACKER_CALL_FNS)
    def test_all_unmatched_detections_are_sorted(self, call_fn: CallFn) -> None:
        """All detections are unmatched when no tracks exist; result is sorted."""
        n_detections = 4
        iou_matrix = np.zeros((0, n_detections))
        _, unmatched_tracks, unmatched_detections = call_fn(0, n_detections, iou_matrix)
        assert unmatched_tracks == []
        assert unmatched_detections == list(range(n_detections))

    @pytest.mark.parametrize("call_fn", _TRACKER_CALL_FNS)
    def test_non_contiguous_unmatched_indices_are_sorted(self, call_fn: CallFn) -> None:
        """Non-contiguous unmatched indices are returned in ascending order.

        Scenario: 4 tracks, 5 detections; track 1 matches detection 2 (IoU=0.9).
        Expected unmatched_tracks  = [0, 2, 3]  (not {0, 2, 3} in arbitrary set order).
        Expected unmatched_detections = [0, 1, 3, 4].
        """
        n_tracks, n_detections = 4, 5
        iou_matrix = np.zeros((n_tracks, n_detections))
        iou_matrix[1, 2] = 0.9  # only match: track 1 ↔ detection 2

        _, unmatched_tracks, unmatched_detections = call_fn(
            n_tracks, n_detections, iou_matrix
        )

        assert unmatched_tracks == [0, 2, 3]
        assert unmatched_detections == [0, 1, 3, 4]
        assert unmatched_tracks == sorted(unmatched_tracks)
        assert unmatched_detections == sorted(unmatched_detections)
