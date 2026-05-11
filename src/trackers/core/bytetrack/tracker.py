# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import ClassVar

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.bytetrack.tracklet import ByteTrackTracklet
from trackers.core.bytetrack.utils import _get_alive_tracklets
from trackers.utils.iou import BaseIoU, IoU
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XYXYStateEstimator,
)


class ByteTrackTracker(BaseTracker):
    """ByteTrack operates online by processing all detector outputs, categorizing them
    by confidence thresholds to enable a two-stage association process. High-score boxes
    are initially linked to tracklets via Kalman filter predictions and IoU-based
    Hungarian matching, optionally enhanced with appearance features. Low-score boxes
    follow in a secondary matching phase using pure motion similarity to revive occluded
    tracks. Tracks without matches are kept briefly for potential re-association,
    preventing premature termination. This inclusive approach addresses common pitfalls
    in detection filtering, establishing ByteTrack as a flexible enhancer for existing
    tracking frameworks.

    ByteTrack excels in dense environments, where its low-score recovery mechanism
    minimizes missed detections and enhances overall trajectory completeness. It
    consistently improves performance across diverse datasets, demonstrating robustness
    and generalization. The tracker's speed remains competitive, facilitating
    integration into production pipelines. On the downside, it is highly dependent on
    detector quality, with performance drops in noisy or low-resolution inputs.
    Additionally, the motion-only secondary association may lead to erroneous matches
    in scenes with similar moving objects.

    Args:
        lost_track_buffer: `int` specifying number of frames to buffer when a
            track is lost. Increasing this value enhances occlusion handling but
            may increase ID switching for disappearing objects.
        frame_rate: `float` specifying video frame rate in frames per second.
            Used to scale the lost track buffer for consistent tracking across
            different frame rates.
        track_activation_threshold: `float` specifying minimum detection
            confidence to create new tracks. Higher values reduce false
            positives but may miss low-confidence objects.
        minimum_consecutive_frames: `int` specifying number of consecutive
            frames before a track is considered valid. Before reaching this
            threshold, tracks are assigned `tracker_id` of `-1`.
        minimum_iou_threshold: `float` specifying IoU threshold for associating
            detections to existing tracks. Higher values require more overlap.
        high_conf_det_threshold: `float` specifying threshold for separating
            high and low confidence detections in the two-stage association.
        state_estimator_class: State estimator class to use for Kalman filter.
            Defaults to `XYXYStateEstimator`. Can also use
            `XCYCSRStateEstimator` for center-based representation.
        iou: IoU similarity metric instance to use for data association.
            Defaults to standard `IoU`. Can be replaced with any `BaseIoU`
            subclass (e.g. GIoU, DIoU, CIoU) to change how bounding-box
            similarity is computed during the association step.
            Passing ``None`` (the default) is equivalent to ``IoU()`` and is
            provided for backward compatibility with existing code that did not
            supply an ``iou`` argument.
    """

    tracker_id = "bytetrack"

    search_space: ClassVar[dict[str, dict]] = {
        "lost_track_buffer": {"type": "randint", "range": [10, 91]},
        "track_activation_threshold": {"type": "uniform", "range": [0.1, 0.9]},
        "minimum_iou_threshold": {"type": "uniform", "range": [0.05, 0.7]},
        "high_conf_det_threshold": {"type": "uniform", "range": [0.3, 0.8]},
        "minimum_consecutive_frames": {"type": "randint", "range": [1, 4]},
    }

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.1,
        high_conf_det_threshold: float = 0.6,
        state_estimator_class: type[BaseStateEstimator] = XYXYStateEstimator,
        iou: BaseIoU | None = None,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold
        self.high_conf_det_threshold = high_conf_det_threshold
        self.tracks: list[ByteTrackTracklet] = []
        self.state_estimator_class = state_estimator_class
        self.iou = iou if iou is not None else IoU()

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
    ) -> sv.Detections:
        """Update tracks state with new detections and return tracked objects.
        Performs Kalman filter prediction, two-stage association (high then low
        confidence), and initializes new tracks for unmatched detections.

        Args:
            detections: `sv.Detections` containing bounding boxes with shape
                `(N, 4)` in `(x_min, y_min, x_max, y_max)` format and optional
                confidence scores.
            frame: Ignored by ByteTrack. If provided (not `None`), a warning is
                emitted.

        Returns:
            sv.Detections with tracker_id assigned for each detection.
            Unmatched detections have tracker_id of -1. Detection order may
            differ from input.
        """
        self._warn_if_frame_unused(frame)
        if len(self.tracks) == 0 and len(detections) == 0:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        out_det_indices: list[int] = []
        out_tracker_ids: list[int] = []

        for tracker in self.tracks:
            tracker.predict()

        detection_boxes = detections.xyxy
        confidences = detections.confidence if detections.confidence is not None else np.zeros(len(detections))

        # Split indices by confidence threshold (no sv.Detections slicing)
        high_mask = confidences >= self.high_conf_det_threshold
        high_indices = np.where(high_mask)[0]
        low_indices = np.where(~high_mask)[0]
        high_boxes = detection_boxes[high_indices]
        low_boxes = detection_boxes[low_indices]

        # Step 1: associate high-confidence detections to all tracks
        predicted_boxes = np.array([t.get_state_bbox() for t in self.tracks]) if self.tracks else np.empty((0, 4))
        iou_matrix = self.iou.compute(predicted_boxes, high_boxes)
        matched, unmatched_tracks, unmatched_high = self._get_associated_indices(iou_matrix, self.minimum_iou_threshold)

        for row, col in matched:
            track = self.tracks[row]
            track.update(high_boxes[col])
            if (
                track.number_of_successful_consecutive_updates >= self.minimum_consecutive_frames
                and track.tracker_id == -1
            ):
                track.tracker_id = ByteTrackTracklet.get_next_tracker_id()
            out_det_indices.append(int(high_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]

        # Step 2: associate low-confidence detections to remaining tracks
        remaining_boxes = predicted_boxes[unmatched_tracks] if unmatched_tracks else np.empty((0, 4))
        iou_matrix = self.iou.compute(remaining_boxes, low_boxes)
        matched, _, unmatched_low = self._get_associated_indices(iou_matrix, self.minimum_iou_threshold)

        for row, col in matched:
            track = remaining_tracks[row]
            track.update(low_boxes[col])
            if (
                track.number_of_successful_consecutive_updates >= self.minimum_consecutive_frames
                and track.tracker_id == -1
            ):
                track.tracker_id = ByteTrackTracklet.get_next_tracker_id()
            out_det_indices.append(int(low_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        # Unmatched low-confidence detections
        for det_local_idx in unmatched_low:
            out_det_indices.append(int(low_indices[det_local_idx]))
            out_tracker_ids.append(-1)

        # Spawn new tracks from unmatched high-confidence detections
        self._spawn_new_tracks(
            detection_boxes,
            confidences,
            unmatched_high,
            high_indices,
            out_det_indices,
            out_tracker_ids,
        )

        self.tracks = _get_alive_tracklets(
            tracklets=self.tracks,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            maximum_frames_without_update=self.maximum_frames_without_update,
        )

        # Build final sv.Detections from original by indexing
        if not out_det_indices:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        idx = np.array(out_det_indices)
        result = detections[idx]
        result.tracker_id = np.array(out_tracker_ids, dtype=int)
        return result

    def _get_associated_indices(
        self,
        similarity_matrix: np.ndarray,
        min_similarity_thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Associate detections to tracks based on Similarity (IoU) using the
        Jonker-Volgenant algorithm approach with no initialization instead of the
        Hungarian algorithm as mentioned in the SORT paper, but it solves the
        assignment problem in an optimal way.

        Args:
            similarity_matrix: Similarity matrix between tracks (rows) and detections (columns).
            min_similarity_thresh: Minimum similarity threshold for a valid match.

        Returns:
            matched: List of ``(tracker_idx, detection_idx)`` tuples for
                associations that meet the similarity threshold.
            unmatched_tracks: Sorted list of track indices not matched to any
                detection.
            unmatched_detections: Sorted list of detection indices not matched
                to any track.
        """
        matched_indices = []
        n_tracks, n_detections = similarity_matrix.shape
        unmatched_tracks = set(range(n_tracks))
        unmatched_detections = set(range(n_detections))

        if n_tracks > 0 and n_detections > 0:
            row_indices, col_indices = linear_sum_assignment(similarity_matrix, maximize=True)
            for row, col in zip(row_indices, col_indices):
                if similarity_matrix[row, col] >= min_similarity_thresh:
                    matched_indices.append((row, col))
                    unmatched_tracks.remove(row)
                    unmatched_detections.remove(col)

        # Return sorted lists for deterministic order across CPython versions.
        return matched_indices, sorted(unmatched_tracks), sorted(unmatched_detections)

    def _spawn_new_tracks(
        self,
        detection_boxes: np.ndarray,
        confidences: np.ndarray,
        unmatched_high_local: list[int],
        high_indices: np.ndarray,
        out_det_indices: list[int],
        out_tracker_ids: list[int],
    ) -> None:
        for det_local_idx in unmatched_high_local:
            global_idx = int(high_indices[det_local_idx])
            conf = float(confidences[global_idx])
            if conf >= self.track_activation_threshold:
                self.tracks.append(
                    ByteTrackTracklet(
                        initial_bbox=detection_boxes[global_idx],
                        state_estimator_class=self.state_estimator_class,
                    )
                )
                out_det_indices.append(global_idx)
                out_tracker_ids.append(-1)

    def reset(self) -> None:
        """Reset tracker state by clearing all tracks and resetting ID counter.
        Call this method when switching to a new video or scene.
        """
        self.tracks = []
        ByteTrackTracklet.count_id = 0
