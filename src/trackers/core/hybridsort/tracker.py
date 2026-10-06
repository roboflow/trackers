# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import ClassVar, cast

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.hybridsort.tracklet import HybridSORTTracklet
from trackers.core.hybridsort.utils import _build_corner_direction_consistency_matrix
from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.utils.detections import default_confidences
from trackers.utils.iou import BaseIoU, HMIoU
from trackers.utils.predict_timing import PredictTiming
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCSRStateEstimator,
)

# Lower bound of the low-confidence band: detections below `high_conf_det_threshold` and at or below this
# confidence never enter association (BoT-SORT uses the same floor). It does not bound the high-confidence band:
# when `high_conf_det_threshold <= 0.1`, every detection at or above the threshold is high confidence and the
# low-confidence stage is empty, as in the reference implementation.
_LOW_CONF_DET_FLOOR = 0.1


def _assign(
    score_matrix: np.ndarray,
    gate_matrix: np.ndarray,
    minimum_score: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve a maximum-score assignment and keep pairs whose gate value passes.

    Args:
        score_matrix: `(n_tracks, n_detections)` matrix maximized by the
            Hungarian algorithm.
        gate_matrix: Matrix of the same shape; a pair is accepted only if its
            value is at least `minimum_score`.
        minimum_score: Acceptance threshold applied to `gate_matrix`.

    Returns:
        matched: `(track_index, detection_index)` pairs.
        unmatched_tracks: Sorted track indices without a match.
        unmatched_detections: Sorted detection indices without a match.
    """
    n_tracks, n_detections = score_matrix.shape
    matched: list[tuple[int, int]] = []
    unmatched_tracks = set(range(n_tracks))
    unmatched_detections = set(range(n_detections))
    if n_tracks > 0 and n_detections > 0:
        for row, col in zip(*linear_sum_assignment(score_matrix, maximize=True)):
            if gate_matrix[row, col] >= minimum_score:
                matched.append((int(row), int(col)))
                unmatched_tracks.discard(int(row))
                unmatched_detections.discard(int(col))
    return matched, sorted(unmatched_tracks), sorted(unmatched_detections)


class HybridSORTTracker(OCSORTTracker):
    """Hybrid-SORT adds weak cues to OC-SORT's strong spatial and motion cues. Where
    OC-SORT associates with IoU and velocity direction alone, Hybrid-SORT also
    uses how a tracklet's detection confidence evolves (Tracklet Confidence
    Modeling, TCM) and the agreement of box heights (Height-Modulated IoU,
    HMIoU). Both are cheap to compute and help exactly where IoU is ambiguous:
    heavily overlapping objects, whose confidence drops while occluded and
    whose heights differ with depth.

    Association runs in three stages. High-confidence detections are matched
    by HMIoU plus a four-corner direction-consistency term (robust OCM),
    penalized by the gap between each detection's confidence and the
    tracklet's Kalman-predicted confidence. Low-confidence detections then
    recover still-unmatched tracklets, ByteTrack style, penalized by the gap to
    a linearly extrapolated confidence. Finally, OC-SORT's recovery stage
    (OCR) matches the remaining high-confidence detections to the last
    observation of the remaining tracklets. Observation-centric re-update
    (ORU) smooths the box and confidence filters after an occlusion.

    Hybrid-SORT keeps OC-SORT's efficiency: it needs no appearance model,
    training, or camera motion compensation, and runs in real time on CPU. Its
    gains are largest on crowded scenes with frequent occlusion and uniform
    appearance, such as group dancing. On oracle boxes, where every detection
    has the same confidence, TCM contributes nothing.

    Note:
        Known limitation in dynamic frame-rate mode (`update` called with a
        `timestamp`): the confidence Kalman filter gets its process noise from
        the shared `KalmanMotionModel`. For an off-nominal frame step, that
        model rebuilds the position noise from the velocity variance alone
        rather than from the configured position noise, and it caches the
        noise matrix keyed on the frame step only, so the matrix built during
        the observation-centric re-update replay (which runs without a frame
        rate) is reused by the next prediction at the same step. Both
        mechanisms are pre-existing shared code that also drives the box
        filters and will be fixed separately. Fixed-rate mode is unaffected.

    Args:
        lost_track_buffer: Non-negative `int` specifying number of 30 FPS frames
            to buffer when a track is lost. `0` deletes a confirmed track on the
            first missed frame. Increasing this value enhances occlusion
            handling but may increase ID switching for similar objects.
        frame_rate: `float` specifying video frame rate in frames per second.
            Must be positive. Used to scale the lost track buffer for consistent
            tracking across different frame rates.
        minimum_consecutive_frames: `int` specifying number of consecutive
            frames before a track is considered valid. Before reaching this
            threshold, tracks are assigned `tracker_id` of `-1`.
        minimum_iou_threshold: `float` specifying the minimum similarity (HMIoU
            by default) for accepting a match. The first (high-confidence) and
            third (last-observation recovery) stages gate on the raw
            similarity. The second (low-confidence) stage gates on the
            penalized score, the similarity minus
            `confidence_weight_second_assoc` times the absolute confidence
            gap, so a larger weight effectively raises its threshold. Higher
            values require more overlap.
        direction_consistency_weight: `float` specifying weight of the
            direction-consistency (OCM) term in the first association stage;
            OC-SORT compares box centers, Hybrid-SORT all four corners. Higher
            values prioritize agreement between each track's recent motion
            direction and the direction to a candidate detection.
        high_conf_det_threshold: `float` specifying threshold for high
            confidence detections, which are matched first and can spawn new
            tracks. Detections below this threshold but strictly above `0.1`
            are only used to recover existing tracks in the second stage; the
            other detections below it never enter association. The `0.1`
            floor bounds only this low-confidence band, so when the threshold
            is `0.1` or lower every detection at or above it is high
            confidence and the second stage is empty. Unmatched detections are
            returned with `tracker_id` of `-1`.
        confidence_weight_first_assoc: `float` specifying weight of the
            confidence-consistency penalty in the first association stage: the
            absolute difference between a high-confidence detection's score and
            the track's Kalman-predicted confidence. `0` disables it.
        confidence_weight_second_assoc: `float` specifying weight of the
            confidence-consistency penalty in the second (low-confidence)
            association stage: the absolute difference between a detection's
            score and the track's linearly extrapolated confidence. `0`
            disables it.
        delta_t: `int` specifying number of past frames to use for velocity
            estimation. Higher values provide more stable direction estimates
            during occlusion.
        state_estimator_class: State estimator class to use for Kalman filter.
            Defaults to `XCYCSRStateEstimator`. Can also use
            `XYXYStateEstimator` for corner-based representation.
        iou: IoU similarity metric instance used by all three association
            stages. Defaults to `HMIoU`, as in the paper. Can be replaced with
            any `BaseIoU` subclass (e.g. `IoU`, `GIoU`).

    Example:
        Track one object across two frames; a new track is confirmed once it
        has been matched:

        >>> import numpy as np
        >>> import supervision as sv
        >>> from trackers import HybridSORTTracker
        >>> tracker = HybridSORTTracker()
        >>> for shift in (0.0, 2.0):
        ...     detections = sv.Detections(
        ...         xyxy=np.array([[10.0 + shift, 10.0, 60.0 + shift, 120.0]]),
        ...         confidence=np.array([0.9]),
        ...     )
        ...     tracked = tracker.update(detections)
        >>> tracked.tracker_id.tolist()
        [0]
    """

    tracker_id = "hybridsort"

    search_space: ClassVar[dict[str, dict]] = {
        "lost_track_buffer": {"type": "randint", "range": [10, 61]},
        "minimum_iou_threshold": {"type": "uniform", "range": [0.05, 0.4]},
        "minimum_consecutive_frames": {"type": "randint", "range": [1, 4]},
        "direction_consistency_weight": {"type": "uniform", "range": [0.0, 0.5]},
        "high_conf_det_threshold": {"type": "uniform", "range": [0.3, 0.8]},
        "confidence_weight_first_assoc": {"type": "uniform", "range": [0.0, 2.0]},
        "confidence_weight_second_assoc": {"type": "uniform", "range": [0.0, 2.0]},
        "delta_t": {"type": "randint", "range": [1, 4]},
    }

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.15,
        direction_consistency_weight: float = 0.2,
        high_conf_det_threshold: float = 0.6,
        confidence_weight_first_assoc: float = 1.0,
        confidence_weight_second_assoc: float = 1.0,
        delta_t: int = 3,
        state_estimator_class: type[BaseStateEstimator] = XCYCSRStateEstimator,
        iou: BaseIoU | None = None,
    ) -> None:
        super().__init__(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
            direction_consistency_weight=direction_consistency_weight,
            high_conf_det_threshold=high_conf_det_threshold,
            delta_t=delta_t,
            state_estimator_class=state_estimator_class,
            iou=iou if iou is not None else HMIoU(),
        )
        self.confidence_weight_first_assoc = confidence_weight_first_assoc
        self.confidence_weight_second_assoc = confidence_weight_second_assoc
        # `list` is invariant and OCSORTTracker declares `self.tracks: list[OCSORTTracklet]`, so narrowing the
        # element type needs this ignore until the parent's `tracks` is generic over the tracklet type. Tracklets
        # are only created by `_spawn_new_tracklets` below, which always builds `HybridSORTTracklet`.
        self.tracks: list[HybridSORTTracklet] = []  # type: ignore[assignment]

    def _spawn_new_tracklets(self, boxes: np.ndarray, confidences: np.ndarray | None = None) -> None:
        """Create new Hybrid-SORT tracklets from bounding boxes and their confidences.

        Overrides the OC-SORT hook so that every spawned tracklet carries the
        confidence state Hybrid-SORT needs; `confidences` is optional to keep
        the parent signature.

        Args:
            boxes: Bounding boxes `(N, 4)` in xyxy format.
            confidences: Detection confidences `(N,)`. `None` treats every box
                as confidence `1.0`, as for detections without scores.
        """
        if confidences is None:
            confidences = np.ones(len(boxes))
        for xyxy, confidence in zip(boxes, confidences):
            self.tracks.append(
                HybridSORTTracklet(
                    xyxy,
                    confidence=float(confidence),
                    delta_t=self.delta_t,
                    state_estimator_class=self.state_estimator_class,
                )
            )

    def _update_matched_tracklet(
        self, track_index: int, box: np.ndarray, confidence: float, timing: PredictTiming
    ) -> int:
        """Update a matched tracklet with its detection and return the resolved tracker ID.

        Args:
            track_index: Index of the tracklet in `self.tracks`.
            box: Matched detection box `[x1, y1, x2, y2]`.
            confidence: Matched detection confidence.
            timing: Predict timing of the current frame.

        Returns:
            The tracker ID, or `-1` while the track is immature.
        """
        tracklet = self.tracks[track_index]
        tracklet.update(box, timing, confidence=confidence)
        return tracklet.resolve_tracker_id(
            self.minimum_consecutive_frames,
            self.frame_count,
            self._allocate_tracker_id,
        )

    def _first_stage_scores(self, iou_matrix: np.ndarray, boxes: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """Score tracks against high-confidence detections: IoU + robust OCM - TCM."""
        scores = iou_matrix.copy()
        if self.direction_consistency_weight != 0:
            scores += self.direction_consistency_weight * self._compute_direction_consistency_matrix(boxes, confidences)
        if self.confidence_weight_first_assoc != 0:
            track_confidences = np.clip(
                [t.kalman_confidence for t in self.tracks], self.high_conf_det_threshold, 1.0
            ).reshape(-1, 1)
            scores -= self.confidence_weight_first_assoc * np.abs(track_confidences - confidences[np.newaxis, :])
        return scores

    def _second_stage_scores(
        self, iou_matrix: np.ndarray, track_indices: list[int], confidences: np.ndarray
    ) -> np.ndarray:
        """Score unmatched tracks against low-confidence detections: IoU - TCM (linear trend)."""
        scores = iou_matrix.copy()
        if self.confidence_weight_second_assoc != 0:
            track_confidences = np.clip(
                [self.tracks[t].linear_confidence for t in track_indices],
                _LOW_CONF_DET_FLOOR,
                self.high_conf_det_threshold,
            ).reshape(-1, 1)
            scores -= self.confidence_weight_second_assoc * np.abs(track_confidences - confidences[np.newaxis, :])
        return scores

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        """Update tracker state with new detections and return tracked objects.
        Performs Kalman filter prediction, the three association stages
        (high-confidence, low-confidence, last-observation recovery), and
        initializes new tracks for unmatched high-confidence detections.

        Args:
            detections: `sv.Detections` containing bounding boxes with shape
                `(N, 4)` in `(x_min, y_min, x_max, y_max)` format and optional
                confidence scores. When `detections.confidence is None`, all
                detections are treated as confidence `1.0` -- they are all
                high-confidence and the confidence cues carry no information.
            frame: Ignored by Hybrid-SORT. If provided (not `None`), a warning
                is emitted.
            timestamp: Absolute time of the current frame in seconds, or ``None``
                for fixed-rate mode (``frame_step = 1.0`` per call).

        Returns:
            sv.Detections with tracker_id assigned for each detection.
            Unmatched or immature tracks, and detections that were not
            associated, have tracker_id of -1. Detection order may differ
            from input.

        Warns:
            UserWarning: If ``frame`` is passed but Hybrid-SORT does not perform
                camera motion compensation (CMC), the frame is ignored.
        """
        self._warn_if_frame_unused(frame)
        timing = self._predict_timing(timestamp)
        if timing.skip_update:
            return self._detections_for_skipped_update(detections)

        if len(self.tracks) == 0 and len(detections) == 0:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        detection_boxes_full = detections.xyxy if len(detections) > 0 else np.empty((0, 4))
        confidences_full = default_confidences(detections)
        high_mask: np.ndarray
        low_mask: np.ndarray
        if detections.confidence is None:
            high_mask = np.ones(len(confidences_full), dtype=bool)
            low_mask = np.zeros(len(confidences_full), dtype=bool)
        else:
            high_mask = confidences_full >= self.high_conf_det_threshold
            low_mask = (confidences_full > _LOW_CONF_DET_FLOOR) & ~high_mask
        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]
        high_boxes = detection_boxes_full[high_indices]
        high_scores = confidences_full[high_indices]
        low_boxes = detection_boxes_full[low_indices]
        low_scores = confidences_full[low_indices]

        # Collect (detection_index, tracker_id) pairs; assembled into the output
        # sv.Detections once at the end. Every input detection appears exactly once.
        track_id_by_detection: dict[int, int] = {}

        # See OCSORTTracker.update: predicted boxes may alias live Kalman state, so
        # nothing may mutate tracklet state between this call and the decode below.
        predicted_boxes_by_tracklet = self._predict_tracklets(self.tracks, timing, return_predictions=True)
        if self._lost_track_time_budget(timing, self.maximum_time_without_update) is not None:
            self.tracks = cast(list[HybridSORTTracklet], self._prune_expired_tracklets(timing))
        predicted_boxes = np.array(
            [predicted_boxes_by_tracklet[t] for t in self.tracks]
            if not timing.skip_predict
            else [t.get_state_bbox() for t in self.tracks]
        ).reshape(-1, 4)

        # 1st association: high-confidence detections (HMIoU + robust OCM - TCM).
        iou_matrix = self.iou.compute(predicted_boxes, high_boxes)
        first_scores = self._first_stage_scores(iou_matrix, high_boxes, high_scores)
        matched, unmatched_tracks, unmatched_high = _assign(first_scores, iou_matrix, self.minimum_iou_threshold)
        for row, col in matched:
            track_id_by_detection[int(high_indices[col])] = self._update_matched_tracklet(
                row, high_boxes[col], float(high_scores[col]), timing
            )

        # 2nd association: low-confidence detections recover unmatched tracks (BYTE + TCM).
        if len(low_indices) > 0 and len(unmatched_tracks) > 0:
            low_iou_matrix = self.iou.compute(predicted_boxes[unmatched_tracks], low_boxes)
            second_scores = self._second_stage_scores(low_iou_matrix, unmatched_tracks, low_scores)
            low_matched, low_unmatched_tracks, _ = _assign(second_scores, second_scores, self.minimum_iou_threshold)
            for row, col in low_matched:
                track_id_by_detection[int(low_indices[col])] = self._update_matched_tracklet(
                    unmatched_tracks[row], low_boxes[col], float(low_scores[col]), timing
                )
            unmatched_tracks = [unmatched_tracks[i] for i in low_unmatched_tracks]

        # 3rd association (OCR): remaining high-confidence detections vs last observations.
        if len(unmatched_high) > 0 and len(unmatched_tracks) > 0:
            last_observations = np.array([self.tracks[t].last_observation for t in unmatched_tracks])
            ocr_iou_matrix = self.iou.compute(last_observations, high_boxes[unmatched_high])
            ocr_matched, ocr_unmatched_tracks, ocr_unmatched_high = _assign(
                ocr_iou_matrix, ocr_iou_matrix, self.minimum_iou_threshold
            )
            for row, col in ocr_matched:
                high_idx = unmatched_high[col]
                track_id_by_detection[int(high_indices[high_idx])] = self._update_matched_tracklet(
                    unmatched_tracks[row], high_boxes[high_idx], float(high_scores[high_idx]), timing
                )
            unmatched_tracks = [unmatched_tracks[i] for i in ocr_unmatched_tracks]
            unmatched_high = [unmatched_high[i] for i in ocr_unmatched_high]

        for track_index in unmatched_tracks:
            self.tracks[track_index].break_confidence_trend()

        self._spawn_new_tracklets(high_boxes[unmatched_high], high_scores[unmatched_high])

        # Post-association budget prune: removes tracks that exceeded budget after predict.
        self.tracks = cast(list[HybridSORTTracklet], self._prune_expired_tracklets(timing))

        # Build output -- a single index into the original detections preserves all
        # metadata (confidence, class_id, mask, data dict). Matched detections first,
        # in association order, then every other detection with tracker_id -1.
        out_det_indices = list(track_id_by_detection)
        out_det_indices += [i for i in range(len(detections)) if i not in track_id_by_detection]
        if out_det_indices:
            result = cast(sv.Detections, detections[out_det_indices])
            result.tracker_id = np.array([track_id_by_detection.get(i, -1) for i in out_det_indices], dtype=int)
        else:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)

        self.frame_count += 1
        return result

    def _compute_direction_consistency_matrix(self, detection_boxes: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """Compute the four-corner (robust OCM) direction consistency matrix, scaled by detection confidence."""
        corner_velocities = np.array(
            [t.corner_velocities if t.corner_velocities is not None else np.zeros((4, 2)) for t in self.tracks]
        ).reshape(-1, 4, 2)
        reference_boxes = np.array(
            [
                previous_obs if (previous_obs := t.get_k_previous_obs()) is not None else t.last_observation
                for t in self.tracks
            ]
        ).reshape(-1, 4)
        velocity_mask = np.array([t.corner_velocities is not None for t in self.tracks], dtype=np.float64)[
            :, np.newaxis
        ]
        matrix = _build_corner_direction_consistency_matrix(
            corner_velocities=corner_velocities,
            reference_boxes=reference_boxes,
            detection_boxes=detection_boxes,
            velocity_mask=velocity_mask,
        )
        return matrix * confidences[np.newaxis, :]
