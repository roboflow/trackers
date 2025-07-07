from __future__ import annotations

import numpy as np
import supervision as sv
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from supervision.detection.utils import box_iou_batch

from trackers.core.base import BaseTracker
from trackers.core.oc_sort.kalman_box_tracker import OCSORTKalmanBoxTracker
from trackers.utils.sort_utils import (
    get_alive_trackers,
    get_iou_matrix,
    update_detections_with_track_ids,
)


def get_center(bbox: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def get_velocity(
    history: list[NDArray[np.float32]], ocm_velocity_time_interval: int = 3
) -> NDArray[np.float32]:
    if len(history) < ocm_velocity_time_interval + 1:
        return np.array([0, 0])
    return (
        get_center(history[-1]) - get_center(history[-ocm_velocity_time_interval - 1])
    ) / ocm_velocity_time_interval


class OCSORTTracker(BaseTracker):
    """Implements OC-SORT (Observation-Centric Simple Online and Realtime Tracking).

    OC-SORT is an extension of SORT that improves tracking robustness, especially
    in scenarios with occlusions and non-linear motion. It introduces Observation-
    Centric Re-Update (ORU), Observation-Centric Momentum (OCM), and
    Observation-Centric Recovery (OCR).

    Args:
        lost_track_buffer (int): Number of frames to buffer when a track is lost.
        frame_rate (float): Frame rate of the video.
        track_activation_threshold (float): Detection confidence for track activation.
        minimum_consecutive_frames (int): Min frames for a track to be valid.
        minimum_iou_threshold (float): IOU threshold for association.
        ocm_cost_weight (float): Weight for the OCM cost term.
        ocm_velocity_time_interval (int): Time interval for calculating velocity
            direction in OCM.
        ocm_observation_history_size (int): The maximum number of past
            observations to store for each track, used for OCM.
    """

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
        ocm_cost_weight: float = 0.2,
        ocm_velocity_time_interval: int = 3,
        ocm_observation_history_size: int = 30,
    ) -> None:
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold
        self.ocm_cost_weight = ocm_cost_weight
        self.ocm_velocity_time_interval = ocm_velocity_time_interval
        self.ocm_observation_history_size = ocm_observation_history_size

        self.trackers: list[OCSORTKalmanBoxTracker] = []

    def _get_ocm_cost_matrix(
        self, trackers: list[OCSORTKalmanBoxTracker], detections: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Calculate the OCM cost matrix based on velocity direction consistency.
        """
        cost_matrix = np.zeros((len(trackers), len(detections)))
        for i, tracker in enumerate(trackers):
            if len(tracker.history_observations) < self.ocm_velocity_time_interval + 1:
                continue
            track_velocity = get_velocity(
                tracker.history_observations, self.ocm_velocity_time_interval
            )
            if np.all(track_velocity == 0):
                continue
            for j, detection in enumerate(detections):
                intention_velocity = (
                    get_center(detection) - get_center(tracker.last_observation)
                ) / (tracker.time_since_update + 1)

                # Calculate angle difference as specified in the paper
                if np.linalg.norm(intention_velocity) < 1e-6:
                    continue

                track_angle = np.arctan2(track_velocity[1], track_velocity[0])
                intention_angle = np.arctan2(
                    intention_velocity[1], intention_velocity[0]
                )
                angle_diff = abs(track_angle - intention_angle)

                # Normalize to [theta, pi] range
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

                # Convert to cost (smaller angle difference = lower cost)
                cost_matrix[i, j] = angle_diff

        return cost_matrix

    def _associate_detections(
        self, iou_matrix: np.ndarray, ocm_matrix: np.ndarray
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        # Combine OCM angle cost with IoU cost
        cost_matrix = self.ocm_cost_weight * ocm_matrix - iou_matrix
        matched_indices = []
        unmatched_trackers = set(range(len(self.trackers)))
        unmatched_detections = set(range(iou_matrix.shape[1]))

        if iou_matrix.shape[0] > 0 and iou_matrix.shape[1] > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] >= self.minimum_iou_threshold:
                    matched_indices.append((row, col))
                    unmatched_trackers.remove(row)
                    unmatched_detections.remove(col)

        return matched_indices, unmatched_trackers, unmatched_detections

    def _ocr_associate(
        self,
        unmatched_trackers: set[int],
        unmatched_detections: set[int],
        detection_boxes: np.ndarray,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        unmatched_tracker_list = list(unmatched_trackers)
        unmatched_detection_list = list(unmatched_detections)

        if not unmatched_tracker_list or not unmatched_detection_list:
            return [], unmatched_trackers, unmatched_detections

        last_observations = np.array(
            [self.trackers[i].last_observation for i in unmatched_tracker_list]
        )
        unmatched_det_boxes = detection_boxes[unmatched_detection_list]

        ocr_iou_matrix = box_iou_batch(last_observations, unmatched_det_boxes)

        ocr_matched_indices_map = []
        if ocr_iou_matrix.shape[0] > 0 and ocr_iou_matrix.shape[1] > 0:
            row, col = linear_sum_assignment(ocr_iou_matrix, maximize=True)
            for r, c in zip(row, col):
                if ocr_iou_matrix[r, c] >= self.minimum_iou_threshold:
                    ocr_matched_indices_map.append((r, c))

        ocr_matched_indices = [
            (unmatched_tracker_list[r], unmatched_detection_list[c])
            for r, c in ocr_matched_indices_map
        ]

        ocr_matched_trackers = {idx[0] for idx in ocr_matched_indices}
        ocr_matched_detections = {idx[1] for idx in ocr_matched_indices}
        new_unmatched_trackers = unmatched_trackers - ocr_matched_trackers
        new_unmatched_detections = unmatched_detections - ocr_matched_detections

        return ocr_matched_indices, new_unmatched_trackers, new_unmatched_detections

    def _spawn_new_trackers(
        self,
        detections: sv.Detections,
        detection_boxes: np.ndarray,
        unmatched_detections: set[int],
    ) -> None:
        for detection_idx in unmatched_detections:
            if (
                detections.confidence is None
                or detection_idx >= len(detections.confidence)
                or detections.confidence[detection_idx]
                >= self.track_activation_threshold
            ):
                new_tracker = OCSORTKalmanBoxTracker(
                    detection_boxes[detection_idx], self.ocm_observation_history_size
                )
                self.trackers.append(new_tracker)

    def update(self, detections: sv.Detections) -> sv.Detections:
        if len(self.trackers) == 0 and len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections

        detection_boxes = (
            detections.xyxy if len(detections) > 0 else np.array([]).reshape(0, 4)
        )

        for tracker in self.trackers:
            tracker.predict()

        iou_matrix = get_iou_matrix(self.trackers, detection_boxes)
        ocm_matrix = self._get_ocm_cost_matrix(self.trackers, detection_boxes)

        matched, unmatched_tr, unmatched_det = self._associate_detections(
            iou_matrix, ocm_matrix
        )

        for row, col in matched:
            tracker = self.trackers[row]
            if tracker.time_since_update > 1:
                tracker.apply_oru(detection_boxes[col])
            tracker.update(detection_boxes[col])

        ocr_matched, unmatched_tr, unmatched_det = self._ocr_associate(
            unmatched_tr, unmatched_det, detection_boxes
        )

        for row, col in ocr_matched:
            tracker = self.trackers[row]
            tracker.update(detection_boxes[col])

        self._spawn_new_trackers(detections, detection_boxes, unmatched_det)

        self.trackers = get_alive_trackers(
            self.trackers,
            self.minimum_consecutive_frames,
            self.maximum_frames_without_update,
        )

        updated_detections = update_detections_with_track_ids(
            self.trackers,
            detections,
            detection_boxes,
            self.minimum_iou_threshold,
            self.minimum_consecutive_frames,
        )

        return updated_detections

    def reset(self) -> None:
        self.trackers = []
        OCSORTKalmanBoxTracker.count_id = 0
