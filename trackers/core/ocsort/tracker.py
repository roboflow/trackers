import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.sort.kalman_box_tracker import SORTKalmanBoxTracker
from trackers.utils.sort_utils import (
    get_alive_trackers,
    get_iou_matrix,
    update_detections_with_track_ids,
)


class OCSORTTracker(BaseTracker):
    """Implements OC-SORT (Observation Centric Simple Online and Realtime Tracking).

    OC-SORT remains Simple, Online, and Real-Time butimproves robustness during occlusion and non-linear motion.
    It recognizes limitations from SORT and the linear motion assumption of the Kalman filter, and adds three
    mechanisms to enhance tracking:
        1. Observation-Centre Re-Update (ORU): runs a predict-update loop with a 'virtual trajectory'
            depending on the last observation and new observation when a track is re-activated after being lost.
        2. Observation-Centric Momentum (OCM): Adjusts predicted positions based on the last observation to better handle occlusions.
        3. Observation-centric Recovery (OCR): A second-stage association step between the last observation of unmatched tracks
            to the unmatched observations after the usual association. It attempts to recover tracks that were lost
            due to object stopping or short-term occlusion.
    Args:
        lost_track_buffer (int): Number of frames to buffer when a track is lost.
            Increasing lost_track_buffer enhances occlusion handling, significantly
            improving tracking through occlusions, but may increase the possibility
            of ID switching for objects with similar appearance.
        frame_rate (float): Frame rate of the video (frames per second).
            Used to calculate the maximum time a track can be lost.
        minimum_consecutive_frames (int): Number of consecutive frames that an object
            must be tracked before it is considered a 'valid' track. Increasing
            `minimum_consecutive_frames` prevents the creation of accidental tracks
            from false detection or double detection, but risks missing shorter
            tracks. Before the tracker is considered valid, it will be assigned
            `-1` as its `tracker_id`.
        minimum_iou_threshold (float): IOU threshold for associating detections to
            existing tracks.
        inertia_weight (float): Weight for inertia term in association cost. Higher values give more importance
            to the angle difference between the motion direction and the association direction.
    """  # noqa: E501

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
        inertia_weight: float = 0.2,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.inertia_weight = inertia_weight

        # Active trackers
        self.trackers: list[SORTKalmanBoxTracker] = []

    def _get_associated_indices(
        self, iou_matrix: np.ndarray, detection_boxes: np.ndarray
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """
        Associate detections to trackers based on IOU

        Args:
            iou_matrix (np.ndarray): IOU cost matrix.
            detection_boxes (np.ndarray): Detected bounding boxes in the
                form [x1, y1, x2, y2].

        Returns:
            tuple[list[tuple[int, int]], set[int], set[int]]: Matched indices,
                unmatched trackers, unmatched detections.
        """
        matched_indices = []
        unmatched_trackers = set(range(len(self.trackers)))
        unmatched_detections = set(range(len(detection_boxes)))

        if len(self.trackers) > 0 and len(detection_boxes) > 0:
            # Find optimal assignment using scipy.optimize.linear_sum_assignment.
            # Note that it uses a a modified Jonker-Volgenant algorithm with no
            # initialization instead of the Hungarian algorithm as mentioned in the
            # SORT paper.
            row_indices, col_indices = linear_sum_assignment(iou_matrix, maximize=True)
            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] >= self.minimum_iou_threshold:
                    matched_indices.append((row, col))
                    unmatched_trackers.remove(row)
                    unmatched_detections.remove(col)

        return matched_indices, unmatched_trackers, unmatched_detections

    def _spawn_new_trackers(
        self,
        detections: sv.Detections,
        detection_boxes: np.ndarray,
        unmatched_detections: set[int],
    ) -> None:
        """
        Create new trackers only for unmatched detections with confidence
        above threshold.

        Args:
            detections (sv.Detections): The latest set of object detections.
            detection_boxes (np.ndarray): Detected bounding boxes in the
                form [x1, y1, x2, y2].
        """
        for detection_idx in unmatched_detections:
            if (
                detections.confidence is None
                or detection_idx >= len(detections.confidence)
                or detections.confidence[detection_idx]
                >= self.track_activation_threshold
            ):
                new_tracker = SORTKalmanBoxTracker(detection_boxes[detection_idx])
                self.trackers.append(new_tracker)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Updates the tracker state with new detections.

        Performs Kalman filter prediction, associates detections with existing
        trackers based on IOU, updates matched trackers, and initializes new
        trackers for unmatched high-confidence detections.

        Args:
            detections (sv.Detections): The latest set of object detections from a frame.

        Returns:
            sv.Detections: A copy of the input detections, augmented with assigned
                `tracker_id` for each successfully tracked object. Detections not
                associated with a track will not have a `tracker_id`.
        """  # noqa: E501

        if len(self.trackers) == 0 and len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections

        # Convert detections to a (N x 4) array (x1, y1, x2, y2)
        detection_boxes = (
            detections.xyxy if len(detections) > 0 else np.array([]).reshape(0, 4)
        )

        # Predict new locations for existing trackers
        for tracker in self.trackers:
            tracker.predict()

        # Build IOU cost matrix between detections and predicted bounding boxes
        iou_matrix = get_iou_matrix(self.trackers, detection_boxes)

        # Associate detections to trackers based on IOU
        matched_indices, _, unmatched_detections = self._get_associated_indices(
            iou_matrix, detection_boxes
        )

        # Update matched trackers with assigned detections
        for row, col in matched_indices:
            self.trackers[row].update(detection_boxes[col])

        self._spawn_new_trackers(detections, detection_boxes, unmatched_detections)

        # Remove dead trackers
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
        """Resets the tracker's internal state.

        Clears all active tracks and resets the track ID counter.
        """
        self.trackers = []
        SORTKalmanBoxTracker.count_id = 0
