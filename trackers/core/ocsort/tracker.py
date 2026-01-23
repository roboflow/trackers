import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.ocsort.tracklet import OCSORTTracklet
from trackers.utils.ocsort_utils import (
    add_track_id_detections,
    build_direction_consistency_matrix,
    get_iou_matrix,
    get_iou_matrix_between_boxes,
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
        direction_consistency_weight (float): Weight for inertia term in association cost. Higher values give more importance
            to the angle difference between the motion direction and the association direction.
        high_conf_det_threshold (float): Confidence threshold to consider a detection as high confidence. If a detection has
            confidence lower than this threshold, it will not be considered for association.

    """  # noqa: E501

    count_id: int = 0

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        minimum_consecutive_frames: int = 3,  # should change this for min_hits?
        minimum_iou_threshold: float = 0.3,
        direction_consistency_weight: float = 0.2,
        high_conf_det_threshold: float = 0.6,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.direction_consistency_weight = direction_consistency_weight
        self.high_conf_det_threshold = high_conf_det_threshold
        # Active trackers
        self.tracks: list[OCSORTTracklet] = []

    def _get_associated_indices(
        self,
        iou_matrix: np.ndarray,
        direction_consistency_matrix: np.ndarray,
        detection_boxes: np.ndarray,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """
        Associate detections to tracks based on IOU

        Args:
            iou_matrix (np.ndarray): IOU cost matrix.
            direction_consistency_matrix (np.ndarray): Direction of the tracklet consistency cost matrix.
            detection_boxes (np.ndarray): Detected bounding boxes in the
                form [x1, y1, x2, y2].

        Returns:
            tuple[list[tuple[int, int]], set[int], set[int]]: Matched indices,
                unmatched tracks, unmatched detections.
        """  # noqa: E501
        matched_indices = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detection_boxes)))
        if len(self.tracks) > 0 and len(detection_boxes) > 0:
            # Find optimal assignment using scipy.optimize.linear_sum_assignment.
            cost_matrix = (
                iou_matrix
                + self.direction_consistency_weight * direction_consistency_matrix
            )
            row_indices, col_indices = linear_sum_assignment(cost_matrix, maximize=True)
            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] >= self.minimum_iou_threshold:
                    matched_indices.append((row, col))
                    unmatched_tracks.remove(row)
                    unmatched_detections.remove(col)

        return matched_indices, unmatched_tracks, unmatched_detections

    def _spawn_new_trackers(
        self,
        detections: sv.Detections,
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
            new_tracker = OCSORTTracklet(detections.xyxy[detection_idx])
            self.tracks.append(new_tracker)

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

        if len(self.tracks) == 0 and len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections

        detections = detections[detections.confidence >= self.high_conf_det_threshold]

        updated_detections: list[
            sv.Detections
        ] = []  # List for returning the updated detections
        # Convert detections to a (N x 4) array (x1, y1, x2, y2)
        detection_boxes = (
            detections.xyxy if len(detections) > 0 else np.array([]).reshape(0, 4)
        )

        # Predict new locations for existing tracks KF
        for tracker in self.tracks:
            tracker.predict()

        # Build IOU cost matrix between detections and predicted bounding boxes
        iou_matrix = get_iou_matrix(self.tracks, detection_boxes)

        direction_consistency_matrix = build_direction_consistency_matrix(
            self.tracks, detection_boxes
        )
        # 1st Association of detections to tracks (OCM)
        matched_indices, unmatched_tracks, unmatched_detections = (
            self._get_associated_indices(
                iou_matrix, direction_consistency_matrix, detection_boxes
            )
        )

        # Update matched trackers with assigned detections
        for row, col in matched_indices:
            self.tracks[row].update(detection_boxes[col])
            add_track_id_detections(
                self.tracks[row], detections[col : col + 1], updated_detections
            )

        # Run 2nd Chance Association (OCR)
        # between the last observation of unmatched tracks to the unmatched observations #noqa: E501
        if len(unmatched_detections) > 0 and len(unmatched_tracks) > 0:
            last_observation_of_tracks = np.array(
                [self.tracks[t_id].last_observation for t_id in list(unmatched_tracks)]
            )

            ocr_iou_matrix = get_iou_matrix_between_boxes(
                last_observation_of_tracks, detection_boxes[list(unmatched_detections)]
            )

            ocr_matched_indices, ocr_unmatched_tracks, ocr_unmatched_detections = (
                self._get_associated_indices(
                    ocr_iou_matrix,
                    np.zeros_like(ocr_iou_matrix),
                    detection_boxes[list(unmatched_detections)],
                )
            )

            for ocr_row, ocr_col in ocr_matched_indices:
                track_idx = list(unmatched_tracks)[ocr_row]
                det_idx = list(unmatched_detections)[ocr_col]
                self.tracks[track_idx].update(detection_boxes[det_idx])
                add_track_id_detections(
                    self.tracks[track_idx],
                    detections[det_idx : det_idx + 1],
                    updated_detections,
                )

            self.tracks = self.activate_or_kill_tracklets()
            self._spawn_new_trackers(
                detections[list(unmatched_detections)], ocr_unmatched_detections
            )
            left_detections = detections[list(ocr_unmatched_detections)]
            left_detections.tracker_id = np.array(
                [-1] * len(left_detections), dtype=int
            )
            updated_detections.append(left_detections)

        else:
            self.tracks = self.activate_or_kill_tracklets()
            self._spawn_new_trackers(detections, unmatched_detections)
            left_detections = detections[list(unmatched_detections)]
            left_detections.tracker_id = np.array(
                [-1] * len(left_detections), dtype=int
            )
            updated_detections.append(left_detections)
        final_updated_detections = sv.Detections.merge(updated_detections)
        if len(final_updated_detections) == 0:
            final_updated_detections.tracker_id = np.array([], dtype=int)
        return final_updated_detections

    def reset(self) -> None:
        """Resets the tracker's internal state.

        Clears all active tracks and resets the track ID counter.
        """
        self.tracks = []
        OCSORTTracklet.count_id = 0

    def activate_or_kill_tracklets(self):
        """Activates or kills tracklets based on their status.

        This method checks each tracklet's status and either activates it
        (assigning a tracker ID) if it meets the criteria for being a valid
        track, or kills it (removing it from active tracking) if it has been
        lost for too long.
        """
        alive_tracklets = []
        for tracklet in self.tracks:
            is_mature = (
                tracklet.number_of_successful_consecutive_updates
                >= self.minimum_consecutive_frames
            )
            # is_active = tracklet.time_since_update == 0
            if tracklet.time_since_update < self.maximum_frames_without_update:
                alive_tracklets.append(tracklet)

            if is_mature and tracklet.tracker_id == -1:
                tracklet.tracker_id = OCSORTTracklet.get_next_tracker_id()
        return alive_tracklets
