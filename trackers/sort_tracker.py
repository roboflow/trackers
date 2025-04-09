from copy import deepcopy

import numpy as np
import supervision as sv
from supervision.detection.utils import box_iou_batch

from trackers.base import BaseTracker


class KalmanBoxTracker:
    """
    The `KalmanBoxTracker` class represents the internals of a single
    tracked object (bounding box), with a Kalman filter to predict and update
    its position.

    References:
        [https://github.com/abewley/sort/blob/master/sort.py](https://github.com/abewley/sort/blob/master/sort.py)

    Attributes:
        id (int): Unique identifier for the tracker.
        hits (int): Number of times the object has been updated successfully.
        time_since_update (int): Number of frames since the last update.
        state (np.ndarray): State vector of the bounding box.
        F (np.ndarray): State transition matrix.
        H (np.ndarray): Measurement matrix.
        Q (np.ndarray): Process noise covariance matrix.
        R (np.ndarray): Measurement noise covariance matrix.
        P (np.ndarray): Error covariance matrix.
        count (int): Class variable to assign unique IDs to each tracker.

    Args:
        bbox (np.ndarray): Initial bounding box in the form [x1, y1, x2, y2].
    """

    count = 0

    def __init__(self, bbox: np.ndarray) -> None:
        # Each track gets a unique ID
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # Number of hits indicates how many times the object has been
        # updated successfully
        self.hits = 1
        # Number of frames since the last update
        self.time_since_update = 0

        # For simplicity, we keep a small state vector:
        # (x, y, x2, y2, vx, vy, vx2, vy2).
        # We'll store the bounding box in "self.state"
        self.state = np.zeros((8, 1), dtype=np.float32)

        # Initialize state directly from the first detection
        self.state[0] = bbox[0]
        self.state[1] = bbox[1]
        self.state[2] = bbox[2]
        self.state[3] = bbox[3]

        # Basic constant velocity model
        self._initialize_kalman_filter()

    def _initialize_kalman_filter(self) -> None:
        """
        Sets up the matrices for the Kalman filter.
        """
        # State transition matrix (F): 8x8
        # We assume a constant velocity model. Positions are incremented by
        # velocity each step.
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix (H): we directly measure x1, y1, x2, y2
        self.H = np.eye(4, 8, dtype=np.float32)  # 4x8

        # Process covariance matrix (Q)
        self.Q = np.eye(8, dtype=np.float32) * 0.01

        # Measurement covariance (R): noise in detection
        self.R = np.eye(4, dtype=np.float32) * 0.1

        # Error covariance matrix (P)
        self.P = np.eye(8, dtype=np.float32)

    def predict(self) -> None:
        """
        Predict the next state of the bounding box (applies the state transition).
        """
        # Predict state
        self.state = self.F @ self.state
        # Predict error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Increase time since update
        self.time_since_update += 1

    def update(self, bbox: np.ndarray) -> None:
        """
        Updates the state with a new detected bounding box.

        Args:
            bbox (np.ndarray): Detected bounding box in the form [x1, y1, x2, y2].
        """
        self.time_since_update = 0
        self.hits += 1

        # Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Residual
        measurement = bbox.reshape((4, 1))
        y = measurement - self.H @ self.state

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        identity_matrix = np.eye(8, dtype=np.float32)
        self.P = (identity_matrix - K @ self.H) @ self.P

    def get_state_bbox(self) -> np.ndarray:
        """
        Returns the current bounding box estimate from the state vector.

        Returns:
            np.ndarray: The bounding box [x1, y1, x2, y2].
        """
        return np.array(
            [
                self.state[0],  # x1
                self.state[1],  # y1
                self.state[2],  # x2
                self.state[3],  # y2
            ],
            dtype=float,
        ).reshape(-1)


class SORTTracker(BaseTracker):
    """
    `SORTTracker` is an implementation of the
    [SORT (Simple Online and Realtime Tracking)](https://arxiv.org/pdf/1602.00763)
    algorithm for object tracking in videos.

    References:
        - [SIMPLE ONLINE AND REALTIME TRACKING](https://arxiv.org/pdf/1602.00763)
        - [https://github.com/abewley/sort/blob/master/sort.py](https://github.com/abewley/sort/blob/master/sort.py)

    ??? example
        ```python
        import numpy as np
        import supervision as sv
        from rfdetr import RFDETRBase
        from rfdetr.util.coco_classes import COCO_CLASSES
        from trackers.sort_tracker import SORTTracker


        model = RFDETRBase(device="mps")
        tracker = SORTTracker()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()


        def callback(frame: np.ndarray, _: int):
            detections = model.predict(frame, threshold=0.5)
            detections = tracker.update(detections)

            labels = [
                f"#{tracker_id} {COCO_CLASSES[class_id]} {confidence:.2f}"
                for tracker_id, class_id, confidence in zip(
                    detections.tracker_id, detections.class_id, detections.confidence
                )
            ]

            annotated_image = frame.copy()
            annotated_image = box_annotator.annotate(annotated_image, detections)
            annotated_image = label_annotator.annotate(
                annotated_image, detections, labels
            )

            return annotated_image


        sv.process_video(
            source_path="data/traffic_video.mp4",
            target_path="data/out.mp4",
            callback=callback,
        )
        ```

    Attributes:
        trackers (list[KalmanBoxTracker]): List of KalmanBoxTracker objects.

    Args:
        lost_track_buffer (int): Number of frames to buffer when a track is lost.
            Increasing lost_track_buffer enhances occlusion handling, significantly
            improving tracking through occlusions, but may increase the possibility
            of ID switching for objects with similar appearance.
        frame_rate (float): Frame rate of the video (frames per second).
            Used to calculate the maximum time a track can be lost.
        track_activation_threshold (float): Detection confidence threshold
            for track activation. Only detections with confidence above this
            threshold will create new tracks. Increasing this threshold
            reduces false positives but may miss real objects with low confidence.
        minimum_consecutive_frames (int): Number of consecutive frames that an object
            must be tracked before it is considered a 'valid' track. Increasing
            `minimum_consecutive_frames` prevents the creation of accidental tracks
            from false detection or double detection, but risks missing shorter
            tracks. Before the tracker is considered valid, it will be assigned
            `-1` as its `tracker_id`.
        minimum_iou_threshold (float): IOU threshold for associating detections to
            existing tracks.
    """

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold

        # Active trackers
        self.trackers: list[KalmanBoxTracker] = []

    def _get_iou_matrix(self, detection_boxes: np.ndarray) -> np.ndarray:
        """
        Build IOU cost matrix between detections and predicted bounding boxes

        Args:
            detection_boxes (np.ndarray): Detected bounding boxes in the
                form [x1, y1, x2, y2].

        Returns:
            np.ndarray: IOU cost matrix.
        """
        predicted_boxes = np.array([t.get_state_bbox() for t in self.trackers])
        if len(predicted_boxes) == 0 and len(self.trackers) > 0:
            # Handle case where get_state_bbox might return empty array
            predicted_boxes = np.zeros((len(self.trackers), 4), dtype=np.float32)

        if len(self.trackers) > 0 and len(detection_boxes) > 0:
            iou_matrix = box_iou_batch(predicted_boxes, detection_boxes)
        else:
            iou_matrix = np.zeros(
                (len(self.trackers), len(detection_boxes)), dtype=np.float32
            )

        return iou_matrix

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

        if iou_matrix.size > 0:
            row_indices, col_indices = np.where(iou_matrix > self.minimum_iou_threshold)
            # Sort in descending order of IOU. Higher = better match.
            sorted_pairs = sorted(
                zip(row_indices, col_indices),
                key=lambda x: iou_matrix[x[0], x[1]],
                reverse=True,
            )
            # keep each unique row/col pair at most once
            used_rows = set()
            used_cols = set()
            for row, col in sorted_pairs:
                if (row not in used_rows) and (col not in used_cols):
                    used_rows.add(row)
                    used_cols.add(col)
                    matched_indices.append((row, col))

            unmatched_trackers = unmatched_trackers - used_rows
            unmatched_detections = unmatched_detections - used_cols

        return matched_indices, unmatched_trackers, unmatched_detections

    def _get_alive_trackers(self) -> list[KalmanBoxTracker]:
        """
        Remove dead or immature lost tracklets and get alive trackers
        that are within maximum_frames_without_update AND (it's mature OR
        it was just updated).
        """
        alive_trackers = []
        for tracker in self.trackers:
            is_mature = tracker.hits >= self.minimum_consecutive_frames
            is_active = tracker.time_since_update == 0
            if tracker.time_since_update < self.maximum_frames_without_update and (
                is_mature or is_active
            ):
                alive_trackers.append(tracker)
        return alive_trackers

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
                new_tracker = KalmanBoxTracker(detection_boxes[detection_idx])
                self.trackers.append(new_tracker)
        self.trackers = self._get_alive_trackers()

    def _update_detections_with_track_ids(
        self, detections: sv.Detections, detection_boxes: np.ndarray
    ) -> sv.Detections:
        """
        The function prepares the updated Detections with track IDs.
        If a tracker is "mature" (>= minimum_consecutive_frames) or recently updated,
        it is assigned an ID to the detection that just updated it.

        Args:
            detections (sv.Detections): The latest set of object detections.
            detection_boxes (np.ndarray): Detected bounding boxes in the
                form [x1, y1, x2, y2].

        Returns:
            sv.Detections: A copy of the detections with `tracker_id` set
                for each detection that is tracked.
        """
        # Re-run association in the same way (could also store direct mapping)
        final_tracker_ids = [-1] * len(detection_boxes)

        # Important: Recalculate predicted_boxes based on current trackers
        # after some may have been removed
        predicted_boxes = np.array([t.get_state_bbox() for t in self.trackers])
        iou_matrix_final = np.zeros(
            (len(self.trackers), len(detection_boxes)), dtype=np.float32
        )

        # Ensure predicted_boxes is properly shaped before the second iou calculation
        if len(predicted_boxes) == 0 and len(self.trackers) > 0:
            predicted_boxes = np.zeros((len(self.trackers), 4), dtype=np.float32)

        if len(self.trackers) > 0 and len(detection_boxes) > 0:
            iou_matrix_final = box_iou_batch(predicted_boxes, detection_boxes)

        row_indices, col_indices = np.where(
            iou_matrix_final > self.minimum_iou_threshold
        )
        sorted_pairs = sorted(
            zip(row_indices, col_indices),
            key=lambda x: iou_matrix_final[x[0], x[1]],
            reverse=True,
        )
        used_rows = set()
        used_cols = set()
        for row, col in sorted_pairs:
            # Double check index is in range
            if row < len(self.trackers):
                tracker_obj = self.trackers[row]
                # Only assign if the track is "mature" or is new but has enough hits
                if (row not in used_rows) and (col not in used_cols):
                    if tracker_obj.hits >= self.minimum_consecutive_frames:
                        final_tracker_ids[col] = tracker_obj.id
                    used_rows.add(row)
                    used_cols.add(col)

        # Assign tracker IDs to the returned Detections
        updated_detections = deepcopy(detections)
        updated_detections.tracker_id = np.array(final_tracker_ids)

        return updated_detections

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the state of tracked objects with the newly received detections
        and returns the updated `sv.Detections` (including tracking IDs).

        Args:
            detections (sv.Detections): The latest set of object detections.

        Returns:
            sv.Detections: A copy of the detections with `tracker_id` set
                for each detection that is tracked.
        """
        if len(self.trackers) == 0 and len(detections) == 0:
            return detections

        # Convert detections to a (N x 4) array (x1, y1, x2, y2)
        detection_boxes = (
            detections.xyxy if len(detections) > 0 else np.array([]).reshape(0, 4)
        )

        # Predict new locations for existing trackers
        for tracker in self.trackers:
            tracker.predict()

        # Build IOU cost matrix between detections and predicted bounding boxes
        iou_matrix = self._get_iou_matrix(detection_boxes)

        # Associate detections to trackers based on IOU
        matched_indices, _, unmatched_detections = self._get_associated_indices(
            iou_matrix, detection_boxes
        )

        # Update matched trackers with assigned detections
        for row, col in matched_indices:
            self.trackers[row].update(detection_boxes[col])

        self._spawn_new_trackers(detections, detection_boxes, unmatched_detections)

        updated_detections = self._update_detections_with_track_ids(
            detections, detection_boxes
        )

        return updated_detections
