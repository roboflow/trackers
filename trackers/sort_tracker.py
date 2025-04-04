import numpy as np
import supervision as sv
from copy import deepcopy


class KalmanBoxTracker:
    """
    Represents the internals of a single tracked object (bounding box),
    with a Kalman filter to predict and update its position.
    """

    count = 0

    def __init__(self, bbox: np.ndarray) -> None:
        """
        Initializes the tracker using the initial bounding box.

        Args:
            bbox (np.ndarray): Initial bounding box in the form [x1, y1, x2, y2].
        """
        # Each track gets a unique ID
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # Number of hits indicates how many times the object has been updated successfully
        self.hits = 1
        # Number of frames since the last update
        self.time_since_update = 0

        # For simplicity, we keep a small state vector: (x, y, x2, y2, vx, vy, vx2, vy2).
        # Real implementations often keep center/width/height or similar.
        # We'll store the bounding box in "self.state"
        self.state = np.zeros((8, 1), dtype=np.float32)

        # Initialize state directly from the first detection
        self.state[0] = bbox[0]
        self.state[1] = bbox[1]
        self.state[2] = bbox[2]
        self.state[3] = bbox[3]

        # Basic constant velocity model
        self._init_kalman_filter()

    def _init_kalman_filter(self) -> None:
        """
        Sets up the matrices for the Kalman filter.
        """
        # State transition matrix (F): 8x8
        # We assume a constant velocity model. Positions are incremented by velocity each step.
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


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Computes IOU (Intersection Over Union) between two arrays of bounding boxes.

    Args:
        bboxes1 (np.ndarray): Nx4 matrix of [x1, y1, x2, y2].
        bboxes2 (np.ndarray): Mx4 matrix of [x1, y1, x2, y2].

    Returns:
        np.ndarray: NxM IOU matrix.
    """
    # Check for empty arrays or incorrect shapes
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)

    # Ensure arrays are 2D with shape (N, 4) and (M, 4)
    if bboxes1.ndim == 1:
        bboxes1 = bboxes1.reshape(1, -1)
    if bboxes2.ndim == 1:
        bboxes2 = bboxes2.reshape(1, -1)

    # Expand dims to broadcast
    x1 = np.maximum(bboxes1[:, None, 0], bboxes2[None, :, 0])
    y1 = np.maximum(bboxes1[:, None, 1], bboxes2[None, :, 1])
    x2 = np.minimum(bboxes1[:, None, 2], bboxes2[None, :, 2])
    y2 = np.minimum(bboxes1[:, None, 3], bboxes2[None, :, 3])

    inter_area = np.clip(x2 - x1, a_min=0, a_max=None) * np.clip(
        y2 - y1, a_min=0, a_max=None
    )

    bboxes1_area = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    bboxes2_area = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    iou = inter_area / (
        bboxes1_area[:, None] + bboxes2_area[None, :] - inter_area + 1e-6
    )
    return iou


class SORTTracker:
    """
    SORTTracker is an implementation of the SORT (Simple Online and Realtime Tracking)
    algorithm for object tracking in videos. It uses a Kalman filter and IOU-based
    data association to track multiple objects.
    """

    def __init__(
        self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3
    ) -> None:
        """
        Args:
            max_age (int): Maximum number of frames to keep a track alive without updates.
            min_hits (int): Minimum number of associated detections before the track is made 'official'.
            iou_threshold (float): IOU threshold for associating detections to existing tracks.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        # Active trackers
        self.trackers = []

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
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

        # 1. Convert detections to a (N x 4) array (x1, y1, x2, y2)
        detection_boxes = (
            detections.xyxy if len(detections) > 0 else np.array([]).reshape(0, 4)
        )

        # 2. Predict new locations for existing trackers
        for tracker in self.trackers:
            tracker.predict()

        # 3. Build IOU cost matrix between detections and predicted bounding boxes
        predicted_boxes = np.array([t.get_state_bbox() for t in self.trackers])
        if len(predicted_boxes) == 0 and len(self.trackers) > 0:
            # Handle case where get_state_bbox might return empty array
            predicted_boxes = np.zeros((len(self.trackers), 4), dtype=np.float32)

        if len(self.trackers) > 0 and len(detection_boxes) > 0:
            iou_matrix = iou_batch(predicted_boxes, detection_boxes)
        else:
            iou_matrix = np.zeros(
                (len(self.trackers), len(detection_boxes)), dtype=np.float32
            )

        # 4. Associate detections to trackers based on IOU
        matched_indices = []
        unmatched_trackers = set(range(len(self.trackers)))
        unmatched_detections = set(range(len(detection_boxes)))

        if iou_matrix.size > 0:
            # Sort in descending order of IOU. Higher = better match.
            row_indices, col_indices = np.where(iou_matrix > self.iou_threshold)

            # For the example, a simple greedy approach:
            #   - sort matches by IOU descending
            #   - keep each unique row/col pair at most once
            sorted_pairs = sorted(
                zip(row_indices, col_indices),
                key=lambda x: iou_matrix[x[0], x[1]],
                reverse=True,
            )
            used_rows = set()
            used_cols = set()
            for row, col in sorted_pairs:
                if (row not in used_rows) and (col not in used_cols):
                    used_rows.add(row)
                    used_cols.add(col)
                    matched_indices.append((row, col))

            unmatched_trackers = unmatched_trackers - used_rows
            unmatched_detections = unmatched_detections - used_cols

        # 5. Update matched trackers with assigned detections
        for row, col in matched_indices:
            self.trackers[row].update(detection_boxes[col])

        # 6. Create new trackers for unmatched detections
        for detection_idx in unmatched_detections:
            new_tracker = KalmanBoxTracker(detection_boxes[detection_idx])
            self.trackers.append(new_tracker)

        # 7. Mark old trackers for removal if they have not been updated in a while
        alive_trackers = []
        for t in self.trackers:
            if t.time_since_update < self.max_age:
                alive_trackers.append(t)
        self.trackers = alive_trackers

        # 8. Prepare the updated Detections with track IDs
        #    If a tracker is "mature" (>= min_hits) or recently updated, we assign its ID
        #    to the detection that just updated it.
        #    In practice, we can also store the last known detection index for each track.

        # For simplicity, re-run association in the same way (could also store direct mapping).
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
            iou_matrix_final = iou_batch(predicted_boxes, detection_boxes)

        row_indices, col_indices = np.where(iou_matrix_final > self.iou_threshold)
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
                    if tracker_obj.hits >= self.min_hits:
                        final_tracker_ids[col] = tracker_obj.id
                    used_rows.add(row)
                    used_cols.add(col)

        # 9. Assign tracker IDs to the returned Detections
        updated_detections = deepcopy(detections)
        updated_detections.tracker_id = np.array(final_tracker_ids)

        return updated_detections
