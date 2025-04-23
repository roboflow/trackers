import numpy as np
import supervision as sv

from trackers.core.base import BaseTracker
from trackers.core.sort.kalman_box_tracker import SORTKalmanBoxTracker
from trackers.utils.sort_utils import (
    get_alive_trackers,
    get_iou_matrix,
    update_detections_with_track_ids,
)


class SORTTracker(BaseTracker):
    """
    `SORTTracker` is an implementation of the
    [SORT (Simple Online and Realtime Tracking)](https://arxiv.org/pdf/1602.00763)
    algorithm for object tracking in videos.

    ??? example
        ```python
        import numpy as np
        import supervision as sv
        from rfdetr import RFDETRBase
        from rfdetr.util.coco_classes import COCO_CLASSES
        from trackers import SORTTracker


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
        trackers (list[SORTKalmanBoxTracker]): List of SORTKalmanBoxTracker objects.

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
        self.trackers = get_alive_trackers(
            self.trackers,
            self.minimum_consecutive_frames,
            self.maximum_frames_without_update,
        )

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
        iou_matrix = get_iou_matrix(self.trackers, detection_boxes)

        # Associate detections to trackers based on IOU
        matched_indices, _, unmatched_detections = self._get_associated_indices(
            iou_matrix, detection_boxes
        )

        # Update matched trackers with assigned detections
        for row, col in matched_indices:
            self.trackers[row].update(detection_boxes[col])

        self._spawn_new_trackers(detections, detection_boxes, unmatched_detections)

        updated_detections = update_detections_with_track_ids(
            self.trackers,
            detections,
            detection_boxes,
            self.minimum_iou_threshold,
            self.minimum_consecutive_frames,
        )

        # If there are no detected objects, set all tracker_id to "None"
        if (
            updated_detections.tracker_id is None
            or len(updated_detections.tracker_id) == 0
        ):
            updated_detections.tracker_id = ["None"] * len(updated_detections)

        return updated_detections

    def reset(self) -> None:
        self.trackers = []
        SORTKalmanBoxTracker.count_id = 0
