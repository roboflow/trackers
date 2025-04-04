import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import supervision as sv


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Computes IoU between two sets of bounding boxes.
    bboxes1: (N, 4) array of bboxes [x1, y1, x2, y2]
    bboxes2: (M, 4) array of bboxes [x1, y1, x2, y2]
    Returns: (N, M) array of IoUs
    """
    bboxes1_area = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    bboxes2_area = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    # Expand dims for broadcasting
    bboxes1 = np.expand_dims(bboxes1, axis=1)  # (N, 1, 4)
    bboxes2 = np.expand_dims(bboxes2, axis=0)  # (1, M, 4)

    # Intersection top-left
    inter_tl_x = np.maximum(bboxes1[:, :, 0], bboxes2[:, :, 0])
    inter_tl_y = np.maximum(bboxes1[:, :, 1], bboxes2[:, :, 1])
    # Intersection bottom-right
    inter_br_x = np.minimum(bboxes1[:, :, 2], bboxes2[:, :, 2])
    inter_br_y = np.minimum(bboxes1[:, :, 3], bboxes2[:, :, 3])

    # Intersection width and height
    inter_w = np.maximum(0.0, inter_br_x - inter_tl_x)
    inter_h = np.maximum(0.0, inter_br_y - inter_tl_y)

    # Intersection area
    inter_area = inter_w * inter_h

    # Union Area
    union_area = (
        np.expand_dims(bboxes1_area, axis=1)
        + np.expand_dims(bboxes2_area, axis=0)
        - inter_area
    )

    # Handle division by zero
    iou = inter_area / (union_area + 1e-6)

    return iou


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Takes a bounding box in the form [x1, y1, x2, y2] and returns z in the form
    [x, y, s, r] where x,y is the centre of the box and s is the scale/area
    and r is the aspect ratio.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is area
    r = w / float(h + 1e-6)  # Add epsilon to avoid division by zero
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    """
    Takes a bounding box in the centre form [x, y, s, r] and returns it
    in the form [x1, y1, x2, y2] where x1, y1 is the top left and x2, y2
    is the bottom right.
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)  # Add epsilon to avoid division by zero
    return np.array(
        [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
    ).reshape((1, 4))


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox: np.ndarray):
        """
        Initialises a tracker using initial bounding box.
        bbox: [x1, y1, x2, y2]
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= (
            1000.0  # Give high uncertainty to the unobservable initial velocities
        )
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox: np.ndarray):
        """
        Updates the state vector with observed bbox.
        bbox: [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        """
        Advances the state vector and returns the predicted bounding box estimate.
        Returns: [x1, y1, x2, y2]
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """
        Returns the current bounding box estimate.
        Returns: [x1, y1, x2, y2]
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(
    detections: np.ndarray, trackers: np.ndarray, iou_threshold: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    detections: (N, 4) array of bboxes [x1, y1, x2, y2]
    trackers: (M, 4) array of bboxes [x1, y1, x2, y2]

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0 or len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty(
                (0), dtype=int
            ),  # Return empty array for unmatched trackers if trackers is empty
        )

    iou_matrix = iou_batch(detections, trackers)

    # Hungarian algorithm for assignment
    # Note: linear_sum_assignment finds the minimum cost, so we use -iou_matrix
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched_indices = np.stack((row_ind, col_ind), axis=1)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matches with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class SORT:
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Sets key parameters for SORT
        Args:
            max_age (int): Maximum number of frames to keep a track alive without associated detections.
            min_hits (int): Minimum number of associated detections before track is initialised.
            iou_threshold (float): Minimum IOU for match.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0  # Reset tracker count for each instance

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Params:
          detections (sv.Detections): Detections for the current frame.
                                     Must contain 'xyxy' and 'confidence'.
        Returns:
          sv.Detections: Filtered detections with assigned tracker IDs in 'tracker_id' field.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1

        # Get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        ret_det = sv.Detections.empty()
        ret_det.tracker_id = np.array([], dtype=int)

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3]]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Remove trackers that produced NaN predictions
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in sorted(to_del, reverse=True):
            self.trackers.pop(t)

        # Extract detections data in [x1, y1, x2, y2, score] format
        # Ensure detections are not empty
        if len(detections) > 0:
            dets = np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))
            # Use only xyxy for association
            det_coords = dets[:, :4]
        else:
            dets = np.empty((0, 5))
            det_coords = np.empty((0, 4))

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_coords, trks, self.iou_threshold
        )

        # Update matched trackers with assigned detections
        for m in matched:
            # m[0] is index in dets, m[1] is index in trackers list *before* potential pop
            # Find the correct tracker index *after* potential pops
            tracker_idx = m[1]
            original_tracker_indices = list(
                range(len(self.trackers))
            )  # Indices before pop
            current_tracker_indices = [
                i for i, trk in enumerate(original_tracker_indices) if i not in to_del
            ]
            # Map matched tracker index back to current list
            current_idx_map = {
                orig_idx: curr_idx
                for curr_idx, orig_idx in enumerate(current_tracker_indices)
            }
            if tracker_idx in current_idx_map:
                current_tracker_idx = current_idx_map[tracker_idx]
                self.trackers[current_tracker_idx].update(
                    dets[m[0], :4]
                )  # Update with xyxy
            else:
                # This case should ideally not happen if indices are handled correctly
                print(
                    f"Warning: Could not find tracker for matched index {tracker_idx}"
                )

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])  # Initialize with xyxy
            self.trackers.append(trk)

        i = len(self.trackers)
        active_tracks_xyxy = []
        active_tracks_ids = []
        active_tracks_conf = []
        active_tracks_class_id = []

        for trk in reversed(self.trackers):
            # Get the current state estimate
            d = trk.get_state()[0]  # Shape (4,)

            # Check if the track should be returned
            # Condition: (updated in current frame OR just initialized) AND (met min_hits OR within initial frames)
            # is_updated = trk.time_since_update == 0
            # meets_min_hits = trk.hit_streak >= self.min_hits
            is_in_init_phase = self.frame_count <= self.min_hits

            # Use hits instead of hit_streak for returning tracks, as hit_streak resets
            meets_min_hits_total = trk.hits >= self.min_hits

            # Return track if it's updated and meets criteria, or if it's young and needs min_hits
            # if (is_updated and (meets_min_hits or is_in_init_phase)):
            if (trk.time_since_update < 1) and (
                meets_min_hits_total or is_in_init_phase
            ):
                active_tracks_xyxy.append(d)
                active_tracks_ids.append(
                    trk.id + 1
                )  # MOT benchmark requires positive IDs

                # Find the associated detection to retrieve confidence and class_id if available
                # This is slightly complex because a tracker might not have a match in the *current* frame
                # We'll associate based on the match list if it exists for this tracker
                associated_detection_idx = -1
                for match in matched:
                    # Find the original tracker index again
                    tracker_idx = match[1]
                    current_idx_map_rev = {
                        curr_idx: orig_idx
                        for orig_idx, curr_idx in current_idx_map.items()
                    }
                    if (
                        i - 1 in current_idx_map_rev
                        and current_idx_map_rev[i - 1] == tracker_idx
                    ):  # i-1 is the current index in reversed loop
                        associated_detection_idx = match[0]
                        break

                # If a match was found *this frame* and detections had conf/class_id
                if associated_detection_idx != -1 and "confidence" in detections.data:
                    active_tracks_conf.append(
                        detections.confidence[associated_detection_idx]
                    )
                    if (
                        "class_id" in detections.data
                        and detections.class_id is not None
                    ):
                        active_tracks_class_id.append(
                            detections.class_id[associated_detection_idx]
                        )
                    else:
                        active_tracks_class_id.append(
                            None
                        )  # Placeholder if class_id not present
                else:
                    # If no match *this frame*, confidence/class is unknown for the prediction
                    active_tracks_conf.append(
                        None
                    )  # Or use a default/placeholder value?
                    active_tracks_class_id.append(None)

            i -= 1
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(active_tracks_xyxy) > 0:
            ret_det.xyxy = np.array(active_tracks_xyxy)
            ret_det.tracker_id = np.array(active_tracks_ids)
            # Handle confidence and class_id - check if all elements are None
            if not all(c is None for c in active_tracks_conf):
                # Replace Nones with a default value (e.g., 0) if mixing occurs, or handle as needed
                active_tracks_conf = [
                    c if c is not None else 0.0 for c in active_tracks_conf
                ]
                ret_det.confidence = np.array(active_tracks_conf)
            if not all(c is None for c in active_tracks_class_id):
                # Ensure all elements have a value before converting to array
                active_tracks_class_id = [
                    c if c is not None else -1 for c in active_tracks_class_id
                ]  # Use -1 for missing class
                ret_det.class_id = np.array(active_tracks_class_id)

        return ret_det
