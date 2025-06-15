"""
Reference Link : https://github.com/noahcao/OC_SORT/blob/master/trackers/ocsort_tracker/ocsort.py
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import supervision as sv
from filterpy.kalman import KalmanFilter

from trackers.core.base import BaseTrackerWithFeatures
from trackers.core.ocsort.association import (
    associate,
    ciou_batch,
    ct_dist,
    diou_batch,
    giou_batch,
    iou_batch,
    linear_assignment,
)


def k_previous_obs(observations: np.ndarray, cur_age: int, k: int) -> np.ndarray:
    """
    Retrieves the last `k` observations for a given track.

    This function extracts the most recent `k` observations from the
    observation history of a particular track based on the current age
    of the track.

    Args:
        observations (np.ndarray): Array of previous observations for the track.
        cur_age (int): The number of frames the track has existed.
        k (int): The number of most recent observations to retrieve.

    Returns:
        np.ndarray: An array containing the last `k` observations for the track.
    """

    if len(observations) == 0:
        return np.array([-1, -1, -1, -1, -1])
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
        max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Converts a bounding box from [x1, y1, x2, y2] format to [x, y, s, r].

    The returned format consists of:
        - x, y: Center coordinates of the bounding box.
        - s: Scale (area) of the bounding box.
        - r: Aspect ratio (width divided by height).

    Args:
        bbox (List[int]): Bounding box in the format [x1, y1, x2, y2].

    Returns:
        List[float]: Converted bounding box in the format [x, y, s, r].
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(
    x: Union[List[int], np.ndarray], score: Optional[float] = None
) -> np.ndarray:
    """
    Converts a bounding box from center format [x, y, s, r]
    to corner format [x1, y1, x2, y2].

    The input format:
        - x, y: Center coordinates of the bounding box.
        - s: Scale (area) of the bounding box.
        - r: Aspect ratio (width divided by height).

    The output format:
        - x1, y1: Top-left corner of the bounding box.
        - x2, y2: Bottom-right corner of the bounding box.

    Args:
        x (List[float] or np.ndarray): Bounding box in the format [x, y, s, r].
        score (float, optional): Detection confidence score to include in the output.

    Returns:
        np.ndarray: Bounding box in the format [x1, y1, x2, y2] or
                    [x1, y1, x2, y2, score] if a score is provided.
    """

    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


def speed_direction(
    bbox1: Union[List[int], np.ndarray], bbox2: Union[List[int], np.ndarray]
) -> float:
    """
    Computes the normalized direction vector of movement between two bounding boxes.

    Calculates the center points of the two bounding boxes and returns a unit
    vector indicating the direction of motion from `bbox1` to `bbox2`.

    Args:
        bbox1 (list or np.ndarray): First bounding box in the format [x1, y1, x2, y2].
        bbox2 (list or np.ndarray): Second bounding box in the format [x1, y1, x2, y2].

    Returns:
        np.ndarray: A 2D unit vector [dy, dx] representing the normalized direction
                    of movement from the first box to the second.
    """
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects
    observed as bbox.
    """

    count = 0

    def __init__(self, bbox: np.ndarray, delta_t: int = 3):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
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
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
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
        position = None
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox: np.ndarray) -> None:
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                self.velocity = speed_direction(previous_box, bbox)
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def predict(self) -> np.ndarray:
        """
        Advances the state vector and returns the predicted bounding box estimate.
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
        """
        return convert_x_to_bbox(self.kf.x)


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note
    that we hardly normalize the cost by all methods to (0,1) which may not be
    the best practice.
"""

ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
}


class OCSORTTracker(BaseTrackerWithFeatures):
    def __init__(
        self,
        det_thresh: float,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False,
    ):
        """
        Initializes the tracker with specified parameters.

        Args:
            det_threh (float): Detection confidence threshold. Detections below this
                threshold are ignored.
            max_age (int, optional): Maximum number of consecutive frames a track is
                kept alive without matching a new detection. Defaults to 30.
            min_hits (int, optional): Minimum number of hits (successful matches)
                required to consider a track valid. Defaults to 3.
            iou_threshold (float, optional): Minimum IOU required to associate a
                detection with an existing track. Defaults to 0.3.
            delta_t (int, optional): Time interval (in frames) considered between
                detection updates. Useful in motion modeling. Defaults to 3.
            asso_func (str, optional): Association metric to use. Must be one of
                {'iou', 'giou', 'ciou', 'diou', 'ct_dist'}, corresponding to keys
                in `ASSO_FUNCS`. Defaults to "iou".
            inertia (float, optional): Blending factor between detection and prediction.
                A higher value places more weight on the prediction. Defaults to 0.2.
            use_byte (bool, optional): Whether to use BYTE association logic for
                tracking. Defaults to False.
        """

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0

    def _existing_track_prediction(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        velocities = np.array(
            [
                trk.velocity if trk.velocity is not None else np.array((0, 0))
                for trk in self.trackers
            ]
        )
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [
                k_previous_obs(trk.observations, trk.age, self.delta_t)
                for trk in self.trackers
            ]
        )

        return trks, velocities, last_boxes, k_observations

    def _byte_association(
        self, trks: np.ndarray, dets_second: np.ndarray, unmatched_trks : np.ndarray
    ) -> np.ndarray:
        """
        Performs BYTE-level association as a secondary matching step.

        This method is typically used after primary association (e.g., IOU-based)
        to match remaining unassociated tracks and detections using a secondary
        association strategy like confidence-guided IOU.

        Args:
            trks (np.ndarray): Array of unmatched track predictions
            dets_second (np.ndarray): Array of unmatched detections in the same format.

        Returns:
            - matches (np.ndarray): Pairs of matched tracks based on Byte Association
        """
        u_trks = trks[unmatched_trks]
        iou_left = self.asso_func(
            dets_second, u_trks
        )  # iou between low score detections and unmatched tracks
        iou_left = np.array(iou_left)
        if iou_left.max() > self.iou_threshold:
            """
            NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
            get a higher performance especially on MOT17/MOT20 datasets. But we keep it
            uniform here for simplicity
            """
            matched_indices = linear_assignment(-iou_left)
            to_remove_trk_indices = []
            for m in matched_indices:
                det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                if iou_left[m[0], m[1]] < self.iou_threshold:
                    continue
                self.trackers[trk_ind].update(dets_second[det_ind, :])
                self.trackers[m[1]].position = m[0]
                to_remove_trk_indices.append(trk_ind)
            unmatched_trks = np.setdiff1d(
                unmatched_trks, np.array(to_remove_trk_indices)
            )

        return unmatched_trks

    def _filter_by_tracker_id(self, detections: sv.Detections) -> sv.Detections:
        """
        Filters all detection attributes to keep only entries with a valid tracker ID.

        This method removes all elements across the detection object's attributes
        (`xyxy`, `confidence`, `class_id`, `tracker_id`) where the `tracker_id` is -1.

        If `tracker_id` is `None`, the method does nothing.

        Returns:
            None
        """

        tracker_id = detections.tracker_id
        if np.max(tracker_id) == -1:
            return detections  # No filtering possible

        valid = tracker_id != -1

        detections.xyxy = detections.xyxy[valid]
        detections.confidence = detections.confidence[valid]
        detections.tracker_id = detections.tracker_id[valid]
        detections.class_id = detections.class_id[valid]

        return detections

    def _update_detection_with_track_ids(
        self, detections: sv.Detections
    ) -> sv.Detections:
        """
        Updation of matched detections with tracking IDs
        """

        ret = []
        to_remove = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
                detections.tracker_id[trk.position] = trk.id + 1
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                to_remove.append(i)

        self.trackers = [
            item for i, item in enumerate(self.trackers) if i not in to_remove
        ]
        return detections

    def _get_unassociated_indices(
        self,
        dets,
        last_boxes,
        unmatched_dets,
        unmatched_trks,
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_dets = dets[unmatched_dets]
        left_trks = last_boxes[unmatched_trks]
        iou_left = self.asso_func(left_dets, left_trks)
        iou_left = np.array(iou_left)
        if iou_left.max() > self.iou_threshold:
            """
            NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
            get a higher performance especially on MOT17/MOT20 datasets. But we keep it
            uniform here for simplicity
            """
            rematched_indices = linear_assignment(-iou_left)
            to_remove_det_indices = []
            to_remove_trk_indices = []
            for m in rematched_indices:
                det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                if iou_left[m[0], m[1]] < self.iou_threshold:
                    continue
                self.trackers[trk_ind].update(dets[det_ind, :])
                to_remove_det_indices.append(det_ind)
                to_remove_trk_indices.append(trk_ind)
            unmatched_dets = np.setdiff1d(
                unmatched_dets, np.array(to_remove_det_indices)
            )
            unmatched_trks = np.setdiff1d(
                unmatched_trks, np.array(to_remove_trk_indices)
            )

        return unmatched_dets, unmatched_trks

    def update(self, detections: sv.Detections) -> np.ndarray:
        """
        Params:
          dets - sv.Detections
        Requires: this method must be called once for each frame
                even with empty detections for frames without detections)
        Returns:
            sv.Detections
        NOTE: The number of objects returned may differ from the number of
            detections provided.
        """

        if len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections

        detections.tracker_id = np.array([-1] * len(detections.xyxy))

        self.frame_count += 1

        detection_boxes = (
            detections.xyxy if len(detections) > 0 else np.array([]).reshape(0, 4)
        )

        detections_confidence = (
            detections.confidence if len(detections) > 0 else np.array([]).reshape(0, 4)
        )

        # Post Processing Detection Boxes
        dets = np.concatenate(
            (detection_boxes, np.expand_dims(detections_confidence, axis=-1)), axis=1
        )
        inds_low = detections_confidence > 0.1
        inds_high = detections_confidence < self.det_thresh
        inds_second = np.logical_and(
            inds_low, inds_high
        )  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = detections_confidence > self.det_thresh
        dets = dets[remain_inds]

        # Predict new locations for existing trackers
        trks, velocities, last_boxes, k_observations = self._existing_track_prediction()

        # First round of association
        matched, unmatched_dets, unmatched_trks = associate(
            dets, trks, self.iou_threshold, velocities, k_observations, self.inertia
        )

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            self.trackers[m[1]].position = m[0]

        # Second round of associaton by OCR
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            unmatched_trks = self._byte_association(trks, dets_second, unmatched_trks)

        # Associate detections to trackers based on Association Function
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            unmatched_dets, unmatched_trks = self._get_unassociated_indices(
                dets, last_boxes, unmatched_dets, unmatched_trks
            )

        # Update unmatched tracks
        for m in unmatched_trks:
            self.trackers[m].update(None)

        # Create new trackers for unmatched detections with confidence above threshold
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], delta_t=self.delta_t)
            trk.position = i
            self.trackers.append(trk)

        # Update detections with tracker IDs
        dets = self._update_detection_with_track_ids(detections)
        dets = self._filter_by_tracker_id(detections)
        return dets

    def reset(self) -> None:
        """
        Resets the tracker's internal state.

        Clears all active tracks and resets the track ID counter.
        """
        self.trackers = []
        KalmanBoxTracker.count = 0
