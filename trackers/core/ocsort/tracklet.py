import numpy as np

from trackers.core.ocsort.kalman_filter_ocsort import (
    KalmanFilterNew as KalmanFilterOCSORT,
)
from trackers.utils.converters import (
    xcycsr_to_xyxy,
    xyxy_to_xcycsr,
)


class OCSORTTracklet:
    """
    The `OCSORTTracklet` class represents the internals of a single
    tracked object (bounding box), with a Kalman filter to predict and update
    its position.
    Attributes:
        age: Age of the tracklet in frames.
        kalman_filter: The Kalman filter instance for this tracklet.
        tracker_id: Unique identifier for the tracker.
        state_transition_matrix: State transition matrix for the Kalman filter. (referred as F)
        number_of_successful_consecutive_updates: Number of times the object has been
            updated successfully in a row.
        time_since_update: Number of frames since the last update.
        last_observation: The last observed bounding box.
        previous_to_last_observation: The bounding box observed before the last one.
        kalman_filter_state_before_being_lost: The Kalman filter state before the tracklet was lost.
        kalman_filter_parameters_before_being_lost:  The Kalman filter parameters before the tracklet was lost.
    """  # noqa: E501

    count_id: int = 0

    def __init__(self, initial_bbox) -> None:
        self.age = 0
        # state format: (x, y, s, r, vx, vy, vs). As detailed in SORT paper, r is the aspect ratio and constant! # noqa: E501
        # self.kalman_filter = KalmanFilter(
        #     bbox=xyxy_to_xcycsr(initial_bbox),
        #     state_dim=7,
        #     state_transition_matrix=np.array(
        #         [
        #             [1, 0, 0, 0, 1, 0, 0],
        #             [0, 1, 0, 0, 0, 1, 0],
        #             [0, 0, 1, 0, 0, 0, 1],
        #             [0, 0, 0, 1, 0, 0, 0],
        #             [0, 0, 0, 0, 1, 0, 0],
        #             [0, 0, 0, 0, 0, 1, 0],
        #             [0, 0, 0, 0, 0, 0, 1],
        #         ]
        #     ),
        # )

        # self.kalman_filter.R[2:, 2:] *= 10.0
        # self.kalman_filter.P[4:, 4:] *= (
        #     1000.0  # give high uncertainty to the unobservable initial velocities
        # )
        # self.kalman_filter.P *= 10.0
        # self.kalman_filter.Q[-1, -1] *= 0.01
        # self.kalman_filter.Q[4:, 4:] *= 0.01
        self.kalman_filter = KalmanFilterOCSORT(dim_x=7, dim_z=4)

        self.kalman_filter.R[2:, 2:] *= 10.0
        self.kalman_filter.P[4:, 4:] *= (
            1000.0  # give high uncertainty to the unobservable initial velocities
        )
        self.kalman_filter.P *= 10.0
        self.kalman_filter.Q[-1, -1] *= 0.01
        self.kalman_filter.Q[4:, 4:] *= 0.01

        self.kalman_filter.x[:4] = xyxy_to_xcycsr(initial_bbox).reshape((4, 1))
        self.last_observation = initial_bbox  # None
        self.previous_to_last_observation = None  # For velocity of track

        # Will be assigned a real ID when the track is considered mature
        self.tracker_id = -1

        # Number of hits indicates how many times the object has been
        # updated successfully
        self.number_of_successful_consecutive_updates = 1
        # Number of frames since the last update
        self.time_since_update = 0
        # self.save_kalman_filter_state()

    # def save_kalman_filter_state(self) -> None:
    #     """Saves the current Kalman filter state and parameters."""
    #     self.kalman_filter_state_before_being_lost = self.kalman_filter.state.copy()
    #     self.kalman_filter_parameters_before_being_lost = {
    #         "H": self.kalman_filter.H.copy(),
    #         "Q": self.kalman_filter.Q.copy(),
    #         "R": self.kalman_filter.R.copy(),
    #         "P": self.kalman_filter.P.copy(),
    #     }

    @classmethod
    def get_next_tracker_id(cls) -> int:
        next_id = cls.count_id
        cls.count_id += 1
        return next_id

    def update(self, bbox: np.ndarray) -> None:
        """Updates the tracklet with a new bounding box observation.

        Args:
            bbox (np.ndarray): The new bounding box in the form [x1, y1, x2, y2].
        """
        if bbox is not None:
            self.kalman_filter.update(xyxy_to_xcycsr(bbox))
            # save the last before being lost KF parameters for re-updating
            self.time_since_update = 0
            self.number_of_successful_consecutive_updates += 1
            self.previous_to_last_observation = self.last_observation
            self.last_observation = bbox
        else:
            self.kalman_filter.update(None)

    def predict(self) -> np.ndarray:
        """Predicts the next bounding box position using the Kalman filter.

        Returns:
            np.ndarray: The predicted bounding box in the form [x1, y1, x2, y2].
        """
        self.kalman_filter.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.number_of_successful_consecutive_updates = 0

        self.time_since_update += 1
        return xcycsr_to_xyxy(self.kalman_filter.x)

    def is_lost(
        self,
    ) -> bool:
        """Determines if the tracklet is considered lost."""
        return self.time_since_update > 1

    # def re_update(self, bbox: np.ndarray) -> None:
    #     """Re-updates the tracklet with the virtual trajectory generated out of the line that joins
    #     the last_observation and the parameter bbox.

    #     Args:
    #         bbox: The new bounding box in the form [x1, y1, x2, y2].
    #     """
    #     self.kalman_filter.state = self.kalman_filter_state_before_being_lost
    #     self.kalman_filter.set_parameters(
    #         **self.kalman_filter_parameters_before_being_lost
    #     )
    #     bbox_xywh = xyxy_to_xywh(np.array([bbox]))[0]
    #     last_observation_xywh = xyxy_to_xywh(np.array([self.last_observation]))[0]
    #     for i in range(1, self.time_since_update + 1):
    #         # Interpolate linearly between last_observation and bbox
    #         virtual_bbox_xywh = last_observation_xywh + (
    #             bbox_xywh - last_observation_xywh
    #         ) * (i / (self.time_since_update))
    #         virtual_bbox_xysa = np.copy(virtual_bbox_xywh)
    #         s = virtual_bbox_xywh[2] * virtual_bbox_xywh[3]  # w*h
    #         virtual_bbox_xysa[2] = s
    #         virtual_bbox_xysa[3] = (
    #             virtual_bbox_xywh[2] / virtual_bbox_xywh[3]
    #         )  # w/h = r

    #         self.kalman_filter.predict()
    #         self.kalman_filter.update(virtual_bbox_xysa)

    #     self.previous_to_last_observation = self.last_observation
    #     self.last_observation = bbox
    #     self.time_since_update = 0

    def get_state_bbox(self) -> np.ndarray:
        """Returns the current bounding box estimate from the Kalman filter.

        Returns:
            np.ndarray: The current bounding box in the form [x1, y1, x2, y2].
        """
        return xcycsr_to_xyxy(self.kalman_filter.x)
