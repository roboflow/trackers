import numpy as np

from trackers.utils.kalman_filter import KalmanFilter
from trackers.utils.ocsort_utils import (
    convert_bbox_to_state_rep,
    convert_state_rep_to_bbox,
)


class OCSORTTracklet:
    """
    The `OCSORTTracklet` class represents the internals of a single
    tracked object (bounding box), with a Kalman filter to predict and update
    its position."""

    count_id: int = 0

    def __init__(self, initial_bbox) -> None:
        self.age = 0
        # state format: (x, y, s, r, vx, vy, vs). As detailed in SORT paper, r is the aspect ratio and constant! # noqa: E501
        self.kalman_filter = KalmanFilter(
            bbox=convert_bbox_to_state_rep(initial_bbox),
            state_dim=7,
            state_transition_matrix=np.array(
                [
                    [1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ]
            ),
        )

        self.kalman_filter.R[2:, 2:] *= 10.0
        self.kalman_filter.P[4:, 4:] *= (
            1000.0  # give high uncertainty to the unobservable initial velocities
        )
        self.kalman_filter.P *= 10.0
        self.kalman_filter.Q[-1, -1] *= 0.01
        self.kalman_filter.Q[4:, 4:] *= 0.01
        self.last_observation = initial_bbox  # None
        self.previous_to_last_observation = None  # For velocity of track

        # Will be assigned a real ID when the track is considered mature
        self.tracker_id = -1

        # Number of hits indicates how many times the object has been
        # updated successfully
        self.number_of_successful_consecutive_updates = 1
        # Number of frames since the last update
        self.time_since_update = 0
        self.kalman_filter_state_before_being_lost = self.kalman_filter.state  # None
        self.kalman_filter_parameters_before_being_lost = {
            "H": self.kalman_filter.H.copy(),
            "Q": self.kalman_filter.Q.copy(),
            "R": self.kalman_filter.R.copy(),
            "P": self.kalman_filter.P.copy(),
        }  # None

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
        if self.is_lost():
            self.re_update(bbox)
        else:
            self.kalman_filter.update(convert_bbox_to_state_rep(bbox))
            # save the last before being lost KF parameters for re-updating
            self.kalman_filter_state_before_being_lost = self.kalman_filter.state.copy()
            # Copying all the time might be slow, is there a better alternative?
            self.kalman_filter_parameters_before_being_lost = {
                "H": self.kalman_filter.H.copy(),
                "Q": self.kalman_filter.Q.copy(),
                "R": self.kalman_filter.R.copy(),
                "P": self.kalman_filter.P.copy(),
            }
        self.time_since_update = 0
        self.number_of_successful_consecutive_updates += 1
        self.previous_to_last_observation = self.last_observation
        self.last_observation = bbox

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
        predicted_bbox = self.kalman_filter.get_state_bbox()
        return convert_state_rep_to_bbox(predicted_bbox)

    def is_lost(
        self,
    ) -> bool:
        """Determines if the tracklet is considered lost."""
        return self.time_since_update > 1

    def re_update(self, bbox: np.ndarray) -> None:
        """Re-updates the tracklet with the virtual trajectory generated out of the line that joins
        the last_observation and the parameter bbox.

        Args:
            bbox (np.ndarray): The new bounding box in the form [x1, y1, x2, y2].
        """  # noqa: E501
        self.kalman_filter.state = self.kalman_filter_state_before_being_lost
        self.kalman_filter.set_parameters(
            **self.kalman_filter_parameters_before_being_lost
        )
        for i in range(1, self.time_since_update + 1):
            # Interpolate linearly between last_observation and bbox
            virtual_bbox = self.last_observation + (bbox - self.last_observation) * (
                i / (self.time_since_update + 1)
            )
            virtual_bbox = convert_bbox_to_state_rep(virtual_bbox)
            self.kalman_filter.predict()
            self.kalman_filter.update(virtual_bbox)

        self.previous_to_last_observation = self.last_observation
        self.last_observation = bbox
        self.time_since_update = 0

    def get_state_bbox(self) -> np.ndarray:
        """Returns the current bounding box estimate from the Kalman filter.

        Returns:
            np.ndarray: The current bounding box in the form [x1, y1, x2, y2].
        """
        return convert_state_rep_to_bbox(self.kalman_filter.get_state_bbox())
