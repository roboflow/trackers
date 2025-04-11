from typing import Optional, Union

import numpy as np

from trackers.core.sort.kalman_box_tracker import SORTKalmanBoxTracker


class DeepSORTKalmanBoxTracker(SORTKalmanBoxTracker):
    def __init__(self, bbox: np.ndarray, feature: Optional[np.ndarray] = None):
        super().__init__(bbox)
        self.features: list[np.ndarray] = []
        if feature is not None:
            self.features.append(feature)

    def update_feature(self, feature: np.ndarray):
        self.features.append(feature)

    def get_feature(self) -> Union[np.ndarray, None]:
        """
        Get the mean feature vector for this tracker.

        Returns:
            np.ndarray: Mean feature vector.
        """
        if len(self.features) > 0:
            # Return the mean of all features, thus (in theory) capturing the
            # "average appearance" of the object, which should be more robust
            # to minor appearance changes. Otherwise, the last feature can
            # also be returned like the following:
            # return self.features[-1]
            return np.mean(self.features, axis=0)
        return None
