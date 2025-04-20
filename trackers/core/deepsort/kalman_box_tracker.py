from typing import Optional, Tuple, Union

import numpy as np
from scipy.linalg import solve_triangular

# Chi-square 0.95 quantile for 4 degrees of freedom (Mahalanobis threshold)
MAHALANOBIS_THRESHOLD = 9.4877


class DeepSORTKalmanBoxTracker:
    """
    The `DeepSORTKalmanBoxTracker` class represents the internals of a single
    tracked object (bounding box), with a Kalman filter to predict and update
    its position. It also maintains a feature vector for the object, which is
    used to identify the object across frames.

    Attributes:
        tracker_id (int): Unique identifier for the tracker.
        number_of_successful_updates (int): Number of times the object has been
            updated successfully.
        time_since_update (int): Number of frames since the last update.
        state (np.ndarray): State vector of the bounding box.
        F (np.ndarray): State transition matrix.
        H (np.ndarray): Measurement matrix.
        Q (np.ndarray): Process noise covariance matrix.
        R (np.ndarray): Measurement noise covariance matrix.
        P (np.ndarray): Error covariance matrix.
        features (list[np.ndarray]): List of feature vectors.
        count_id (int): Class variable to assign unique IDs to each tracker.

    Args:
        bbox (np.ndarray): Initial bounding box in the form [x1, y1, x2, y2].
        feature (Optional[np.ndarray]): Optional initial feature vector.
    """

    count_id = 0

    @classmethod
    def get_next_tracker_id(cls) -> int:
        """
        Class method that returns the next available tracker ID.

        Returns:
            int: The next available tracker ID.
        """
        next_id = cls.count_id
        cls.count_id += 1
        return next_id

    def __init__(self, bbox: np.ndarray, feature: Optional[np.ndarray] = None):
        # Initialize with a temporary ID of -1
        # Will be assigned a real ID when the track is considered mature
        self.tracker_id = -1

        # Number of hits indicates how many times the object has been
        # updated successfully
        self.number_of_successful_updates = 1
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

        # Initialize features list
        self.features: list[np.ndarray] = []
        if feature is not None:
            self.features.append(feature)

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

    def project(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects the current state distribution to measurement space.

        As per the Kalman Filter formulation mentioned implicitly in
        Section 2.1 of the DeepSORT paper, this function computes:
            (y_i, S_i) = (H·μ_i, H·Σ_i·H^T + R)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Projected mean (y_i) and innovation
                covariance (S_i) for gating and association.
        """
        # Project state mean to measurement space: y_i = H·μ_i
        projected_mean = self.H @ self.state

        # Project state covariance to measurement space: H·Σ_i·H^T
        projected_covariance = self.H @ self.P @ self.H.T

        # Add measurement noise: S_i = H·Σ_i·H^T + R
        innovation_covariance = projected_covariance + self.R

        return projected_mean, innovation_covariance

    def compute_gating_distance(self, measurements: np.ndarray) -> np.ndarray:
        """
        Computes the squared Mahalanobis distance between the track and
        measurements.

        This function is used for gating (ruling out) unlikely associations
        as described in Eq. (1)-(2) of the DeepSORT paper:
        d^(1)(i,j) = (d_j - y_i)^T · S_i^(-1) · (d_j - y_i)

        Args:
            measurements (np.ndarray): An Nx4 matrix of N measurements, each in
                format [x1, y1, x2, y2] representing detected bounding boxes.

        Returns:
            np.ndarray: An array of length N, where the i-th element contains the
                squared Mahalanobis distance between the track and measurements[i].
        """
        # Project current state to measurement space
        mean, covariance = self.project()
        mean = mean.reshape(1, 4)
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        # Solve the system L·z = d^T efficiently using triangular solver
        # This gives us z where z = L^(-1)·d^T
        z = solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False)
        # Compute squared Mahalanobis distance as the squared norm of z
        # d_m^2 = z^T·z = d^T·S^(-1)·d
        return np.sum(z * z, axis=0)

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
        self.number_of_successful_updates += 1

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
