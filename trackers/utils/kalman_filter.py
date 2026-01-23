import numpy as np
from numpy.typing import NDArray


class KalmanFilter:
    """
    The `OCSORTKalmanBoxTracker` class represents the internals of a single
    tracked object (bounding box), with a Kalman filter to predict and update
    its position.

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

    Args:
        bbox (np.ndarray): Initial bounding box in the form [x1, y1, x2, y2].
        state_dim (int): Dimension of the state vector. Default is 7 using (x, y, s, r, vx, vy, vs).
        state_transition_matrix (np.ndarray): State transition matrix F. Default is identity.
    """  # noqa: E501

    state: NDArray[np.float32]
    F: NDArray[np.float32]
    H: NDArray[np.float32]
    Q: NDArray[np.float32]
    R: NDArray[np.float32]
    P: NDArray[np.float32]

    def __init__(
        self,
        bbox: NDArray[np.float64],
        state_dim: int = 7,
        state_transition_matrix: np.ndarray = np.eye(7),
    ) -> None:
        # Number of hits indicates how many times the object has been
        # updated successfully
        self.number_of_successful_updates = 1
        # Number of frames since the last update
        self.time_since_update = 0
        self.state_dim = state_dim
        # For simplicity, we keep a small state vector:
        # We'll store the bounding box in "self.state"
        self.state = np.zeros((state_dim, 1), dtype=np.float32)
        self.F = state_transition_matrix.astype(np.float32)
        # Initialize state directly from the first detection
        bbox_float: NDArray[np.float32] = bbox.astype(np.float32)
        self.state[0, 0] = bbox_float[0]
        self.state[1, 0] = bbox_float[1]
        self.state[2, 0] = bbox_float[2]
        self.state[3, 0] = bbox_float[3]

        # Basic constant velocity model
        self._initialize_kalman_filter()

    def _initialize_kalman_filter(self) -> None:
        """
        Sets up the matrices for the Kalman filter.
        """
        # We assume a constant velocity model. Positions are incremented by
        # velocity each step.

        # Measurement matrix (H)
        self.H = np.eye(4, self.state_dim, dtype=np.float32)  # 4xself.state_dim

        # Process covariance matrix (Q)
        self.Q = np.eye(self.state_dim, dtype=np.float32)  # * 0.01

        # Measurement covariance (R): noise in detection
        self.R = np.eye(4, dtype=np.float32)  # * 0.1

        # Error covariance matrix (P)
        self.P = np.eye(self.state_dim, dtype=np.float32)

    def predict(self) -> None:
        """
        Predict the next state of the bounding box (applies the state transition).
        """
        # Predict state
        self.state = (self.F @ self.state).astype(np.float32)
        # Predict error covariance
        self.P = (self.F @ self.P @ self.F.T + self.Q).astype(np.float32)

        # Increase time since update
        self.time_since_update += 1

    def update(self, bbox: NDArray[np.float64]) -> None:
        """
        Updates the state with a new detected bounding box.

        Args:
            bbox (np.ndarray): Detected bounding box in the form [x1, y1, x2, y2].
        """
        # self.time_since_update = 0
        # self.number_of_successful_updates += 1

        # Kalman Gain
        S: NDArray[np.float32] = self.H @ self.P @ self.H.T + self.R
        K: NDArray[np.float32] = (self.P @ self.H.T @ np.linalg.inv(S)).astype(
            np.float32
        )

        # Residual
        measurement: NDArray[np.float32] = bbox.reshape((4, 1)).astype(np.float32)
        y: NDArray[np.float32] = (
            measurement - self.H @ self.state
        )  # y should be float32 (4,1)

        # Update state
        self.state = (self.state + K @ y).astype(np.float32)

        # Update covariance
        identity_matrix: NDArray[np.float32] = np.eye(self.state_dim, dtype=np.float32)
        self.P = ((identity_matrix - K @ self.H) @ self.P).astype(np.float32)

    def set_parameters(
        self,
        H: NDArray[np.float32],
        Q: NDArray[np.float32],
        R: NDArray[np.float32],
        P: NDArray[np.float32],
    ) -> None:
        """
        Sets the Kalman filter parameters.

        Args:
            H (np.ndarray): Measurement matrix.
            Q (np.ndarray): Process noise covariance matrix.
            R (np.ndarray): Measurement noise covariance matrix.
            P (np.ndarray): Error covariance matrix.
        """
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P

    def get_state_bbox(self) -> NDArray[np.float32]:
        """
        Returns the current bounding box estimate from the state vector.

        Returns:
            np.ndarray: The bounding box [x1, y1, x2, y2]
        """
        return self.state[:4, 0].flatten().astype(np.float32)
