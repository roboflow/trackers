import numpy as np
from numpy.typing import NDArray


class SORTKalmanBoxTracker:
    """
    The `SORTKalmanBoxTracker` class represents the internals of a single
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
        count_id (int): Class variable to assign unique IDs to each tracker.

    Args:
        bbox (np.ndarray): Initial bounding box in the form [x1, y1, x2, y2].
    """

    count_id: int = 0
    state: NDArray[np.float32]
    F: NDArray[np.float32]
    H: NDArray[np.float32]
    Q: NDArray[np.float32]
    R: NDArray[np.float32]
    P: NDArray[np.float32]

    @classmethod
    def get_next_tracker_id(cls) -> int:
        next_id = cls.count_id
        cls.count_id += 1
        return next_id

    def __init__(self, bbox: NDArray[np.float64]) -> None:
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
        self.time_since_update = 0
        self.number_of_successful_updates += 1

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
        identity_matrix: NDArray[np.float32] = np.eye(8, dtype=np.float32)
        self.P = ((identity_matrix - K @ self.H) @ self.P).astype(np.float32)

    def get_state_bbox(self) -> NDArray[np.float32]:
        """
        Returns the current bounding box estimate from the state vector.

        Returns:
            np.ndarray: The bounding box [x1, y1, x2, y2]
        """
        return self.state[:4, 0].flatten().astype(np.float32)
