# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np


class BoTSORTKalmanBoxTracker:
    """
    Kalman-filter-based state estimator for a single tracked object.

    This class maintains the motion state of one object using a linear Kalman filter
    with a constant-velocity model. The tracker stores the object state internally in
    center-width-height form, but accepts detections and returns boxes in standard
    corner format.

    Internal state vector:
        [xc, yc, w, h, vxc, vyc, vw, vh]^T

    where:
        xc, yc:
            Bounding box center coordinates.
        w, h:
            Bounding box width and height.
        vxc, vyc:
            Velocities of the center coordinates.
        vw, vh:
            Velocities of the width and height.

    Public input/output convention:
        - input detections to `__init__()` and `update()` are expected in xyxy format:
          [x1, y1, x2, y2]
        - output from `get_state_bbox()` is returned in xyxy format:
          [x1, y1, x2, y2]

    Kalman filter matrices used in this class:
        F:
            State transition matrix. Propagates the state from one frame to the next
            under a constant-velocity assumption.
        H:
            Measurement matrix. Maps the internal 8D state to the observable 4D
            measurement space [xc, yc, w, h].
        Q:
            Process noise covariance. Models uncertainty in the motion model used
            during prediction.
        R:
            Measurement noise covariance. Models uncertainty in incoming detections
            during the update step.
        P:
            State covariance matrix. Represents the current uncertainty of the full
            8D state estimate.

    Lifecycle-related attributes:
        tracker_id:
            Permanent track identifier. Starts at -1 and is assigned later by the
            outer tracking logic once the track is considered mature.
        number_of_successful_updates:
            Number of successful detection-based updates received by this track.
        time_since_update:
            Number of consecutive prediction steps since the last measurement update.

    Notes:
        - The process and measurement noise are scaled using the current object width
          and height. This makes the uncertainty proportional to object size.
        - Width and height are constrained to remain positive after prediction and
          update to avoid degenerate boxes.
    """

    count_id = 0

    @classmethod
    def get_next_tracker_id(cls) -> int:
        next_id = cls.count_id
        cls.count_id += 1
        return next_id

    def __init__(self, bbox: np.ndarray):
        """
        Initialize a new track from the first observed bounding box.

        Args:
            bbox:
                Initial detection in xyxy format: [x1, y1, x2, y2].

        Initialization steps:
            1) Set track-management attributes such as `tracker_id`,
               `number_of_successful_updates`, and `time_since_update`.
            2) Allocate the internal 8D Kalman state vector:
                   [xc, yc, w, h, vxc, vyc, vw, vh]^T
            3) Convert the input bounding box from xyxy to xywh form:
                   [xc, yc, w, h]
            4) Store that measurement in the position/size part of the state.
            5) Initialize the Kalman filter matrices F, H, Q, R, and P.

        Notes:
            - Initial velocities are set to zero.
            - The initial covariance matrix P is set in `_initialize_kalman_filter()`
              and reflects uncertainty about both position/size and velocity.
        """
        self.tracker_id = -1
        self.number_of_successful_updates = 1
        self.time_since_update = 0

        # State mean: [xc, yc, w, h, vxc, vyc, vw, vh]^T
        self.state = np.zeros((8, 1), dtype=np.float32)

        # Initialize from first detection in xyxy
        measurement = self.xyxy_to_xywh(bbox)
        self.state[0:4, 0] = measurement

        self._initialize_kalman_filter(measurement)

    @staticmethod
    def xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
        """
        Convert a bounding box from corner format to center-size format.

        Args:
            bbox:
                Bounding box in xyxy format: [x1, y1, x2, y2].

        Returns:
            Bounding box in xywh format: [xc, yc, w, h].
        """
        x1, y1, x2, y2 = bbox.astype(np.float32)
        w = x2 - x1
        h = y2 - y1
        xc = x1 + w / 2.0
        yc = y1 + h / 2.0
        return np.array([xc, yc, w, h], dtype=np.float32)

    @staticmethod
    def xywh_to_xyxy(state_xywh: np.ndarray) -> np.ndarray:
        """
        Convert a bounding box from center-size format to corner format.

        Args:
            state_xywh:
                Bounding box in xywh format: [xc, yc, w, h].

        Returns:
            Bounding box in xyxy format: [x1, y1, x2, y2].
        """
        xc, yc, w, h = state_xywh.astype(np.float32)
        x1 = xc - w / 2.0
        y1 = yc - h / 2.0
        x2 = xc + w / 2.0
        y2 = yc + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _initialize_kalman_filter(self, measurement: np.ndarray) -> None:
        """
        Initialize the Kalman filter matrices for the current track.

        Args:
            measurement:
                Initial object measurement in xywh format:
                [xc, yc, w, h].

        This method initializes the following matrices:

        State transition matrix:
            F is an 8x8 matrix defining how the state evolves from one frame to the
            next. It implements a constant-velocity model:
                xc <- xc + vxc
                yc <- yc + vyc
                w  <- w  + vw
                h  <- h  + vh
            while the velocity terms are carried forward unchanged.

        Measurement matrix:
            H is a 4x8 matrix mapping the internal 8D state
                [xc, yc, w, h, vxc, vyc, vw, vh]^T
            to the observable 4D measurement
                [xc, yc, w, h]^T.
            In other words, only the first four state components are directly observed.

        Process noise covariance:
            Q is an 8x8 diagonal matrix representing uncertainty in the motion model
            used during prediction. Larger values allow the predicted state to change
            more freely from frame to frame.

        Measurement noise covariance:
            R is a 4x4 diagonal matrix representing uncertainty in the detector
            measurements used during correction/update.

        State covariance:
            P is the initial 8x8 covariance matrix representing uncertainty in the
            initial state estimate. The velocity terms are initialized with larger
            uncertainty than the position/size terms because they are not directly
            observed in the first frame.

        Noise scaling:
            The diagonal entries of Q, R, and P are scaled using the initial object
            width and height. This makes the uncertainty proportional to object size:
            larger objects are allowed proportionally larger absolute motion and noise.

        Notes:
            - `sigma_p` controls the scale of position/size process noise.
            - `sigma_v` controls the scale of velocity process noise.
            - `sigma_m` controls the scale of measurement noise.
            - All covariance matrices are diagonal in this implementation.
        """
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.eye(4, 8, dtype=np.float32)

        # BoT-SORT-style scale-aware noise using width/height.
        sigma_p = 0.05
        sigma_v = 0.00625
        sigma_m = 0.05

        w, h = measurement[2], measurement[3]

        q_diag = np.array(
            [
                (sigma_p * w) ** 2,
                (sigma_p * h) ** 2,
                (sigma_p * w) ** 2,
                (sigma_p * h) ** 2,
                (sigma_v * w) ** 2,
                (sigma_v * h) ** 2,
                (sigma_v * w) ** 2,
                (sigma_v * h) ** 2,
            ],
            dtype=np.float32,
        )
        self.Q = np.diag(q_diag)

        r_diag = np.array(
            [
                (sigma_m * w) ** 2,
                (sigma_m * h) ** 2,
                (sigma_m * w) ** 2,
                (sigma_m * h) ** 2,
            ],
            dtype=np.float32,
        )
        self.R = np.diag(r_diag)

        # Initial covariance, as in original BoT-SORT KF
        p_diag = np.array(
            [
                (2 * sigma_p * w) ** 2,
                (2 * sigma_p * h) ** 2,
                (2 * sigma_p * w) ** 2,
                (2 * sigma_p * h) ** 2,
                (10 * sigma_v * w) ** 2,
                (10 * sigma_v * h) ** 2,
                (10 * sigma_v * w) ** 2,
                (10 * sigma_v * h) ** 2,
            ],
            dtype=np.float32,
        )
        self.P = np.diag(p_diag)

    def _update_process_and_measurement_noise(self) -> None:
        """
        Recompute the process and measurement noise covariances from the current box
        size.

        This method updates:

        Q:
            Process noise covariance, used in the prediction step.
            It models uncertainty in how the state changes from one frame to the next.

        R:
            Measurement noise covariance, used in the update step.
            It models uncertainty in the current detection measurement.

        Why this update is needed:
            The scale of the uncertainty should depend on the current object size.
            For example, a 2-pixel error is relatively more important for a small object
            than for a large one. Therefore, the diagonal entries of Q and R are
            computed from the current predicted width and height stored in the state.

        Implementation details:
            - Width and height are read from the current state:
                  w = state[2], h = state[3]
            - They are clamped to a small positive minimum to avoid zero or negative
              values.
            - The resulting Q and R matrices remain diagonal.

        Notes:
            This method does not update P directly. It only refreshes the noise models
            used later in `predict()` and `update()`.
        """
        sigma_p = 0.05
        sigma_v = 0.00625
        sigma_m = 0.05

        w = max(float(self.state[2, 0]), 1e-3)
        h = max(float(self.state[3, 0]), 1e-3)

        q_diag = np.array(
            [
                (sigma_p * w) ** 2,
                (sigma_p * h) ** 2,
                (sigma_p * w) ** 2,
                (sigma_p * h) ** 2,
                (sigma_v * w) ** 2,
                (sigma_v * h) ** 2,
                (sigma_v * w) ** 2,
                (sigma_v * h) ** 2,
            ],
            dtype=np.float32,
        )
        self.Q = np.diag(q_diag)

        r_diag = np.array(
            [
                (sigma_m * w) ** 2,
                (sigma_m * h) ** 2,
                (sigma_m * w) ** 2,
                (sigma_m * h) ** 2,
            ],
            dtype=np.float32,
        )
        self.R = np.diag(r_diag)

    def predict(self) -> None:
        """
        Predict the next state and covariance using the Kalman motion model.

        This method performs the Kalman filter prediction step:

            state <- F @ state
            P     <- F @ P @ F.T + Q

        where:
            F:
                State transition matrix.
            P:
                Current state covariance matrix.
            Q:
                Process noise covariance.

        Effect of the prediction:
            - The center position and box size are advanced using their current
              velocities.
            - The covariance matrix P is propagated forward and increased by Q to
              reflect additional uncertainty introduced during motion prediction.

        Additional behavior:
            - The process and measurement noise matrices are refreshed first by calling
              `_update_process_and_measurement_noise()`.
            - Width and height are clamped to remain positive after prediction.
            - `time_since_update` is incremented because this frame has not yet received
              a measurement update.

        Notes:
            This method does not use any detection input. It only extrapolates the track
            state forward in time.
        """
        self._update_process_and_measurement_noise()

        # Predict state
        self.state = self.F @ self.state  # type: ignore[assignment]

        # Predict error (uncertainty) covariance
        self.P = self.F @ self.P @ self.F.T + self.Q  # type: ignore[assignment]

        # Prevent degenerate box shape
        self.state[2, 0] = max(self.state[2, 0], 1e-3)
        self.state[3, 0] = max(self.state[3, 0], 1e-3)

        # Increase time since update
        self.time_since_update += 1

    def update(self, bbox: np.ndarray) -> None:
        """
        Correct the predicted state using a new detection.

        Args:
            bbox:
                Detection bounding box in xyxy format: [x1, y1, x2, y2].

        This method performs the Kalman filter correction/update step:

            measurement = xyxy_to_xywh(bbox)
            S = H @ P @ H.T + R
            K = P @ H.T @ inv(S)
            y = measurement - H @ state
            state = state + K @ y
            P = (I - K @ H) @ P

        where:
            measurement:
                Observed bounding box converted to [xc, yc, w, h].
            S:
                Innovation covariance. Represents uncertainty in the predicted
                measurement.
            K:
                Kalman gain. Controls how strongly the state is corrected toward
                the new measurement.
            y:
                Innovation (also called residual), i.e. the difference between the
                observed measurement and the predicted measurement.
            I:
                Identity matrix of appropriate size.

        Effect of the update:
            - The predicted state is corrected toward the observed detection.
            - The covariance matrix P is reduced to reflect increased confidence
              after receiving a measurement.

        Additional behavior:
            - `time_since_update` is reset to zero.
            - `number_of_successful_updates` is incremented.
            - Width and height are clamped to remain positive after correction.

        Notes:
            The measurement only directly observes [xc, yc, w, h], not the velocity
            terms. However, the velocity estimates can still change indirectly through
            the Kalman gain and the state covariance structure.
        """
        self.time_since_update = 0
        self.number_of_successful_updates += 1

        measurement = self.xyxy_to_xywh(bbox).reshape((4, 1))
        self._update_process_and_measurement_noise()

        # Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Innovation (residual)
        y = measurement - self.H @ self.state

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        identity_matrix = np.eye(8, dtype=np.float32)
        self.P = (identity_matrix - K @ self.H) @ self.P  # type: ignore[assignment]

        self.state[2, 0] = max(self.state[2, 0], 1e-3)
        self.state[3, 0] = max(self.state[3, 0], 1e-3)

    def get_state_bbox(self) -> np.ndarray:
        """
        Return current predicted box in xyxy format.
        """
        return self.xywh_to_xyxy(self.state[0:4, 0])
