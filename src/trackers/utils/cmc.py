# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
from deprecate import deprecated

if TYPE_CHECKING:
    from trackers.utils.base_tracklet import BaseTracklet
    from trackers.utils.state_representations import BaseStateEstimator

logger = logging.getLogger("trackers.cmc")

CMCMethod = Literal["orb", "sift", "sparseOptFlow", "ecc"]
CMCTMethod = CMCMethod  # back-compat alias; use CMCMethod. Removed in v3.0.


@dataclass
class CMCConfig:
    """
    Configuration for camera motion compensation (CMC).

    The CMC module estimates a global 2D affine transform `H` (2x3) between consecutive
    frames. This transform is then applied to predicted track states before data
    association.

    Attributes:
        method:
            Camera motion estimation method.

            - "orb": Feature matching using
              FAST keypoints + ORB descriptors + BFMatcher (Hamming),
              followed by robust affine estimation (RANSAC).
              Optionally masks out detection boxes so features are extracted from
              background.
            - "sift": Feature matching using
              SIFT keypoints + SIFT descriptors + BFMatcher (L2),
              followed by robust affine estimation (RANSAC).
              Optionally masks out detection boxes so features are extracted from
              background. "sift" generally produces fewer but more distinctive matches
              than ORB at higher compute cost.
            - "sparseOptFlow": Sparse optical flow using corner tracking:
              goodFeaturesToTrack -> calcOpticalFlowPyrLK -> robust affine estimation
              (RANSAC).
            - "ecc": Global image alignment using the Enhanced Correlation Coefficient
              (ECC) optimization method. This estimates a 2D Euclidean transform
              directly from grayscale image intensities rather than from sparse feature
              correspondences.

        downscale:
            Integer downscale factor applied to frames before running CMC.

            Purpose:
            - Speeds up feature extraction / optical flow.

            Behavior:
            - Frames are resized to (W//downscale, H//downscale) for motion estimation.
            - The resulting affine translation components H[0,2], H[1,2] are scaled back
              by multiplying by `downscale`, so the transform is in original image
              coordinates.

        fast_threshold:
            (ORB only) Threshold for the FAST keypoint detector.
            Higher values yield fewer keypoints (more selective); lower values yield
            more keypoints.

        ransac_reproj_threshold:
            (ORB and SIFT) RANSAC reprojection threshold in pixels passed to
            OpenCV's affine estimation. It controls how far a point is allowed to
            deviate from the estimated model while still being counted as an inlier.
            Smaller values are stricter (reject more matches); larger values are more
            tolerant.

        max_spatial_distance_frac:
            (ORB and SIFT) Maximum allowed spatial displacement for a tentative match,
            expressed as a fraction of (image width, image height) *after downscale*.

            Example:
                If max_spatial_distance_frac = 0.25 and the downscaled frame is (W, H),
                then a match is rejected if |dx| >= 0.25*W or |dy| >= 0.25*H.

            Motivation:
                Reject obviously incorrect descriptor matches whose displacement is
                implausibly large.

        roi_min_frac:
            (ORB and SIFT) Lower bound of the region-of-interest (ROI) used to select
            keypoints, expressed as a fraction of frame size. Points outside the ROI
            are masked out.

            Example:
                roi_min_frac=0.02 means we ignore a ~2% border on each side.

        roi_max_frac:
            (ORB and SIFT) Upper bound of the ROI used to select keypoints (fraction of
            frame size). Together with roi_min_frac, it defines a central rectangle:
                [roi_min_frac..roi_max_frac] in both x and y.

        sift_n_octave_layers:
            (SIFT only) Number of octave layers used by SIFT when constructing the
            scale-space pyramid. Increasing this can increase sensitivity to scale
            changes, at higher compute cost.

        sift_contrast_threshold:
            (SIFT only) Threshold controlling how sensitive SIFT is
            to low-contrast keypoints. Lower values generally produce more keypoints;
            higher values are stricter.

        sift_edge_threshold:
            (SIFT only) Threshold controlling rejection of keypoints on edges.
            Lower values reject more edge-like responses; higher values are more
            permissive.

        sof_max_corners:
            (SparseOptFlow only) `maxCorners` passed to `cv2.goodFeaturesToTrack`.
            Maximum number of corners to detect for tracking.
            Larger values can improve robustness (more points), but cost more compute.

        sof_quality_level:
            (SparseOptFlow only) `qualityLevel` passed to `cv2.goodFeaturesToTrack`.
            Minimum accepted quality of corners. A higher value keeps only stronger
            corners; a lower value yields more corners (including weaker ones).

        sof_min_distance:
            (SparseOptFlow only) `minDistance` passed to `cv2.goodFeaturesToTrack`.
            Minimum Euclidean distance (in pixels) between returned corners.
            Higher values produce more spatially spread points; lower values allow
            clustering.

        sof_block_size:
            (SparseOptFlow only) `blockSize` passed to `cv2.goodFeaturesToTrack`.
            Size of the neighborhood used to compute corner quality (structure tensor
            window).

        sof_use_harris:
            (SparseOptFlow only) `useHarrisDetector` passed to
            `cv2.goodFeaturesToTrack`. If True, uses the Harris corner measure;
            if False, uses the Shi-Tomasi measure.

        sof_k:
            (SparseOptFlow only) `k` passed to `cv2.goodFeaturesToTrack`.
            Harris detector free parameter. Ignored if `sof_use_harris` is False.

        ecc_number_of_iterations:
            (ECC only) Maximum number of optimization iterations used by the ECC
            alignment procedure.

        ecc_termination_eps:
            (ECC only) Convergence tolerance used by the ECC optimizer.
            Smaller values require a more precise fit and may increase runtime.

        ecc_gaussian_filter_size:
            (ECC only) Gaussian filter size parameter passed to OpenCV's
            `findTransformECC`. This can help stabilize optimization on noisy frames.
            A value of 1 matches the current implementation.
    """

    method: CMCMethod = "sparseOptFlow"
    downscale: int = 2

    # Shared ORB and SIFT parameters (_estimate_feature_affine)
    ransac_reproj_threshold: float = 3.0
    max_spatial_distance_frac: float = 0.25
    roi_min_frac: float = 0.02
    roi_max_frac: float = 0.98

    # ORB parameters
    fast_threshold: int = 20

    # SIFT parameters
    sift_n_octave_layers: int = 3
    sift_contrast_threshold: float = 0.02
    sift_edge_threshold: int = 20

    # Sparse optical flow parameters (goodFeaturesToTrack)
    sof_max_corners: int = 1000
    sof_quality_level: float = 0.01
    sof_min_distance: int = 1
    sof_block_size: int = 3
    sof_use_harris: bool = False
    sof_k: float = 0.04

    # ECC parameters

    # BoT-SORT's original, which significantly increases runtime
    # ecc_number_of_iterations: int = 5000
    # ecc_termination_eps: float = 1e-6

    # Adjusted
    ecc_number_of_iterations: int = 50
    ecc_termination_eps: float = 1e-4

    ecc_gaussian_filter_size: int = 1


class CMC:
    """
    Camera motion compensation estimator.

    Estimates a global 2D affine transform H (2x3) between consecutive frames and
    provides helpers to apply that transform to predicted track states.

    The ``estimate()`` method returns a 2x3 affine matrix ``H`` each frame. Pass it to
    ``CMC.apply_batch`` or ``BoTSORTTracklet.apply_cmc`` to warp Kalman states before
    data association. ``apply_batch`` is BoT-SORT-specific and requires tracklets with
    a ``state_estimator.kf`` attribute; see :meth:`apply_batch` for details.

    Typical usage (integrating CMC into a tracker loop)::

        cmc = CMC(CMCConfig(method="sparseOptFlow"))

        for frame_bgr, detections in video:
            for trk in tracker.tracks:
                trk.predict()
            H = cmc.estimate(frame_bgr, detections.xyxy)
            CMC.apply_batch(H, tracker.tracks)
            # then run data association …

    Internal state:
        - Keeps previous-frame features / points depending on the chosen method.
        - On the first frame (or after reset), returns identity transform.

    Notes:
        - H maps points from previous frame coordinates to current frame coordinates.
        - This class does not perform any drawing/visualization; it estimates motion
          and applies affine warps to Kalman states when using ``apply_batch``.
    """

    def __init__(self, cfg: CMCConfig | None = None) -> None:
        """
        Initialize CMC.

        Args:
            cfg: Optional configuration. If None, defaults are used.

        Notes:
            - Detector/extractor/matcher are only created if method is "orb" or "sift".
            - feature_params are only created if method is "sparseOptFlow".
            - ECC optimization settings are created for "ecc".
        """
        self.cfg = cfg or CMCConfig()
        self.downscale = max(1, int(self.cfg.downscale))

        # ORB init (only if needed)
        self.detector: Any | None = None
        self.extractor: Any | None = None
        self.matcher: Any | None = None
        if self.cfg.method == "orb":
            self.detector = cv2.FastFeatureDetector_create(self.cfg.fast_threshold)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif self.cfg.method == "sift":
            self.detector = cv2.SIFT_create(
                nOctaveLayers=self.cfg.sift_n_octave_layers,
                contrastThreshold=self.cfg.sift_contrast_threshold,
                edgeThreshold=int(self.cfg.sift_edge_threshold),
            )
            self.extractor = cv2.SIFT_create(
                nOctaveLayers=self.cfg.sift_n_octave_layers,
                contrastThreshold=self.cfg.sift_contrast_threshold,
                edgeThreshold=int(self.cfg.sift_edge_threshold),
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif self.cfg.method == "sparseOptFlow":
            self.feature_params = {
                "maxCorners": self.cfg.sof_max_corners,
                "qualityLevel": self.cfg.sof_quality_level,
                "minDistance": self.cfg.sof_min_distance,
                "blockSize": self.cfg.sof_block_size,
                "useHarrisDetector": self.cfg.sof_use_harris,
                "k": self.cfg.sof_k,
            }
        elif self.cfg.method == "ecc":
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.cfg.ecc_number_of_iterations,
                self.cfg.ecc_termination_eps,
            )
        else:
            valid = ("orb", "sift", "sparseOptFlow", "ecc")
            raise ValueError(f"Unknown CMC method {self.cfg.method!r}. Valid options are: {valid}.")

        self.frames_failed = 0
        self.reset()

    def reset(self) -> None:
        """
        Reset internal state.

        After calling reset:
        - The next `estimate()` call returns identity and initializes prev-frame state.
        - This should be called when starting a new sequence or after a scene cut.
        """
        self._initialized = False
        self.frames_failed = 0

        # ORB state
        self._prev_kps = None
        self._prev_desc: np.ndarray | None = None

        # SparseOptFlow state
        self._prev_frame_gray: np.ndarray | None = None

        # shape (N,1,2) from goodFeaturesToTrack
        self._prev_points: np.ndarray | None = None

    def estimate(self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray | None = None) -> np.ndarray:
        """
        Estimate global affine transform H (2x3) from previous frame to current frame.

        Args:
            frame_bgr: Current frame in BGR format (uint8), shape (H, W, 3).
            dets_xyxy: Optional detections (N,4) in xyxy format, in original image
                scale. Used by feature-based methods (ORB and SIFT) to mask out object
                regions during motion estimation.

        Returns:
            Affine transform matrix of shape (2, 3), dtype float32.
            Identity if not enough correspondences or if not initialized yet.

        Examples:
            >>> import numpy as np
            >>> cmc = CMC(CMCConfig(method="sparseOptFlow"))
            >>> frame = np.zeros((240, 320, 3), dtype=np.uint8)
            >>> H = cmc.estimate(frame)  # first frame always returns identity
            >>> H.shape
            (2, 3)
        """
        if frame_bgr is None:
            return np.eye(2, 3, dtype=np.float32)

        if self.cfg.method == "orb" or self.cfg.method == "sift":
            return self._estimate_feature_affine(frame_bgr, dets_xyxy)

        if self.cfg.method == "sparseOptFlow":
            return self._estimate_sparse_optflow(frame_bgr)

        if self.cfg.method == "ecc":
            return self._estimate_ecc(frame_bgr)

        # fallback
        return np.eye(2, 3, dtype=np.float32)

    def _estimate_feature_affine(self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray | None = None) -> np.ndarray:
        """
        Feature affine estimation. ORB-based or SIFT-based
        (different initializations of self.detector, self.extractor and self.matcher for
        ORB and SIFT)

        Steps:
            1) Convert to grayscale (+ optional downscale).
            2) Create ROI mask and optionally mask out detections (background emphasis).
            3) Detect FAST keypoints and compute ORB or SIFT descriptors.
            4) KNN match descriptors against previous frame (ratio test).
            5) Filter matches by max spatial displacement and by 2.5*std inliers.
            6) Estimate affine transform with RANSAC.
            7) Scale translation back up if downscaled.

        Args:
            frame_bgr: Current BGR frame.
            dets_xyxy: Optional detection boxes for masking (original image scale).

        Returns:
            Affine transform matrix (2, 3) mapping previous frame to current, float32.
        """
        H_img, W_img = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.downscale > 1:
            gray = cv2.resize(gray, (W_img // self.downscale, H_img // self.downscale))
        H, W = gray.shape[:2]

        # Build mask: central ROI + remove detections (background features)
        mask = np.zeros_like(gray, dtype=np.uint8)
        y0 = int(self.cfg.roi_min_frac * H)
        y1 = int(self.cfg.roi_max_frac * H)
        x0 = int(self.cfg.roi_min_frac * W)
        x1 = int(self.cfg.roi_max_frac * W)
        mask[y0:y1, x0:x1] = 255

        if dets_xyxy is not None and len(dets_xyxy) > 0:
            dets = np.asarray(dets_xyxy, dtype=np.float32) / float(self.downscale)
            dets = dets.astype(np.int32)

            # Safety clipping to avoid negative/out-of-bounds slicing
            dets[:, 0] = np.clip(dets[:, 0], 0, W - 1)
            dets[:, 2] = np.clip(dets[:, 2], 0, W - 1)
            dets[:, 1] = np.clip(dets[:, 1], 0, H - 1)
            dets[:, 3] = np.clip(dets[:, 3], 0, H - 1)

            for x1b, y1b, x2b, y2b in dets:
                if x2b > x1b and y2b > y1b:
                    mask[y1b:y2b, x1b:x2b] = 0

        # Detect + describe (ORB / SIFT). Mypy cannot narrow instance attrs here.
        kps = self.detector.detect(gray, mask)  # type: ignore[union-attr]
        kps, desc = self.extractor.compute(gray, kps)  # type: ignore[union-attr]

        H_aff = np.eye(2, 3, dtype=np.float32)

        # First frame init
        if not self._initialized:
            self._prev_kps = copy.copy(kps)
            self._prev_desc = None if desc is None else copy.copy(desc)
            self._initialized = True
            return H_aff

        if self._prev_desc is None or desc is None or len(desc) == 0 or self._prev_kps is None:
            self._prev_kps = copy.copy(kps)
            self._prev_desc = None if desc is None else copy.copy(desc)
            return H_aff

        knn = self.matcher.knnMatch(self._prev_desc, desc, k=2)  # type: ignore[union-attr]
        if len(knn) == 0:
            self._prev_kps = copy.copy(kps)
            self._prev_desc = copy.copy(desc)
            return H_aff

        max_spatial = self.cfg.max_spatial_distance_frac * np.array([W, H], dtype=np.float32)

        prev_pts: list[np.ndarray] = []
        curr_pts: list[np.ndarray] = []
        spatial_deltas: list[np.ndarray] = []

        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.9 * n.distance:
                p_prev = np.array(self._prev_kps[m.queryIdx].pt, dtype=np.float32)
                p_curr = np.array(kps[m.trainIdx].pt, dtype=np.float32)
                d = p_prev - p_curr
                if (abs(d[0]) < max_spatial[0]) and (abs(d[1]) < max_spatial[1]):
                    spatial_deltas.append(d)
                    prev_pts.append(p_prev)
                    curr_pts.append(p_curr)

        if len(prev_pts) >= 5:
            spatial_arr = np.asarray(spatial_deltas, dtype=np.float32)
            mean = spatial_arr.mean(axis=0)
            std = spatial_arr.std(axis=0) + 1e-6
            inl = np.logical_and(
                np.abs(spatial_arr[:, 0] - mean[0]) < 2.5 * std[0],
                np.abs(spatial_arr[:, 1] - mean[1]) < 2.5 * std[1],
            )
            prev_pts_np = np.asarray(prev_pts, dtype=np.float32)[inl]
            curr_pts_np = np.asarray(curr_pts, dtype=np.float32)[inl]

            if len(prev_pts_np) >= 5:
                H_est, _ = cv2.estimateAffinePartial2D(
                    prev_pts_np,
                    curr_pts_np,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.cfg.ransac_reproj_threshold,
                )
                if H_est is not None:
                    H_aff = H_est.astype(np.float32)
                    if self.downscale > 1:
                        H_aff[0, 2] *= self.downscale
                        H_aff[1, 2] *= self.downscale

        self._prev_kps = copy.copy(kps)
        self._prev_desc = copy.copy(desc)
        return H_aff

    def _estimate_sparse_optflow(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Sparse optical-flow-based affine estimation.

        Steps:
            1) grayscale (+ optional downscale)
            2) detect corners using goodFeaturesToTrack
            3) compute correspondences via calcOpticalFlowPyrLK(prev, curr, prev_points)
            4) keep only points with status == 1
            5) estimate affine transform with RANSAC
            6) scale translation back up if downscaled

        Args:
            frame_bgr: Current BGR frame.

        Returns:
            Affine transform matrix (2, 3) mapping previous frame to current, float32.
        """
        H_img, W_img = frame_bgr.shape[:2]
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        H_aff = np.eye(2, 3, dtype=np.float32)

        # Downscale
        if self.downscale > 1:
            frame = cv2.resize(frame, (W_img // self.downscale, H_img // self.downscale))

        # Find keypoints in current frame
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # First frame: init and return identity
        if not self._initialized:
            self._prev_frame_gray = frame.copy()
            self._prev_points = copy.copy(keypoints)
            self._initialized = True
            return H_aff

        # If we don't have points, re-init
        if self._prev_frame_gray is None or self._prev_points is None or keypoints is None:
            self._prev_frame_gray = frame.copy()
            self._prev_points = copy.copy(keypoints)
            return H_aff

        # Optical flow correspondences
        # calcOpticalFlowPyrLK will throw or return nonsense if we give it None
        matched, status, _err = cv2.calcOpticalFlowPyrLK(self._prev_frame_gray, frame, self._prev_points, None)

        if status is None or matched is None:
            self._prev_frame_gray = frame.copy()
            self._prev_points = copy.copy(keypoints)
            return H_aff

        # Keep only good correspondences
        prev_pts: list[np.ndarray] = []
        curr_pts: list[np.ndarray] = []
        # status is (N,1) or (N,)
        status_flat = status.reshape(-1)

        for i in range(len(status_flat)):
            if status_flat[i]:
                prev_pts.append(self._prev_points[i])
                curr_pts.append(matched[i])

        prev_pts_np = np.array(prev_pts)
        curr_pts_np = np.array(curr_pts)

        # Find rigid matrix
        if (np.size(prev_pts_np, 0) > 4) and (np.size(prev_pts_np, 0) == np.size(curr_pts_np, 0)):
            H_est, _ = cv2.estimateAffinePartial2D(
                prev_pts_np,
                curr_pts_np,
                method=cv2.RANSAC,
            )
            if H_est is not None:
                H_aff = H_est.astype(np.float32)

                # Handle downscale translation back to original image coords
                if self.downscale > 1:
                    H_aff[0, 2] *= self.downscale
                    H_aff[1, 2] *= self.downscale
        else:
            logger.warning("CMC: not enough matching points for motion estimation")
            self.frames_failed += 1

        # Store to next iteration
        self._prev_frame_gray = frame.copy()
        # self._prev_points = copy.copy(keypoints)
        self._prev_points = None if keypoints is None else keypoints.copy()

        return H_aff

    def _estimate_ecc(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        ECC-based affine motion estimation.

        This method estimates a global 2D Euclidean transform between the previous
        frame and the current frame using OpenCV's Enhanced Correlation Coefficient
        (ECC) image alignment algorithm.

        Steps:
            1) Convert the current frame to grayscale.
            2) Optionally smooth and downscale the frame.
            3) If this is the first frame, store it and return identity.
            4) Optimize a 2x3 warp matrix aligning the previous frame to the current
               frame.
            5) If optimization succeeds, return the estimated transform.
               Otherwise, keep the identity transform.
            6) Store the current frame for the next call.

        Args:
            frame_bgr: Current frame in BGR format.

        Returns:
            Affine transform matrix of shape (2, 3), dtype float32, mapping
            previous-frame coordinates to current-frame coordinates. Returns
            identity if initialization has not yet occurred or if ECC
            optimization fails.
        """
        H_img, W_img = frame_bgr.shape[:2]
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        H_aff = np.eye(2, 3, dtype=np.float32)

        if self.downscale > 1:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (W_img // self.downscale, H_img // self.downscale))

        if not self._initialized:
            self._prev_frame_gray = frame.copy()
            self._initialized = True
            return H_aff

        if self._prev_frame_gray is None:
            self._prev_frame_gray = frame.copy()
            return H_aff

        try:
            _cc, H_est = cv2.findTransformECC(
                self._prev_frame_gray,
                frame,
                H_aff,
                self.warp_mode,
                self.criteria,
                None,
                self.cfg.ecc_gaussian_filter_size,
            )
            if H_est is not None:
                H_aff = H_est.astype(np.float32)
                if self.downscale > 1:
                    H_aff[0, 2] *= self.downscale
                    H_aff[1, 2] *= self.downscale
        except cv2.error:
            logger.warning("CMC: ECC motion estimation failed, using identity")
            self.frames_failed += 1

        # NOTE: this line is not included in the original BoT-SORT. However,
        # in a working recurrent estimator, you do need to update the previous frame
        # after each call. Otherwise the next call would keep aligning against an old
        # frame.
        self._prev_frame_gray = frame.copy()

        return H_aff

    @staticmethod
    def warp_xyxy_corners(
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
        R: np.ndarray,
        t: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform four box corners via R/t and return the enclosing axis-aligned box.

        Takes per-axis coordinate arrays (not a packed (N, 4) array) and returns
        per-axis result arrays of the same shape. The four corners are
        ``(x1, y1)``, ``(x2, y1)``, ``(x2, y2)``, ``(x1, y2)``; after the affine
        transform the axis-aligned bounding box of the transformed corners is returned.

        Transforming only two corners can invert the box or produce invalid geometry
        under rotation or reflection; transforming all four corners and taking the
        enclosing axis-aligned box always yields a valid result.

        Args:
            x1: Left edge coordinate(s) as a NumPy scalar (shape ``()``) or 1-D
                array (shape ``(N,)``).
            y1: Top edge coordinate(s), same shape as ``x1``.
            x2: Right edge coordinate(s), same shape as ``x1``.
            y2: Bottom edge coordinate(s), same shape as ``x1``.
            R: 2x2 rotation/shear sub-matrix of the affine transform.
            t: Optional 2-element translation vector.

        Returns:
            Tuple ``(new_x1, new_y1, new_x2, new_y2)`` -- per-axis min and max of
            the four transformed corners, same shape as inputs.

        Examples:
            >>> import numpy as np
            >>> R = np.eye(2, dtype=np.float64)
            >>> t = np.array([5.0, -3.0])
            >>> x1 = np.array([10.0, 20.0])
            >>> y1 = np.array([20.0, 30.0])
            >>> x2 = np.array([50.0, 60.0])
            >>> y2 = np.array([80.0, 90.0])
            >>> nx1, ny1, nx2, ny2 = CMC.warp_xyxy_corners(x1, y1, x2, y2, R, t)
            >>> nx1.tolist()
            [15.0, 25.0]
        """
        corners = np.stack(
            [
                np.stack([x1, y1], axis=-1),
                np.stack([x2, y1], axis=-1),
                np.stack([x2, y2], axis=-1),
                np.stack([x1, y2], axis=-1),
            ],
            axis=-2,
        )  # (..., 4, 2)
        out = corners @ R.T
        if t is not None:
            out = out + t
        lo = out.min(axis=-2)
        hi = out.max(axis=-2)
        return lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1]

    @staticmethod
    @deprecated(
        target=warp_xyxy_corners.__func__,  # type: ignore[attr-defined]
        deprecated_in="2.5",
        remove_in="3.0",
    )
    def apply_to_xyxy(  # type: ignore[empty-body]
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
        R: np.ndarray,
        t: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Deprecated alias for :meth:`CMC.warp_xyxy_corners`.

        .. deprecated:: 2.5
            Renamed to :meth:`CMC.warp_xyxy_corners`. This wrapper forwards all
            calls to the new name and will be removed in v3.0.

        Args:
            x1: Left edge coordinate(s).
            y1: Top edge coordinate(s).
            x2: Right edge coordinate(s).
            y2: Bottom edge coordinate(s).
            R: 2x2 rotation/shear sub-matrix of the affine transform.
            t: Optional 2-element translation vector.

        Returns:
            Tuple ``(new_x1, new_y1, new_x2, new_y2)`` -- see
            :meth:`CMC.warp_xyxy_corners` for full details.

        Examples:
            >>> import numpy as np
            >>> import warnings
            >>> R = np.eye(2, dtype=np.float64)
            >>> t = np.array([5.0, -3.0])
            >>> x1 = np.array([10.0, 20.0])
            >>> y1 = np.array([20.0, 30.0])
            >>> x2 = np.array([50.0, 60.0])
            >>> y2 = np.array([80.0, 90.0])
            >>> with warnings.catch_warnings():
            ...     warnings.simplefilter("ignore")
            ...     nx1, _, _, _ = CMC.apply_to_xyxy(x1, y1, x2, y2, R, t)
            >>> nx1.tolist()
            [15.0, 25.0]
        """

    @staticmethod
    def apply_batch(H: np.ndarray | None, tracklets: Sequence[BaseTracklet]) -> None:
        """Apply a 2x3 affine camera-motion transform to a list of BoT-SORT tracklets.

        .. note::
            This method is BoT-SORT-specific. It requires each tracklet to expose a
            ``state_estimator`` with a ``kf.x`` state-vector column (``(dim, 1)``) and
            ``kf.P`` covariance matrix, matching the layout of
            ``XCYCWHStateEstimator`` / ``XYXYStateEstimator``. Passing arbitrary
            tracklets without this layout will raise ``AttributeError`` at runtime.

        All tracklets in the list must share the same state representation type.
        Pass a heterogeneous list and ``TypeError`` is raised immediately.

        For XYXY-state tracks, positions and velocities are updated via four-corner
        enclosure (``CMC.warp_xyxy_corners``) so that axis-alignment is preserved under
        rotation, reflection, and shear. The covariance matrix ``P`` is updated with
        the block-diagonal rotation matrix only when ``R`` is axis-aligned
        (off-diagonals < 1e-6). When ``R`` has cross-axis terms, ``P`` is left
        unchanged.

        For XCYCWH-state tracks, only the centre position and velocity are rotated;
        width/height and their velocities are not transformed.

        Args:
            H: 2x3 affine transform matrix returned by ``CMC.estimate()``. If
                ``None``, this method is a no-op.
            tracklets: Homogeneous list of BoT-SORT tracklets, each with a
                ``state_estimator.kf.x`` state vector and ``kf.P`` covariance.

        Raises:
            TypeError: If tracklets in the list have different state estimator types.

        Examples:
            >>> import numpy as np
            >>> from trackers.core.botsort.tracklet import BoTSORTTracklet
            >>> bbox = np.array([10.0, 20.0, 50.0, 80.0])
            >>> track = BoTSORTTracklet(bbox)
            >>> H = np.eye(2, 3, dtype=np.float32)
            >>> CMC.apply_batch(H, [track])  # identity H -- state unchanged
            >>> CMC.apply_batch(None, [track])  # None H -- no-op
        """
        from trackers.utils.state_representations import XYXYStateEstimator

        if H is None or len(tracklets) == 0:
            return

        R = H[:2, :2].astype(np.float64)
        t = H[:2, 2].astype(np.float64)

        first_estimator: BaseStateEstimator = tracklets[0].state_estimator
        if not all(type(t.state_estimator) is type(first_estimator) for t in tracklets):
            mismatch = next(t for t in tracklets if type(t.state_estimator) is not type(first_estimator))
            raise TypeError(
                f"CMC.apply_batch requires homogeneous state types; "
                f"got {type(first_estimator).__name__!r} and {type(mismatch.state_estimator).__name__!r}."
            )
        dim = first_estimator.kf.x.shape[0]
        is_xyxy = isinstance(first_estimator, XYXYStateEstimator)

        # Stack states (N, dim) and covariances (N, dim, dim)
        states = np.array([trk.state_estimator.kf.x.reshape(-1) for trk in tracklets])
        Ps = np.array([trk.state_estimator.kf.P for trk in tracklets])

        if is_xyxy:
            # XYXY boxes must remain axis-aligned after CMC. For transforms with
            # rotation/reflection/shear, applying the affine matrix only to the
            # top-left and bottom-right corners can invert the box or produce
            # invalid geometry. Transform all four corners, then rebuild the
            # enclosing axis-aligned box with per-axis min/max.
            states[:, 0], states[:, 1], states[:, 2], states[:, 3] = CMC.warp_xyxy_corners(
                states[:, 0], states[:, 1], states[:, 2], states[:, 3], R, t
            )
            # Keep XYXY velocity ordering valid under mixed-axis transforms by
            # applying the same corner-wise normalization to the paired velocity
            # components.
            states[:, 4], states[:, 5], states[:, 6], states[:, 7] = CMC.warp_xyxy_corners(
                states[:, 4], states[:, 5], states[:, 6], states[:, 7], R
            )
        else:
            # Batch-transform centre positions: x' = x @ R.T + t
            states[:, 0:2] = states[:, 0:2] @ R.T + t
            # Batch-transform centre velocities: v' = v @ R.T
            states[:, 4:6] = states[:, 4:6] @ R.T

        A = None
        if is_xyxy:
            # atol=1e-6: float32 CMC (sparseOptFlow/ORB/SIFT/ECC) carries ~1e-7
            # to 1e-6 residuals on off-diagonals even for pure-translation H;
            # default atol=1e-8 misclassifies those as cross-axis transforms.
            if np.isclose(R[0, 1], 0.0, atol=1e-6) and np.isclose(R[1, 0], 0.0, atol=1e-6):
                A = np.eye(dim, dtype=np.float64)
                A[0:2, 0:2] = R
                A[2:4, 2:4] = R
                A[4:6, 4:6] = R
                A[6:8, 6:8] = R
        else:
            A = np.eye(dim, dtype=np.float64)
            A[0:2, 0:2] = R
            A[4:6, 4:6] = R

        if A is not None:
            Ps = A @ Ps @ A.T

        for i, trk in enumerate(tracklets):
            trk.state_estimator.kf.x = states[i].reshape(-1, 1)
            trk.state_estimator.kf.P = Ps[i]
