# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Conditional camera motion compensation, independently usable by trackers.

The ORB preset and XCYCWH state warp follow the original McByte++ research version. Estimator organization and the
alternative methods follow Trackers' CMC. This separate implementation adds transform rejection and safe per-track state
updates without changing CMC.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from trackers.utils.base_tracklet import BaseTracklet

logger = logging.getLogger("trackers.conditional_cmc")
CMCMethod = Literal["orb", "sift", "sparseOptFlow", "ecc"]


@dataclass
class ConditionalCMCConfig:
    """Configure conditional camera motion compensation (CMC).

    Camera motion is estimated as a global 2D affine transform with shape
    (2, 3), mapping the previous image to the current image. The transform
    is checked against the configured scale limits before it can correct
    predicted Kalman states, ahead of detection-to-track association.

    The defaults select ORB with downscale 4 and the research McByte++ match
    and scale thresholds. The selected method processes one image at a time.
    Scale checks apply to all four methods; feature-matching settings
    apply only to ORB/SIFT, and the ``sof_`` and ``ecc_`` settings apply only
    to their respective methods. Candidate track states are also checked
    for finite values and valid box dimensions before being committed.

    Attributes:
        method:
            Camera motion estimation method. Defaults to ``"orb"``.

            - ``"orb"``: FAST keypoint detection, ORB descriptors and Hamming
              descriptor matching, followed by robust affine estimation with
              RANSAC.
            - ``"sift"``: SIFT keypoint detection and descriptors, L2 descriptor
              matching, and robust affine estimation with RANSAC.
            - ``"sparseOptFlow"``: Corner detection with
              ``cv2.goodFeaturesToTrack``, tracking with
              ``cv2.calcOpticalFlowPyrLK``, and affine estimation with RANSAC.
            - ``"ecc"``: Enhanced Correlation Coefficient (ECC) alignment of
              grayscale image intensities. Estimates a Euclidean transform
              containing rotation and translation.

            ORB/SIFT use a background feature mask: keypoints outside the ROI
            or inside supplied detection boxes are excluded. Sparse optical
            flow and ECC do not use the detection boxes or ROI settings.

        downscale:
            Image downscale factor, converted to an integer and clamped to
            at least one. Defaults to 4. Smaller images reduce the work of
            feature extraction, optical flow or ECC alignment. Translation
            components are multiplied by this factor after estimation so
            the returned transform uses original-image pixel coordinates.
            If either image dimension divided by this factor is less than
            two pixels after integer division, estimation returns identity
            and clears the previous-image reference.

        minimum_matches:
            ORB/SIFT candidate matches required after descriptor-ratio and
            spatial-displacement filtering, but before statistical filtering
            of displacement outliers. Defaults to the research ORB value of
            12. At least five matches must remain after statistical filtering
            before RANSAC is attempted. Must be an integer of at least five;
            booleans are not accepted. Sparse optical flow independently
            requires five correspondences and does not use this setting.

        minimum_scale:
            Inclusive lower limit for the Euclidean norm of each column of
            the affine matrix's 2x2 linear part. Defaults to 0.3. Both column
            norms must pass the lower and upper limits for any estimation
            method. These checks reject extreme transforms but do not
            guarantee accurate motion or valid transformed box geometry.

        maximum_scale:
            Inclusive upper limit for those column norms. Defaults to 3.0.
            Both scale limits must be finite and satisfy
            ``0 < minimum_scale <= 1 <= maximum_scale``. The defaults match
            the research tracker's active ORB scale guard. These settings
            are checked by both ``estimate`` and ``apply_batch``.

        ransac_reproj_threshold:
            ORB/SIFT RANSAC reprojection tolerance in downscaled-image
            pixels. Defaults to 3.0. Smaller values require matches to fit
            the estimated transform more closely. This setting is not
            passed to the sparse optical-flow method's RANSAC call.

        max_spatial_distance_frac:
            ORB/SIFT displacement limit for each tentative feature match,
            as a fraction of the downscaled width and height. Defaults to
            0.25: a match is rejected when ``abs(dx) >= 0.25 * W`` or
            ``abs(dy) >= 0.25 * H``. This filters individual matches; it is
            not a separate translation limit on the final transform.

        roi_min_frac:
            ORB/SIFT region-of-interest (ROI) lower bound on both axes,
            expressed as a fraction of the downscaled image dimensions.
            Defaults to 0.02. Together with ``roi_max_frac``, selects the
            central rectangle in which keypoints may be detected.

        roi_max_frac:
            ORB/SIFT ROI upper bound on both axes. Defaults to 0.98, leaving
            approximately a 2% border on each side with the default lower
            bound. Supplied detection boxes are also excluded from this ROI.

        fast_threshold:
            ORB-only FAST keypoint detector threshold. Defaults to 20.
            Higher values are more selective and generally yield fewer
            keypoints; lower values allow weaker responses.

        sift_n_octave_layers:
            SIFT-only number of layers in each scale-space octave.
            Defaults to 3. More layers increase scale sampling and work.

        sift_contrast_threshold:
            SIFT-only threshold for rejecting low-contrast keypoints.
            Defaults to 0.02. Lower values generally retain more keypoints.

        sift_edge_threshold:
            SIFT-only threshold for rejecting edge-like keypoints.
            Defaults to 20. Lower values reject more edge-like responses;
            higher values are more permissive.

        sof_max_corners:
            Sparse optical-flow ``maxCorners`` argument for
            ``cv2.goodFeaturesToTrack``. Defaults to 1000. Limits how many
            corners are detected; more corners increase tracking work.

        sof_quality_level:
            Sparse optical-flow ``qualityLevel`` argument. Defaults to
            0.01. A corner must meet this fraction of the best corner's
            quality; higher values keep only stronger corners.

        sof_min_distance:
            Sparse optical-flow ``minDistance`` argument. Defaults to 1.
            Minimum Euclidean distance between detected corners, in
            downscaled-image pixels.

        sof_block_size:
            Sparse optical-flow ``blockSize`` argument. Defaults to 3.
            Neighborhood size used to compute corner quality.

        sof_use_harris:
            Sparse optical-flow ``useHarrisDetector`` argument. Defaults to
            False, selecting the Shi-Tomasi corner measure. True selects
            the Harris corner measure instead.

        sof_k:
            Sparse optical-flow Harris detector sensitivity, passed as
            ``k``. Defaults to 0.04. Used only when ``sof_use_harris`` is True.

        ecc_number_of_iterations:
            Maximum ECC optimization iterations. Defaults to 50, using
            Trackers' reduced iteration budget. Optimization may stop
            earlier when the convergence tolerance is reached.

        ecc_termination_eps:
            ECC convergence tolerance for the change in correlation.
            Defaults to 1e-4. A smaller tolerance may require more iterations.

        ecc_gaussian_filter_size:
            Internal Gaussian filter size passed to ``cv2.findTransformECC``.
            Defaults to 1. Independently of this setting, the ECC path applies
            a fixed 3x3 Gaussian blur before resizing when ``downscale > 1``.
    """

    method: CMCMethod = "orb"
    downscale: int = 4

    # ORB/SIFT candidate-match requirement, before statistical filtering
    minimum_matches: int = 12

    # Transform acceptance limits shared by all four estimation methods
    minimum_scale: float = 0.3
    maximum_scale: float = 3.0

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

    # ECC parameters (Trackers' reduced iteration budget and convergence tolerance)
    ecc_number_of_iterations: int = 50
    ecc_termination_eps: float = 1e-4
    ecc_gaussian_filter_size: int = 1


class ConditionalCMC:
    """Conditional camera motion compensation (CMC) for predicted track states.

    Every supplied BGR uint8 image advances estimation. No future images, final
    sequence length, frame interval or cached motion extrapolation are used.
    ``estimate`` returns a previous-to-current 2x3 float32 affine matrix, or
    identity when motion cannot be estimated or fails the configured scale and
    finiteness checks. Feature-poor frames advance the reference; OpenCV errors
    clear it so the next image initializes a fresh reference. Original image
    resolution changes also start a fresh reference. Unexpected errors propagate.

    ``apply_batch`` validates the matrix again, then computes candidate means
    and covariances before modifying any track. For XCYCWH states, it retains
    the research linear transformation of centre, size and both velocity pairs.
    A candidate with non-finite values or non-positive dimensions is rejected
    for that track, retaining its uncorrected prediction and covariance. This
    prevents invalid updates but deliberately retains the research size
    approximation when dimensions remain positive. XYXY and XCYCSR use
    Trackers' existing representation-specific transformations.

    Mask images and identities are never transformed. Call ``reset`` between
    sequences. ``frames_failed`` counts OpenCV failures, rejected affine
    estimates and insufficient sparse-flow correspondences since reset.

    Typical usage within a tracker's prediction and association loop::

        cmc = ConditionalCMC(ConditionalCMCConfig(method="orb", downscale=4))

        for frame_bgr, detections in video:
            for track in tracker.tracks:
                track.predict()
            affine = cmc.estimate(frame_bgr, detections.xyxy)
            cmc.apply_batch(affine, tracker.tracks)
            # Associate detections with the corrected predictions, then update
            # matched tracks and manage track creation/removal in the tracker.

        cmc.reset()  # Start the next independent sequence with no image history.

    Call ``apply_batch`` on the same instance so the transform is checked
    against that instance's configured scale limits. Estimation and state
    correction are separate operations; this class does not run the tracker
    loop, perform association or draw annotations itself.

    Args:
        cfg: Configuration; defaults to ORB, downscale 4 and research guards.
    """

    def __init__(self, cfg: ConditionalCMCConfig | None = None) -> None:
        """Initialize conditional camera motion compensation (CMC).

        Args:
            cfg: Optional configuration. If None, defaults are used.

        Notes:
            - Detector/extractor/matcher are only created if method is "orb" or "sift".
            - feature_params are only created if method is "sparseOptFlow".
            - ECC optimization settings are created for "ecc".
        """
        self.cfg = cfg or ConditionalCMCConfig()
        if (
            isinstance(self.cfg.minimum_matches, bool)
            or not isinstance(self.cfg.minimum_matches, int)
            or self.cfg.minimum_matches < 5
        ):
            raise ValueError("minimum_matches must be an integer of at least five.")
        if not (
            np.isfinite(self.cfg.minimum_scale)
            and np.isfinite(self.cfg.maximum_scale)
            and 0 < self.cfg.minimum_scale <= 1 <= self.cfg.maximum_scale
        ):
            raise ValueError("Scale bounds must be finite and satisfy 0 < minimum_scale <= 1 <= maximum_scale.")
        self.downscale = max(1, int(self.cfg.downscale))

        # Feature detector, descriptor extractor and matcher (ORB/SIFT only)
        self.detector: Any | None = None
        self.extractor: Any | None = None
        self.matcher: Any | None = None
        if self.cfg.method == "orb":
            self.detector = cv2.FastFeatureDetector_create(self.cfg.fast_threshold)  # type: ignore[attr-defined]
            self.extractor = cv2.ORB_create()  # type: ignore[attr-defined]
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif self.cfg.method == "sift":
            self.detector = cv2.SIFT_create(  # type: ignore[attr-defined]
                nOctaveLayers=self.cfg.sift_n_octave_layers,
                contrastThreshold=self.cfg.sift_contrast_threshold,
                edgeThreshold=int(self.cfg.sift_edge_threshold),
            )
            self.extractor = cv2.SIFT_create(  # type: ignore[attr-defined]
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
        """Clear sequence references and failure counts; retain settings and detectors."""
        self.frames_failed = 0
        self._frame_shape: tuple[int, int] | None = None
        self._clear_reference()

    def _clear_reference(self) -> None:
        """Discard correspondence history without resetting diagnostics or settings."""
        self._initialized = False
        self._prev_kps = None
        self._prev_desc: np.ndarray | None = None
        self._prev_frame_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None

    def _is_valid_affine(self, affine_mtx: np.ndarray | None) -> bool:
        """Check shape, finite entries and the research column-norm scale bounds."""
        if affine_mtx is None or affine_mtx.shape != (2, 3) or not np.isfinite(affine_mtx).all():
            return False
        with np.errstate(over="ignore", invalid="ignore"):
            scales = np.linalg.norm(affine_mtx[:, :2], axis=0)
        return bool(np.all(scales >= self.cfg.minimum_scale) and np.all(scales <= self.cfg.maximum_scale))

    def estimate(self, frame_bgr: np.ndarray | None, dets_xyxy: np.ndarray | None = None) -> np.ndarray:
        """Estimate a validated previous-to-current image transform.

        Args:
            frame_bgr: Current BGR uint8 image of shape (H, W, 3). ``None``
                returns identity without advancing the reference.
            dets_xyxy: Optional finite (N, 4) detection boxes in original image
                coordinates. ORB/SIFT exclude their clipped regions from feature
                extraction; optical flow and ECC do not use them.

        Returns:
            A finite 2x3 float32 affine matrix. Initialization, tiny images,
            resolution changes, unavailable features, rejected transforms and
            expected OpenCV failures return identity. A supplied real image
            advances or clears the reference even when compensation is skipped.

        Raises:
            ValueError: If the image or detection-array contract is invalid.
            Exception: Unexpected implementation errors propagate to the caller.
        """
        identity = np.eye(2, 3, dtype=np.float32)
        if frame_bgr is None:
            return identity
        if (
            frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.dtype != np.uint8
            or min(frame_bgr.shape[:2]) < 1
        ):
            raise ValueError("Conditional CMC requires a non-empty BGR uint8 image of shape (H, W, 3).")
        if dets_xyxy is not None and (
            dets_xyxy.ndim != 2 or dets_xyxy.shape[1] != 4 or not np.isfinite(dets_xyxy).all()
        ):
            raise ValueError("CMC detections must be a finite (N, 4) array.")
        shape = frame_bgr.shape[:2]
        if self._frame_shape is not None and self._frame_shape != shape:
            self._clear_reference()
        self._frame_shape = shape
        if min(shape) // self.downscale < 2:
            self._clear_reference()
            return identity
        try:
            if self.cfg.method in ("orb", "sift"):
                affine = self._estimate_feature_affine(frame_bgr, dets_xyxy)
            elif self.cfg.method == "sparseOptFlow":
                affine = self._estimate_sparse_optflow(frame_bgr)
            else:
                affine = self._estimate_ecc(frame_bgr)
        except cv2.error as error:
            # A failed matcher/alignment must not leave a multi-frame reference
            # that would later be treated as a single-update camera correction.
            self.frames_failed += 1
            self._clear_reference()
            logger.warning("Conditional CMC: OpenCV estimation failed (%s), using identity", error)
            return identity
        with np.errstate(over="ignore", invalid="ignore"):
            affine = affine.astype(np.float32, copy=False) if affine is not None else None
        if affine is None or not self._is_valid_affine(affine):
            self.frames_failed += 1
            logger.debug("Conditional CMC: rejected affine estimate, using identity")
            return identity
        return affine

    def _estimate_feature_affine(self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray | None = None) -> np.ndarray:
        """Feature affine estimation.

        ORB-based or SIFT-based (different initializations of self.detector, self.extractor and self.matcher for
        ORB and SIFT)

        Steps:
            1) Convert to grayscale (+ optional downscale).
            2) Create ROI mask and optionally mask out detections (background emphasis).
            3) Detect FAST keypoints and compute ORB descriptors, or use SIFT
               for both keypoint detection and descriptor computation.
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
        img_h, img_w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.downscale > 1:
            new_w = max(1, img_w // self.downscale)
            new_h = max(1, img_h // self.downscale)
            gray = cv2.resize(gray, (new_w, new_h))
        H, W = gray.shape[:2]

        # Build mask: central ROI + remove detections (background features)
        mask = np.zeros_like(gray, dtype=np.uint8)
        y0 = int(self.cfg.roi_min_frac * H)
        y1 = int(self.cfg.roi_max_frac * H)
        x0 = int(self.cfg.roi_min_frac * W)
        x1 = int(self.cfg.roi_max_frac * W)
        mask[y0:y1, x0:x1] = 255

        if dets_xyxy is not None and len(dets_xyxy) > 0:
            dets = np.asarray(dets_xyxy) / float(self.downscale)

            # Clip before casting to avoid out-of-range int32 conversions.
            # W/H are exclusive slice ends, allowing exclusion through the last pixel.
            dets[:, 0] = np.clip(dets[:, 0], 0, W)
            dets[:, 2] = np.clip(dets[:, 2], 0, W)
            dets[:, 1] = np.clip(dets[:, 1], 0, H)
            dets[:, 3] = np.clip(dets[:, 3], 0, H)
            dets = dets.astype(np.int32)

            for x1b, y1b, x2b, y2b in dets:
                if x2b > x1b and y2b > y1b:
                    mask[y1b:y2b, x1b:x2b] = 0

        affine_mtx = np.eye(2, 3, dtype=np.float32)

        # Let cv2.error from detection, descriptors, matching or affine fitting
        # reach estimate(); it clears the previous-frame reference and returns identity.
        kps = self.detector.detect(gray, mask)  # type: ignore[union-attr]
        kps, desc = self.extractor.compute(gray, kps)  # type: ignore[union-attr]

        # First frame init
        if not self._initialized:
            self._prev_kps = copy.copy(kps)
            self._prev_desc = None if desc is None else copy.copy(desc)
            self._initialized = True
            return affine_mtx

        if self._prev_desc is None or desc is None or len(desc) == 0 or self._prev_kps is None:
            self._prev_kps = copy.copy(kps)
            self._prev_desc = None if desc is None else copy.copy(desc)
            return affine_mtx

        knn = self.matcher.knnMatch(self._prev_desc, desc, k=2)  # type: ignore[union-attr]
        if len(knn) == 0:
            self._prev_kps = copy.copy(kps)
            self._prev_desc = copy.copy(desc)
            return affine_mtx

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

        if len(prev_pts) >= self.cfg.minimum_matches:
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
                affine_est, _ = cv2.estimateAffinePartial2D(
                    prev_pts_np,
                    curr_pts_np,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.cfg.ransac_reproj_threshold,
                )
                if affine_est is not None:
                    affine_mtx = affine_est.astype(np.float32)
                    if self.downscale > 1:
                        affine_mtx[0, 2] *= self.downscale
                        affine_mtx[1, 2] *= self.downscale

        self._prev_kps = copy.copy(kps)
        self._prev_desc = copy.copy(desc)
        return affine_mtx

    def _estimate_sparse_optflow(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Sparse optical-flow-based affine estimation.

        Steps:
            1) grayscale (+ optional downscale)
            2) detect corners using goodFeaturesToTrack
            3) compute correspondences via calcOpticalFlowPyrLK(prev, curr, prev_points)
            4) keep only points with a nonzero status
            5) estimate affine transform with RANSAC
            6) scale translation back up if downscaled

        Args:
            frame_bgr: Current BGR frame.

        Returns:
            Affine transform matrix (2, 3) mapping previous frame to current, float32.
        """
        img_h, img_w = frame_bgr.shape[:2]
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        affine_mtx = np.eye(2, 3, dtype=np.float32)

        # Downscale
        if self.downscale > 1:
            new_w = max(1, img_w // self.downscale)
            new_h = max(1, img_h // self.downscale)
            frame = cv2.resize(frame, (new_w, new_h))

        # Find keypoints in current frame
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)  # type: ignore[call-overload]

        # First frame: init and return identity
        if not self._initialized:
            self._prev_frame_gray = frame.copy()
            self._prev_points = copy.copy(keypoints)
            self._initialized = True
            return affine_mtx

        # No usable prior state (no previous frame / points) or no keypoints in
        # the current frame: re-init and return identity.
        if self._prev_frame_gray is None or self._prev_points is None or keypoints is None:
            self._prev_frame_gray = frame.copy()
            self._prev_points = None if keypoints is None else keypoints.copy()
            return affine_mtx

        # Optical flow correspondences
        # calcOpticalFlowPyrLK will throw or return nonsense if we give it None
        matched, status, _err = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
            self._prev_frame_gray, frame, self._prev_points, None
        )

        if status is None or matched is None:
            self._prev_frame_gray = frame.copy()
            self._prev_points = copy.copy(keypoints)
            return affine_mtx

        # Keep only good correspondences
        # status is (N,1) or (N,)
        good = status.reshape(-1) != 0
        prev_pts_np = self._prev_points[good]
        curr_pts_np = matched[good]

        # Find rigid matrix
        if np.size(prev_pts_np, 0) > 4:
            affine_est, _ = cv2.estimateAffinePartial2D(
                prev_pts_np,
                curr_pts_np,
                method=cv2.RANSAC,
            )
            if affine_est is not None:
                affine_mtx = affine_est.astype(np.float32)

                # Handle downscale translation back to original image coords
                if self.downscale > 1:
                    affine_mtx[0, 2] *= self.downscale
                    affine_mtx[1, 2] *= self.downscale
        else:
            logger.debug("Conditional CMC: not enough matching points for motion estimation")
            self.frames_failed += 1

        # Store to next iteration
        self._prev_frame_gray = frame.copy()
        self._prev_points = None if keypoints is None else keypoints.copy()

        return affine_mtx

    def _estimate_ecc(self, frame_bgr: np.ndarray) -> np.ndarray:
        """ECC-based affine motion estimation.

        This method estimates a global 2D Euclidean transform between the previous
        frame and the current frame using OpenCV's Enhanced Correlation Coefficient
        (ECC) image alignment algorithm.

        Steps:
            1) Convert the current frame to grayscale.
            2) Optionally smooth and downscale the frame.
            3) If this is the first frame, store it and return identity.
            4) Optimize a 2x3 warp matrix aligning the previous frame to the current
               frame.
            5) Restore the translation to original-image pixels.
            6) Store the current frame for the next call. OpenCV failures
               propagate to estimate(), which clears the reference and returns
               identity.

        Args:
            frame_bgr: Current frame in BGR format.

        Returns:
            Affine transform matrix of shape (2, 3), dtype float32, mapping
            previous-frame coordinates to current-frame coordinates. Returns
            identity during initialization. OpenCV failures are handled by
            estimate(), which also clears the reference.
        """
        img_h, img_w = frame_bgr.shape[:2]
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        affine_mtx = np.eye(2, 3, dtype=np.float32)

        if self.downscale > 1:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            new_w = max(1, img_w // self.downscale)
            new_h = max(1, img_h // self.downscale)
            frame = cv2.resize(frame, (new_w, new_h))

        if not self._initialized:
            self._prev_frame_gray = frame.copy()
            self._initialized = True
            return affine_mtx

        if self._prev_frame_gray is None:
            self._prev_frame_gray = frame.copy()
            return affine_mtx

        _cc, affine_est = cv2.findTransformECC(  # type: ignore[call-overload]
            self._prev_frame_gray,
            frame,
            affine_mtx,
            self.warp_mode,
            self.criteria,
            None,
            self.cfg.ecc_gaussian_filter_size,
        )
        if affine_est is not None:
            affine_mtx = affine_est.astype(np.float32)
            if self.downscale > 1:
                affine_mtx[0, 2] *= self.downscale
                affine_mtx[1, 2] *= self.downscale

        # The next estimate must use this image, not the initial sequence frame.
        self._prev_frame_gray = frame.copy()
        return affine_mtx

    @staticmethod
    def warp_xyxy_corners(
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
        rot_mtx: np.ndarray,
        translation: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform four box corners via rot_mtx/translation and return the enclosing axis-aligned box.

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
            rot_mtx: 2x2 rotation/shear sub-matrix of the affine transform.
            translation: Optional 2-element translation vector.

        Returns:
            Tuple ``(new_x1, new_y1, new_x2, new_y2)`` -- per-axis min and max of
            the four transformed corners, same shape as inputs.

        Examples:
            >>> import numpy as np
            >>> rot_mtx = np.eye(2, dtype=np.float64)
            >>> translation = np.array([5.0, -3.0])
            >>> x1 = np.array([10.0, 20.0])
            >>> y1 = np.array([20.0, 30.0])
            >>> x2 = np.array([50.0, 60.0])
            >>> y2 = np.array([80.0, 90.0])
            >>> nx1, ny1, nx2, ny2 = ConditionalCMC.warp_xyxy_corners(x1, y1, x2, y2, rot_mtx, translation)
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
        out = corners @ rot_mtx.T
        if translation is not None:
            out = out + translation
        lo = out.min(axis=-2)
        hi = out.max(axis=-2)
        return lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1]

    def apply_batch(self, affine_mtx: np.ndarray | None, tracklets: Sequence[BaseTracklet]) -> None:
        """Conditionally apply a camera-motion transform to tracklets' Kalman states.

        Call after predicting track states and before detection-to-track
        association. The affine matrix is validated against this instance's
        scale limits, even if it was already validated by ``estimate``.
        ``None``, an empty tracklet sequence, or a matrix with invalid shape,
        non-finite entries or out-of-range column norms leaves all tracks
        unchanged.

        Each tracklet must expose ``state_estimator.kf.state``, a state-vector
        column with shape (dim, 1), and ``state_estimator.kf.state_covariance``
        with shape (dim, dim). All tracklets must share the same concrete
        estimator type, based on ``XCYCWHStateEstimator`` (8 dimensions),
        ``XYXYStateEstimator`` (8) or ``XCYCSRStateEstimator`` (7).

        Candidate transformations depend on the state representation:

        - XCYCWH: Apply the affine matrix's 2x2 linear part to each of the
          pairs ``(xc, yc)``, ``(w, h)``, ``(vx, vy)`` and ``(vw, vh)``.
          Add translation only to the centre. With the four copies of that
          linear part forming block-diagonal matrix ``M``, transform the
          covariance as ``M @ P @ M.T``. This preserves the research McByte++
          size approximation, including its limitations under rotation.
        - XYXY: Transform all four box corners and take their axis-aligned
          enclosure. Use the same enclosure operation for the corner
          velocities, without translation. Transform covariance with the
          block-diagonal linear map only when both off-diagonal entries of
          the 2x2 linear part are within 1e-6 of zero; otherwise retain it.
        - XCYCSR: Transform the centre and its velocity, adding translation
          only to the centre. Keep area, aspect ratio and area velocity
          unchanged. Apply the corresponding block-diagonal covariance
          transformation. This and the XYXY mapping follow the existing CMC.

        Compute candidates in temporary batch arrays before changing any
        live filter. Reject a track's candidate if its state or covariance
        contains non-finite values, or if its box geometry is invalid:
        non-positive width/height for XCYCWH, non-positive area/aspect ratio
        for XCYCSR, or unordered corner coordinates for XYXY. Accept or
        reject each track's state and covariance together; rejecting one
        track does not prevent valid candidates for other tracks from being
        applied. Rejected tracks retain their original predictions.

        Only live Kalman states and covariances are changed. This method
        does not advance the image reference, update ``frames_failed``,
        transform masks or alter track identities and observation histories.

        Args:
            affine_mtx: Previous-to-current affine matrix with shape (2, 3),
                expressed in original-image coordinates, typically returned
                by ``estimate``. ``None`` skips compensation.
            tracklets: Sequence of tracks sharing one supported estimator
                type and exposing the Kalman state/covariance layout above.

        Returns:
            None. Accepted state and covariance corrections are written
            directly to the tracklets' Kalman filters.

        Raises:
            TypeError: If estimator types are mixed or unsupported, when a
                valid matrix and a non-empty tracklet sequence are supplied.
            AttributeError: If a tracklet lacks the required estimator or
                Kalman-filter attributes.

        Examples:
            >>> from trackers.core.mcbyteplusplus.tracklet import McBytePlusPlusTracklet
            >>> track = McBytePlusPlusTracklet(np.array([10.0, 20.0, 50.0, 80.0]))
            >>> cmc = ConditionalCMC()
            >>> affine = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
            >>> cmc.apply_batch(affine, [track])
            >>> track.get_state_bbox().tolist()
            [15.0, 17.0, 55.0, 77.0]
            >>> cmc.apply_batch(None, [track])  # No-op.
        """
        from trackers.utils.state_representations import XCYCSRStateEstimator, XCYCWHStateEstimator, XYXYStateEstimator

        if affine_mtx is None or not tracklets or not self._is_valid_affine(affine_mtx):
            return
        estimator = tracklets[0].state_estimator
        if not all(type(track.state_estimator) is type(estimator) for track in tracklets):
            raise TypeError("ConditionalCMC.apply_batch requires homogeneous state types.")
        if not isinstance(estimator, (XCYCWHStateEstimator, XCYCSRStateEstimator, XYXYStateEstimator)):
            raise TypeError("Conditional CMC supports XCYCWH, XCYCSR and XYXY state estimators.")
        states = np.array([track.state_estimator.kf.state.reshape(-1) for track in tracklets])
        covariances = np.array([track.state_estimator.kf.state_covariance for track in tracklets])
        is_xyxy = isinstance(estimator, XYXYStateEstimator)
        # Work on the existing batch buffers. No track is changed until all
        # candidate calculations and validity checks have completed.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            states, covariances = _warp_kalman_states(
                states,
                covariances,
                affine_mtx,
                is_xyxy=is_xyxy,
                is_xcycwh=isinstance(estimator, XCYCWHStateEstimator),
            )
        valid = np.isfinite(states).all(axis=1) & np.isfinite(covariances).all(axis=(1, 2))
        if is_xyxy:
            valid &= (states[:, 2] > states[:, 0]) & (states[:, 3] > states[:, 1])
        else:
            valid &= (states[:, 2] > 0) & (states[:, 3] > 0)
        for track, state, covariance, accepted in zip(tracklets, states, covariances, valid):
            if accepted:
                track.state_estimator.kf.state = state.reshape(-1, 1)
                track.state_estimator.kf.state_covariance = covariance


def _warp_kalman_states(
    states: np.ndarray,
    covariances: np.ndarray,
    affine_mtx: np.ndarray,
    *,
    is_xyxy: bool,
    is_xcycwh: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform owned batch buffers before per-track acceptance.

    XCYCWH retains the research size/velocity approximation. Other layouts use the same mappings as the existing CMC
    utility. Inputs are temporary arrays, so invalid results can be discarded without rolling back live filters.
    """
    rot_mtx = affine_mtx[:2, :2].astype(np.float64)
    translation = affine_mtx[:2, 2].astype(np.float64)
    dim = states.shape[1]

    if is_xcycwh:
        research_transform = np.kron(np.eye(4), rot_mtx)
        states = states @ research_transform.T
        states[:, :2] += translation
        covariances = research_transform @ covariances @ research_transform.T
        return states, covariances

    if is_xyxy:
        # XYXY boxes must remain axis-aligned after CMC. For transforms with
        # rotation/reflection/shear, applying the affine matrix only to the
        # top-left and bottom-right corners can invert the box or produce
        # invalid geometry. Transform all four corners, then rebuild the
        # enclosing axis-aligned box with per-axis min/max.
        states[:, 0], states[:, 1], states[:, 2], states[:, 3] = ConditionalCMC.warp_xyxy_corners(
            states[:, 0], states[:, 1], states[:, 2], states[:, 3], rot_mtx, translation
        )
        # Keep XYXY velocity ordering valid under mixed-axis transforms by
        # applying the same corner-wise normalization to the paired velocity
        # components.
        states[:, 4], states[:, 5], states[:, 6], states[:, 7] = ConditionalCMC.warp_xyxy_corners(
            states[:, 4], states[:, 5], states[:, 6], states[:, 7], rot_mtx
        )
    else:
        # Batch-transform centre positions: pos' = pos @ rot_mtx.T + translation
        states[:, 0:2] = states[:, 0:2] @ rot_mtx.T + translation
        # Batch-transform centre velocities: vel' = vel @ rot_mtx.T
        states[:, 4:6] = states[:, 4:6] @ rot_mtx.T

    block_rot = None
    if is_xyxy:
        # atol=1e-6: float32 CMC (sparseOptFlow/ORB/SIFT/ECC) carries ~1e-7
        # to 1e-6 residuals on off-diagonals even for pure-translation affine_mtx;
        # default atol=1e-8 misclassifies those as cross-axis transforms.
        if np.isclose(rot_mtx[0, 1], 0.0, atol=1e-6) and np.isclose(rot_mtx[1, 0], 0.0, atol=1e-6):
            block_rot = np.eye(dim, dtype=np.float64)
            block_rot[0:2, 0:2] = rot_mtx
            block_rot[2:4, 2:4] = rot_mtx
            block_rot[4:6, 4:6] = rot_mtx
            block_rot[6:8, 6:8] = rot_mtx
    else:
        block_rot = np.eye(dim, dtype=np.float64)
        block_rot[0:2, 0:2] = rot_mtx
        block_rot[4:6, 4:6] = rot_mtx

    if block_rot is not None:
        covariances = block_rot @ covariances @ block_rot.T

    return states, covariances
