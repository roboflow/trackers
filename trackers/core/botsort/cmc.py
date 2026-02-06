# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import copy
import numpy as np
import cv2


@dataclass
class CMCConfig:
    downscale: int = 2
    fast_threshold: int = 20

    # Affine estimation
    ransac_reproj_threshold: float = 3.0

    # Filtering matches by spatial displacement (fraction of image size)
    max_spatial_distance_frac: float = 0.25

    # Keep features from central ROI (avoid borders)
    roi_min_frac: float = 0.02
    roi_max_frac: float = 0.98


class CMC:
    def __init__(self, cfg: Optional[CMCConfig] = None) -> None:
        self.cfg = cfg or CMCConfig()
        self.downscale = max(1, int(self.cfg.downscale))

        self.detector = cv2.FastFeatureDetector_create(self.cfg.fast_threshold)
        self.extractor = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self._initialized = False
        self._prev_kps = None
        self._prev_desc: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._initialized = False
        self._prev_kps = None
        self._prev_desc = None

    def estimate(self, frame_bgr: np.ndarray, dets_xyxy: Optional[np.ndarray] = None) -> np.ndarray:
        if frame_bgr is None:
            return np.eye(2, 3, dtype=np.float32)

        H_img, W_img = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Downscale for speed / robustness
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
            dets[:, 0] = np.clip(dets[:, 0], 0, W - 1)
            dets[:, 2] = np.clip(dets[:, 2], 0, W - 1)
            dets[:, 1] = np.clip(dets[:, 1], 0, H - 1)
            dets[:, 3] = np.clip(dets[:, 3], 0, H - 1)
            for x1b, y1b, x2b, y2b in dets:
                if x2b > x1b and y2b > y1b:
                    mask[y1b:y2b, x1b:x2b] = 0

        # Detect + describe
        kps = self.detector.detect(gray, mask)
        kps, desc = self.extractor.compute(gray, kps)

        H_aff = np.eye(2, 3, dtype=np.float32)

        # First frame: only initialize
        if not self._initialized:
            self._prev_kps = copy.copy(kps)
            self._prev_desc = None if desc is None else copy.copy(desc)
            self._initialized = True
            return H_aff

        # If missing descriptors
        if self._prev_desc is None or desc is None or len(desc) == 0:
            self._prev_kps = copy.copy(kps)
            self._prev_desc = None if desc is None else copy.copy(desc)
            return H_aff

        # KNN match (k=2) + ratio test
        knn = self.matcher.knnMatch(self._prev_desc, desc, k=2)
        if len(knn) == 0:
            self._prev_kps = copy.copy(kps)
            self._prev_desc = copy.copy(desc)
            return H_aff

        max_spatial = self.cfg.max_spatial_distance_frac * np.array([W, H], dtype=np.float32)

        prev_pts = []
        curr_pts = []
        spatial = []

        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.9 * n.distance:
                p_prev = np.array(self._prev_kps[m.queryIdx].pt, dtype=np.float32)
                p_curr = np.array(kps[m.trainIdx].pt, dtype=np.float32)
                d = p_prev - p_curr
                if (abs(d[0]) < max_spatial[0]) and (abs(d[1]) < max_spatial[1]):
                    spatial.append(d)
                    prev_pts.append(p_prev)
                    curr_pts.append(p_curr)

        if len(prev_pts) >= 5:
            spatial = np.asarray(spatial, dtype=np.float32)
            mean = spatial.mean(axis=0)
            std = spatial.std(axis=0) + 1e-6
            inl = np.logical_and(
                np.abs(spatial[:, 0] - mean[0]) < 2.5 * std[0],
                np.abs(spatial[:, 1] - mean[1]) < 2.5 * std[1],
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

        # Update prev
        self._prev_kps = copy.copy(kps)
        self._prev_desc = copy.copy(desc)

        return H_aff

    @staticmethod
    def apply_to_tracks(tracks: list, H: np.ndarray) -> None:
        if H is None or len(tracks) == 0:
            return

        H = H.astype(np.float32)
        R = H[:2, :2]
        t = H[:2, 2:3]  # (2,1)

        # A4 maps [x1,y1,x2,y2]
        A4 = np.zeros((4, 4), dtype=np.float32)
        A4[0:2, 0:2] = R
        A4[2:4, 2:4] = R

        # A8 maps state (pos and vel blocks)
        A8 = np.zeros((8, 8), dtype=np.float32)
        A8[0:4, 0:4] = A4
        A8[4:8, 4:8] = A4

        trans4 = np.array([t[0, 0], t[1, 0], t[0, 0], t[1, 0]], dtype=np.float32).reshape(4, 1)

        for trk in tracks:
            trk.state = (A8 @ trk.state).astype(np.float32)
            trk.state[0:4] += trans4
            trk.P = (A8 @ trk.P @ A8.T).astype(np.float32)
