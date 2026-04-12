# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.calibration.projection import (
    apply_homography,
    bottom_center_from_xywh,
)


def test_apply_homography_with_identity_matrix() -> None:
    points = np.array([[10.0, 20.0], [30.0, 40.0]])
    homography = np.eye(3)

    projected = apply_homography(points, homography)

    np.testing.assert_allclose(projected, points)


def test_bottom_center_from_xywh() -> None:
    boxes = np.array([[10.0, 20.0, 30.0, 40.0]])

    anchors = bottom_center_from_xywh(boxes)

    np.testing.assert_allclose(anchors, np.array([[25.0, 60.0]]))
