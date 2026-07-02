# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import Literal

import numpy as np
import pytest

from trackers.utils.cmc import CMC, CMCConfig


@pytest.mark.parametrize(
    "method",
    ["sparseOptFlow", "orb", "sift", "ecc"],
)
def test_cmc_downscale_zero_dimension(method: Literal["sparseOptFlow", "orb", "sift", "ecc"]) -> None:
    """Test that cmc downscaling does not crash when passed an image smaller than the downscale factor."""
    cfg = CMCConfig(method=method, downscale=2)
    cmc = CMC(cfg)

    # 1x1 image - would result in 0x0 dimension if img_w // 2 is used
    frame = np.zeros((1, 1, 3), dtype=np.uint8)

    # First frame always returns identity (init)
    H1 = cmc.estimate(frame)
    assert H1.shape == (2, 3)
    np.testing.assert_array_equal(H1, np.eye(2, 3, dtype=np.float32))

    # Second frame triggers the actual feature matching / alignment logic
    # Should not crash with cv2.error
    H2 = cmc.estimate(frame)

    assert H2.shape == (2, 3)
    # The result should be identity since it's the exact same 1x1 black frame
    np.testing.assert_array_equal(H2, np.eye(2, 3, dtype=np.float32))
