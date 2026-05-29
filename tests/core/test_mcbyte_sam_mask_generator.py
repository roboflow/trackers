# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from trackers.core.mcbyte.masks.base import TrackletSnapshot
from trackers.core.mcbyte.masks.sam import SAMBoxMaskGenerator


def test_sam_box_mask_generator_converts_sam_masks_to_expected_shape() -> None:
    # We only need _convert_masks(). Don't call __init__(), as it loads SAM checkpoint.
    generator = object.__new__(SAMBoxMaskGenerator)

    masks = torch.zeros((2, 1, 100, 120), dtype=torch.bool)
    masks[0, 0, 10:20, 30:40] = True
    masks[1, 0, 50:70, 80:90] = True

    converted_masks = generator._convert_masks(masks)

    assert converted_masks.shape == (2, 100, 120)
    assert converted_masks.dtype == bool

    assert converted_masks[0, 10:20, 30:40].all()
    # No extra True pixels elsewhere.
    assert converted_masks[0].sum() == 10 * 10
    
    assert converted_masks[1, 50:70, 80:90].all()
    assert converted_masks[1].sum() == 20 * 10