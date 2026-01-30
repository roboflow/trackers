# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Adapted from TrackEval (https://github.com/JonathonLuiten/TrackEval)
# Copyright (c) Jonathon Luiten. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from copy import deepcopy
from typing import Literal

import numpy as np

# Epsilon for floating point comparisons - must match TrackEval exactly
EPS = np.finfo("float").eps

BoxFormat = Literal["xyxy", "xywh"]


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from xywh format to xyxy format.

    Args:
        boxes: Array of shape `(N, 4)` in xywh format `(x, y, width, height)`.

    Returns:
        Array of shape `(N, 4)` in xyxy format `(x0, y0, x1, y1)`.
    """
    boxes = deepcopy(boxes)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    return boxes


def box_iou(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    box_format: BoxFormat = "xyxy",
) -> np.ndarray:
    """Calculate the IoU (Intersection over Union) between two sets of boxes.
    Compute pairwise IoU between all boxes in boxes1 and boxes2, returning a
    matrix of shape `(N, M)` where N is the number of boxes in boxes1 and M is
    the number of boxes in boxes2.

    Args:
        boxes1: First set of boxes with shape `(N, 4)`.
        boxes2: Second set of boxes with shape `(M, 4)`.
        box_format: Format of the input boxes. Either `"xyxy"` for
            `(x0, y0, x1, y1)` or `"xywh"` for `(x, y, width, height)`.
            Defaults to `"xyxy"`.

    Returns:
        IoU matrix of shape `(N, M)` where element `[i, j]` is the IoU between
        `boxes1[i]` and `boxes2[j]`. Values are in range `[0, 1]`.

    Raises:
        ValueError: If box_format is not `"xyxy"` or `"xywh"`.

    Examples:
        ```python
        import numpy as np
        from trackers.eval import box_iou

        boxes1 = np.array([
            [0,   0, 10, 10],
            [20, 20, 30, 30],
            [5,   5, 15, 15],
        ])
        boxes2 = np.array([
            [5,   5, 15, 15],
            [0,   0, 10, 10],
        ])
        box_iou(boxes1, boxes2, box_format="xyxy")
        # array([[0.14285714, 1.        ],
        #        [0.        , 0.        ],
        #        [1.        , 0.14285714]])
        ```

        ```python
        import numpy as np
        from trackers.eval import box_iou

        boxes1 = np.array([
            [0,  0, 10, 10],
            [10, 0, 10, 10],
        ])
        boxes2 = np.array([
            [5, 0, 10, 10],
        ])
        box_iou(boxes1, boxes2, box_format="xywh")
        # array([[0.33333333],
        #        [0.33333333]])
        ```
    """
    return _calculate_box_ious(boxes1, boxes2, box_format=box_format, do_ioa=False)


def box_ioa(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    box_format: BoxFormat = "xyxy",
) -> np.ndarray:
    """Calculate the IoA (Intersection over Area) between two sets of boxes.
    IoA is calculated as intersection divided by the area of boxes1. This is
    commonly used to determine if detections fall within crowd ignore regions.

    Args:
        boxes1: First set of boxes with shape `(N, 4)`. The area of these boxes
            is used as the denominator.
        boxes2: Second set of boxes with shape `(M, 4)`.
        box_format: Format of the input boxes. Either `"xyxy"` for
            `(x0, y0, x1, y1)` or `"xywh"` for `(x, y, width, height)`.
            Defaults to `"xyxy"`.

    Returns:
        IoA matrix of shape `(N, M)` where element `[i, j]` is the IoA between
        `boxes1[i]` and `boxes2[j]`. Values are in range `[0, 1]`.

    Raises:
        ValueError: If box_format is not `"xyxy"` or `"xywh"`.

    Examples:
        ```python
        import numpy as np
        from trackers.eval import box_ioa

        boxes1 = np.array([
            [5,  5, 15, 15],
            [0,  0, 10, 10],
        ])
        boxes2 = np.array([
            [0,  0, 20, 20],
            [5,  0, 15, 10],
        ])
        box_ioa(boxes1, boxes2, box_format="xyxy")
        # array([[1. , 0.5],
        #        [1. , 0.5]])
        ```

        ```python
        import numpy as np
        from trackers.eval import box_ioa

        boxes1 = np.array([
            [0,  0, 10, 10],
            [10, 0, 10, 10],
        ])
        boxes2 = np.array([
            [5, 0, 10, 10],
        ])
        box_ioa(boxes1, boxes2, box_format="xywh")
        # array([[0.5],
        #        [0.5]])
        ```
    """
    return _calculate_box_ious(boxes1, boxes2, box_format=box_format, do_ioa=True)


def _calculate_box_ious(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    box_format: BoxFormat = "xyxy",
    do_ioa: bool = False,
) -> np.ndarray:
    """Calculate IoU or IoA between two sets of boxes.

    Args:
        boxes1: First set of boxes with shape `(N, 4)`.
        boxes2: Second set of boxes with shape `(M, 4)`.
        box_format: Format of the input boxes. Either `"xyxy"` or `"xywh"`.
        do_ioa: If `True`, calculate IoA (intersection over area of boxes1).
            If `False`, calculate IoU (intersection over union).

    Returns:
        IoU/IoA matrix of shape `(N, M)`.

    Raises:
        ValueError: If box_format is not `"xyxy"` or `"xywh"`.
    """
    # Handle empty input arrays
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)

    # Convert xywh to xyxy if needed
    if box_format == "xywh":
        boxes1 = _xywh_to_xyxy(boxes1)
        boxes2 = _xywh_to_xyxy(boxes2)
    elif box_format != "xyxy":
        raise ValueError(f"box_format must be 'xyxy' or 'xywh', got '{box_format}'")

    # Ensure float64 for numerical precision
    boxes1 = np.asarray(boxes1, dtype=np.float64)
    boxes2 = np.asarray(boxes2, dtype=np.float64)

    # Calculate intersection coordinates
    # boxes1: (N, 4), boxes2: (M, 4) -> broadcasting to (N, M, 4)
    min_ = np.minimum(boxes1[:, np.newaxis, :], boxes2[np.newaxis, :, :])
    max_ = np.maximum(boxes1[:, np.newaxis, :], boxes2[np.newaxis, :, :])

    # Intersection: max of left edges to min of right edges
    # min_[..., 2] is min of x1 values, max_[..., 0] is max of x0 values
    intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(
        min_[..., 3] - max_[..., 1], 0
    )

    # Area of boxes1
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])

    if do_ioa:
        # IoA: Intersection over Area of boxes1
        ioas = np.zeros_like(intersection)
        valid_mask = area1 > 0 + EPS
        ioas[valid_mask, :] = (
            intersection[valid_mask, :] / area1[valid_mask][:, np.newaxis]
        )
        return ioas
    else:
        # IoU: Intersection over Union
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection

        # Handle edge cases to avoid division issues
        intersection[area1 <= 0 + EPS, :] = 0
        intersection[:, area2 <= 0 + EPS] = 0
        intersection[union <= 0 + EPS] = 0
        union[union <= 0 + EPS] = 1

        ious = intersection / union
        return ious
