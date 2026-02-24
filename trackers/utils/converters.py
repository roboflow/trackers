# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np


def xyxy_to_xcycsr(xyxy: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from corner to center-scale-ratio format.

    Args:
        xyxy: Bounding boxes `[x_min, y_min, x_max, y_max]` with shape `(4,)`
            for a single box or `(N, 4)` for multiple boxes.

    Returns:
        Bounding boxes `[x_center, y_center, scale, aspect_ratio]` with same
            shape as input, where `scale` is area (`width * height`) and
            `aspect_ratio` is `width / height`.

    Examples:
        >>> import numpy as np
        >>> from trackers import xyxy_to_xcycsr
        >>>
        >>> boxes = np.array([
        ...     [0,   0, 10, 10],
        ...     [0,   0, 20, 10],
        ...     [0,   0, 10, 20],
        ... ])
        >>>
        >>> xyxy_to_xcycsr(boxes)
        array([[  5.        ,   5.        , 100.        ,   0.9999999 ],
               [ 10.        ,   5.        , 200.        ,   1.9999998 ],
               [  5.        ,  10.        , 200.        ,   0.49999998]])
    """
    single = xyxy.ndim == 1
    if single:
        xyxy = xyxy[np.newaxis, :]

    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    x = xyxy[:, 0] + w / 2.0
    y = xyxy[:, 1] + h / 2.0
    s = w * h
    r = w / (h + 1e-6)

    result = np.stack([x, y, s, r], axis=1)
    return result[0] if single else result


def xcycsr_to_xyxy(xcycsr: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from center-scale-ratio to corner format.

    Args:
        xcycsr: Bounding boxes `[x_center, y_center, scale, aspect_ratio]` with
            shape `(4,)` for a single box or `(N, 4)` for multiple boxes,
            where `scale` is area and `aspect_ratio` is `width / height`.

    Returns:
        Bounding boxes `[x_min, y_min, x_max, y_max]` with same shape as input.

    Examples:
        >>> import numpy as np
        >>> from trackers import xcycsr_to_xyxy
        >>>
        >>> boxes = np.array([
        ...     [  5.,   5., 100., 1.],
        ...     [ 10.,   5., 200., 2.],
        ...     [  5.,  10., 200., 0.5],
        ... ])
        >>>
        >>> xcycsr_to_xyxy(boxes)
        array([[ 0.,  0., 10., 10.],
               [ 0.,  0., 20., 10.],
               [ 0.,  0., 10., 20.]])
    """
    single = xcycsr.ndim == 1
    if single:
        xcycsr = xcycsr[np.newaxis, :]

    w = np.sqrt(xcycsr[:, 2] * xcycsr[:, 3])
    h = xcycsr[:, 2] / w
    x, y = xcycsr[:, 0], xcycsr[:, 1]

    result = np.stack([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0], axis=1)
    return result[0] if single else result
