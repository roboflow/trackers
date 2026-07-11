# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np


def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from corner to center-width-height format.

    Args:
        xyxy: Bounding boxes `[x_min, y_min, x_max, y_max]` with shape `(4,)`
            for a single box or `(N, 4)` for multiple boxes.

    Returns:
        Bounding boxes `[x_center, y_center, width, height]` with same shape
            as input.
    """
    if xyxy.ndim == 1:
        x1, y1, x2, y2 = xyxy.astype(np.float64)
        w = x2 - x1
        h = y2 - y1
        return np.array([x1 + w * 0.5, y1 + h * 0.5, w, h], dtype=np.float64)

    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    result = np.empty((xyxy.shape[0], 4), dtype=np.float64)
    result[:, 0] = xyxy[:, 0] + w * 0.5
    result[:, 1] = xyxy[:, 1] + h * 0.5
    result[:, 2] = w
    result[:, 3] = h
    return result


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from center-width-height to corner format.

    Args:
        xywh: Bounding boxes `[x_center, y_center, width, height]` with shape
            `(4,)` for a single box or `(N, 4)` for multiple boxes.

    Returns:
        Bounding boxes `[x_min, y_min, x_max, y_max]` with same shape as input.
    """
    if xywh.ndim == 1:
        single_xc, single_yc, single_w, single_h = xywh.astype(np.float64)
        single_hw, single_hh = single_w * 0.5, single_h * 0.5
        return np.array(
            [
                single_xc - single_hw,
                single_yc - single_hh,
                single_xc + single_hw,
                single_yc + single_hh,
            ],
            dtype=np.float64,
        )

    xc = xywh[:, 0]
    yc = xywh[:, 1]
    w = xywh[:, 2]
    h = xywh[:, 3]
    result = np.empty((xywh.shape[0], 4), dtype=np.float64)
    hw = w * 0.5
    hh = h * 0.5
    result[:, 0] = xc - hw
    result[:, 1] = yc - hh
    result[:, 2] = xc + hw
    result[:, 3] = yc + hh
    return result


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
    if xyxy.ndim == 1:
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        return np.array(
            [
                xyxy[0] + w * 0.5,
                xyxy[1] + h * 0.5,
                w * h,
                w / (h + 1e-6),
            ]
        )

    # Batch path — pre-allocated array avoids np.stack overhead
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    result = np.empty((xyxy.shape[0], 4), dtype=np.float64)
    result[:, 0] = xyxy[:, 0] + w * 0.5
    result[:, 1] = xyxy[:, 1] + h * 0.5
    result[:, 2] = w * h
    result[:, 3] = w / (h + 1e-6)
    return result


def xcycsr_to_xyxy(xcycsr: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from center-scale-ratio to corner format.

    Args:
        xcycsr: Bounding boxes `[x_center, y_center, scale, aspect_ratio]` with
            shape `(4,)` for a single box or `(N, 4)` for multiple boxes,
            where `scale` is area and `aspect_ratio` is `width / height`.

    Returns:
        Bounding boxes `[x_min, y_min, x_max, y_max]` with same shape as input.

    Note:
        When ``scale`` or ``aspect_ratio`` is zero, ``w`` collapses to ``0.0``
        and the box degenerates to a point at ``(x_center, y_center)`` — all four
        coordinates equal the center. No ``NaN`` or ``Inf`` is produced.

        When the product ``scale * aspect_ratio`` underflows to exactly ``0.0``
        due to subnormal float arithmetic (e.g. both values near ``1e-200``), the
        zero-guard fires and returns ``h=0.0`` even though the mathematically
        correct value may be non-zero.

        Negative ``scale`` yields ``NaN`` entries because ``sqrt`` of a negative
        value is undefined.

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
        >>>
        >>> # degenerate: zero scale collapses to a point box
        >>> xcycsr_to_xyxy(np.array([10., 20., 0., 1.]))
        array([10., 20., 10., 20.])
    """
    if xcycsr.ndim == 1:
        w = np.sqrt(xcycsr[2] * xcycsr[3])
        h = xcycsr[2] / w if w != 0 else np.float64(0.0)
        hw, hh = w * 0.5, h * 0.5
        return np.array(
            [
                xcycsr[0] - hw,
                xcycsr[1] - hh,
                xcycsr[0] + hw,
                xcycsr[1] + hh,
            ]
        )

    # Batch path — pre-allocated array avoids np.stack overhead
    w = np.sqrt(xcycsr[:, 2] * xcycsr[:, 3])
    # Inner np.where substitutes 1.0 for zero denominators to suppress the
    # eager-evaluation divide-by-zero warning; outer np.where replaces those results with 0.0.
    h = np.where(w != 0, xcycsr[:, 2] / np.where(w != 0, w, 1.0), 0.0)
    result = np.empty((xcycsr.shape[0], 4), dtype=xcycsr.dtype)
    result[:, 0] = xcycsr[:, 0] - w * 0.5
    result[:, 1] = xcycsr[:, 1] - h * 0.5
    result[:, 2] = xcycsr[:, 0] + w * 0.5
    result[:, 3] = xcycsr[:, 1] + h * 0.5
    return result
