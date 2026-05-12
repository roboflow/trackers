# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseIoU(ABC):
    """Abstract base for IoU similarity metrics used in tracker association.

    Subclasses implement a specific Intersection over Union variant
    (e.g. standard IoU, GIoU, DIoU, CIoU, BIoU) that computes a pairwise
    similarity matrix between two sets of bounding boxes.

    The resulting matrix is used as a cost/similarity signal in the
    Hungarian algorithm during the data association step.

    Examples:
        Subclass BaseIoU to create a custom metric::

            class MyIoU(BaseIoU):
                def _compute(
                    self, boxes_1: np.ndarray, boxes_2: np.ndarray
                ) -> np.ndarray:
                    return np.ones((len(boxes_1), len(boxes_2)))

            metric = MyIoU()
            matrix = metric.compute(np.array([[0, 0, 10, 10]]), np.array([[5, 5, 15, 15]]))

    Note:
        Subclasses must override :meth:`_compute`, **not** :meth:`compute`.
        Overriding ``compute`` directly bypasses the empty-input guard which
        returns a zero matrix when either input has zero rows.
    """

    def compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity between two sets of bounding boxes.

        Handles the empty-input edge case (returns a correctly-shaped zero
        matrix) and delegates to subclass `_compute` method for the actual math.

        Args:
            boxes_1: ``(N, 4)`` array of boxes in ``[x1, y1, x2, y2]`` format.
            boxes_2: ``(M, 4)`` array of boxes in ``[x1, y1, x2, y2]`` format.

        Returns:
            ``(N, M)`` similarity matrix where entry ``(i, j)`` is the
            similarity between ``boxes_1[i]`` and ``boxes_2[j]``.

        Note:
            Input boxes are assumed well-formed (``x1 <= x2`` and
            ``y1 <= y2``). No validation is performed; malformed boxes
            produce undefined output.
        """
        if not np.isfinite(boxes_1).all():
            raise ValueError("boxes_1 contains non-finite values (NaN or inf)")
        if not np.isfinite(boxes_2).all():
            raise ValueError("boxes_2 contains non-finite values (NaN or inf)")
        if len(boxes_1) == 0 or len(boxes_2) == 0:
            return np.zeros((len(boxes_1), len(boxes_2)), dtype=np.float64)
        return self._compute(boxes_1, boxes_2)

    @abstractmethod
    def _compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        """Subclass hook — compute similarity for non-empty inputs.

        Args:
            boxes_1: ``(N, 4)`` array of boxes in ``[x1, y1, x2, y2]`` format.
                Guaranteed ``N > 0``.
            boxes_2: ``(M, 4)`` array of boxes in ``[x1, y1, x2, y2]`` format.
                Guaranteed ``M > 0``.

        Returns:
            ``(N, M)`` similarity matrix.
        """

    def normalize_for_fusion(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Normalize similarity values for score fusion in BoT-SORT association.

        By default returns the matrix unchanged. Signed variants (GIoU, DIoU,
        CIoU) override this to shift ``[-1, 1]`` scores into ``[0, 1]`` via
        ``(matrix + 1) / 2`` so that score fusion preserves ranking for both
        overlapping and non-overlapping box pairs.

        Args:
            similarity_matrix: ``(N, M)`` similarity matrix from :meth:`compute`.

        Returns:
            ``(N, M)`` matrix suitable for element-wise multiplication with
            detection confidence scores.
        """
        return similarity_matrix


class IoU(BaseIoU):
    """Standard Intersection over Union.

    Computes the ratio of the intersection area to the union area for
    every pair of boxes. Values range from 0 (no overlap) to 1 (perfect
    overlap). This is the classic metric used in SORT.

    Examples:
        >>> import numpy as np
        >>> metric = IoU()
        >>> boxes_a = np.array([[0.0, 0.0, 10.0, 10.0]])
        >>> boxes_b = np.array([[5.0, 5.0, 15.0, 15.0]])
        >>> metric.compute(boxes_a, boxes_b)
        array([[0.14285715]], dtype=float32)
    """

    def _compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        import supervision as sv

        return sv.box_iou_batch(boxes_1, boxes_2)


class BIoU(BaseIoU):
    """Buffered Intersection over Union.

    Computes IoU after expanding each box by a configurable relative margin
    around its center:

    - ``x1' = x1 - r * w``
    - ``y1' = y1 - r * h``
    - ``x2' = x2 + r * w``
    - ``y2' = y2 + r * h``

    where ``w = x2 - x1``, ``h = y2 - y1``, and ``r`` is ``buffer_ratio``.

    In practice, this makes association more tolerant to small localization
    gaps while preserving familiar IoU behavior. Setting
    ``buffer_ratio=0`` recovers standard IoU exactly.

    Reference: https://arxiv.org/pdf/2211.14317

    Examples:
        Buffer expands boxes before computing IoU — useful when detections
        are slightly outside the track's predicted region::

            >>> import numpy as np
            >>> metric = BIoU(buffer_ratio=0.1)
            >>> boxes_a = np.array([[0.0, 0.0, 10.0, 10.0]])
            >>> boxes_b = np.array([[11.0, 0.0, 21.0, 10.0]])
            >>> float(metric.compute(boxes_a, boxes_b)[0, 0]) > 0
            True
    """

    def __init__(self, buffer_ratio: float = 0.1) -> None:
        """Initialise BIoU with a configurable buffer ratio.

        Args:
            buffer_ratio: Non-negative relative margin to expand each box
                before computing IoU. ``0`` recovers standard IoU exactly;
                larger values tolerate wider localization gaps.

        Raises:
            ValueError: If ``buffer_ratio`` is negative.
        """
        if buffer_ratio < 0:
            raise ValueError(f"buffer_ratio must be non-negative, got {buffer_ratio}")
        self.buffer_ratio = buffer_ratio

    def _compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        import supervision as sv

        if self.buffer_ratio == 0:
            return sv.box_iou_batch(boxes_1, boxes_2)

        boxes_1_b = boxes_1.astype(np.float64, copy=True)
        boxes_2_b = boxes_2.astype(np.float64, copy=True)

        w1 = boxes_1_b[:, 2] - boxes_1_b[:, 0]
        h1 = boxes_1_b[:, 3] - boxes_1_b[:, 1]
        w2 = boxes_2_b[:, 2] - boxes_2_b[:, 0]
        h2 = boxes_2_b[:, 3] - boxes_2_b[:, 1]

        r = self.buffer_ratio
        boxes_1_b[:, 0] -= r * w1
        boxes_1_b[:, 1] -= r * h1
        boxes_1_b[:, 2] += r * w1
        boxes_1_b[:, 3] += r * h1

        boxes_2_b[:, 0] -= r * w2
        boxes_2_b[:, 1] -= r * h2
        boxes_2_b[:, 2] += r * w2
        boxes_2_b[:, 3] += r * h2

        return sv.box_iou_batch(boxes_1_b, boxes_2_b)


def _compute_iou_and_enclosing(
    boxes_1: np.ndarray, boxes_2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared geometry used by GIoU, DIoU, CIoU and other variants.

    Args:
        boxes_1: ``(N, 4)`` array in ``[x1, y1, x2, y2]`` format.
        boxes_2: ``(M, 4)`` array in ``[x1, y1, x2, y2]`` format.

    Returns:
        Tuple of ``(iou, intersection, union, enclosing_area, enclosing_diagonal_sq)``
        each with shape ``(N, M)``.
    """
    # Intersection
    inter_x1 = np.maximum(boxes_1[:, np.newaxis, 0], boxes_2[np.newaxis, :, 0])
    inter_y1 = np.maximum(boxes_1[:, np.newaxis, 1], boxes_2[np.newaxis, :, 1])
    inter_x2 = np.minimum(boxes_1[:, np.newaxis, 2], boxes_2[np.newaxis, :, 2])
    inter_y2 = np.minimum(boxes_1[:, np.newaxis, 3], boxes_2[np.newaxis, :, 3])
    intersection = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    # Areas and union
    area_1 = (boxes_1[:, 2] - boxes_1[:, 0]) * (boxes_1[:, 3] - boxes_1[:, 1])
    area_2 = (boxes_2[:, 2] - boxes_2[:, 0]) * (boxes_2[:, 3] - boxes_2[:, 1])
    union = area_1[:, np.newaxis] + area_2[np.newaxis, :] - intersection

    iou = np.divide(
        intersection,
        union,
        out=np.zeros(intersection.shape, dtype=np.float64),
        where=union > 0,
    )

    # Smallest enclosing box C
    enc_x1 = np.minimum(boxes_1[:, np.newaxis, 0], boxes_2[np.newaxis, :, 0])
    enc_y1 = np.minimum(boxes_1[:, np.newaxis, 1], boxes_2[np.newaxis, :, 1])
    enc_x2 = np.maximum(boxes_1[:, np.newaxis, 2], boxes_2[np.newaxis, :, 2])
    enc_y2 = np.maximum(boxes_1[:, np.newaxis, 3], boxes_2[np.newaxis, :, 3])

    enc_w = enc_x2 - enc_x1
    enc_h = enc_y2 - enc_y1
    enclosing_area = enc_w * enc_h
    enclosing_diagonal_sq = enc_w**2 + enc_h**2

    return iou, intersection, union, enclosing_area, enclosing_diagonal_sq


class GIoU(BaseIoU):
    """Generalized Intersection over Union (Rezatofighi et al., 2019).

    Extends standard IoU by penalizing the empty area within the smallest
    enclosing box that is not covered by either box. This provides a
    meaningful gradient even when the two boxes do not overlap.

    ``GIoU = IoU - |C \\ (A U B)| / |C|``

    Values are in ``[-1, 1]``: near -1 for far-apart boxes, 1 for perfect overlap.

    Reference: https://arxiv.org/abs/1902.09630

    Examples:
        GIoU is negative for non-overlapping boxes, providing gradient signal
        unavailable from standard IoU::

            >>> import numpy as np
            >>> metric = GIoU()
            >>> boxes_a = np.array([[0.0, 0.0, 1.0, 1.0]])
            >>> boxes_b = np.array([[5.0, 5.0, 6.0, 6.0]])
            >>> float(metric.compute(boxes_a, boxes_b)[0, 0]) < 0
            True
    """

    def _compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        iou, _, union, enclosing_area, _ = _compute_iou_and_enclosing(boxes_1, boxes_2)

        penalty = np.divide(
            enclosing_area - union,
            enclosing_area,
            out=np.zeros(enclosing_area.shape, dtype=np.float64),
            where=enclosing_area > 0,
        )

        return iou - penalty

    def normalize_for_fusion(self, similarity_matrix: np.ndarray) -> np.ndarray:
        return (similarity_matrix + 1.0) / 2.0


class DIoU(BaseIoU):
    """Distance Intersection over Union (Zheng et al., 2019).

    Extends IoU by penalizing the normalized Euclidean distance between
    bounding-box centers, using the diagonal length of the smallest
    enclosing rectangle as the scale. This yields a smooth signal when
    boxes overlap or are separated and aligns with how many detectors
    localize objects (center-based error).

    ``DIoU = IoU - d^2 / (c^2 + epsilon)``

    where `d` is the center-to-center distance, `c` is the enclosing
    diagonal, and ``\\epsilon`` avoids division by zero (same convention as
    :func:`torchvision.ops.distance_box_iou`).

    Because the penalty is nonnegative, ``DIoU ≤ IoU`` for every pair.
    Values typically lie in ``[-1, 1]`` for well-formed boxes.

    Reference: https://arxiv.org/abs/1911.08287

    Examples:
        DIoU penalizes center distance, so concentric boxes score 1.0::

            >>> import numpy as np
            >>> metric = DIoU()
            >>> boxes_a = np.array([[0.0, 0.0, 4.0, 4.0]])
            >>> boxes_b = np.array([[1.0, 1.0, 3.0, 3.0]])
            >>> float(metric.compute(boxes_a, boxes_b)[0, 0]) <= 1.0
            True
    """

    _EPS = 1e-7

    def _compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        iou, _, _, _, enclosing_diagonal_sq = _compute_iou_and_enclosing(boxes_1, boxes_2)

        cx1 = (boxes_1[:, 0] + boxes_1[:, 2]) / 2
        cy1 = (boxes_1[:, 1] + boxes_1[:, 3]) / 2
        cx2 = (boxes_2[:, 0] + boxes_2[:, 2]) / 2
        cy2 = (boxes_2[:, 1] + boxes_2[:, 3]) / 2

        dx = cx1[:, np.newaxis] - cx2[np.newaxis, :]
        dy = cy1[:, np.newaxis] - cy2[np.newaxis, :]
        center_dist_sq = dx * dx + dy * dy

        denom = enclosing_diagonal_sq + self._EPS
        return iou - center_dist_sq / denom

    def normalize_for_fusion(self, similarity_matrix: np.ndarray) -> np.ndarray:
        return (similarity_matrix + 1.0) / 2.0


class CIoU(BaseIoU):
    """Complete Intersection over Union (Zheng et al., 2019).

    Builds on **DIoU** by adding a penalty for mismatched aspect ratio between
    boxes (via a term ``v`` on the difference of box arctan aspect ratios).
    The trade-off is weighted by ``\\alpha`` that depends on IoU and ``v``,
    matching :func:`torchvision.ops.complete_box_iou`.

    ``CIoU = DIoU - alpha * v``, with
    ``alpha = v / (1 - IoU + v + epsilon)``.

    So **CIoU ≤ DIoU ≤ IoU** when widths and heights are positive.
    Scores are in ``[-1, 1]``, matching the range of
    :func:`torchvision.ops.complete_box_iou`.

    Reference: https://arxiv.org/abs/1911.08287

    Examples:
        CIoU adds an aspect-ratio penalty to DIoU, so CIoU <= DIoU::

            >>> import numpy as np
            >>> metric_d = DIoU()
            >>> metric_c = CIoU()
            >>> boxes_a = np.array([[0.0, 0.0, 10.0, 5.0]])
            >>> boxes_b = np.array([[8.0, 3.0, 15.0, 8.0]])
            >>> lhs = float(metric_c.compute(boxes_a, boxes_b)[0, 0])
            >>> rhs = float(metric_d.compute(boxes_a, boxes_b)[0, 0]) + 1e-6
            >>> lhs <= rhs
            True
    """

    _EPS = 1e-7

    def _compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        iou, _, _, _, enclosing_diagonal_sq = _compute_iou_and_enclosing(boxes_1, boxes_2)

        cx1 = (boxes_1[:, 0] + boxes_1[:, 2]) / 2
        cy1 = (boxes_1[:, 1] + boxes_1[:, 3]) / 2
        cx2 = (boxes_2[:, 0] + boxes_2[:, 2]) / 2
        cy2 = (boxes_2[:, 1] + boxes_2[:, 3]) / 2

        dx = cx1[:, np.newaxis] - cx2[np.newaxis, :]
        dy = cy1[:, np.newaxis] - cy2[np.newaxis, :]
        center_dist_sq = dx * dx + dy * dy

        denom = enclosing_diagonal_sq + self._EPS
        diou = iou - center_dist_sq / denom

        w1 = boxes_1[:, 2] - boxes_1[:, 0]
        h1 = boxes_1[:, 3] - boxes_1[:, 1]
        w2 = boxes_2[:, 2] - boxes_2[:, 0]
        h2 = boxes_2[:, 3] - boxes_2[:, 1]

        w_pred = w1[:, np.newaxis]
        h_pred = h1[:, np.newaxis]
        w_gt = w2[np.newaxis, :]
        h_gt = h2[np.newaxis, :]

        safe_h_pred = np.maximum(h_pred, self._EPS)
        safe_h_gt = np.maximum(h_gt, self._EPS)

        v = (4.0 / (np.pi**2)) * (np.arctan(w_pred / safe_h_pred) - np.arctan(w_gt / safe_h_gt)) ** 2
        alpha = np.divide(v, 1.0 - iou + v, out=np.zeros_like(v), where=(1.0 - iou + v) > 0)
        return diou - alpha * v

    def normalize_for_fusion(self, similarity_matrix: np.ndarray) -> np.ndarray:
        return (similarity_matrix + 1.0) / 2.0
