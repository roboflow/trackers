# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import supervision as sv


class BaseIoU(ABC):
    """Abstract base for IoU similarity metrics used in tracker association.

    Subclasses implement a specific variant of Intersection over Union
    (e.g. standard IoU, GIoU, DIoU, CIoU) that computes a pairwise
    similarity matrix between two sets of bounding boxes.

    The resulting matrix is used as a cost/similarity signal in the
    Hungarian algorithm during the data association step.
    """

    def compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity between two sets of bounding boxes.

        Handles the empty-input edge case (returns a correctly-shaped zero
        matrix) and delegates to :meth:`_compute` for the actual math.

        Args:
            boxes_1: ``(N, 4)`` array of boxes in ``[x1, y1, x2, y2]`` format.
            boxes_2: ``(M, 4)`` array of boxes in ``[x1, y1, x2, y2]`` format.

        Returns:
            ``(N, M)`` similarity matrix where entry ``(i, j)`` is the
            similarity between ``boxes_1[i]`` and ``boxes_2[j]``.
        """
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


class IoU(BaseIoU):
    """Standard Intersection over Union.

    Computes the ratio of the intersection area to the union area for
    every pair of boxes. Values range from 0 (no overlap) to 1 (perfect
    overlap). This is the metric used in the original SORT paper.
    """

    def _compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        return sv.box_iou_batch(boxes_1, boxes_2)


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
    intersection = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(
        inter_y2 - inter_y1, 0
    )

    # Areas and union
    area_1 = (boxes_1[:, 2] - boxes_1[:, 0]) * (boxes_1[:, 3] - boxes_1[:, 1])
    area_2 = (boxes_2[:, 2] - boxes_2[:, 0]) * (boxes_2[:, 3] - boxes_2[:, 1])
    union = area_1[:, np.newaxis] + area_2[np.newaxis, :] - intersection

    iou = np.where(union > 0, intersection / union, 0.0)

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

    ``GIoU = IoU - |C \\ (A ∪ B)| / |C|``

    Values range from -1 (boxes far apart) to 1 (perfect overlap).

    Reference: https://arxiv.org/abs/1902.09630
    """

    def _compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        iou, _, union, enclosing_area, _ = _compute_iou_and_enclosing(
            boxes_1, boxes_2
        )

        giou = iou - np.where(
            enclosing_area > 0,
            (enclosing_area - union) / enclosing_area,
            0.0,
        )

        return giou
