import numpy as np


def intersection_over_union(
    bounding_boxes_batch_1: np.ndarray, bounding_boxes_batch_2: np.ndarray
) -> np.ndarray:
    """
    Computes IOU (Intersection Over Union) between two arrays of bounding boxes.
    Args:
        bounding_boxes_batch_1 (np.ndarray): Nx4 matrix of [x1, y1, x2, y2].
        bounding_boxes_batch_2 (np.ndarray): Mx4 matrix of [x1, y1, x2, y2].
    Returns:
        np.ndarray: NxM IOU matrix.
    """
    # Check for empty arrays or incorrect shapes
    if len(bounding_boxes_batch_1) == 0 or len(bounding_boxes_batch_2) == 0:
        return np.zeros(
            (len(bounding_boxes_batch_1), len(bounding_boxes_batch_2)), dtype=np.float32
        )

    # Ensure arrays are 2D with shape (N, 4) and (M, 4)
    if bounding_boxes_batch_1.ndim == 1:
        bounding_boxes_batch_1 = bounding_boxes_batch_1.reshape(1, -1)
    if bounding_boxes_batch_2.ndim == 1:
        bounding_boxes_batch_2 = bounding_boxes_batch_2.reshape(1, -1)

    # Expand dims to broadcast
    x1 = np.maximum(
        bounding_boxes_batch_1[:, None, 0], bounding_boxes_batch_2[None, :, 0]
    )
    y1 = np.maximum(
        bounding_boxes_batch_1[:, None, 1], bounding_boxes_batch_2[None, :, 1]
    )
    x2 = np.minimum(
        bounding_boxes_batch_1[:, None, 2], bounding_boxes_batch_2[None, :, 2]
    )
    y2 = np.minimum(
        bounding_boxes_batch_1[:, None, 3], bounding_boxes_batch_2[None, :, 3]
    )

    inter_area = np.clip(x2 - x1, a_min=0, a_max=None) * np.clip(
        y2 - y1, a_min=0, a_max=None
    )

    bounding_boxes_batch_1_area = (
        bounding_boxes_batch_1[:, 2] - bounding_boxes_batch_1[:, 0]
    ) * (bounding_boxes_batch_1[:, 3] - bounding_boxes_batch_1[:, 1])
    bounding_boxes_batch_2_area = (
        bounding_boxes_batch_2[:, 2] - bounding_boxes_batch_2[:, 0]
    ) * (bounding_boxes_batch_2[:, 3] - bounding_boxes_batch_2[:, 1])

    iou = inter_area / (
        bounding_boxes_batch_1_area[:, None]
        + bounding_boxes_batch_2_area[None, :]
        - inter_area
        + 1e-6
    )
    return iou
