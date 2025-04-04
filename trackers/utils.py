import numpy as np


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Computes IOU (Intersection Over Union) between two arrays of bounding boxes.

    Args:
        bboxes1 (np.ndarray): Nx4 matrix of [x1, y1, x2, y2].
        bboxes2 (np.ndarray): Mx4 matrix of [x1, y1, x2, y2].

    Returns:
        np.ndarray: NxM IOU matrix.
    """
    # Check for empty arrays or incorrect shapes
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)

    # Ensure arrays are 2D with shape (N, 4) and (M, 4)
    if bboxes1.ndim == 1:
        bboxes1 = bboxes1.reshape(1, -1)
    if bboxes2.ndim == 1:
        bboxes2 = bboxes2.reshape(1, -1)

    # Expand dims to broadcast
    x1 = np.maximum(bboxes1[:, None, 0], bboxes2[None, :, 0])
    y1 = np.maximum(bboxes1[:, None, 1], bboxes2[None, :, 1])
    x2 = np.minimum(bboxes1[:, None, 2], bboxes2[None, :, 2])
    y2 = np.minimum(bboxes1[:, None, 3], bboxes2[None, :, 3])

    inter_area = np.clip(x2 - x1, a_min=0, a_max=None) * np.clip(
        y2 - y1, a_min=0, a_max=None
    )

    bboxes1_area = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    bboxes2_area = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    iou = inter_area / (
        bboxes1_area[:, None] + bboxes2_area[None, :] - inter_area + 1e-6
    )
    return iou
