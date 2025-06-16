import numpy as np
import supervision as sv
import networkx as nx
from trackers.core.base import BaseTracker

class KSPTracker(BaseTracker):
    """
    Offline tracker using the K-Shortest Paths (KSP) algorithm.
    """
    def __init__(self,
                 max_gap: int = 30,
                 min_confidence: float = 0.3,
                 max_paths: int = 1000,
                 max_distance: float = 0.3,
    ):
        self.max_gap = max_gap
        self.min_confidence = min_confidence
        self.max_paths = max_paths
        self.max_distance = max_distance
        self.reset()

    def _calc_iou(self, bbox1: np.ndarray, bbox2: np.ndarray):
        x1, y1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
        x2, y2 = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
        eps = 1e-5 # To not allow division by 0

        x_inter, y_inter = max(0, x2 - x1) , min(0, y2 - y1)
        intersection_area = x_inter * y_inter
        area1 = (bbox1[3] - bbox1[1]) * (bbox1[2] - bbox1[0])
        area2 = (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])
        union = area1 + area2 - intersection_area

        return intersection_area / (union + eps)


    def update(self, detections: sv.Detections) -> sv.Detections:
        self.detection_buffer.append(detections)
        return detections

    def reset(self) -> None:
        pass