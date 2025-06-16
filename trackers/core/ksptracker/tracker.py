import numpy as np
import supervision as sv
from trackers.core.base import BaseTracker


class KSPTracker(BaseTracker):
    """
    Offline tracker using the K-Shortest Paths (KSP) algorithm.
    """
    def __init__(self):
        self.reset()

    def update(self, detections: sv.Detections) -> sv.Detections:
        self.detection_buffer.append(detections)
        return detections

    def reset(self) -> None:
        pass