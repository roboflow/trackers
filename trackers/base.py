from abc import ABC, abstractmethod

import numpy as np
import supervision as sv


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: sv.Detections) -> sv.Detections:
        pass


class BaseTrackerWithFeatures(ABC):
    @abstractmethod
    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        pass
