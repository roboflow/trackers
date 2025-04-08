from abc import ABC, abstractmethod

import supervision as sv


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: sv.Detections) -> sv.Detections:
        pass
