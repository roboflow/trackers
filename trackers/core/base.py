from abc import ABC, abstractmethod

import numpy as np
import supervision as sv
from typing import Any, Callable, List, Optional


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: sv.Detections) -> sv.Detections:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class BaseTrackerWithFeatures(ABC):
    @abstractmethod
    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

class BaseOfflineTracker(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def track(self, 
            source_path: str,
            get_model_detections: Callable[[np.ndarray], sv.Detections],
            num_of_tracks: Optional[int] = None
        ) -> List[sv.Detections]:
        pass
            
