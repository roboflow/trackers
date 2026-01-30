# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod

import numpy as np
import supervision as sv


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
