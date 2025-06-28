from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import supervision as sv

from trackers.core.base import BaseTracker
from trackers.core.ksptracker.solver import KSP_Solver, TrackNode


class KSPTracker(BaseTracker):
    """
    Offline tracker using K-Shortest Paths (KSP) algorithm.
    """

    def __init__(
        self,
    ) -> None:
        self._solver = KSP_Solver()
        
        self._solver.reset()
        self.reset()

    def reset(self) -> None:
        self._solver.reset()

    def update(self, detections: sv.Detections) -> sv.Detections:
        self._solver.append_frame(detections)
        return detections
    
    def assign_tracker_ids_from_paths(self, paths: List[List[TrackNode]], num_frames: int) -> Dict[int, sv.Detections]:
        """
        Converts paths (list of list of TrackNode) into frame-wise sv.Detections with tracker IDs assigned.
        """
        frame_to_dets = {frame: [] for frame in range(num_frames)}

        for tracker_id, path in enumerate(paths, start=1):
            for node in path:
                if isinstance(node, TrackNode):
                    frame_to_dets[node.frame_id].append({
                        "xyxy": node.bbox,
                        "confidence": node.confidence,
                        "class_id": node.class_id,
                        "tracker_id": tracker_id,
                    })

        frame_to_detections = {}
        for frame, dets_list in frame_to_dets.items():
            if not dets_list:
                continue
            xyxy = np.array([d["xyxy"] for d in dets_list], dtype=np.float32)
            confidence = np.array([d["confidence"] for d in dets_list], dtype=np.float32)
            class_id = np.array([d["class_id"] for d in dets_list], dtype=int)
            tracker_id = np.array([d["tracker_id"] for d in dets_list], dtype=int)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                tracker_id=tracker_id,
            )
            frame_to_detections[frame] = detections

        return frame_to_detections

    def process_tracks(self) -> Dict[int, sv.Detections]:
        paths = self._solver.solve()
        print(f"Extracted {len(paths)} tracks.")

        if not paths:
            print("No valid paths found.")
            return {}
        
        return self.assign_tracker_ids_from_paths(paths, num_frames=len(self._solver.detection_per_frame))
