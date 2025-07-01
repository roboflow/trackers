from collections import defaultdict
from typing import Any, List

import numpy as np
import supervision as sv

from trackers.core.base import BaseTracker
from trackers.core.ksptracker.solver import KSP_Solver, TrackNode


class KSPTracker(BaseTracker):
    """
    Offline tracker using K-Shortest Paths (KSP) algorithm.
    """

    def __init__(self) -> None:
        """
        Initialize the KSPTracker and its solver.
        """
        self._solver = KSP_Solver()
        self._solver.reset()
        self.reset()

    def reset(self) -> None:
        """
        Reset the solver and clear any stored state.
        """
        self._solver.reset()

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Add detections for the current frame to the solver.

        Args:
            detections (sv.Detections): Detections for the current frame.

        Returns:
            sv.Detections: The same detections passed in.
        """
        self._solver.append_frame(detections)
        return detections

    def assign_tracker_ids_from_paths(
        self, paths: List[List[TrackNode]]
    ) -> List[sv.Detections]:
        """
        Assigns each detection a unique tracker ID by preferring the path with
        the least motion change (displacement).

        Args:
            paths (List[List[TrackNode]]): List of tracks, each a list of TrackNode.

        Returns:
            Dict[int, sv.Detections]: Mapping from frame index to sv.Detections
                                      with tracker IDs assigned.
        """
        # Track where each node appears
        framed_nodes = defaultdict(list)
        node_to_candidates = defaultdict(list)
        for tracker_id, path in enumerate(paths, start=1):
            for i, node in enumerate(path):
                next_node: Any = path[i + 1] if i + 1 < len(path) else None
                node_to_candidates[node].append((tracker_id, next_node))
                framed_nodes[node.frame_id].append(node)

        # Select best tracker for each node based on minimal displacement
        node_to_tracker = {}
        for node, candidates in node_to_candidates.items():
            min_displacement = float("inf")
            selected_tracker = -1
            for tracker_id, next_node in candidates:
                if next_node is not None:
                    dx = node.position[0] - next_node.position[0]
                    dy = node.position[1] - next_node.position[1]
                    displacement = dx * dx + dy * dy  # squared distance
                else:
                    displacement = 0  # last node in path, no penalty

                if displacement < min_displacement:
                    min_displacement = displacement
                    selected_tracker = tracker_id

            node_to_tracker[node] = selected_tracker

        # Organize detections by frame
        frame_to_dets = defaultdict(list)

        for node, tracker_id in node_to_tracker.items():
            frame_to_dets[node.frame_id].append(
                {
                    "xyxy": node.bbox,
                    "confidence": node.confidence,
                    "class_id": node.class_id,
                    "tracker_id": tracker_id,
                }
            )

        # Convert into sv.Detections
        frame_to_detections = []
        for frame, dets_list in frame_to_dets.items():
            xyxy = np.array([d["xyxy"] for d in dets_list], dtype=np.float32)
            confidence = np.array(
                [d["confidence"] for d in dets_list], dtype=np.float32
            )
            class_id = np.array([d["class_id"] for d in dets_list], dtype=int)
            tracker_id = np.array([d["tracker_id"] for d in dets_list], dtype=int)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                tracker_id=tracker_id,
            )
            frame_to_detections.append(detections)

        return frame_to_detections

    def process_tracks(self, num_of_tracks=None) -> List[sv.Detections]:
        """
        Run the KSP solver and assign tracker IDs to detections.

        Returns:
            List[sv.Detections]: Mapping from frame index to sv.Detections
                                      with tracker IDs assigned.
        """
        paths = self._solver.solve(num_of_tracks)
        if not paths:
            return []
        return self.assign_tracker_ids_from_paths(paths)
