from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import supervision as sv

from trackers.core.base import BaseTracker


@dataclass(frozen=True)
class TrackNode:
    """
    Represents a detection node in the tracking graph.

    Attributes:
        frame_id (int): Frame index where detection occurred
        grid_cell_id (int): Discretized grid cell ID of detection center
        position (tuple): Grid coordinates (x_bin, y_bin)
        confidence (float): Detection confidence score
    """

    frame_id: int
    grid_cell_id: int
    position: tuple
    confidence: float

    def __hash__(self) -> int:
        """
        Generate hash using frame and grid cell.

        Returns:
            int: Hash value for node
        """
        return hash((self.frame_id, self.grid_cell_id))

    def __eq__(self, other: Any) -> bool:
        """
        Compare nodes by frame and grid cell ID.

        Args:
            other (Any): Object to compare

        Returns:
            bool: True if same node, False otherwise
        """
        if not isinstance(other, TrackNode):
            return False
        return (self.frame_id, self.grid_cell_id) == (
            other.frame_id,
            other.grid_cell_id,
        )


class KSPTracker(BaseTracker):
    """
    Offline tracker using K-Shortest Paths (KSP).

    Attributes:
        grid_size (int): Size of each grid cell (in pixels)
        max_gap (int): Max frame gap between connections
        min_confidence (float): Minimum detection confidence
        max_distance (float): Max dissimilarity (1 - IoU) allowed
    """

    def __init__(
        self,
        grid_size: int = 20,
        max_gap: int = 30,
        min_confidence: float = 0.3,
        max_distance: float = 0.3,
    ) -> None:
        """
        Initialize KSP tracker with config parameters.

        Args:
            grid_size (int): Pixel size of each grid cell
            max_gap (int): Max frames between connected detections
            min_confidence (float): Min detection confidence
            max_distance (float): Max allowed dissimilarity
        """
        self.grid_size = grid_size
        self.max_gap = max_gap
        self.min_confidence = min_confidence
        self.max_distance = max_distance
        self.reset()

    def reset(self) -> None:
        """
        Reset the internal detection buffer.
        """
        self.detection_buffer: List[sv.Detections] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Append new detections to the buffer.

        Args:
            detections (sv.Detections): Frame detections

        Returns:
            sv.Detections: Same as input
        """
        self.detection_buffer.append(detections)
        return detections

    def _discretized_grid_cell_id(self, bbox: np.ndarray) -> tuple:
        """
        Get grid cell ID from bbox center.

        Args:
            bbox (np.ndarray): Bounding box coordinates

        Returns:
            tuple: Grid (x_bin, y_bin)
        """
        x_center = (bbox[2] - bbox[0]) / 2
        y_center = (bbox[3] - bbox[1]) / 2
        grid_x_center = int(x_center // self.grid_size)
        grid_y_center = int(y_center // self.grid_size)
        return (grid_x_center, grid_y_center)

    def _build_graph(self, all_detections: List[sv.Detections]) -> nx.DiGraph:
        """
        Build graph from all buffered detections.

        Args:
            all_detections (List[sv.Detections]): All video detections

        Returns:
            nx.DiGraph: Directed graph with detection nodes
        """
        G = nx.DiGraph()
        G.add_node("source")
        G.add_node("sink")

        self.node_to_detection: Dict[TrackNode, tuple] = {}
        node_dict: Dict[int, List[TrackNode]] = {}

        for frame_idx, dets in enumerate(all_detections):
            node_dict[frame_idx] = []
            for det_idx in range(len(dets)):
                if dets.confidence[det_idx] < self.min_confidence:
                    continue

                pos = self._discretized_grid_cell_id(np.array(dets.xyxy[det_idx]))
                cell_id = hash(pos)

                node = TrackNode(
                    frame_id=frame_idx,
                    grid_cell_id=cell_id,
                    position=pos,
                    confidence=dets.confidence[det_idx],
                )

                G.add_node(node)
                node_dict[frame_idx].append(node)
                self.node_to_detection[node] = (frame_idx, det_idx)

                if frame_idx == 0:
                    G.add_edge("source", node, weight=max(-node.confidence, 0.001))
                if frame_idx == len(all_detections) - 1:
                    G.add_edge(node, "sink", weight=0)

        for i in range(len(all_detections) - 1):
            for node in node_dict[i]:
                for node_next in node_dict[i + 1]:
                    dist = np.linalg.norm(
                        np.array(node.position) - np.array(node_next.position)
                    )
                    if dist <= 2:
                        G.add_edge(
                            node_next,
                            node,
                            weight=max(dist - node_next.confidence, 0.001),
                        )

        return G

    def _update_detections_with_tracks(
        self, assignments: List[List[TrackNode]]
    ) -> sv.Detections:
        """
        Assign track IDs to detections.

        Args:
            assignments (Dict): Paths from KSP with track IDs

        Returns:
            sv.Detections: Merged detections with tracker IDs
        """
        all_detections = []
        all_tracker_ids = []

        assigned = set()
        assignment_map = {}
        for track_id, path in enumerate(assignments, start=1):
            for node in path:
                det_key = self.node_to_detection.get(node)
                if det_key and det_key not in assigned:
                    assignment_map[det_key] = track_id
                    assigned.add(det_key)

        for frame_idx, dets in enumerate(self.detection_buffer):
            frame_tracker_ids = [-1] * len(dets)
            for det_idx in range(len(dets)):
                key = (frame_idx, det_idx)
                if key in assignment_map:
                    frame_tracker_ids[det_idx] = assignment_map[key]

            all_detections.append(dets)
            all_tracker_ids.extend(frame_tracker_ids)

        final_detections = sv.Detections.merge(all_detections)
        final_detections.tracker_id = np.array(all_tracker_ids)
        return final_detections

    def ksp(self, graph: nx.DiGraph) -> List[List[TrackNode]]:
        """
        Find multiple disjoint shortest paths.

        Args:
            graph (nx.DiGraph): Detection graph

        Returns:
            List[List[TrackNode]]: Disjoint detection paths
        """
        paths: List[List[TrackNode]] = []
        G_copy = graph.copy()

        while True:
            try:
                path = nx.shortest_path(
                    G_copy, source="source", target="sink", weight="weight"
                )
                if len(path) < 2:
                    break
                paths.append(path[1:-1])
                for node in path[1:-1]:
                    G_copy.remove_node(node)
            except nx.NetworkXNoPath:
                break

        return paths

    def process_tracks(self) -> sv.Detections:
        """
        Run tracker and assign detections to tracks.

        Returns:
            sv.Detections: Final detections with track IDs
        """
        graph = self._build_graph(self.detection_buffer)
        disjoint_paths = self.ksp(graph)
        return self._update_detections_with_tracks(assignments=disjoint_paths)
