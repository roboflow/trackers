from dataclasses import dataclass
from typing import Any, Dict, List, Union

import networkx as nx
import numpy as np
import supervision as sv

from trackers.core.base import BaseTracker


@dataclass(frozen=True)
class TrackNode:
    """Represents a detection node in the tracking graph.

    Attributes:
        frame_id (int): Frame index where detection occurred
        grid_cell_id (int): Discretized grid cell id where the detection center
        position (tuple): Grid coordinates of the detection center (x_center, y_center)
        confidence (float): Detection confidence score
    """

    frame_id: int
    grid_cell_id: int
    position: tuple
    confidence: float

    def __hash__(self) -> int:
        """Generates hash based on frame_id and detection_id.

        Returns:
            int: Hash value of the node
        """
        return hash((self.frame_id, self.grid_cell_id))

    def __eq__(self, other: Any) -> bool:
        """Compares equality based on frame_id and detection_id.

        Args:
            other (Any): Object to compare with

        Returns:
            bool: True if nodes are equal, False otherwise
        """
        if not isinstance(other, TrackNode):
            return False
        return (self.frame_id, self.grid_cell_id) == (
            other.frame_id,
            other.grid_cell_id,
        )


class KSPTracker(BaseTracker):
    """Offline tracker using K-Shortest Paths (KSP) algorithm.

    Attributes:
        max_gap (int): Maximum allowed frame gap between detections in a track
        min_confidence (float): Minimum confidence threshold for detections
        max_paths (int): Maximum number of paths to find in KSP algorithm
        max_distance (float): Maximum allowed dissimilarity (1 - IoU) for edges
        detection_buffer (List[sv.Detections]): Buffer storing all frame detections
    """

    def __init__(
        self,
        grid_size: int = 20,
        max_gap: int = 30,
        min_confidence: float = 0.3,
        max_paths: int = 1000,
        max_distance: float = 0.3,
    ) -> None:
        """Initialize KSP tracker with configuration parameters.

        Args:
            grid_size (int): Size (in pixels) of each square cell in the spatial grid
            max_gap (int): Max frame gap between connected detections
            min_confidence (float): Minimum detection confidence
            max_paths (int): Maximum number of paths to find
            max_distance (float): Max dissimilarity (1-IoU) for connections
        """
        self.grid_size = grid_size
        self.max_gap = max_gap
        self.min_confidence = min_confidence
        self.max_paths = max_paths
        self.max_distance = max_distance
        self.reset()

    def reset(self) -> None:
        """Reset the tracker's internal state."""
        self.detection_buffer: List[sv.Detections] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker with new detections (stores without processing).

        Args:
            detections (sv.Detections): New detections for current frame

        Returns:
            sv.Detections: Input detections (unmodified)
        """
        self.detection_buffer.append(detections)
        return detections
    
    def _discretized_grid_cell_id(self, bbox: np.ndarray) -> tuple:
        x_center = (bbox[2] - bbox[0]) / 2
        y_center = (bbox[3] - bbox[1]) / 2
        grid_x_center = int(x_center // self.grid_size)
        grid_y_center = int(y_center // self.grid_size)
        
        return (grid_x_center, grid_y_center)

    def _build_graph(self, all_detections: List[sv.Detections]) -> nx.DiGraph:
        """Build directed graph from all detections.

        Args:
            all_detections (List[sv.Detections]): List of detections from frames

        Returns:
            nx.DiGraph: Directed graph with detection nodes and edges
        """
        G = nx.DiGraph()
        G.add_node("source")
        G.add_node("sink")

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

                if frame_idx == 0:
                    G.add_edge("source", node, weight=-node.confidence)
                
                if frame_idx == len(all_detections) - 1:
                    G.add_edge(node, "sink", weight=0)

        for i in range(len(all_detections) - 1):
            for node in node_dict[i]:
                for node_next in node_dict[i + 1]:
                    dist = np.linalg.norm(np.array(node.position) - np.array(node_next.position))
                    if dist <= 2:
                        G.add_edge(node, node_next, weight=dist - node_next.confidence)

        return G

    def _update_detections_with_tracks(self, assignments: Dict) -> sv.Detections:
        """Update detections with track IDs based on assignments.

        Args:
            assignments (Dict): Maps (frame_id, det_id) to track_id

        Returns:
            sv.Detections: Updated detections with tracker_ids assigned
        """
        all_detections = []
        all_tracker_ids = []

        for frame_idx, dets in enumerate(self.detection_buffer):
            frame_tracker_ids = [-1] * len(dets)

            for det_idx in range(len(dets)):
                key = (frame_idx, det_idx)
                if key in assignments:
                    frame_tracker_ids[det_idx] = assignments[key]

            all_detections.append(dets)
            all_tracker_ids.extend(frame_tracker_ids)

        final_detections = sv.Detections.merge(all_detections)
        final_detections.tracker_id = np.array(all_tracker_ids)

        return final_detections

    def ksp(self, graph: nx.DiGraph) -> List[List[TrackNode]]:
        """Find K-shortest paths in the graph.

        Args:
            graph (nx.DiGraph): Directed graph of detection nodes

        Returns:
            List[List[TrackNode]]: List of paths, each path is list of TrackNodes
        """
        paths: List[List[TrackNode]] = []
        for path in nx.shortest_simple_paths(graph, "source", "sink", weight="weight"):
            if len(paths) >= self.max_paths:
                break
            # Remove source and sink nodes from path
            paths.append(path[1:-1])
        return paths

    def process_tracks(self) -> sv.Detections:
        """Process all buffered detections to create final tracks.

        Returns:
            sv.Detections: Detections with assigned track IDs
        """
        graph = self._build_graph(self.detection_buffer)
        paths = self.ksp(graph)

        # Assign track IDs
        assignments = {}
        for track_id, path in enumerate(paths, start=1):
            for node in path:
                for det_idx, det in enumerate(self.detection_buffer[node.frame_id]):
                   #  print(det)
                    pos = self._discretized_grid_cell_id(np.array(det[0]))
                    if pos == node.position:
                        assignments[(node.frame_id, node.grid_cell_id)] = track_id

        return self._update_detections_with_tracks(assignments=assignments)
