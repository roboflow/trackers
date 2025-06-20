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
        frame_id (int): Frame index where detection occurred.
        grid_cell_id (int): Discretized grid cell ID of detection center.
        position (tuple): Grid coordinates (x_bin, y_bin).
        confidence (float): Detection confidence score.
    """

    frame_id: int
    grid_cell_id: int
    position: tuple
    confidence: float

    def __hash__(self) -> int:
        """Generate hash using frame and grid cell."""
        return hash((self.frame_id, self.grid_cell_id))

    def __eq__(self, other: Any) -> bool:
        """Compare nodes by frame and grid cell ID."""
        if not isinstance(other, TrackNode):
            return False
        return (self.frame_id, self.grid_cell_id) == (other.frame_id, other.grid_cell_id)


class KSPTracker(BaseTracker):
    """
    Offline tracker using K-Shortest Paths (KSP) algorithm.

    Attributes:
        grid_size (int): Size of each grid cell (pixels).
        min_confidence (float): Minimum detection confidence to consider.
        max_distance (float): Maximum spatial distance between nodes to connect.
        max_paths (int): Maximum number of paths (tracks) to find.
    """

    def __init__(
        self,
        grid_size: int = 20,
        max_paths: int = 20,
        min_confidence: float = 0.3,
        max_distance: float = 0.3,
    ) -> None:
        """
        Initialize the KSP tracker.

        Args:
            grid_size (int): Pixel size of grid cells.
            max_paths (int): Max number of paths to find.
            min_confidence (float): Minimum confidence to keep detection.
            max_distance (float): Max allowed spatial distance between nodes.
        """
        self.grid_size = grid_size
        self.min_confidence = min_confidence
        self.max_distance = max_distance
        self.max_paths = max_paths
        self.G = nx.DiGraph()
        self.reset()

    def reset(self) -> None:
        """Reset the detection buffer."""
        self.detection_buffer: List[sv.Detections] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Append new detections to the buffer.

        Args:
            detections (sv.Detections): Detections for the current frame.

        Returns:
            sv.Detections: The same detections passed in.
        """
        self.detection_buffer.append(detections)
        return detections

    def _discretized_grid_cell_id(self, bbox: np.ndarray) -> tuple:
        """
        Compute discretized grid cell coordinates from bbox center.

        Args:
            bbox (np.ndarray): Bounding box [x1, y1, x2, y2].

        Returns:
            tuple: (grid_x, grid_y) discretized coordinates.
        """
        x_center = (bbox[2] + bbox[0]) / 2
        y_center = (bbox[3] + bbox[1]) / 2
        grid_x = int(x_center // self.grid_size)
        grid_y = int(y_center // self.grid_size)
        return (grid_x, grid_y)

    def _edge_cost(self, confidence: float) -> float:
        """
        Compute edge cost from detection confidence.

        Args:
            confidence (float): Detection confidence score.

        Returns:
            float: Edge cost for KSP (should be non-negative after transform).
        """
        # Add small epsilon to denominator to avoid division by zero
        return -np.log(confidence / ((1 - confidence) + 1e-6))

    def _build_graph(self, all_detections: List[sv.Detections]) -> nx.DiGraph:
        """
        Build a directed graph from buffered detections.

        Args:
            all_detections (List[sv.Detections]): List of detections per frame.

        Returns:
            nx.DiGraph: Directed graph representing detections and edges.
        """
        self.G = nx.DiGraph()
        self.G.add_node("source")
        self.G.add_node("sink")

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

                self.G.add_node(node)
                node_dict[frame_idx].append(node)
                self.node_to_detection[node] = (frame_idx, det_idx)

                if frame_idx == 0:
                    self.G.add_edge("source", node, weight=0)
                if frame_idx == len(all_detections) - 1:
                    self.G.add_edge(node, "sink", weight=0)

        for i in range(len(all_detections) - 1):
            for node in node_dict[i]:
                for node_next in node_dict[i + 1]:
                    dist = np.linalg.norm(
                        np.array(node.position) - np.array(node_next.position)
                    )
                    if dist <= 2:
                        self.G.add_edge(
                            node,
                            node_next,
                            weight=self._edge_cost(confidence=node.confidence),
                        )


    def _update_detections_with_tracks(
        self, assignments: List[List[TrackNode]]
    ) -> List[sv.Detections]:
        """
        Assign track IDs to detections based on paths.

        Args:
            assignments (List[List[TrackNode]]): List of detection paths.

        Returns:
            List[sv.Detections]: Detections with assigned tracker IDs.
        """
        all_detections = []

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

            dets.tracker_id = np.array(frame_tracker_ids)
            all_detections.append(dets)

        return all_detections

    def _shortest_path(self) -> tuple:
        """
        Compute shortest path from 'source' to 'sink' using Bellman-Ford.

        Args:
            self.G (nx.DiGraph): Graph with possible negative edges.

        Returns:
            tuple: (path, total_cost, lengths) where path is list of nodes,
                total_cost is the total weight of that path, and lengths is
                dict of shortest distances from source.

        Raises:
            RuntimeError: If negative cycle detected.
            KeyError: If sink unreachable.
        """
        try:
            lengths, paths = nx.single_source_bellman_ford(self.G, "source", weight="weight")
            if "sink" not in paths:
                raise KeyError("No path found from source to sink.")
            return paths["sink"], lengths["sink"], lengths
        except nx.NetworkXUnbounded:
            raise RuntimeError("Graph contains a negative weight cycle.")

    def _extend_graph(self, paths: List[List[TrackNode]]) -> nx.DiGraph:
        """
        Remove nodes used in previous paths to enforce disjointness.

        Args:
            self.G (nx.DiGraph): Original graph.
            paths (List[List[TrackNode]]): Previously found paths.

        Returns:
            nx.DiGraph: Extended graph with used nodes removed.
        """
        G_extended = self.G.copy()
        for path in paths:
            for node in path:
                if node in G_extended and node not in {"source", "sink"}:
                    G_extended.remove_node(node)
        return G_extended

    def _transform_edge_cost(
        self, shortest_costs: Dict[Any, float]
    ) -> nx.DiGraph:
        """
        Apply cost transformation to ensure non-negative edge weights.

        Args:
            self.G (nx.DiGraph): Graph with possibly negative weights.
            shortest_costs (dict): Shortest path distances from source.

        Returns:
            nx.DiGraph: Cost-transformed graph.
        """
        Gc = nx.DiGraph()
        for u, v, data in self.G.edges(data=True):
            if u not in shortest_costs or v not in shortest_costs:
                continue
            original = data["weight"]
            transformed = original + shortest_costs[u] - shortest_costs[v]
            Gc.add_edge(u, v, weight=transformed)
        return Gc

    def _interlace_paths(
        self, current_paths: List[List[TrackNode]], new_path: List[TrackNode]
    ) -> List[TrackNode]:
        """
        Remove nodes from new_path that conflict with current_paths.

        Args:
            current_paths (List[List[TrackNode]]): Existing disjoint paths.
            new_path (List[TrackNode]): New candidate path.

        Returns:
            List[TrackNode]: Interlaced path without conflicts.
        """
        used_nodes = set()
        for path in current_paths:
            for node in path:
                if isinstance(node, TrackNode):
                    used_nodes.add((node.frame_id, node.grid_cell_id))

        interlaced = []
        for node in new_path:
            if isinstance(node, TrackNode):
                key = (node.frame_id, node.grid_cell_id)
                if key not in used_nodes:
                    interlaced.append(node)

        return interlaced

    def ksp(self) -> List[List[TrackNode]]:
        """
        Compute k disjoint shortest paths using KSP algorithm.

        Args:
            self.G (nx.DiGraph): Detection graph.

        Returns:
            List[List[TrackNode]]: List of disjoint detection paths.
        """
        path, cost, lengths = self._shortest_path(self.G)
        P = [path]
        cost_P = [cost]

        for l in range(1, self.max_paths):
            if l != 1 and cost_P[-1] >= cost_P[-2]:
                return P  # early termination

            Gl = self._extend_graph(self.G, P)
            Gc_l = self._transform_edge_cost(Gl, lengths)

            try:
                lengths, paths_dict = nx.single_source_dijkstra(Gc_l, "source", weight="weight")

                if "sink" not in paths_dict:
                    break

                new_path = paths_dict["sink"]
                cost_P.append(lengths["sink"])

                interlaced_path = self._interlace_paths(P, new_path)
                P.append(interlaced_path)

            except nx.NetworkXNoPath:
                break

        return P

    def process_tracks(self) -> List[sv.Detections]:
        """
        Run the tracking algorithm and assign track IDs to detections.

        Returns:
            List[sv.Detections]: Detections updated with tracker IDs.
        """
        self._build_graph(self.detection_buffer)
        disjoint_paths = self.ksp()
        return self._update_detections_with_tracks(assignments=disjoint_paths)
