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
        min_confidence (float): Minimum detection confidence
        max_distance (float): Max dissimilarity (1 - IoU) allowed
    """

    def __init__(
        self,
        grid_size: int = 20,
        max_paths: int = 20,
        min_confidence: float = 0.3,
        max_distance: float = 0.3,
    ) -> None:
        """
        Initialize KSP tracker with config parameters.

        Args:
            grid_size (int): Pixel size of each grid cell
            min_confidence (float): Min detection confidence
            max_distance (float): Max allowed dissimilarity
        """
        self.grid_size = grid_size
        self.min_confidence = min_confidence
        self.max_distance = max_distance
        self.max_paths = max_paths
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
    
    def _edge_cost(self, confidence: float):
        return -np.log( confidence / ((1 - confidence) + 1e6))

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
                    G.add_edge("source", node, weight=0)
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
                            node,
                            node_next,
                            weight=self._edge_cost(confidence=node.confidence),
                        )
        print(f"Number of detections in frame 0: {len(node_dict.get(0, []))}")
        print(f"Number of detections in last frame: {len(node_dict.get(len(all_detections) - 1, []))}")

        print("Edges from source:")
        for u, v in G.edges("source"):
            print(f"source -> {v}")

        print("Edges to sink:")
        for u, v in G.in_edges("sink"):
            print(f"{u} -> sink")

        for i in range(len(all_detections) - 1):
            edges_between = 0
            for node in node_dict[i]:
                for node_next in node_dict[i + 1]:
                    if G.has_edge(node, node_next):
                        edges_between += 1
            print(f"Edges between frame {i} and {i+1}: {edges_between}")


        return G

    def _update_detections_with_tracks(
        self, assignments: List[List[TrackNode]]
    ) -> List[sv.Detections]:
        """
        Assign track IDs to detections.

        Args:
            assignments (Dict): Paths from KSP with track IDs

        Returns:
            List[sv.Detections]: Merged detections with tracker IDs
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

    def _shortest_path(self, G: nx.DiGraph) -> tuple:
        """
        Compute the shortest path from 'source' to 'sink' using Bellman-Ford.

        Args:
            G (nx.DiGraph): Graph with possible negative edge weights.

        Returns:
            tuple: (path, total_cost, lengths) where path is a list of nodes, 
                total_cost is the path weight, and lengths is a dict of all
                shortest distances from source.

        Raises:
            RuntimeError: If a negative weight cycle is detected.
            KeyError: If 'sink' is not reachable.
        """
        try:
            lengths, paths = nx.single_source_bellman_ford(G, "source", weight="weight")
            if "sink" not in paths:
                raise KeyError("No path found from source to sink.")
            return paths["sink"], lengths["sink"], lengths
        except nx.NetworkXUnbounded:
            raise RuntimeError("Graph contains a negative weight cycle.")

    def _extend_graph(self, G: nx.DiGraph, paths: List[List[TrackNode]]) -> nx.DiGraph:
        """
        Extend the graph by removing previously used nodes.

        Args:
            G (nx.DiGraph): Original detection graph.
            paths (List[List[TrackNode]]): Found disjoint paths.

        Returns:
            nx.DiGraph: Modified graph with used nodes removed.
        """
        G_extended = G.copy()
        for path in paths:
            for node in path:
                if node in G_extended:
                    if node in G_extended and node not in {"source", "sink"}:
                        G_extended.remove_node(node)
        return G_extended

    def _transform_edge_cost(self, G: nx.DiGraph, shortest_costs: Dict[Any, float]) -> nx.DiGraph:
        """
        Transform edge weights to non-negative using cost shifting.

        Args:
            G (nx.DiGraph): Graph with possibly negative weights.
            shortest_costs (dict): Shortest distances from source.

        Returns:
            nx.DiGraph: Cost-shifted graph.
        """
        Gc = nx.DiGraph()
        for u, v, data in G.edges(data=True):
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
        Filter out nodes from the new path that conflict with existing paths.

        Args:
            current_paths (List[List[TrackNode]]): Existing disjoint paths.
            new_path (List[TrackNode]): New shortest path candidate.

        Returns:
            List[TrackNode]: Cleaned, non-conflicting path.
        """
        used_nodes = set()
        for path in current_paths:
            for node in path:
                if isinstance(node, TrackNode):  # Skip 'source'/'sink' nodes
                    used_nodes.add((node.frame_id, node.grid_cell_id))

        interlaced = []
        for node in new_path:
            if isinstance(node, TrackNode):
                key = (node.frame_id, node.grid_cell_id)
                if key not in used_nodes:
                    interlaced.append(node)

        return interlaced



    def ksp(self, G: nx.DiGraph) -> List[List[TrackNode]]:
        """
        Find multiple disjoint shortest paths.

        Args:
            G (nx.DiGraph): Detection graph.

        Returns:
            List[List[TrackNode]]: List of disjoint detection paths.
        """
        path, cost, lengths = self._shortest_path(G)
        P = [path]
        cost_P = [cost]

        for l in range(1, self.max_paths):
            if l != 1 and cost_P[-1] >= cost_P[-2]:
                return P  # early termination

            Gl = self._extend_graph(G, P)
            Gc_l = self._transform_edge_cost(Gl, lengths)

            try:
                dkslengths, paths_dict = nx.single_source_dijkstra(Gc_l, "source", weight="weight")

                if "sink" not in paths_dict:
                    break

                new_path = paths_dict["sink"]
                cost_P.append(lengths["sink"])
                print(new_path)
                interlaced_path = self._interlace_paths(P, new_path)
                P.append(interlaced_path)

                lengths = dkslengths
            except nx.NetworkXNoPath:
                break

        return P

    def process_tracks(self) -> List[sv.Detections]:
        """
        Run tracker and assign detections to tracks.

        Returns:
            List[sv.Detections]: Final detections with track IDs
        """
        graph = self._build_graph(self.detection_buffer)
        disjoint_paths = self.ksp(graph)
        return self._update_detections_with_tracks(assignments=disjoint_paths)
