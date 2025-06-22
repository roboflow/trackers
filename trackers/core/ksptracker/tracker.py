from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import supervision as sv

from trackers.core.base import BaseTracker

import matplotlib.pyplot as plt
import networkx as nx

import itertools


def visualize_tracking_graph_debug(G: nx.DiGraph, max_edges=500):
    import matplotlib.pyplot as plt
    import networkx as nx

    plt.figure(figsize=(18, 8))

    # Collect all TrackNode nodes
    track_nodes = [n for n in G.nodes if isinstance(n, TrackNode)]
    frames = sorted(set(n.frame_id for n in track_nodes))

    frame_to_x = {f: i for i, f in enumerate(frames)}

    # Group nodes by frame and sort by det_idx (or confidence)
    nodes_by_frame = {}
    for node in track_nodes:
        nodes_by_frame.setdefault(node.frame_id, []).append(node)
    for frame in nodes_by_frame:
        nodes_by_frame[frame].sort(key=lambda n: n.det_idx)  # or key=lambda n: -n.confidence for sorting by confidence

    pos = {}
    for node in G.nodes:
        if node == "source":
            pos[node] = (-1, 0)
        elif node == "sink":
            pos[node] = (len(frames), 0)
        elif isinstance(node, TrackNode):
            x = frame_to_x[node.frame_id]
            # vertical spacing: spread nodes evenly in y-axis
            idx = nodes_by_frame[node.frame_id].index(node)
            total = len(nodes_by_frame[node.frame_id])
            # spread vertically between 0 and total, centered around 0
            y = idx - total / 2
            pos[node] = (x, y)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color="lightblue", alpha=0.9)

    # Labels
    labels = {}
    for node in G.nodes:
        if node == "source":
            labels[node] = "SRC"
        elif node == "sink":
            labels[node] = "SNK"
        elif isinstance(node, TrackNode):
            labels[node] = f"F{node.frame_id},D{node.det_idx}"
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Edges (limit number)
    edges_to_draw = list(G.edges(data=True))[:max_edges]
    edge_list = [(u, v) for u, v, _ in edges_to_draw]
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, arrowstyle='->', arrowsize=15, edge_color='gray')

    # Edge weights
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges_to_draw}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5)

    plt.title("Tracking Graph - Directed Timeline Layout with Vertical Spacing")
    plt.xlabel("Frame Index")
    plt.ylabel("Detection Vertical Position (Jittered)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_all_shortest_bellman_ford_paths(G: nx.DiGraph):
    try:
        all_paths = list(nx.all_shortest_paths(G, source="source", target="sink", weight="weight", method="bellman-ford"))
        if not all_paths:
            print("No path found from source to sink.")
            return
        shortest_path_length = sum(G[u][v]['weight'] for u, v in zip(all_paths[0][:-1], all_paths[0][1:]))
        print(f"Number of shortest paths from source to sink: {len(all_paths)}")
        print(f"Each shortest path has cost: {shortest_path_length:.2f}")

    except nx.NetworkXUnbounded:
        print("Negative weight cycle detected.")
        return

    plt.figure(figsize=(18, 8))

    # Layout
    track_nodes = [n for n in G.nodes if isinstance(n, TrackNode)]
    frames = sorted(set(n.frame_id for n in track_nodes))
    frame_to_x = {f: i for i, f in enumerate(frames)}

    nodes_by_frame = {}
    for node in track_nodes:
        nodes_by_frame.setdefault(node.frame_id, []).append(node)
    for frame in nodes_by_frame:
        nodes_by_frame[frame].sort(key=lambda n: n.det_idx)

    pos = {}
    for node in G.nodes:
        if node == "source":
            pos[node] = (-1, 0)
        elif node == "sink":
            pos[node] = (len(frames), 0)
        elif isinstance(node, TrackNode):
            x = frame_to_x[node.frame_id]
            idx = nodes_by_frame[node.frame_id].index(node)
            total = len(nodes_by_frame[node.frame_id])
            y = idx - total / 2
            pos[node] = (x, y)

    # Draw all nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color="lightgray", alpha=0.7)
    labels = {
        node: (
            "SRC" if node == "source"
            else "SNK" if node == "sink"
            else f"F{node.frame_id},D{node.det_idx}"
        )
        for node in G.nodes
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Draw all edges (light gray)
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", alpha=0.3)

    # Draw edge weights for all edges
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # Collect all edges in all shortest paths
    edges_in_paths = set()
    nodes_in_paths = set()
    for path in all_paths:
        nodes_in_paths.update(path)
        edges_in_paths.update(zip(path[:-1], path[1:]))

    # Draw all nodes in shortest paths highlighted (orange)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_paths, node_color="orange")

    # Draw all edges in shortest paths highlighted (red, thicker)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_in_paths,
        edge_color="red",
        width=2.5,
        arrowstyle="->",
        arrowsize=15,
        label="Shortest Paths"
    )

    plt.title("Bellman-Ford All Shortest Paths Visualization")
    plt.xlabel("Frame Index")
    plt.ylabel("Detection Index Offset")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_path(G, path, pos, path_num, color="red"):
    plt.figure(figsize=(12, 6))

    # Draw all nodes and edges lightly
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color="lightgray", alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", alpha=0.3)
    labels = {
        node: (
            "SRC" if node == "source"
            else "SNK" if node == "sink"
            else f"F{node.frame_id},D{node.det_idx}"
        )
        for node in G.nodes
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Highlight the path nodes and edges
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=color)
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=color, width=3, arrowstyle="->", arrowsize=15)

    plt.title(f"Node-Disjoint Shortest Path #{path_num}")
    plt.xlabel("Frame Index")
    plt.ylabel("Detection Index Offset")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def path_to_detections(path) -> sv.Detections:
    track_nodes = [node for node in path if hasattr(node, "frame_id") and hasattr(node, "det_idx")]

    xyxys = []
    confidences = []
    class_ids = []

    for node in track_nodes:
        det_idx = node.det_idx
        dets = node.dets  # sv.Detections

        bbox = dets.xyxy[det_idx]
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"Invalid bbox at node {node}: {bbox}")

        xyxys.append(bbox)
        confidences.append(dets.confidence[det_idx])

        if hasattr(dets, "class_id") and len(dets.class_id) > det_idx:
            class_ids.append(dets.class_id[det_idx])
        else:
            class_ids.append(0)

    xyxys = np.array(xyxys)
    confidences = np.array(confidences)
    class_ids = np.array(class_ids)

    if xyxys.ndim != 2 or xyxys.shape[1] != 4:
        raise ValueError(f"xyxy must be 2D array with shape (_,4), got shape {xyxys.shape}")

    return sv.Detections(xyxy=xyxys, confidence=confidences, class_id=class_ids)


def find_and_visualize_disjoint_paths(G_orig, source="source", sink="sink", weight="weight") -> List[sv.Detections]:
    G = G_orig.copy()

    # Compute layout positions once for consistency
    track_nodes = [n for n in G.nodes if hasattr(n, "frame_id") and hasattr(n, "det_idx")]
    frames = sorted(set(n.frame_id for n in track_nodes))
    frame_to_x = {f: i for i, f in enumerate(frames)}
    nodes_by_frame = {}
    for node in track_nodes:
        nodes_by_frame.setdefault(node.frame_id, []).append(node)
    for frame in nodes_by_frame:
        nodes_by_frame[frame].sort(key=lambda n: n.det_idx)
    pos = {}
    for node in G.nodes:
        if node == source:
            pos[node] = (-1, 0)
        elif node == sink:
            pos[node] = (len(frames), 0)
        elif hasattr(node, "frame_id") and hasattr(node, "det_idx"):
            x = frame_to_x[node.frame_id]
            idx = nodes_by_frame[node.frame_id].index(node)
            total = len(nodes_by_frame[node.frame_id])
            y = idx - total / 2
            pos[node] = (x, y)
        else:
            pos[node] = (0, 0)  # fallback

    all_detections = []
    colors = itertools.cycle(["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"])

    while True:
        try:
            length, paths = nx.single_source_bellman_ford(G, source=source, weight=weight)
            if sink not in paths:
                print("No more paths found.")
                break

            shortest_path = paths[sink]
            cost = length[sink]
            color = next(colors)
            print(f"Found path with cost {cost:.2f}: {shortest_path}")

            visualize_path(G_orig, shortest_path, pos, len(all_detections) + 1, color=color)

            dets = path_to_detections(shortest_path)
            all_detections.append(dets)

            # Remove intermediate nodes to enforce node-disjointness
            intermediate_nodes = shortest_path[1:-1]
            G.remove_nodes_from(intermediate_nodes)

        except nx.NetworkXNoPath:
            print("No more paths found.")
            break
        except nx.NetworkXUnbounded:
            print("Negative weight cycle detected.")
            break

    print(f"Total node-disjoint shortest paths found: {len(all_detections)}")
    return all_detections

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
    det_idx: int
    position: tuple
    confidence: float
    dets: Any

    def __hash__(self) -> int:
        """Generate hash using frame and grid cell."""
        return hash((self.frame_id, self.grid_cell_id))

    def __eq__(self, other: Any) -> bool:
        """Compare nodes by frame and grid cell ID."""
        if not isinstance(other, TrackNode):
            return False
        return (self.frame_id, self.grid_cell_id) == (other.frame_id, other.grid_cell_id)
    
    def __str__(self):
        return str(self.frame_id) + " " + str(self.det_idx)


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

    def _edge_cost(self, confidence: float, dist: float) -> float:
        """
        Compute edge cost from detection confidence.

        Args:
            confidence (float): Detection confidence score.

        Returns:
            float: Edge cost for KSP (non-negative after transform).
        """
        return -np.log(confidence)

    def _build_graph(self, all_detections: List[sv.Detections]) -> None:
        """
        Build a directed graph from buffered detections.

        Args:
            all_detections (List[sv.Detections]): List of detections per frame.

        Side Effects:
            Sets self.G to the constructed graph.
            Populates self.node_to_detection mapping.
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
                    det_idx=det_idx,
                    position=pos,
                    dets=dets,
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

                    self.G.add_edge(
                            node,
                            node_next,
                            weight=self._edge_cost(confidence=node.confidence, dist=dist),
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

        for frame_idx, dets in enumerate(all_detections):
            tracker_ids = dets.tracker_id
            num_tracked = np.sum(tracker_ids != -1)
            print(f"[Frame {frame_idx}] Total Detections: {len(tracker_ids)} | Tracked: {num_tracked}")

            for det_idx, tid in enumerate(tracker_ids):
                if tid == -1:
                    print(f"  ⛔ Untracked Detection {det_idx}: BBox={dets.xyxy[det_idx]}, Conf={dets.confidence[det_idx]:.2f}")
                else:
                    print(f"  ✅ Track {tid} assigned to Detection {det_idx}: BBox={dets.xyxy[det_idx]}, Conf={dets.confidence[det_idx]:.2f}")


        return all_detections

    def _shortest_path(self) -> tuple:
        """
        Compute shortest path from 'source' to 'sink' using Bellman-Ford.

        Returns:
            tuple: (path, total_cost, lengths)

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
        self, G: nx.DiGraph, shortest_costs: Dict[Any, float]
    ) -> nx.DiGraph:
        """
        Apply cost transformation to ensure non-negative edge weights.

        Args:
            G (nx.DiGraph): Graph with possibly negative weights.
            shortest_costs (dict): Shortest path distances from source.

        Returns:
            nx.DiGraph: Cost-transformed graph.
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

        Returns:
            List[List[TrackNode]]: List of disjoint detection paths.
        """
        path, cost, lengths = self._shortest_path()
        P = [path]
        cost_P = [cost]

        for l in range(1, self.max_paths):
            if l != 1 and cost_P[-1] >= cost_P[-2]:
                return P  # early termination

            Gl = self._extend_graph(P)
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
        visualize_tracking_graph_debug(self.G)
        detections_list  = find_and_visualize_disjoint_paths(self.G)
        return detections_list 
        # disjoint_paths = self.ksp()
        # return self._update_detections_with_tracks(assignments=disjoint_paths)
