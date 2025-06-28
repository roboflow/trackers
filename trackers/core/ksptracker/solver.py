from dataclasses import dataclass
from collections import defaultdict
from typing import Any, List, Optional

import supervision as sv
import networkx as nx
import numpy as np


@dataclass(frozen=True)
class TrackNode:
    """
    Represents a detection node in the tracking graph.

    Attributes:
        frame_id (int): Frame index where detection occurred.
        det_idx (int): Detection index in the frame.
        class_id (int): Class ID of the detection.
        position (tuple): Center position of the detection.
        bbox (np.ndarray): Bounding box coordinates.
        confidence (float): Detection confidence score.
    """
    frame_id: int
    det_idx: int
    class_id: int
    position: tuple
    bbox: np.ndarray
    confidence: float

    def __hash__(self):
        return hash((self.frame_id, self.det_idx))

    def __eq__(self, other: Any):
        return (
            isinstance(other, TrackNode)
            and (self.frame_id, self.det_idx) == (other.frame_id, other.det_idx)
        )

    def __str__(self):
        return f"{self.frame_id}:{self.det_idx}@{self.position}"


class KSP_Solver:
    """
    Solver for the K-Shortest Paths (KSP) tracking problem.
    Builds a graph from detections and extracts multiple disjoint paths.
    """

    def __init__(self, base_penalty: float = 10.0, weight_key: str = "weight"):
        """
        Initialize the KSP_Solver.

        Args:
            base_penalty (float): Penalty for edge reuse in successive paths.
            weight_key (str): Edge attribute to use for weights.
        """
        self.base_penalty = base_penalty
        self.weight_key = weight_key
        self.source = "SOURCE"
        self.sink = "SINK"
        self.detection_per_frame: List[sv.Detections] = []
        self.reset()

    def reset(self):
        """
        Reset the solver state and clear all detections and graph.
        """
        self.detection_per_frame: List[sv.Detections] = []
        self.graph: nx.DiGraph = nx.DiGraph()

    def append_frame(self, detections: sv.Detections):
        """
        Add detections for a new frame.

        Args:
            detections (sv.Detections): Detections for the frame.
        """
        self.detection_per_frame.append(detections)

    def _get_center(self, bbox):
        """
        Compute the center of a bounding box.

        Args:
            bbox (np.ndarray): Bounding box coordinates.

        Returns:
            np.ndarray: Center coordinates.
        """
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def _iou(self, a, b):
        """
        Compute Intersection over Union (IoU) between two bounding boxes.

        Args:
            a (np.ndarray): First bounding box.
            b (np.ndarray): Second bounding box.

        Returns:
            float: IoU value.
        """
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _edge_cost(self, a, b, conf_a, conf_b, iou_w=0.5, dist_w=0.3, size_w=0.1, conf_w=0.1):
        """
        Compute the cost of connecting two detections.

        Args:
            a, b (np.ndarray): Bounding boxes.
            conf_a, conf_b (float): Detection confidences.
            iou_w, dist_w, size_w, conf_w (float): Weights for cost components.

        Returns:
            float: Edge cost.
        """
        center_dist = np.linalg.norm(self._get_center(a) - self._get_center(b))
        iou_penalty = 1 - self._iou(a, b)

        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        size_penalty = np.log((max(area_a, area_b) / (min(area_a, area_b) + 1e-6)) + 1e-6)

        conf_penalty = 1 - min(conf_a, conf_b)

        return iou_w * iou_penalty + dist_w * center_dist + size_w * size_penalty + conf_w * conf_penalty

    def _build_graph(self):
        """
        Build the tracking graph from all buffered detections.
        """
        G = nx.DiGraph()
        G.add_node(self.source)
        G.add_node(self.sink)

        node_frames: List[List[TrackNode]] = []

        for frame_id, detections in enumerate(self.detection_per_frame):
            frame_nodes = []
            for det_idx, bbox in enumerate(detections.xyxy):
                node = TrackNode(
                    frame_id=frame_id,
                    det_idx=det_idx,
                    class_id=int(detections.class_id[det_idx]),
                    position=tuple(self._get_center(bbox)),
                    bbox=bbox,
                    confidence=float(detections.confidence[det_idx])
                )
                G.add_node(node)
                frame_nodes.append(node)
            node_frames.append(frame_nodes)

        for t in range(len(node_frames) - 1):
            for node_a in node_frames[t]:
                for node_b in node_frames[t + 1]:
                    cost = self._edge_cost(node_a.bbox, node_b.bbox, node_a.confidence, node_b.confidence)
                    G.add_edge(node_a, node_b, weight=cost)

        for node in node_frames[0]:
            G.add_edge(self.source, node, weight=0.0)
        for node in node_frames[-1]:
            G.add_edge(node, self.sink, weight=0.0)

        self.graph = G

    def solve(self, k: Optional[int] = None) -> List[List[TrackNode]]:
        """
        Extract up to k node-disjoint shortest paths from the graph.

        Args:
            k (Optional[int]): Maximum number of paths to extract.

        Returns:
            List[List[TrackNode]]: List of node-disjoint paths (tracks).
        """
        self._build_graph()

        G_base = self.graph.copy()
        edge_reuse = defaultdict(int)
        paths: List[List[TrackNode]] = []

        if k is None:
            k = max(len(f.xyxy) for f in self.detection_per_frame)

        for _ in range(k):
            G_mod = G_base.copy()

            for u, v, data in G_mod.edges(data=True):
                base = data[self.weight_key]
                penalty = self.base_penalty * edge_reuse[(u, v)] * base
                data[self.weight_key] = base + penalty

            try:
                _, path = nx.single_source_dijkstra(G_mod, self.source, self.sink, weight=self.weight_key)
            except nx.NetworkXNoPath:
                break

            if path[1:-1] in paths:
                break

            paths.append(path[1:-1])

            for u, v in zip(path[:-1], path[1:]):
                edge_reuse[(u, v)] += 1

        return paths