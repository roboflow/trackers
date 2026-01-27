from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import supervision as sv
from tqdm.auto import tqdm


@dataclass(frozen=True)
class TrackNode:
    frame_id: int
    det_idx: int
    class_id: int
    position: tuple
    bbox: np.ndarray
    confidence: float

    def __hash__(self):
        return hash((self.frame_id, self.det_idx))

    def __eq__(self, other: Any):
        return isinstance(other, TrackNode) and (self.frame_id, self.det_idx) == (
            other.frame_id,
            other.det_idx,
        )

    def __str__(self):
        return f"{self.frame_id}:{self.det_idx}@{self.position}"


class KSPSolver:
    def __init__(
        self,
        path_overlap_penalty: float = 40,
        iou_weight: float = 0.9,
        dist_weight: float = 0.1,
        size_weight: float = 0.1,
        conf_weight: float = 0.1,
        entry_weight: float = 2.0,
        exit_weight: float = 2.0,
    ):
        self.path_overlap_penalty = (
            path_overlap_penalty if path_overlap_penalty is not None else 40
        )
        self.weight_key = "weight"
        self.source = "SOURCE"
        self.sink = "SINK"
        self.detection_per_frame: List[sv.Detections] = []
        self.weights = {"iou": 0.9, "dist": 0.1, "size": 0.1, "conf": 0.1}
        self.entry_weight = entry_weight
        self.exit_weight = exit_weight

        if path_overlap_penalty is not None:
            self.path_overlap_penalty = path_overlap_penalty
        if iou_weight is not None:
            self.weights["iou"] = iou_weight
        if dist_weight is not None:
            self.weights["dist"] = dist_weight
        if size_weight is not None:
            self.weights["size"] = size_weight
        if conf_weight is not None:
            self.weights["conf"] = conf_weight

        # Entry/exit region settings
        self.entry_exit_regions: List[
            Tuple[int, int, int, int]
        ] = []  # (x1, y1, x2, y2)

        # Border region settings
        self.use_border_regions = True
        self.active_borders: Set[str] = {"left", "right", "top", "bottom"}
        self.border_margin = 40
        self.frame_size = (1920, 1080)

        self.reset()

    def reset(self) -> None:
        """
        Reset the solver state, clearing all buffered detections and the graph.
        """
        self.detection_per_frame = []
        self.graph = nx.DiGraph()

    def append_frame(self, detections: sv.Detections) -> None:
        """
        Add detections for the current frame to the solver's buffer.

        Args:
            detections (sv.Detections): Detections for the current frame.
        """
        self.detection_per_frame.append(detections)

    def _get_center(self, bbox: np.ndarray) -> np.ndarray:
        """
        Compute the center (x, y) of a bounding box.

        Args:
            bbox (np.ndarray): Bounding box as [x1, y1, x2, y2].

        Returns:
            np.ndarray: Center coordinates as (x, y).
        """
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def set_entry_exit_regions(self, regions: List[Tuple[int, int, int, int]]) -> None:
        """
        Set rectangular entry/exit zones (x1, y1, x2, y2).

        Args:
            regions (List[Tuple[int, int, int, int]]): List of rectangular regions.
        """
        self.entry_exit_regions = regions

    def set_border_entry_exit(
        self,
        use_border: bool = True,
        borders: Optional[Set[str]] = None,
        margin: int = 40,
        frame_size: Tuple[int, int] = (1920, 1080),
    ) -> None:
        """
        Configure border-based entry/exit zones.

        Args:
            use_border (bool): Enable/disable border-based entry/exit.
            borders (Optional[Set[str]]): Set of borders to use.
            margin (int): Border thickness in pixels.
            frame_size (Tuple[int, int]): Size of the image (width, height).
        """
        self.use_border_regions = use_border
        self.active_borders = (
            borders if borders is not None else {"left", "right", "top", "bottom"}
        )
        self.border_margin = margin
        self.frame_size = frame_size

    def _in_door(self, node: TrackNode) -> bool:
        """
        Check if a node is inside any entry/exit region (rectangular or border).

        Args:
            node (TrackNode): The node to check.

        Returns:
            bool: True if in any entry/exit region, else False.
        """
        x, y = node.position

        # Check custom rectangular regions
        for x1, y1, x2, y2 in self.entry_exit_regions:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True

        # Check image border zones
        if self.use_border_regions:
            width, height = self.frame_size
            m = self.border_margin

            if "left" in self.active_borders and x <= m:
                return True
            if "right" in self.active_borders and x >= width - m:
                return True
            if "top" in self.active_borders and y <= m:
                return True
            if "bottom" in self.active_borders and y >= height - m:
                return True

        return False

    def _edge_cost(self, nodeU: TrackNode, nodeV: TrackNode) -> float:
        """
        Compute the cost of linking two detections (nodes) in the graph.

        Args:
            nodeU (TrackNode): Source node.
            nodeV (TrackNode): Destination node.

        Returns:
            float: Edge cost based on IoU, distance, size, and confidence weights.
        """
        bboxU, bboxV = nodeU.bbox, nodeV.bbox
        conf_u, conf_v = nodeU.confidence, nodeV.confidence

        center_dist = np.linalg.norm(self._get_center(bboxU) - self._get_center(bboxV))
        iou_penalty = 1 - sv.box_iou_batch(np.array([bboxU]), np.array([bboxV]))

        area_a = (bboxU[2] - bboxU[0]) * (bboxU[3] - bboxU[1])
        area_b = (bboxV[2] - bboxV[0]) * (bboxV[3] - bboxV[1])
        size_penalty = np.log(
            (max(area_a, area_b) / (min(area_a, area_b) + 1e-6)) + 1e-6
        )

        conf_penalty = 1 - min(conf_u, conf_v)

        return (
            self.weights["iou"] * iou_penalty
            + self.weights["dist"] * center_dist
            + self.weights["size"] * size_penalty
            + self.weights["conf"] * conf_penalty
        )

    def _build_graph(self):
        """
        Build the directed graph of detections for KSP computation.
        Nodes represent detections; edges represent possible associations.
        """
        G = nx.DiGraph()
        G.add_node(self.source)
        G.add_node(self.sink)

        node_frames = []

        for frame_id, detections in enumerate(self.detection_per_frame):
            frame_nodes = []
            for det_idx, bbox in enumerate(detections.xyxy):
                node = TrackNode(
                    frame_id=frame_id,
                    det_idx=det_idx,
                    class_id=int(detections.class_id[det_idx]),
                    position=tuple(self._get_center(bbox)),
                    bbox=bbox,
                    confidence=float(detections.confidence[det_idx]),
                )
                G.add_node(node)
                frame_nodes.append(node)
            node_frames.append(frame_nodes)

        for t in range(len(node_frames) - 1):
            for node_a in node_frames[t]:
                if self._in_door(node_a):
                    G.add_edge(self.source, node_a, weight=t * self.entry_weight)
                    G.add_edge(
                        node_a,
                        self.sink,
                        weight=(len(node_frames) - 1 - t) * self.exit_weight,
                    )

                for node_b in node_frames[t + 1]:
                    cost = self._edge_cost(node_a, node_b)
                    G.add_edge(node_a, node_b, weight=cost)

        for node in node_frames[0]:
            G.add_edge(self.source, node, weight=0.0)
        for node in node_frames[-1]:
            G.add_edge(node, self.sink, weight=0.0)

        self.graph = G

    def solve(self, k: Optional[int] = None) -> List[List[TrackNode]]:
        """
        Solve the K-Shortest Paths problem on the constructed detection graph.

        This method extracts up to k node-disjoint paths from the source to the sink in
        the detection graph, assigning each path as a unique object track. Edge reuse is
        penalized to encourage distinct tracks. The cost of each edge is determined by
        the configured weights for IoU, distance, size, and confidence.

        Args:
            k (Optional[int]): The number of tracks (paths) to extract. If None, uses
                the maximum number of detections in any frame as the default.

        Returns:
            List[List[TrackNode]]: A list of tracks, each track is a list of TrackNode
                objects representing the detections assigned to that track.
        """
        self._build_graph()

        G_base = self.graph.copy()
        edge_reuse: defaultdict[Tuple[Any, Any], int] = defaultdict(int)
        paths: List[List[TrackNode]] = []

        if k is None:
            k = max(len(f.xyxy) for f in self.detection_per_frame)

        for _i in tqdm(range(k), desc="Extracting k-shortest paths", leave=True):
            G_mod = G_base.copy()

            for u, v, data in G_mod.edges(data=True):
                base = data[self.weight_key]
                penalty = self.path_overlap_penalty * 1000 * edge_reuse[(u, v)] * base
                data[self.weight_key] = base + penalty

            try:
                _, path = nx.single_source_dijkstra(
                    G_mod, self.source, self.sink, weight=self.weight_key
                )
            except nx.NetworkXNoPath:
                print(f"No path found from source to sink at {_i}th iteration")
                break

            if path[1:-1] in paths:
                print("Duplicate path found!")
                continue

            paths.append(path[1:-1])

            for u, v in zip(path[:-1], path[1:]):
                edge_reuse[(u, v)] += 1

        return paths
