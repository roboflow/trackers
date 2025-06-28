from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import supervision as sv

from trackers.core.base import BaseTracker

import pprint
p = pprint.PrettyPrinter()

@dataclass(frozen=True)
class TrackNode:
    """
    Represents a detection node in the tracking graph.
    """
    frame_id: int
    grid_cell_id: int
    det_idx: int
    position: tuple
    bbox: Any
    confidence: float

    def __hash__(self) -> int:
        return hash((self.frame_id, self.det_idx))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TrackNode):
            return False
        return (self.frame_id, self.det_idx) == (other.frame_id, other.det_idx)

    def __str__(self):
        return f"{self.frame_id} {self.det_idx} {self.grid_cell_id}"

def compute_edge_cost(
    det_a_xyxy, det_b_xyxy, conf_a, conf_b, class_a, class_b,
    iou_weight=0.5, dist_weight=0.3, size_weight=0.1, conf_weight=0.1
):
    print("TEST")
    print(det_a_xyxy)
    def get_center(box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    center_a = get_center(det_a_xyxy)
    center_b = get_center(det_b_xyxy)
    euclidean_dist = np.linalg.norm(center_a - center_b)

    def box_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union_area = boxA_area + boxB_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    iou = box_iou(det_a_xyxy, det_b_xyxy)
    iou_penalty = 1 - iou

    area_a = (det_a_xyxy[2] - det_a_xyxy[0]) * (det_a_xyxy[3] - det_a_xyxy[1])
    area_b = (det_b_xyxy[2] - det_b_xyxy[0]) * (det_b_xyxy[3] - det_b_xyxy[1])
    size_ratio = max(area_a, area_b) / (min(area_a, area_b) + 1e-5)
    size_penalty = np.log(size_ratio + 1e-5)

    conf_penalty = 1 - min(conf_a, conf_b)

    cost = (
        iou_weight * iou_penalty +
        dist_weight * euclidean_dist +
        size_weight * size_penalty +
        conf_weight * conf_penalty
    )
    return cost

def build_tracking_graph_with_source_sink(detections_per_frame: List[sv.Detections]):
    G = nx.DiGraph()
    node_list_per_frame = []
    print("SDFSDFSD")
    # Add all detection nodes
    for frame_idx, detections in enumerate(detections_per_frame):
        frame_nodes = []
        for det_idx, bbox in enumerate(detections.xyxy):
            position = (
                float(bbox[0] + bbox[2]) / 2.0,
                float(bbox[1] + bbox[3]) / 2.0
            )
            node = TrackNode(
                frame_id=frame_idx,
                det_idx=det_idx,
                position=position,
                bbox=bbox,
                confidence=float(detections.confidence[det_idx]),
                grid_cell_id=None
            )
            G.add_node(node)
            frame_nodes.append(node)
        node_list_per_frame.append(frame_nodes)

    # Add edges between detections in consecutive frames
    for t in range(len(node_list_per_frame) - 1):
        nodes_a = node_list_per_frame[t]
        nodes_b = node_list_per_frame[t + 1]
        dets_a = detections_per_frame[t]
        dets_b = detections_per_frame[t + 1]

        for i, node_a in enumerate(nodes_a):
            for j, node_b in enumerate(nodes_b):
                class_a = str(dets_a.data["class_name"][i])
                class_b = str(dets_b.data["class_name"][j])
                conf_a = float(dets_a.confidence[i])
                conf_b = float(dets_b.confidence[j])

                cost = compute_edge_cost(
                    node_a.bbox, node_b.bbox, conf_a, conf_b, class_a, class_b
                )
                if cost < float('inf'):
                    G.add_edge(node_a, node_b, weight=cost)

    # Add SOURCE and SINK nodes
    G.add_node("SOURCE")
    G.add_node("SINK")

    # Connect SOURCE to all detections in the first frame
    for node in node_list_per_frame[0]:
        G.add_edge("SOURCE", node, weight=0)

    # Connect all detections in the last frame to SINK
    for node in node_list_per_frame[-1]:
        G.add_edge(node, "SINK", weight=0)

    return G

def assign_tracker_ids_from_paths(paths: List[List[TrackNode]], num_frames: int) -> Dict[int, sv.Detections]:
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
                    "class_id": 0,
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

def path_cost(G: nx.DiGraph, path: list, weight_key="weight") -> float:
    return sum(G[u][v][weight_key] for u, v in zip(path[:-1], path[1:]))

# ------------- Main Function ----------------
from collections import defaultdict

def discourage_path_edges(G, path, weight_key="weight", penalty=1e6):
    """
    Add penalty to edges along the given path to discourage reuse.
    """
    for u, v in zip(path[:-1], path[1:]):
        if G.has_edge(u, v):
            G[u][v][weight_key] += penalty

def iterative_k_shortest_paths_soft_penalty(
    G: nx.DiGraph,
    source="SOURCE",
    sink="SINK",
    k=5,
    weight_key="weight",
    base_penalty=10.0,  # smaller penalty
):
    G_base = G.copy()
    paths = []
    edge_reuse_count = defaultdict(int)

    for iteration in range(k):
        G_mod = G_base.copy()

        # Increase edge weights softly according to reuse count
        for (u, v, data) in G_mod.edges(data=True):
            base_cost = data[weight_key]
            reuse_pen = base_penalty * edge_reuse_count[(u, v)] * base_cost
            data[weight_key] = base_cost + reuse_pen

        try:
            length, path = nx.single_source_dijkstra(G_mod, source, sink, weight=weight_key)
        except nx.NetworkXNoPath:
            print(f"No more paths found after {len(paths)} iterations.")
            break

        if path in paths:
            print("Duplicate path found, stopping early.")
            break

        print(f"Found path {iteration + 1} with cost {length}")
        paths.append(path)

        # Update reuse counts for edges in this path
        for u, v in zip(path[:-1], path[1:]):
            edge_reuse_count[(u, v)] += 1

    return paths
class KSPTracker(BaseTracker):
    """
    Offline tracker using K-Shortest Paths (KSP) algorithm.
    """

    def __init__(
        self,
        grid_size: int = 25,
        max_paths: int = 20,
        min_confidence: float = 0.3,
        max_distance: float = 0.3,
        img_dim: tuple = (512, 512),
    ) -> None:
        self.grid_size = grid_size
        self.min_confidence = min_confidence
        self.max_distance = max_distance
        self.max_paths = max_paths
        self.img_dim = img_dim
        self.frames = []
        self.G = nx.DiGraph()
        self.reset()

    def set_image_dim(self, dim: tuple):
        self.img_dim = dim

    def reset(self) -> None:
        self.detection_buffer: List[sv.Detections] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        self.detection_buffer.append(detections)
        return detections

    def process_tracks(self) -> Dict[int, sv.Detections]:
        max_detection_obj = len(max(self.detection_buffer, key=lambda d: len(d.xyxy)))

        print(f"Number of detections: {max_detection_obj}")
        G = build_tracking_graph_with_source_sink(self.detection_buffer)
        paths = iterative_k_shortest_paths_soft_penalty(G, source="SOURCE", sink="SINK", k=max_detection_obj)
        print(f"Extracted {len(paths)} tracks.")
        for i in paths:
           p.pprint(i)
        # print(paths[0] == paths[1])
        # for path in paths:
        #     print(path)
        #     print([p.position for p in path[1:-1]])
        if not paths:
            print("No valid paths found.")
            return {}
        return assign_tracker_ids_from_paths(paths, num_frames=len(self.detection_buffer))
