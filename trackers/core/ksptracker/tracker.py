from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import supervision as sv

from trackers.core.base import BaseTracker

import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import cv2
import itertools
from copy import deepcopy
from pyvis.network import Network
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from trackers.core.ksptracker.InteractivePMapViewer import InteractivePMapViewer

import pprint
p = pprint.PrettyPrinter(4)
import copy

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
    bbox: Any
    confidence: float

    def __hash__(self) -> int:
        """Generate hash using frame and grid cell."""
        return hash((self.frame_id, self.grid_cell_id))

    def __eq__(self, other: Any) -> bool:
        """Compare nodes by frame and grid cell ID."""
        if not isinstance(other, TrackNode):
            return False
        return (self.frame_id, self.grid_cell_id) == (other.frame_id, other.grid_cell_id)
    
    def __str__(self):
        return f"{self.frame_id} {self.det_idx} {self.grid_cell_id}"

def compute_edge_cost(
    det_a_xyxy, det_b_xyxy, conf_a, conf_b, class_a, class_b,
    iou_weight=0.5, dist_weight=0.3, size_weight=0.1, conf_weight=0.1
):
    # Block if class doesn't match
    if class_a != class_b:
        return float('inf')

    # Get box centers
    def get_center(box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    center_a = get_center(det_a_xyxy)
    center_b = get_center(det_b_xyxy)
    euclidean_dist = np.linalg.norm(center_a - center_b)

    # IoU between boxes
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
    iou_penalty = 1 - iou  # lower IoU = higher cost

    # Size ratio penalty
    area_a = (det_a_xyxy[2] - det_a_xyxy[0]) * (det_a_xyxy[3] - det_a_xyxy[1])
    area_b = (det_b_xyxy[2] - det_b_xyxy[0]) * (det_b_xyxy[3] - det_b_xyxy[1])
    size_ratio = max(area_a, area_b) / (min(area_a, area_b) + 1e-5)
    size_penalty = np.log(size_ratio + 1e-5)  # higher penalty for large size change

    # Confidence penalty
    conf_penalty = 1 - min(conf_a, conf_b)

    # Weighted sum
    cost = (
        iou_weight * iou_penalty +
        dist_weight * euclidean_dist +
        size_weight * size_penalty +
        conf_weight * conf_penalty
    )
    return cost

def build_tracking_graph_with_source_sink(detections_per_frame: List):
    G = nx.DiGraph()
    node_data = {}

    # Add all detection nodes
    for frame_idx, detections in enumerate(detections_per_frame):
        for det_idx, bbox in enumerate(detections.xyxy):
            node_id = (frame_idx, det_idx)
            G.add_node(node_id)
            node_data[node_id] = {
                "bbox": bbox,
                "confidence": float(detections.confidence[det_idx]),
                "class": str(detections.data["class_name"][det_idx])
            }

    # Add edges between detections in consecutive frames
    for t in range(len(detections_per_frame) - 1):
        dets_a = detections_per_frame[t]
        dets_b = detections_per_frame[t + 1]

        for i, box_a in enumerate(dets_a.xyxy):
            for j, box_b in enumerate(dets_b.xyxy):
                node_a = (t, i)
                node_b = (t + 1, j)

                class_a = str(dets_a.data["class_name"][i])
                class_b = str(dets_b.data["class_name"][j])
                conf_a = float(dets_a.confidence[i])
                conf_b = float(dets_b.confidence[j])

                cost = compute_edge_cost(
                    box_a, box_b, conf_a, conf_b, class_a, class_b
                )

                if cost < float('inf'):
                    G.add_edge(node_a, node_b, weight=cost)

    # Add SOURCE and SINK nodes
    G.add_node("SOURCE")
    G.add_node("SINK")

    # Connect SOURCE to all detections in the first frame
    for det_idx in range(len(detections_per_frame[0].xyxy)):
        G.add_edge("SOURCE", (0, det_idx), weight=0)

    # Connect all detections in the last frame to SINK
    last_frame = len(detections_per_frame) - 1
    for det_idx in range(len(detections_per_frame[-1].xyxy)):
        G.add_edge((last_frame, det_idx), "SINK", weight=0)

    return G, node_data

def visualize_tracking_graph_with_path_pyvis(
    G, node_data, path=None, output_file="graph.html"
):
    net = Network(height="800px", width="100%", directed=True)
    net.toggle_physics(False)
    spacing_x = 300
    spacing_y = 50

    path_edges = set(zip(path, path[1:])) if path else set()

    # Collect frames and assign vertical positions
    frame_positions = {}
    for node in G.nodes():
        if isinstance(node, tuple) and len(node) == 2:
            frame, det_idx = node
            frame_positions.setdefault(frame, []).append(det_idx)
    for frame in frame_positions:
        frame_positions[frame].sort()

    frames = list(frame_positions.keys())
    max_frame = max(frames) if frames else 0

    for node in G.nodes():
        if node == "SOURCE":
            x, y = -spacing_x, 0
            label = "SOURCE"
            color = "green"
            title = "Source node"
        elif node == "SINK":
            x, y = spacing_x * (max_frame + 1), 0
            label = "SINK"
            color = "red"
            title = "Sink node"
        else:
            frame, det_idx = node
            x = spacing_x * frame
            y = spacing_y * frame_positions[frame].index(det_idx)
            label = f"{frame}:{det_idx}"
            color = "orange" if path and node in path else "lightgray"
            data = node_data.get(node, {})
            title = f"Frame: {frame}<br>ID: {det_idx}"
            if "confidence" in data:
                title += f"<br>Conf: {data['confidence']:.2f}"
            if "class" in data:
                title += f"<br>Class: {data['class']}"
        net.add_node(
            str(node), label=label, color=color, x=x, y=y,
            physics=False, title=title
        )

    for u, v, data in G.edges(data=True):
        u_id, v_id = str(u), str(v)
        color = "orange" if (u, v) in path_edges else "gray"
        width = 3 if (u, v) in path_edges else 1
        weight = float(data.get("weight", 0))
        label = f"{weight:.2f}"
        net.add_edge(u_id, v_id, color=color, width=width, label=label)

    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 14
        }
      },
      "edges": {
        "font": {
          "size": 10,
          "align": "middle"
        },
        "smooth": false
      },
      "physics": {
        "enabled": false
      }
    }
    """)

    net.write_html(output_file, notebook=False, open_browser=True)

def assign_tracker_ids_from_paths(paths, node_data):
    """
    Converts paths and node data into frame-wise sv.Detections with tracker IDs assigned.
    
    Args:
        paths (list of lists): Each inner list is a tracking path (list of nodes).
        node_data (dict): Maps node tuples (frame, det_idx) to dicts containing detection info,
                          e.g. 'xyxy', 'confidence', 'class_id', etc.
    
    Returns:
        dict: frame_index -> sv.Detections for that frame with tracker_id assigned.
    """
    # Normalize input: if a single path, wrap into a list
    if isinstance(paths, list) and len(paths) > 0 and not isinstance(paths[0], list):
        paths = [paths]

    frame_to_raw_dets = {}

    for tracker_id, path in enumerate(paths, start=1):
        for node in path:
            if node in ("SOURCE", "SINK"):
                continue
            try:
                frame, det_idx = node
            except Exception:
                continue

            det_info = node_data.get(node)
            if det_info is None:
                continue
            
            # Store detection info per frame
            frame_to_raw_dets.setdefault(frame, []).append({
                "xyxy": det_info["bbox"],
                "confidence": det_info["confidence"],
                "class_id": 0,
                "tracker_id": tracker_id,
            })

    # Now convert each frame detections to sv.Detections objects
    frame_to_detections = {}

    for frame, dets_list in frame_to_raw_dets.items():
        xyxy = np.array([d["xyxy"] for d in dets_list], dtype=np.float32)
        confidence = np.array([d["confidence"] for d in dets_list], dtype=np.float32)
        class_id = np.array([d["class_id"] for d in dets_list], dtype=int)
        tracker_id = np.array([d["tracker_id"] for d in dets_list], dtype=int)

        # Construct sv.Detections with tracker_id as an attribute
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
        )
        frame_to_detections[frame] = detections

    return frame_to_detections

def extend_graph(
    G: nx.DiGraph, path: list, weight_key="weight", discourage_weight=1e6
):
    """
    Given a path, extend the graph by:
    - Splitting each node (except SOURCE/SINK) into v_in and v_out
    - Redirecting incoming edges to v_in and outgoing edges from v_out
    - Adding a zero-weight edge between v_in and v_out
    - Reversing the used path and adding high-weight edges to discourage reuse
    """
    extended_G = nx.DiGraph()
    extended_G.add_nodes_from(G.nodes(data=True))
    
    # Add all original edges
    for u, v, data in G.edges(data=True):
        extended_G.add_edge(u, v, **data)

    for node in path:
        if node in ("SOURCE", "SINK"):
            continue

        node_in = f"{node}_in"
        node_out = f"{node}_out"

        # Split node
        extended_G.add_node(node_in, **G.nodes[node])
        extended_G.add_node(node_out, **G.nodes[node])
        extended_G.add_edge(node_in, node_out, **{weight_key: 0.0})

        # Rewire incoming edges to node_in
        for pred in list(G.predecessors(node)):
            extended_G.add_edge(pred, node_in, **G.edges[pred, node])

        # Rewire outgoing edges from node_out
        for succ in list(G.successors(node)):
            extended_G.add_edge(node_out, succ, **G.edges[node, succ])

        # Remove original node edges
        extended_G.remove_node(node)

    # Discourage reusing this path by reversing it and setting high weights
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if u in ("SOURCE", "SINK") or v in ("SOURCE", "SINK"):
            continue

        u_out = f"{u}_out" if f"{u}_out" in extended_G else u
        v_in = f"{v}_in" if f"{v}_in" in extended_G else v

        # Reverse with discourage weight
        extended_G.add_edge(v_in, u_out, **{weight_key: discourage_weight})

    return extended_G

def greedy_disjoint_paths_with_extension(
    G: nx.DiGraph, node_data, source="SOURCE", sink="SINK", max_paths=10000
) -> list:
    """
    Extract disjoint paths using graph extension and discourage reuse.
    """
    G = copy.deepcopy(G)  # Don't mutate original
    paths = []

    for _ in range(max_paths):
        try:
            path = nx.shortest_path(G, source=source, target=sink, weight="weight")
            paths.append(path)
            G = extend_graph(G, path)  # Extend to discourage reuse, preserve structure
        except nx.NetworkXNoPath:
            print("no path")
            
            break
    return paths

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
        grid_size: int = 25,
        max_paths: int = 20,
        min_confidence: float = 0.3,
        max_distance: float = 0.3,
        img_dim: tuple = (512, 512),
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
        self.img_dim = img_dim
        self.frames = []
        self.G = nx.DiGraph()
        self.reset()

    def set_image_dim(self, dim: tuple):
        self.img_dim = dim

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
        
    def process_tracks(self) -> List[sv.Detections]:
        G, node_data = build_tracking_graph_with_source_sink(self.detection_buffer)
        paths = greedy_disjoint_paths_with_extension(G, node_data)
        print(len(paths))

        if not paths:
            print("No valid paths found.")
            return []

        return assign_tracker_ids_from_paths(paths, node_data)
        # self._build_graph(self.detection_buffer)
        # # visualize_tracking_graph_debug(self.G)
        # # detections_list  = find_and_visualize_disjoint_paths(self.G)
        # # return detections_list
