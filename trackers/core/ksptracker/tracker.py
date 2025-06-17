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
        detection_id (int): Detection index within the frame
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
        confidence (float): Detection confidence score
    """

    frame_id: int
    detection_id: int
    bbox: tuple
    confidence: float

    def __hash__(self) -> int:
        """Generates hash based on frame_id and detection_id.

        Returns:
            int: Hash value of the node
        """
        return hash((self.frame_id, self.detection_id))

    def __eq__(self, other: Any) -> bool:
        """Compares equality based on frame_id and detection_id.

        Args:
            other (Any): Object to compare with

        Returns:
            bool: True if nodes are equal, False otherwise
        """
        if not isinstance(other, TrackNode):
            return False
        return (self.frame_id, self.detection_id) == (
            other.frame_id,
            other.detection_id,
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
        max_gap: int = 30,
        min_confidence: float = 0.3,
        max_paths: int = 1000,
        max_distance: float = 0.3,
    ) -> None:
        """Initialize KSP tracker with configuration parameters.

        Args:
            max_gap (int): Max frame gap between connected detections
            min_confidence (float): Minimum detection confidence
            max_paths (int): Maximum number of paths to find
            max_distance (float): Max dissimilarity (1-IoU) for connections
        """
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

    def _calc_iou(
        self, bbox1: Union[np.ndarray, tuple], bbox2: Union[np.ndarray, tuple]
    ) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1 (Union[np.ndarray, tuple]): First bounding box (x1, y1, x2, y2)
            bbox2 (Union[np.ndarray, tuple]): Second bounding box (x1, y1, x2, y2)

        Returns:
            float: IoU value between 0.0 and 1.0
        """
        bbox1 = np.array(bbox1)
        bbox2 = np.array(bbox2)

        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter_area + 1e-5  # epsilon to avoid div by 0

        return inter_area / union

    def _can_connect_nodes(self, node1: TrackNode, node2: TrackNode) -> bool:
        """Determine if two nodes can be connected based on IoU threshold.

        Args:
            node1 (TrackNode): First track node
            node2 (TrackNode): Second track node

        Returns:
            bool: True if nodes can be connected, False otherwise
        """
        if node2.frame_id <= node1.frame_id:
            return False
        if node2.frame_id - node1.frame_id > self.max_gap:
            return False
        iou = self._calc_iou(node1.bbox, node2.bbox)
        return iou >= (1 - self.max_distance)

    def _edge_cost(self, node1: TrackNode, node2: TrackNode) -> float:
        """Calculate edge cost between two nodes.

        Args:
            node1 (TrackNode): First track node
            node2 (TrackNode): Second track node

        Returns:
            float: Edge cost based on IoU and frame gap
        """
        iou = self._calc_iou(node1.bbox, node2.bbox)
        frame_gap = node2.frame_id - node1.frame_id
        return -iou * (1.0 / frame_gap)

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

        # Add detection nodes and edges
        for frame_idx, dets in enumerate(all_detections):
            for det_idx in range(len(dets)):
                if dets.confidence[det_idx] < self.min_confidence:
                    continue

                node = TrackNode(
                    frame_id=frame_idx,
                    detection_id=det_idx,
                    bbox=tuple(dets.xyxy[det_idx]),
                    confidence=dets.confidence[det_idx],
                )
                G.add_node(node)

                # Connect to source if first frame
                if frame_idx == 0:
                    G.add_edge("source", node, weight=-node.confidence)

                # Connect to sink if last frame
                if frame_idx == len(all_detections) - 1:
                    G.add_edge(node, "sink", weight=0)

                # Connect to future frames within max_gap
                future_range = range(
                    frame_idx + 1,
                    min(frame_idx + self.max_gap + 1, len(all_detections)),
                )
                for future_idx in future_range:
                    future_dets = all_detections[future_idx]
                    for future_det_idx in range(len(future_dets)):
                        if future_dets.confidence[future_det_idx] < self.min_confidence:
                            continue

                        future_node = TrackNode(
                            frame_id=future_idx,
                            detection_id=future_det_idx,
                            bbox=tuple(future_dets.xyxy[future_det_idx]),
                            confidence=future_dets.confidence[future_det_idx],
                        )

                        if self._can_connect_nodes(node, future_node):
                            weight = self._edge_cost(node, future_node)
                            G.add_edge(node, future_node, weight=weight)

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
                assignments[(node.frame_id, node.detection_id)] = track_id

        return
