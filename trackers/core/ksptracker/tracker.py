from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

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
        detection_id (int): Detection index within the frame
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
        confidence (float): Detection confidence score
    """

    frame_id: int
    detection_id: int
    bbox: tuple
    confidence: float

    def __hash__(self) -> int:
        """Generates hash based on frame_id and detection_id."""
        return hash((self.frame_id, self.detection_id))

    def __eq__(self, other: Any) -> bool:
        """Compares equality based on frame_id and detection_id."""
        return isinstance(other, TrackNode) and (self.frame_id, self.detection_id) == (
            other.frame_id,
            other.detection_id,
        )


class KSPTracker(BaseTracker):
    """
    Offline tracker using K-Shortest Paths (KSP) algorithm.

    Attributes:
        max_gap (int): Maximum allowed frame gap between detections in a track
        min_confidence (float): Minimum confidence threshold for detections
        max_paths (int): Maximum number of paths to find in KSP algorithm
        max_distance (float): Maximum allowed dissimilarity (1 - IoU) for edge connections
    """

    def __init__(
        self,
        max_gap: int = 30,
        min_confidence: float = 0.3,
        max_paths: int = 1000,
        max_distance: float = 0.3,
    ) -> None:
        """
        Initializes KSP tracker with configuration parameters.

        Args:
            max_gap (int): Max frame gap between connected detections (default: 30)
            min_confidence (float): Minimum detection confidence (default: 0.3)
            max_paths (int): Maximum number of paths to find (default: 1000)
            max_distance (float): Max dissimilarity (1-IoU) for connections (default: 0.3)
        """
        self.max_gap = max_gap
        self.min_confidence = min_confidence
        self.max_paths = max_paths
        self.max_distance = max_distance
        self.reset()

    def reset(self) -> None:
        """Resets the tracker's internal state."""
        self.detection_buffer: List[sv.Detections] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates tracker with new detections (stores without processing).

        Args:
            detections (sv.Detections): New detections for current frame

        Returns:
            detections (sv.Detections): Input detections
        """
        self.detection_buffer.append(detections)
        return detections

    def _calc_iou(
        self, bbox1: Union[np.ndarray, tuple], bbox2: Union[np.ndarray, tuple]
    ) -> float:
        """
        Calculates Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1 (Union[np.ndarray, tuple]): First bounding box (x1, y1, x2, y2)
            bbox2 (Union[np.ndarray, tuple]): Second bounding box (x1, y1, x2, y2)

        Returns:
            IOU (float): IoU value between 0.0 and 1.0
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
        """
        Determines if two nodes can be connected based on IoU threshold.

        Args:
            node1 (TrackNode): First track node
            node2 (TrackNode): Second track node

        Returns:
            bool: True if IoU >= (1 - max_distance), False otherwise
        """
        return self._calc_iou(node1.bbox, node2.bbox) >= (1 - self.max_distance)

    def _edge_cost(self, node1: TrackNode, node2: TrackNode) -> float:
        """
        Computes edge cost between two nodes for graph weighting.

        Args:
            node1 (TrackNode): Source node
            node2 (TrackNode): Target node

        Returns:
            float: Negative IoU normalized by frame gap
        """
        iou = self._calc_iou(node1.bbox, node2.bbox)
        frame_gap = node2.frame_id - node1.frame_id
        return -iou * (1.0 / frame_gap)

    def _build_graph(self, all_detections: List[sv.Detections]) -> nx.DiGraph:
        """
        Constructs tracking graph from buffered detections.

        Args:
            all_detections (List[sv.Detections]): List of detections per frame

        Returns:
            directed_graph (nx.DiGraph): Directed graph with connections between nodes with calculated weights
        """
        diGraph = nx.DiGraph()

        # Add the 2 virtual nodes
        diGraph.add_node("source")
        diGraph.add_node("sink")

        for frame_idx, detections in enumerate(all_detections):
            for det_idx in range(len(detections)):
                node = TrackNode(
                    frame_id=frame_idx,
                    detection_id=det_idx,
                    bbox=tuple(detections.xyxy[det_idx]),
                    confidence=detections.confidence[det_idx],
                )
                diGraph.add_node(node)

                # Connect to source for first frame
                if frame_idx == 0:
                    diGraph.add_edge("source", node, weight=-node.confidence)

                # Connect to sink for last frame
                if frame_idx == len(all_detections) - 1:
                    diGraph.add_edge(node, "sink", weight=0)

                # Create edges to future frames
                for next_frame in range(
                    frame_idx + 1, min(frame_idx + self.max_gap, len(all_detections))
                ):
                    for next_idx in range(len(all_detections[next_frame])):
                        future_node = TrackNode(
                            frame_id=next_frame,
                            detection_id=next_idx,
                            bbox=tuple(all_detections[next_frame].xyxy[next_idx]),
                            confidence=all_detections[next_frame].confidence[next_idx],
                        )

                        if self._can_connect_nodes(node, future_node):
                            diGraph.add_edge(
                                node,
                                future_node,
                                weight=self._edge_cost(node, future_node),
                            )

        return diGraph

    def ksp(self, diGraph: nx.DiGraph) -> List[List[TrackNode]]:
        """
        Finds K-Shortest Paths in the tracking graph.

        Args:
            graph (nx.DiGraph): Constructed tracking graph

        Returns:
            paths (List[List[TrackNode]]): List of track paths (each path is list of TrackNodes)
        """
        paths: List[List[TrackNode]] = []
        try:
            gen_paths = nx.shortest_simple_paths(
                diGraph, "source", "sink", weight="weight"
            )
            for i, path in enumerate(gen_paths):
                if i >= self.max_paths:
                    break
                paths.append(path[1:-1])  # strip 'source' and 'sink'
        except nx.NetworkXNoPath:
            pass
        return paths

    def _update_detections_with_tracks(self, assignments: dict) -> sv.Detections:
        """
        Assigns track IDs to detections based on path assignments.

        Args:
            assignments (dict): Mapping of (frame_id, detection_id) to track_id

        Returns:
            detections (sv.Detections): Detections with tracker_id attribute populated
        """
        output: List[sv.Detections] = []

        for frame_idx, detections in enumerate(self.detection_buffer):
            tracker_ids: List[int] = []
            for det_idx in range(len(detections)):
                tracker_ids.append(
                    assignments.get((frame_idx, det_idx), -1)
                )  # -1 for unassigned

            # Attach tracker IDs to current frame detections
            detections.tracker_id = np.array(tracker_ids)
            output.append(detections)

        # Merge all updated detections
        return sv.Detections.merge(output)

    def process_tracks(self) -> sv.Detections:
        """
        Processes buffered detections to generate tracks.

        Steps:
        1. Build tracking graph
        2. Find K-Shortest Paths
        3. Assign track IDs
        4. Return detections with tracker IDs

        Returns:
            detections (sv.Detections): Merged detections with tracker_id attribute populated
        """
        if not self.detection_buffer:
            return sv.Detections.empty()

        graph: nx.DiGraph = self._build_graph(self.detection_buffer)
        paths: List[List[TrackNode]] = self.ksp(graph)

        # Create track ID assignments
        assignments: Dict[Tuple[int, int], int] = {}
        for track_id, path in enumerate(paths, start=1):
            for node in path:
                assignments[(node.frame_id, node.detection_id)] = track_id

        return self._update_detections_with_tracks(assignments)
