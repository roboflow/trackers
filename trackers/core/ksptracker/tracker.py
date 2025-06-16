import numpy as np
import supervision as sv
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from trackers.core.base import BaseTracker

@dataclass(frozen=True)
class TrackNode:
    frame_id: int
    detection_id: int
    bbox: tuple
    confidence: float

    def __hash__(self):
        return hash((self.frame_id, self.detection_id))

    def __eq__(self, other):
        return isinstance(other, TrackNode) and (self.frame_id, self.detection_id) == (other.frame_id, other.detection_id)

class KSPTracker(BaseTracker):
    """
    Offline tracker using the K-Shortest Paths (KSP) algorithm.
    """
    def __init__(self,
                 max_gap: int = 30,
                 min_confidence: float = 0.3,
                 max_paths: int = 1000,
                 max_distance: float = 0.3):
        self.max_gap = max_gap
        self.min_confidence = min_confidence
        self.max_paths = max_paths
        self.max_distance = max_distance
        self.reset()

    def reset(self) -> None:
        self.detection_buffer: List[sv.Detections] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        self.detection_buffer.append(detections)
        return detections

    def _calc_iou(self, bbox1: Union[np.ndarray, tuple], bbox2: Union[np.ndarray, tuple]) -> float:
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
        return self._calc_iou(node1.bbox, node2.bbox) >= (1 - self.max_distance)

    def _edge_cost(self, node1: TrackNode, node2: TrackNode) -> float:
        iou = self._calc_iou(node1.bbox, node2.bbox)
        frame_gap = node2.frame_id - node1.frame_id
        return -iou * (1.0 / frame_gap)

    def _build_graph(self, all_detections: List[sv.Detections]) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_node("source")
        graph.add_node("sink")

        for frame_idx, detections in enumerate(all_detections):
            for det_idx in range(len(detections)):
                node = TrackNode(
                    frame_id=frame_idx,
                    detection_id=det_idx,
                    bbox=tuple(detections.xyxy[det_idx]),
                    confidence=detections.confidence[det_idx]
                )
                graph.add_node(node)

                if frame_idx == 0:
                    graph.add_edge("source", node, weight=-node.confidence)

                if frame_idx == len(all_detections) - 1:
                    graph.add_edge(node, "sink", weight=0)

                for next_frame in range(frame_idx + 1, min(frame_idx + self.max_gap, len(all_detections))):
                    for next_idx in range(len(all_detections[next_frame])):
                        future_node = TrackNode(
                            frame_id=next_frame,
                            detection_id=next_idx,
                            bbox=tuple(all_detections[next_frame].xyxy[next_idx]),
                            confidence=all_detections[next_frame].confidence[next_idx]
                        )

                        if self._can_connect_nodes(node, future_node):
                            graph.add_edge(node, future_node, weight=self._edge_cost(node, future_node))

        return graph

    def ksp(self, graph: nx.DiGraph) -> List[List[TrackNode]]:
        paths = []
        try:
            gen_paths = nx.shortest_simple_paths(graph, "source", "sink", weight="weight")
            for i, path in enumerate(gen_paths):
                if i >= self.max_paths:
                    break
                paths.append(path[1:-1])  # strip 'source' and 'sink'
        except nx.NetworkXNoPath:
            pass
        return paths

    def _update_detections_with_tracks(self, assignments: dict) -> sv.Detections:
        output = []
        
        for frame_idx, detections in enumerate(self.detection_buffer):
            tracker_ids = []
            for det_idx in range(len(detections)):
                tracker_ids.append(assignments.get((frame_idx, det_idx), -1))  # Default to -1 if not assigned

            # Attach tracker IDs to current frame detections
            detections.tracker_id = np.array(tracker_ids)
            output.append(detections)

        # Merge all updated detections into a single sv.Detections object
        return sv.Detections.merge(output)

    def process_tracks(self) -> sv.Detections:
        if not self.detection_buffer:
            return sv.Detections.empty()

        graph = self._build_graph(self.detection_buffer)
        paths = self.ksp(graph)

        assignments = {}
        for track_id, path in enumerate(paths, start=1):
            for node in path:
                assignments[(node.frame_id, node.detection_id)] = track_id

        return self._update_detections_with_tracks(assignments)
