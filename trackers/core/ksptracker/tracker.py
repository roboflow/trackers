import numpy as np
import supervision as sv
import networkx as nx
from dataclasses import dataclass
from typing import List
from trackers.core.base import BaseTracker

@dataclass
class TrackNode:
    frame_id: int
    detection_id: int
    bbox: np.ndarray
    confidence: float

class KSPTracker(BaseTracker):
    """
    Offline tracker using the K-Shortest Paths (KSP) algorithm.
    """
    def __init__(self,
                 max_gap: int = 30,
                 min_confidence: float = 0.3,
                 max_paths: int = 1000,
                 max_distance: float = 0.3,
    ):
        self.max_gap = max_gap
        self.min_confidence = min_confidence
        self.max_paths = max_paths
        self.max_distance = max_distance
        self.reset()

    def _can_connect_nodes(self, node1: TrackNode, node2: TrackNode) -> bool:
        iou = self._calc_iou(node1.bbox, node2.bbox)
        eps = (1 - self.max_distance)

        return iou / eps

    def _calc_iou(self, bbox1: np.ndarray, bbox2: np.ndarray):
        x1, y1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
        x2, y2 = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
        eps = 1e-5 # To not allow division by 0

        x_inter, y_inter = max(0, x2 - x1) , min(0, y2 - y1)
        intersection_area = x_inter * y_inter
        area1 = (bbox1[3] - bbox1[1]) * (bbox1[2] - bbox1[0])
        area2 = (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])
        union = area1 + area2 - intersection_area

        return intersection_area / (union + eps)
    
    def _edge_cost(self, node1: TrackNode, node2: TrackNode) -> float:
        iou = self._calc_iou(node1.bbox, node2.bbox)
        frame_gap = node2.frame_id - node1.frame_id
        return -iou * (1.0 / frame_gap)
    
    def _build_graph(self, all_detections: List[sv.Detections]):
        diGraph = nx.DiGraph() # Create a directed graph

        # Adding your 2 virtual locations/nodes
        diGraph.add_node("source")
        diGraph.add_node("sink")

        for frame_idx, detections in enumerate(all_detections):
            for det_idx in range(len(detections)):
                node = TrackNode(
                    frame_id=frame_idx,
                    detection_id=det_idx,
                    bbox=detections.xyxy[det_idx],
                    confidence=detections.confidence[det_idx]
                )

                diGraph.add_node(node)

                # Connecting the source node to the first frame
                if (frame_idx == 0):
                    diGraph.add_edge("source", node, weight=-node.confidence)

                # Connecting the last frame to the sink node 
                if frame_idx == len(all_detections) - 1:
                    diGraph.add_edge(node, "sink", weight=0)

                for future_frame_idx in range(frame_idx + 1, min(frame_idx + self.max_gap, len(all_detections))):
                    future_dets = all_detections[future_frame_idx]

                    for future_det_idx in range(len(future_dets)):
                        future_node = TrackNode(
                            frame_id=future_frame_idx,
                            detection_id=future_det_idx,
                            bbox=future_dets.xyxy[future_det_idx],
                            confidence=future_dets.confidence[future_det_idx]
                        )
                        
                        if self._can_connect_nodes(node, future_node):
                            weight = self._edge_cost(node, future_node)
                            diGraph.add_edge(node, future_node, weight=weight)
        return diGraph                


    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Run KSP algorithm on all detections (offline).
        """
        self.detection_buffer.append(detections)
        return detections

    def reset(self) -> None:
        pass