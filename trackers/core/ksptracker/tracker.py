import os
from collections import defaultdict
from typing import Any, Callable, List, Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from trackers.core.base import BaseTracker
from trackers.core.ksptracker.solver import KSPSolver, TrackNode


class KSPTracker(BaseTracker):
    """
    Offline tracker using K-Shortest Paths (KSP) algorithm.
    """

    def __init__(
        self, iou_weight=0.9, dist_weight=0.1, size_weight=0.1, conf_weight=0.1
    ) -> None:
        """
        Initialize the KSPTracker and its solver.
        """
        self._solver = KSPSolver()
        self._solver.append_config(
            iou_weight=iou_weight,
            dist_weight=dist_weight,
            size_weight=size_weight,
            conf_weight=conf_weight,
        )
        self.reset()

    def reset(self) -> None:
        """
        Reset the solver and clear any stored state.
        """
        self._solver.reset()

    def update_config(
        self, iou_weight=0.9, dist_weight=0.1, size_weight=0.1, conf_weight=0.1
    ):
        """
        Update the configuration weights for the KSP algorithm.

        Args:
            iou_weight (float): Weight for IoU component.
            dist_weight (float): Weight for distance component.
            size_weight (float): Weight for size component.
            conf_weight (float): Weight for confidence component.
        """
        self._solver.append_config(
            iou_weight=iou_weight,
            dist_weight=dist_weight,
            size_weight=size_weight,
            conf_weight=conf_weight,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Add detections for the current frame to the solver.

        Args:
            detections (sv.Detections): Detections for the current frame.

        Returns:
            sv.Detections: The same detections passed in.
        """
        self._solver.append_frame(detections)
        return detections

    def assign_tracker_ids_from_paths(
        self, paths: List[List[TrackNode]]
    ) -> List[sv.Detections]:
        """
        Assigns each detection a unique tracker ID by preferring the path with
        the least motion change (displacement).

        Args:
            paths (List[List[TrackNode]]): List of tracks, each a list of TrackNode.

        Returns:
            Dict[int, sv.Detections]: Mapping from frame index to sv.Detections
                                      with tracker IDs assigned.
        """
        # Track where each node appears
        framed_nodes = defaultdict(list)
        node_to_candidates = defaultdict(list)
        for tracker_id, path in enumerate(paths, start=1):
            for i, node in enumerate(path):
                next_node: Any = path[i + 1] if i + 1 < len(path) else None
                node_to_candidates[node].append((tracker_id, next_node))
                framed_nodes[node.frame_id].append(node)

        # Select best tracker for each node based on minimal displacement
        node_to_tracker = {}
        for node, candidates in node_to_candidates.items():
            min_displacement = float("inf")
            selected_tracker = -1
            for tracker_id, next_node in candidates:
                if next_node is not None:
                    dx = node.position[0] - next_node.position[0]
                    dy = node.position[1] - next_node.position[1]
                    displacement = dx * dx + dy * dy  # squared distance
                else:
                    displacement = 0  # last node in path, no penalty

                if displacement < min_displacement:
                    min_displacement = displacement
                    selected_tracker = tracker_id

            node_to_tracker[node] = selected_tracker

        # Organize detections by frame
        frame_to_dets = defaultdict(list)

        for node, tracker_id in node_to_tracker.items():
            frame_to_dets[node.frame_id].append(
                {
                    "xyxy": node.bbox,
                    "confidence": node.confidence,
                    "class_id": node.class_id,
                    "tracker_id": tracker_id,
                }
            )

        # Convert into sv.Detections
        frame_to_detections = []
        for frame, dets_list in frame_to_dets.items():
            xyxy = np.array([d["xyxy"] for d in dets_list], dtype=np.float32)
            confidence = np.array(
                [d["confidence"] for d in dets_list], dtype=np.float32
            )
            class_id = np.array([d["class_id"] for d in dets_list], dtype=int)
            tracker_id = np.array([d["tracker_id"] for d in dets_list], dtype=int)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                tracker_id=tracker_id,
            )
            frame_to_detections.append(detections)

        return frame_to_detections

    def process_tracks(
        self,
        source_path: Optional[str] = None,
        get_model_detections: Optional[Callable[[np.ndarray], sv.Detections]] = None,
        num_of_tracks: Optional[int] = None,
    ) -> List[sv.Detections]:
        """
        Run the KSP solver and assign tracker IDs to detections.

        Args:
            source_path (Optional[str]): Path to video file or directory of frames.
            get_model_detections (Optional[Callable[[np.ndarray], sv.Detections]]):
                Function that takes an image (np.ndarray) and returns sv.Detections.
            num_of_tracks (Optional[int]): Number of tracks to extract (K).

        Returns:
            List[sv.Detections]: List of sv.Detections with tracker IDs assigned.
        """
        if not source_path:
            raise ValueError(
                "`source_path` must be a string path to a directory or an .mp4 file."
            )
        if not get_model_detections:
            raise TypeError(
                "`get_model_detections` must be a callable that returns an "
                "instance of `sv.Detections`."
            )
        if source_path.lower().endswith(".mp4"):
            frames_generator = sv.get_video_frames_generator(source_path=source_path)
            video_info = sv.VideoInfo.from_video_path(video_path=source_path)
            for frame in tqdm(
                frames_generator,
                total=video_info.total_frames,
                desc="Extracting detections and buffering from video",
                dynamic_ncols=True,
            ):
                detections = get_model_detections(frame)
                self.update(detections)
        elif os.path.isdir(source_path):
            frame_paths = sorted(
                [
                    os.path.join(source_path, f)
                    for f in os.listdir(source_path)
                    if f.lower().endswith(".jpg")
                ]
            )
            for frame_path in tqdm(
                frame_paths,
                desc="Extracting detections and buffering directory",
                dynamic_ncols=True,
            ):
                image = cv2.imread(frame_path)
                detections = get_model_detections(image)
                self.update(detections)
        else:
            raise ValueError(f"{source_path} not found!")
        paths = self._solver.solve(num_of_tracks)
        if not paths:
            return []
        return self.assign_tracker_ids_from_paths(paths)
