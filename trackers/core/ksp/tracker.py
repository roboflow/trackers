import os
from collections import defaultdict
from typing import Callable, List, Optional

import cv2
import numpy as np
import PIL
import supervision as sv
from tqdm.auto import tqdm

from trackers.core.base import BaseOfflineTracker
from trackers.core.ksp.solver import KSPSolver, TrackNode


class KSPTracker(BaseOfflineTracker):
    """
    Offline tracker using K-Shortest Paths (KSP) algorithm.
    """

    def __init__(
        self,
        path_overlap_penalty: Optional[int] = None,
        iou_weight: Optional[int] = None,
        dist_weight: Optional[int] = None,
        size_weight: Optional[int] = None,
        conf_weight: Optional[int] = None,
    ) -> None:
        """
        Initialize the KSPTracker and its solver.

        Args:
            path_overlap_penalty (Optional[int]): Penalty for reusing the same edge
                (detection pairing) in multiple tracks. Increasing this value encourages
                the tracker to produce more distinct, non-overlapping tracks by
                discouraging shared detections between tracks.

            iou_weight (Optional[int]): Weight for the Intersection-over-Union (IoU)
                penalty in the edge cost. Higher values make the tracker favor linking
                detections with greater spatial overlap, which helps maintain track
                continuity for objects that move smoothly.

            dist_weight (Optional[int]): Weight for the Euclidean distance between
                detection centers in the edge cost. Increasing this value penalizes
                large jumps between detections in consecutive frames, promoting
                smoother, more physically plausible tracks.

            size_weight (Optional[int]): Weight for the size difference penalty in the
                edge cost. Higher values penalize linking detections with significantly
                different bounding box areas, which helps prevent identity switches when
                object size changes abruptly.

            conf_weight (Optional[int]): Weight for the confidence penalty in the edge
                cost. Higher values penalize edges between detections with lower
                confidence scores, making the tracker prefer more reliable detections
                and reducing the impact of false positives.
        """
        self._solver = KSPSolver(
            path_overlap_penalty=path_overlap_penalty,
            iou_weight=iou_weight,
            dist_weight=dist_weight,
            size_weight=size_weight,
            conf_weight=conf_weight,
        )
        self.reset()

    def reset(self) -> None:
        """
        Reset the KSPTracker and its solver state.

        This clears all buffered detections and resets the underlying solver.
        """
        self._solver.reset()

    def _update(self, detections: sv.Detections) -> sv.Detections:
        """
        Add detections for the current frame to the solver.

        Args:
            detections (sv.Detections): Detections for the current frame.

        Returns:
            sv.Detections: The same detections passed in.
        """
        self._solver.append_frame(detections)
        return detections

    def _assign_tracker_ids_from_paths(
        self, paths: List[List[TrackNode]]
    ) -> List[sv.Detections]:
        """
        Assigns each detection a unique tracker ID directly from node-disjoint paths.

        Args:
            paths (List[List[TrackNode]]): List of tracks, each a list of TrackNode.

        Returns:
            List[sv.Detections]: List of sv.Detections with tracker IDs assigned
                for each frame.
        """
        # Map from frame to list of dicts with detection info + tracker_id
        frame_to_dets = defaultdict(list)

        # Assign each node a unique tracker ID (path index + 1)
        for tracker_id, path in enumerate(paths, start=1):
            for node in path:
                frame_to_dets[node.frame_id].append(
                    {
                        "xyxy": node.bbox,
                        "confidence": node.confidence,
                        "class_id": node.class_id,
                        "tracker_id": tracker_id,
                    }
                )

        # Convert detections per frame into sv.Detections objects
        frame_to_detections = []
        for frame in sorted(frame_to_dets.keys()):
            dets_list = frame_to_dets[frame]
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

    def track(
        self,
        source: str | List[PIL.Image.Image],
        get_model_detections: Callable[[np.ndarray], sv.Detections],
        num_of_tracks: Optional[int] = None,
    ) -> List[sv.Detections]:
        """
        Run the KSP solver and assign tracker IDs to detections.

        Args:
            source_path (str): Path to video file or directory of frames.
            get_model_detections (Callable[[np.ndarray], sv.Detections]):
                Function that takes an image (np.ndarray) and returns sv.Detections.
            num_of_tracks (Optional[int]): Number of tracks to extract (K).

        Returns:
            List[sv.Detections]: List of sv.Detections with tracker IDs assigned.
        """
        if not source:
            raise ValueError(
                "`source_path` must be a string path to a directory or an .mp4 file."
            )
        if get_model_detections is None:
            raise TypeError(
                "`get_model_detections` must be a callable that returns an "
                "instance of `sv.Detections`."
            )
        if source.lower().endswith(".mp4"):
            frames_generator = sv.get_video_frames_generator(source_path=source)
            video_info = sv.VideoInfo.from_video_path(video_path=source)
            for frame in tqdm(
                frames_generator,
                total=video_info.total_frames,
                desc="Extracting detections and buffering from video",
                dynamic_ncols=True,
            ):
                detections = get_model_detections(frame)
                self._update(detections)
        elif os.path.isdir(source):
            frame_paths = sorted(
                [
                    os.path.join(source, f)
                    for f in os.listdir(source)
                    if f.lower().endswith(".jpg")
                ][:100]
            )
            for frame_path in tqdm(
                frame_paths,
                desc="Extracting detections and buffering directory",
                dynamic_ncols=True,
            ):
                image = cv2.imread(frame_path)
                detections = get_model_detections(image)
                self._update(detections)
        else:
            raise ValueError(f"{source} not a valid path or list of PIL.Image.Image.")
        paths = self._solver.solve(num_of_tracks)
        for i in paths:
            print(len(i))
        if not paths:
            return []
        return self._assign_tracker_ids_from_paths(paths)
