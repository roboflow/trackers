import os
from collections import defaultdict
from typing import Callable, List, Optional, Set, Tuple, Union

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
        path_overlap_penalty: float = 40,
        iou_weight: float = 0.9,
        dist_weight: float = 0.1,
        size_weight: float = 0.1,
        conf_weight: float = 0.1,
        entry_exit_regions: Optional[List[Tuple[int, int, int, int]]] = None,
        use_border: bool = True,
        borders: Optional[Set[str]] = None,
        border_margin: int = 40,
        frame_size: Tuple[int, int] = (1920, 1080),
    ) -> None:
        """
        Initialize the KSPTracker and its underlying solver with region and cost
        configuration.

        Args:
            path_overlap_penalty (float): Penalty for reusing the same edge
                (detection pairing) in multiple tracks. Higher values encourage the
                tracker to produce more distinct, non-overlapping tracks by
                discouraging shared detections between tracks. Default is 40.
            iou_weight (float): Weight for the IoU penalty in the edge cost.
                Higher values make the tracker favor linking detections with greater
                spatial overlap, which helps maintain track continuity for objects
                that move smoothly. Default is 0.9.
            dist_weight (float): Weight for the Euclidean distance between
                detection centers in the edge cost. Increasing this value penalizes
                large jumps between detections in consecutive frames, promoting
                smoother, more physically plausible tracks. Default is 0.1.
            size_weight (float): Weight for the size difference penalty in
                the edge cost. Higher values penalize linking detections with
                significantly different bounding box areas, which helps prevent
                identity switches when object size changes abruptly. Default is 0.1.
            conf_weight (float): Weight for the confidence penalty in the
                edge cost. Higher values penalize edges between detections with lower
                confidence scores, making the tracker prefer more reliable detections
                and reducing the impact of false positives. Default is 0.1.
            entry_exit_regions (Optional[List[Tuple[int, int, int, int]]]): List of
                rectangular entry/exit regions, each as (x1, y1, x2, y2) in pixels.
                Used to determine when objects enter or exit the scene. Default is
                an empty list.
            use_border (bool): Whether to enable border-based entry/exit
                logic. If True, objects entering or exiting through the image borders
                (as defined by `borders` and `border_margin`) are considered for
                entry/exit events. Default is True.
            borders (Optional[Set[str]]): Set of border sides to use for entry/exit
                logic. Valid values are any subset of {"left", "right", "top",
                "bottom"}. Default is all four borders.
            border_margin (int): Thickness of the border region (in pixels)
                used for entry/exit detection. Default is 40.
            frame_size (Tuple[int, int]): Size of the image frames as
                (width, height). Used to determine border region extents. Default is
                (1920, 1080).
        """
        self.entry_exit_regions: List[Tuple[int, int, int, int]] = (
            entry_exit_regions if entry_exit_regions is not None else []
        )
        self.use_border: bool = use_border if use_border is not None else True
        self.borders: Set[str] = (
            borders if borders is not None else {"left", "right", "top", "bottom"}
        )
        self.border_margin: int = border_margin if border_margin is not None else 40
        self.frame_size: Tuple[int, int] = (
            frame_size if frame_size is not None else (1920, 1080)
        )
        self._solver = KSPSolver(
            path_overlap_penalty=path_overlap_penalty,
            iou_weight=iou_weight,
            dist_weight=dist_weight,
            size_weight=size_weight,
            conf_weight=conf_weight,
        )
        self._solver.set_entry_exit_regions(self.entry_exit_regions)
        self._solver.set_border_entry_exit(
            use_border=self.use_border,
            borders=self.borders,
            margin=self.border_margin,
            frame_size=self.frame_size,
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

    def set_entry_exit_regions(self, regions: List[Tuple[int, int, int, int]]) -> None:
        """
        Set rectangular entry/exit zones (x1, y1, x2, y2) and update both the
        tracker and solver.

        Args:
            regions (List[Tuple[int, int, int, int]]): List of rectangular
                regions for entry/exit logic.
        """
        self.entry_exit_regions = regions
        self._solver.set_entry_exit_regions(regions)

    def set_border_entry_exit(
        self,
        use_border: bool = True,
        borders: Optional[Set[str]] = None,
        margin: int = 40,
        frame_size: Tuple[int, int] = (1920, 1080),
    ) -> None:
        """
        Configure border-based entry/exit zones and update both the tracker and
        solver.

        Args:
            use_border (bool): Enable/disable border-based entry/exit.
            borders (Optional[Set[str]]): Set of borders to use. Each value should
                be one of "left", "right", "top", "bottom".
            margin (int): Border thickness in pixels.
            frame_size (Tuple[int, int]): Size of the image (width, height).
        """
        self.use_border = use_border
        self.borders = (
            borders if borders is not None else {"left", "right", "top", "bottom"}
        )
        self.border_margin = margin
        self.frame_size = frame_size
        self._solver.set_border_entry_exit(
            use_border=self.use_border,
            borders=self.borders,
            margin=self.border_margin,
            frame_size=self.frame_size,
        )

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
        source: Union[str, List[PIL.Image.Image]],
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
        if isinstance(source, str) and source.lower().endswith(".mp4"):
            frames_generator = sv.get_video_frames_generator(source_path=source)
            video_info = sv.VideoInfo.from_video_path(video_path=source)

            self._solver.set_border_entry_exit(
                self.use_border,
                self.borders,
                self.border_margin,
                (video_info.width, video_info.height),
            )

            for frame in tqdm(
                frames_generator,
                total=video_info.total_frames,
                desc="Extracting detections and buffering from video",
                dynamic_ncols=True,
            ):
                detections = get_model_detections(frame)
                self._update(detections)
        elif isinstance(source, str) and os.path.isdir(source):
            frame_paths = sorted(
                [
                    os.path.join(source, f)
                    for f in os.listdir(source)
                    if f.lower().endswith(".jpg")
                ]
            )

            has_set_frame_size = False

            for frame_path in tqdm(
                frame_paths,
                desc="Extracting detections and buffering directory",
                dynamic_ncols=True,
            ):
                image = cv2.imread(frame_path)
                height, width = image.shape[:2]

                if not has_set_frame_size:
                    self._solver.set_border_entry_exit(
                        self.use_border,
                        self.borders,
                        self.border_margin,
                        (width, height),
                    )

                detections = get_model_detections(image)
                self._update(detections)
        else:
            raise ValueError(f"{source} not a valid path or list of PIL.Image.Image.")
        paths = self._solver.solve(num_of_tracks)

        if not paths:
            return []
        return self._assign_tracker_ids_from_paths(paths)
