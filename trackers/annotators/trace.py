# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import supervision as sv
from supervision.annotators.utils import ColorLookup, resolve_color
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position

from trackers.motion.transformation import (
    CoordinatesTransformation,
    IdentityTransformation,
)


class MotionAwareTraceAnnotator:
    """Draws object trajectories with camera motion compensation.

    This annotator maintains a history of object positions in world coordinates
    and draws them as trajectories (traces) on each frame. When used with camera
    motion compensation, trajectories appear stable even when the camera moves.

    The API is compatible with supervision annotators, using the same color
    resolution strategy and position anchoring.

    Args:
        color: The color to draw the trace. Can be a single `Color` or a
            `ColorPalette`. Defaults to `ColorPalette.DEFAULT`.
        position: The anchor position on the bounding box for the trace point.
            Defaults to `Position.CENTER`.
        trace_length: Maximum number of points to store per trajectory.
            Defaults to `30`.
        thickness: Line thickness for drawing traces. Defaults to `2`.
        color_lookup: Strategy for mapping colors to annotations.
            Options are `INDEX`, `CLASS`, `TRACK`. Defaults to `ColorLookup.TRACK`.

    Example:
        ```python
        import cv2
        import supervision as sv
        from inference import get_model

        from trackers import (
            ByteTrackTracker,
            MotionAwareTraceAnnotator,
            MotionEstimator,
        )

        model = get_model("rfdetr-nano")
        tracker = ByteTrackTracker()
        motion_estimator = MotionEstimator()
        trace_annotator = MotionAwareTraceAnnotator()

        cap = cv2.VideoCapture("moving_camera.mp4")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            coord_transform = motion_estimator.update(frame)

            result = model.infer(frame)[0]
            detections = sv.Detections.from_inference(result)
            detections = tracker.update(detections)

            frame = trace_annotator.annotate(
                scene=frame,
                detections=detections,
                coord_transform=coord_transform,
            )
        ```
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette, None] = None,
        position: Optional[Position] = None,
        trace_length: int = 30,
        thickness: int = 2,
        color_lookup: Optional[ColorLookup] = None,
    ) -> None:
        self.color: Union[Color, ColorPalette] = (
            color if color is not None else ColorPalette.DEFAULT
        )
        self.position: Position = position if position is not None else Position.CENTER
        self.trace_length = trace_length
        self.thickness = thickness
        self.color_lookup: ColorLookup = (
            color_lookup if color_lookup is not None else ColorLookup.TRACK
        )

        self._trajectories: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    def _get_anchor_points(self, detections: sv.Detections) -> np.ndarray:
        """Extract anchor points from detections based on position setting.

        Args:
            detections: Detections object with xyxy boxes.

        Returns:
            Array of shape `(N, 2)` with `(x, y)` anchor points.
        """
        return detections.get_anchors_coordinates(self.position)

    def annotate(
        self,
        scene: np.ndarray,
        detections: sv.Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
        coord_transform: Optional[CoordinatesTransformation] = None,
    ) -> np.ndarray:
        """Draw motion-compensated trace paths on the scene.

        Updates internal trajectory storage with new detection positions (converted
        to world coordinates), then draws all trajectories transformed back to
        frame coordinates.

        Args:
            scene: The image on which traces will be drawn. Modified in place.
            detections: Detections with `tracker_id` field populated.
            custom_color_lookup: Optional custom color lookup array to override
                the default color mapping strategy.
            coord_transform: Coordinate transformation for the current frame.
                If None, uses identity transformation (no motion compensation).

        Returns:
            The annotated image.

        Raises:
            ValueError: If detections don't have tracker_id field.
        """
        if detections.tracker_id is None:
            raise ValueError(
                "The `tracker_id` field is missing in the provided detections. "
                "See: https://supervision.roboflow.com/latest/how_to/track_objects"
            )

        if coord_transform is None:
            coord_transform = IdentityTransformation()

        anchor_points = self._get_anchor_points(detections)

        if len(anchor_points) > 0:
            world_points = coord_transform.rel_to_abs(anchor_points)

            for tracker_id, world_point in zip(detections.tracker_id, world_points):
                if tracker_id is None or tracker_id < 0:
                    continue
                tracker_id = int(tracker_id)
                trajectory = self._trajectories[tracker_id]
                trajectory.append((float(world_point[0]), float(world_point[1])))

                if len(trajectory) > self.trace_length:
                    self._trajectories[tracker_id] = trajectory[-self.trace_length :]

        for detection_idx in range(len(detections)):
            tracker_id = detections.tracker_id[detection_idx]
            if tracker_id is None or tracker_id < 0:
                continue
            tracker_id = int(tracker_id)

            trajectory = self._trajectories.get(tracker_id, [])
            if len(trajectory) < 2:
                continue

            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=(
                    self.color_lookup
                    if custom_color_lookup is None
                    else custom_color_lookup
                ),
            )

            world_points = np.array(trajectory, dtype=np.float32)
            frame_points = coord_transform.abs_to_rel(world_points)
            points = frame_points.astype(np.int32)

            scene = cv2.polylines(
                scene,
                [points],
                isClosed=False,
                color=color.as_bgr(),
                thickness=self.thickness,
            )

        return scene

    def reset(self) -> None:
        """Clear all stored trajectories.

        Call this when switching videos or when you want to reset trajectory history.
        """
        self._trajectories.clear()

    def clear_tracker(self, tracker_id: int) -> None:
        """Clear the trajectory for a specific tracker ID.

        Args:
            tracker_id: The tracker ID to clear.
        """
        self._trajectories.pop(tracker_id, None)
