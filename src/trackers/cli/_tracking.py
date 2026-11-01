# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Sequence options and result writing for tracker inspection and benchmarking.

McByte and McByte++ import these helpers independently. The McByte++ benchmark also uses the MOT writer. Tracker
configuration and execution remain in their respective command modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import supervision as sv

DetectionFileFormat = Literal["mot_tlwh", "xyxy"]


@dataclass
class SequenceOptions:
    """Frames, detections, and the frame range to process.

    Attributes:
        image_dir: Directory containing sequence frames.
        det_file: Path to the detection file.
        start_frame: First frame number to process, inclusive.
        end_frame: Last frame number to process, inclusive.
        det_format: Detection-file column format. ``mot_tlwh`` expects
            ``frame,id,left,top,width,height,confidence,...``; ``xyxy`` expects
            ``frame,x1,y1,x2,y2,confidence``.
        frame_rate: Sequence frame rate used to scale the lost-track buffer.
    """

    image_dir: Path
    det_file: Path
    start_frame: int
    end_frame: int
    det_format: DetectionFileFormat = "mot_tlwh"
    frame_rate: float = 30.0


def append_mot_results(
    *,
    results_file: TextIO,
    frame_number: int,
    tracked_detections: sv.Detections,
) -> None:
    """Append confirmed tracker outputs in MOTChallenge result format.

    Detections with negative tracker IDs are omitted.

    Args:
        results_file: Open text stream receiving result rows.
        frame_number: Sequence frame number written in the first column.
        tracked_detections: Detections with tracker IDs and optional confidences.
    """
    tracker_ids = tracked_detections.tracker_id
    if tracker_ids is None:
        return

    confidences = tracked_detections.confidence
    if confidences is None:
        confidences = np.ones(len(tracked_detections), dtype=np.float32)

    for xyxy, tracker_id_value, confidence_value in zip(
        tracked_detections.xyxy,
        tracker_ids,
        confidences,
    ):
        tracker_id = int(tracker_id_value)
        if tracker_id < 0:
            continue

        left, top, right, bottom = map(float, xyxy)
        width = right - left
        height = bottom - top
        confidence = float(confidence_value)

        results_file.write(
            f"{frame_number},{tracker_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},{confidence:.6f},-1,-1,-1\n"
        )
