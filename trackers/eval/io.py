# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trackers.eval.box import box_iou


@dataclass
class MOTFrameData:
    """Detection data for a single frame from a MOT format file.

    Attributes:
        ids: Track IDs for each detection. Shape `(N,)` where N is number
            of detections in this frame.
        boxes: Bounding boxes in xywh format (x, y, width, height).
            Shape `(N, 4)`.
        confidences: Detection confidence scores. Shape `(N,)`. For GT files,
            this indicates whether the detection should be considered (0=ignore).
        classes: Class IDs for each detection. Shape `(N,)`. In MOT Challenge,
            1=pedestrian, 2-13=other classes (distractors, vehicles, etc.).
    """

    ids: np.ndarray
    boxes: np.ndarray
    confidences: np.ndarray
    classes: np.ndarray


@dataclass
class MOTSequenceData:
    """Prepared sequence data ready for metric evaluation.

    This dataclass contains all data needed by CLEAR, HOTA, and Identity
    metrics. IDs are remapped to 0-indexed contiguous values because metrics
    use IDs as array indices for efficient accumulation.

    Attributes:
        gt_ids: Ground truth track IDs per frame, 0-indexed. Each element is
            an array of shape `(num_gt_in_frame,)`. Used by all metrics to
            track which GT objects are present.
        tracker_ids: Tracker track IDs per frame, 0-indexed. Each element is
            an array of shape `(num_tracker_in_frame,)`. Used by all metrics
            to track which predictions are present.
        similarity_scores: IoU similarity matrices per frame. Each element is
            shape `(num_gt_in_frame, num_tracker_in_frame)`. Used for matching
            GT to predictions and computing MOTP/LocA.
        num_frames: Total number of frames in the sequence. Used by Count
            metrics and for validation.
        num_gt_ids: Count of unique GT track IDs. Used to allocate accumulator
            arrays in HOTA/Identity metrics.
        num_tracker_ids: Count of unique tracker track IDs. Used to allocate
            accumulator arrays in HOTA/Identity metrics.
        num_gt_dets: Total GT detections across all frames. Used for MOTA
            denominator and early-exit conditions.
        num_tracker_dets: Total tracker detections across all frames. Used
            for FP counting and early-exit conditions.
        gt_id_mapping: Mapping from original GT IDs to 0-indexed values.
            Useful for debugging and tracing results back to source files.
        tracker_id_mapping: Mapping from original tracker IDs to 0-indexed
            values. Useful for debugging and tracing results back to source.
    """

    gt_ids: list[np.ndarray]
    tracker_ids: list[np.ndarray]
    similarity_scores: list[np.ndarray]
    num_frames: int
    num_gt_ids: int
    num_tracker_ids: int
    num_gt_dets: int
    num_tracker_dets: int
    gt_id_mapping: dict[int, int]
    tracker_id_mapping: dict[int, int]


def load_mot_file(path: str | Path) -> dict[int, MOTFrameData]:
    """Load a MOT Challenge format file.

    Parse a text file in the standard MOT format where each line represents
    one detection with comma-separated values:
    `<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, ...`

    Args:
        path: Path to the MOT format text file.

    Returns:
        Dictionary mapping frame numbers (1-based, as in the file) to
        `MOTFrameData` containing all detections for that frame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or has invalid format.

    Examples:
        ```python
        from trackers.eval import load_mot_file

        gt_data = load_mot_file("data/gt/MOT17-02/gt/gt.txt")
        len(gt_data)
        # 600
        len(gt_data[1].ids)
        # 12
        ```
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MOT file not found: {path}")

    frame_data: dict[int, list[list[str]]] = {}

    with open(path) as f:
        # Check if file is empty
        f.seek(0, 2)
        if f.tell() == 0:
            raise ValueError(f"MOT file is empty: {path}")
        f.seek(0)

        # Auto-detect CSV dialect
        sample = f.readline()
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",; \t")
            dialect.skipinitialspace = True
        except csv.Error:
            dialect = csv.excel
            dialect.skipinitialspace = True

        reader = csv.reader(f, dialect)
        for row in reader:
            if not row or (len(row) == 1 and row[0].strip() == ""):
                continue

            while row and row[-1] == "":
                row = row[:-1]

            if len(row) < 6:
                raise ValueError(
                    f"Invalid MOT format in {path}: expected at least 6 columns, "
                    f"got {len(row)} in row: {row}"
                )

            try:
                frame = int(float(row[0]))
            except ValueError as e:
                raise ValueError(f"Invalid frame number in {path}: {row[0]}") from e

            if frame not in frame_data:
                frame_data[frame] = []
            frame_data[frame].append(row)

    if not frame_data:
        raise ValueError(f"No valid data found in MOT file: {path}")

    result: dict[int, MOTFrameData] = {}
    for frame, rows in frame_data.items():
        try:
            data = np.array(rows, dtype=np.float64)
        except ValueError as e:
            raise ValueError(
                f"Cannot convert data to float in {path}, frame {frame}"
            ) from e

        ids = data[:, 1].astype(np.intp)
        boxes = data[:, 2:6]
        confidences = data[:, 6] if data.shape[1] > 6 else np.ones(len(data))
        classes = (
            data[:, 7].astype(np.intp)
            if data.shape[1] > 7
            else np.ones(len(data), dtype=np.intp)
        )

        result[frame] = MOTFrameData(
            ids=ids,
            boxes=boxes,
            confidences=confidences,
            classes=classes,
        )

    return result


def prepare_mot_sequence(
    gt_data: dict[int, MOTFrameData],
    tracker_data: dict[int, MOTFrameData],
    num_frames: int | None = None,
) -> MOTSequenceData:
    """Prepare GT and tracker data for metric evaluation.

    Compute IoU similarity matrices between GT and tracker detections for each
    frame, and remap track IDs to 0-indexed contiguous values as required by
    CLEAR, HOTA, and Identity metrics.

    Args:
        gt_data: Ground truth data from `load_mot_file`.
        tracker_data: Tracker predictions from `load_mot_file`.
        num_frames: Total number of frames in the sequence. If `None`,
            auto-detected from the maximum frame number in the data.

    Returns:
        `MOTSequenceData` containing prepared data ready for metric evaluation.

    Examples:
        ```python
        from trackers.eval import load_mot_file, prepare_mot_sequence

        gt = load_mot_file("data/gt/MOT17-02/gt/gt.txt")
        tracker = load_mot_file("data/trackers/MOT17-02.txt")
        data = prepare_mot_sequence(gt, tracker)
        data.num_frames
        # 600
        data.num_gt_ids
        # 54
        ```
    """
    gt_frames = set(gt_data.keys()) if gt_data else set()
    tracker_frames = set(tracker_data.keys()) if tracker_data else set()
    all_frames = gt_frames | tracker_frames

    if num_frames is None:
        num_frames = max(all_frames) if all_frames else 0

    all_gt_ids: set[int] = set()
    all_tracker_ids: set[int] = set()

    for frame in range(1, num_frames + 1):
        if frame in gt_data:
            all_gt_ids.update(gt_data[frame].ids.tolist())
        if frame in tracker_data:
            all_tracker_ids.update(tracker_data[frame].ids.tolist())

    # Build ID mappings (original -> 0-indexed)
    sorted_gt_ids = sorted(all_gt_ids)
    sorted_tracker_ids = sorted(all_tracker_ids)
    gt_id_mapping = {orig_id: idx for idx, orig_id in enumerate(sorted_gt_ids)}
    tracker_id_mapping = {
        orig_id: idx for idx, orig_id in enumerate(sorted_tracker_ids)
    }

    gt_ids_list: list[np.ndarray] = []
    tracker_ids_list: list[np.ndarray] = []
    similarity_scores_list: list[np.ndarray] = []
    num_gt_dets = 0
    num_tracker_dets = 0

    for frame in range(1, num_frames + 1):
        # Get GT data for this frame
        if frame in gt_data:
            gt_frame = gt_data[frame]
            gt_boxes = gt_frame.boxes
            gt_ids_orig = gt_frame.ids
            # Remap IDs to 0-indexed
            gt_ids_remapped = np.array(
                [gt_id_mapping[int(gid)] for gid in gt_ids_orig], dtype=np.intp
            )
            num_gt_dets += len(gt_ids_remapped)
        else:
            gt_boxes = np.empty((0, 4), dtype=np.float64)
            gt_ids_remapped = np.array([], dtype=np.intp)

        # Get tracker data for this frame
        if frame in tracker_data:
            tracker_frame = tracker_data[frame]
            tracker_boxes = tracker_frame.boxes
            tracker_ids_orig = tracker_frame.ids
            # Remap IDs to 0-indexed
            tracker_ids_remapped = np.array(
                [tracker_id_mapping[int(tid)] for tid in tracker_ids_orig],
                dtype=np.intp,
            )
            num_tracker_dets += len(tracker_ids_remapped)
        else:
            tracker_boxes = np.empty((0, 4), dtype=np.float64)
            tracker_ids_remapped = np.array([], dtype=np.intp)

        # Compute IoU similarity matrix
        similarity = box_iou(gt_boxes, tracker_boxes, box_format="xywh")

        gt_ids_list.append(gt_ids_remapped)
        tracker_ids_list.append(tracker_ids_remapped)
        similarity_scores_list.append(similarity)

    return MOTSequenceData(
        gt_ids=gt_ids_list,
        tracker_ids=tracker_ids_list,
        similarity_scores=similarity_scores_list,
        num_frames=num_frames,
        num_gt_ids=len(sorted_gt_ids),
        num_tracker_ids=len(sorted_tracker_ids),
        num_gt_dets=num_gt_dets,
        num_tracker_dets=num_tracker_dets,
        gt_id_mapping=gt_id_mapping,
        tracker_id_mapping=tracker_id_mapping,
    )
