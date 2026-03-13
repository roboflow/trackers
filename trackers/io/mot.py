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
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.eval.box import box_iou
from trackers.eval.constants import EPS

_DISTRACTOR_IOU_THRESHOLD = 0.5


@dataclass
class _MOTFrameData:
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


def _mot_frame_to_detections(frame_data: _MOTFrameData) -> sv.Detections:
    return sv.Detections(
        xyxy=sv.xywh_to_xyxy(frame_data.boxes),
        confidence=frame_data.confidences,
        class_id=frame_data.classes.astype(int),
    )


@dataclass
class _MOTSequenceData:
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


def _load_mot_file(path: str | Path) -> dict[int, _MOTFrameData]:
    """Load a MOT Challenge format file.

    Parse a text file in the standard MOT format where each line represents
    one detection with comma-separated values:
    `<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, ...`

    Args:
        path: Path to the MOT format text file.

    Returns:
        Dictionary mapping frame numbers (1-based, as in the file) to
        `_MOTFrameData` containing all detections for that frame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or has invalid format.

    Examples:
        >>> from trackers import load_mot_file  # doctest: +SKIP
        >>>
        >>> gt_data = load_mot_file("data/gt/MOT17-02/gt/gt.txt")  # doctest: +SKIP
        >>>
        >>> len(gt_data)  # doctest: +SKIP
        600
        >>>
        >>> len(gt_data[1].ids)  # doctest: +SKIP
        12
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

    result: dict[int, _MOTFrameData] = {}
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

        result[frame] = _MOTFrameData(
            ids=ids,
            boxes=boxes,
            confidences=confidences,
            classes=classes,
        )

    return result


def _resolve_num_frames(
    ground_truth_data: dict[int, _MOTFrameData],
    tracker_data: dict[int, _MOTFrameData],
    num_frames: int | None,
) -> int:
    """Determine the total frame count from data if not explicitly provided."""
    if num_frames is not None:
        return num_frames
    ground_truth_frames = set(ground_truth_data.keys()) if ground_truth_data else set()
    tracker_frames = set(tracker_data.keys()) if tracker_data else set()
    all_frames = ground_truth_frames | tracker_frames
    return max(all_frames) if all_frames else 0


def _build_id_mappings(
    ground_truth_data: dict[int, _MOTFrameData],
    tracker_data: dict[int, _MOTFrameData],
    num_frames: int,
) -> tuple[dict[int, int], dict[int, int]]:
    """Collect valid IDs across all frames and build original-to-0-indexed maps.

    Returns:
        Tuple of (ground_truth_id_map, tracker_id_map) where each maps original
        track IDs to contiguous 0-indexed values.
    """
    # Reference: trackeval/datasets/mot_challenge_2d_box.py:402-421
    unique_ground_truth_ids: set[int] = set()
    unique_tracker_ids: set[int] = set()

    for frame in range(1, num_frames + 1):
        if frame in ground_truth_data:
            valid_mask = ground_truth_data[frame].confidences > 0
            unique_ground_truth_ids.update(
                ground_truth_data[frame].ids[valid_mask].tolist()
            )
        if frame in tracker_data:
            confirmed_mask = tracker_data[frame].ids >= 0
            unique_tracker_ids.update(tracker_data[frame].ids[confirmed_mask].tolist())

    sorted_ground_truth_ids = sorted(unique_ground_truth_ids)
    sorted_tracker_ids = sorted(unique_tracker_ids)

    ground_truth_id_map = {
        original_id: index for index, original_id in enumerate(sorted_ground_truth_ids)
    }
    tracker_id_map = {
        original_id: index for index, original_id in enumerate(sorted_tracker_ids)
    }
    return ground_truth_id_map, tracker_id_map


def _extract_ground_truth_frame(
    ground_truth_data: dict[int, _MOTFrameData],
    frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract and split ground truth data for a single frame.

    Returns:
        Tuple of (valid_boxes, valid_ids, all_boxes, distractor_mask).
        For missing frames, returns empty arrays with correct shapes.
    """
    # Reference: trackeval/datasets/mot_challenge_2d_box.py:390-400
    if frame in ground_truth_data:
        frame_data = ground_truth_data[frame]
        valid_mask = frame_data.confidences > 0
        return (
            frame_data.boxes[valid_mask],
            frame_data.ids[valid_mask],
            frame_data.boxes,
            ~valid_mask,
        )

    empty_boxes = np.empty((0, 4), dtype=np.float64)
    empty_ids = np.array([], dtype=np.intp)
    empty_mask = np.array([], dtype=bool)
    return empty_boxes, empty_ids, empty_boxes, empty_mask


def _extract_tracker_frame(
    tracker_data: dict[int, _MOTFrameData],
    frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract tracker detections for a single frame, keeping only confirmed tracks.

    Returns:
        Tuple of (boxes, ids) with unconfirmed tracks (id < 0) removed.
    """
    # Reference: trackeval/datasets/mot_challenge_2d_box.py:385-386
    if frame in tracker_data:
        frame_data = tracker_data[frame]
        confirmed_mask = frame_data.ids >= 0
        return frame_data.boxes[confirmed_mask], frame_data.ids[confirmed_mask]

    return np.empty((0, 4), dtype=np.float64), np.array([], dtype=np.intp)


def _remove_distractor_matches(
    all_ground_truth_boxes: np.ndarray,
    distractor_mask: np.ndarray,
    tracker_boxes: np.ndarray,
    tracker_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove tracker detections matched to distractor ground truth regions.

    Uses the Hungarian algorithm to match tracker detections against ALL ground
    truth boxes (including distractors). Tracker detections that best-match a
    distractor (conf==0) are removed so they are neither penalized as FP nor
    rewarded as TP.

    Returns:
        Filtered (boxes, ids) with distractor-matched detections removed.
    """
    # Reference: trackeval/datasets/mot_challenge_2d_box.py:360-386
    if not distractor_mask.any() or len(tracker_ids) == 0:
        return tracker_boxes, tracker_ids

    distractor_iou_matrix = box_iou(
        all_ground_truth_boxes, tracker_boxes, box_format="xywh"
    )
    distractor_iou_matrix[distractor_iou_matrix < _DISTRACTOR_IOU_THRESHOLD - EPS] = 0

    matched_gt_indices, matched_tracker_indices = linear_sum_assignment(
        -distractor_iou_matrix
    )

    actually_matched = (
        distractor_iou_matrix[matched_gt_indices, matched_tracker_indices] > 0 + EPS
    )
    matched_gt_indices = matched_gt_indices[actually_matched]
    matched_tracker_indices = matched_tracker_indices[actually_matched]

    matched_to_distractor = distractor_mask[matched_gt_indices]
    keep_mask = np.ones(len(tracker_ids), dtype=bool)
    keep_mask[matched_tracker_indices[matched_to_distractor]] = False

    return tracker_boxes[keep_mask], tracker_ids[keep_mask]


def _remap_ids(
    ids: np.ndarray,
    id_map: dict[int, int],
) -> np.ndarray:
    """Remap original track IDs to 0-indexed contiguous values."""
    # Reference: trackeval/datasets/mot_challenge_2d_box.py:407-421
    if len(ids) == 0:
        return np.array([], dtype=np.intp)
    return np.array([id_map[int(original_id)] for original_id in ids], dtype=np.intp)


def _prepare_mot_sequence(
    ground_truth_data: dict[int, _MOTFrameData],
    tracker_data: dict[int, _MOTFrameData],
    num_frames: int | None = None,
) -> _MOTSequenceData:
    """Prepare GT and tracker data for metric evaluation.

    Compute IoU similarity matrices between GT and tracker detections for each
    frame, and remap track IDs to 0-indexed contiguous values as required by
    CLEAR, HOTA, and Identity metrics.

    Args:
        ground_truth_data: Ground truth data from `load_mot_file`.
        tracker_data: Tracker predictions from `load_mot_file`.
        num_frames: Total number of frames in the sequence. If `None`,
            auto-detected from the maximum frame number in the data.

    Returns:
        `_MOTSequenceData` containing prepared data ready for metric evaluation.
    """
    num_frames = _resolve_num_frames(ground_truth_data, tracker_data, num_frames)
    ground_truth_id_map, tracker_id_map = _build_id_mappings(
        ground_truth_data, tracker_data, num_frames
    )

    per_frame_ground_truth_ids: list[np.ndarray] = []
    per_frame_tracker_ids: list[np.ndarray] = []
    per_frame_similarity: list[np.ndarray] = []
    total_ground_truth_detections = 0
    total_tracker_detections = 0

    for frame in range(1, num_frames + 1):
        ground_truth_boxes, ground_truth_ids, all_boxes, distractor_mask = (
            _extract_ground_truth_frame(ground_truth_data, frame)
        )
        tracker_boxes, tracker_ids = _extract_tracker_frame(tracker_data, frame)
        tracker_boxes, tracker_ids = _remove_distractor_matches(
            all_boxes, distractor_mask, tracker_boxes, tracker_ids
        )

        remapped_ground_truth_ids = _remap_ids(ground_truth_ids, ground_truth_id_map)
        remapped_tracker_ids = _remap_ids(tracker_ids, tracker_id_map)
        similarity = box_iou(ground_truth_boxes, tracker_boxes, box_format="xywh")

        per_frame_ground_truth_ids.append(remapped_ground_truth_ids)
        per_frame_tracker_ids.append(remapped_tracker_ids)
        per_frame_similarity.append(similarity)
        total_ground_truth_detections += len(remapped_ground_truth_ids)
        total_tracker_detections += len(remapped_tracker_ids)

    return _MOTSequenceData(
        gt_ids=per_frame_ground_truth_ids,
        tracker_ids=per_frame_tracker_ids,
        similarity_scores=per_frame_similarity,
        num_frames=num_frames,
        num_gt_ids=len(ground_truth_id_map),
        num_tracker_ids=len(tracker_id_map),
        num_gt_dets=total_ground_truth_detections,
        num_tracker_dets=total_tracker_detections,
        gt_id_mapping=ground_truth_id_map,
        tracker_id_mapping=tracker_id_map,
    )


class _MOTOutput:
    """Context manager for MOT format file writing."""

    def __init__(self, path: Path | None):
        self.path = path
        self._file = None

    def write(self, frame_idx: int, detections: sv.Detections) -> None:
        """Write detections for a frame in MOT format."""
        if self._file is None or len(detections) == 0:
            return

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            w, h = x2 - x1, y2 - y1

            track_id = (
                int(detections.tracker_id[i])
                if detections.tracker_id is not None
                else -1
            )
            conf = (
                float(detections.confidence[i])
                if detections.confidence is not None
                else -1.0
            )

            self._file.write(
                f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                f"{conf:.4f},-1,-1,-1\n"
            )

    def __enter__(self):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "w")
        return self

    def __exit__(self, *_):
        if self._file is not None:
            self._file.close()
