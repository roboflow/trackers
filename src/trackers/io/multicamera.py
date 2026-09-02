# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""AI City 2024 multicamera file I/O and world-plane HOTA preparation."""

from __future__ import annotations

import json
import numbers
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from trackers.io.mot import _remap_ids

# Column layout for AI City 2024 9-column text files.
_COL_CAMERA = 0
_COL_ID = 1
_COL_FRAME = 2
_COL_XWORLD = 7
_COL_YWORLD = 8
_NUM_COLUMNS = 9

_SUPPORTED_FILE_FORMATS = ("aicity-2024",)
_MAX_SAFE_INTEGER = 2**53 - 1


@dataclass
class _MultiCameraSequenceData:
    """Prepared IDs and similarity matrices for HOTA evaluation."""

    gt_ids: list[NDArray[np.intp]]
    tracker_ids: list[NDArray[np.intp]]
    similarity_scores: list[NDArray[np.float64]]


@dataclass
class _PreparedRows:
    """Compact rows after filter, stable deduplication, rounding, and sorting."""

    frame_ids: NDArray[np.intp]
    object_ids: NDArray[np.intp]
    points: NDArray[np.float64]


def load_multicamera_file(
    path: str | Path,
    *,
    file_format: Literal["aicity-2024"] = "aicity-2024",
    camera_ids: Sequence[int] | None = None,
) -> NDArray[np.float64]:
    """Load an AI City multicamera tracking file into an ``(N, 9)`` array.

    Each row is ``camera_id obj_id frame_id x y w h xworld yworld``
    (space-delimited). ``camera_id``, ``obj_id``, and ``frame_id`` must be
    non-negative integral values no larger than ``2**53 - 1``, the largest
    integer stored exactly in the returned float64 array. World coordinates
    may be negative. Negative identifiers raise for the whole file, whereas
    NVIDIA's evaluator silently drops rows with a negative ``frame_id``.

    Args:
        path: Path to the text file.
        file_format: Edition of the on-disk format. Only ``"aicity-2024"`` is
            supported.
        camera_ids: If provided, keep only rows whose camera ID is in this set.
            Applied identically to ground truth and predictions. An empty result
            after filtering raises ``ValueError``.

    Returns:
        Float array of shape ``(N, 9)`` in file order after optional camera
        filtering. Rounding, deduplication, and the frame-ID offset are applied
        later during evaluation.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If ``camera_ids`` contains a non-integer.
        ValueError: If ``file_format`` is unsupported, the file is empty /
            malformed, identifiers are invalid, or filtering removes all rows.

    Examples:
        >>> from trackers.io.multicamera import load_multicamera_file  # doctest: +SKIP
        >>>
        >>> rows = load_multicamera_file(  # doctest: +SKIP
        ...     "tests/data/multicamera/gt/scene_a/ground_truth.txt",
        ...     camera_ids=[1],
        ... )
        >>> rows.shape  # doctest: +SKIP
        (5, 9)
    """
    return _load_multicamera_columns(
        path,
        file_format=file_format,
        camera_ids=camera_ids,
        columns=None,
    )


def _load_multicamera_columns(
    path: str | Path,
    *,
    file_format: str,
    camera_ids: Sequence[int] | None,
    columns: tuple[int, ...] | None,
    allow_empty: bool = False,
) -> NDArray[np.float64]:
    """Load all columns for the public API or a compact internal projection."""
    if file_format not in _SUPPORTED_FILE_FORMATS:
        raise ValueError(
            f"Unsupported file_format={file_format!r}. "
            f"Supported: {list(_SUPPORTED_FILE_FORMATS)}. "
            "AI City 2025/2026 JSON editions are not parsed yet."
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Multicamera file not found: {path}")

    # An empty prediction file is a valid zero-score input; loadtxt warns on it.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="loadtxt: input contained no data", category=UserWarning)
        try:
            data = np.loadtxt(path, dtype=np.float64, ndmin=2)
        except ValueError as exc:
            raise ValueError(f"Invalid AI City 2024 file: {path}") from exc

    output_columns = _NUM_COLUMNS if columns is None else len(columns)
    if data.size == 0 and allow_empty:
        return np.empty((0, output_columns), dtype=np.float64)
    if data.size == 0:
        raise ValueError(f"Multicamera file is empty: {path}")
    if data.shape[1] != _NUM_COLUMNS:
        raise ValueError(f"Expected {_NUM_COLUMNS} columns in {path}, got {data.shape[1]}.")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Multicamera file contains non-finite values: {path}")

    identifiers = data[:, [_COL_CAMERA, _COL_ID, _COL_FRAME]]
    if np.any(identifiers < 0) or np.any(identifiers != np.floor(identifiers)):
        raise ValueError(f"camera_id, obj_id, and frame_id must be non-negative integers: {path}")
    if np.any(identifiers > _MAX_SAFE_INTEGER):
        raise ValueError(f"Identifiers in {path} exceed the exact float64-safe bound 2**53 - 1.")

    camera_set = _validate_camera_ids(camera_ids) if camera_ids is not None else None
    if camera_set is not None:
        data = data[np.isin(data[:, _COL_CAMERA], tuple(camera_set))]
    if len(data) == 0 and allow_empty:
        return np.empty((0, output_columns), dtype=np.float64)
    if len(data) == 0:
        if camera_set is not None:
            raise ValueError(f"Multicamera file is empty after camera filtering ({sorted(camera_set)}): {path}")
        raise ValueError(f"Multicamera file is empty: {path}")
    return data if columns is None else data[:, columns]


def load_scene_camera_map(path: str | Path) -> dict[str, list[int]]:
    """Load NVIDIA's ``scene_name_2_cam_id`` JSON mapping.

    A scene name that appears in more than one entry has its camera lists
    concatenated, matching NVIDIA's evaluator.

    Args:
        path: Path to a JSON list of
            ``{"scene_name": "scene_061", "camera_ids": [535, ...]}``
            objects.

    Returns:
        Mapping from scene name to camera ID list (order preserved).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the JSON schema is invalid.

    Examples:
        >>> from trackers.io.multicamera import load_scene_camera_map  # doctest: +SKIP
        >>>
        >>> load_scene_camera_map("tests/data/multicamera/scene_camera_map.json")  # doctest: +SKIP
        {'scene_a': [1, 2], 'scene_b': [3, 4]}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scene camera map not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Scene camera map is not valid JSON: {path}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"Scene camera map must be a JSON list: {path}")

    mapping: dict[str, list[int]] = {}
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"Scene camera map entry {index} must be an object: {path}")
        if "scene_name" not in entry or "camera_ids" not in entry:
            raise ValueError(f"Scene camera map entry {index} must contain 'scene_name' and 'camera_ids': {path}")
        scene_name = entry["scene_name"]
        camera_ids = entry["camera_ids"]
        if not isinstance(scene_name, str):
            raise ValueError(f"scene_name must be a string at entry {index}: {path}")
        if not isinstance(camera_ids, list) or not all(
            isinstance(camera_id, numbers.Integral) and not isinstance(camera_id, (bool, np.bool_))
            for camera_id in camera_ids
        ):
            raise ValueError(f"camera_ids must be a list of ints at entry {index}: {path}")
        mapping.setdefault(scene_name, []).extend(camera_ids)
    return mapping


def _validate_camera_ids(camera_ids: Sequence[int]) -> set[int]:
    """Validate camera filters without lossy or ambiguous integer coercion."""
    validated: set[int] = set()
    for camera_id in camera_ids:
        if not isinstance(camera_id, numbers.Integral) or isinstance(camera_id, (bool, np.bool_)):
            raise TypeError(f"camera_ids must contain integers, got {camera_id!r}.")
        value = int(camera_id)
        if value < 0 or value > _MAX_SAFE_INTEGER:
            raise ValueError(f"camera_ids must be in [0, 2**53 - 1], got {value}.")
        validated.add(value)
    return validated


def _validate_zero_distance(zero_distance: float) -> None:
    if not np.isfinite(zero_distance) or zero_distance <= 0:
        raise ValueError(f"zero_distance must be finite and > 0, got {zero_distance}")


def _euclidean_similarity(
    points1: NDArray[np.float64],
    points2: NDArray[np.float64],
    zero_distance: float,
) -> NDArray[np.float64]:
    """Convert pairwise Euclidean distances into similarities in ``[0, 1]``.

    ``sim = max(0, 1 - dist / zero_distance)``. Matches TrackEval's
    ``_calculate_euclidean_similarity`` (MOT15_3D / AI City 2024), which
    subtracts coordinates directly; a Gram-matrix expansion loses precision at
    world-plane magnitudes and flips matches sitting on an alpha threshold.
    """
    if len(points1) == 0 or len(points2) == 0:
        return np.empty((len(points1), len(points2)), dtype=np.float64)
    similarity = cdist(points1, points2)
    similarity /= zero_distance
    similarity *= -1.0
    similarity += 1.0
    np.maximum(similarity, 0.0, out=similarity)
    return similarity


def _dedup_sort_prepared(
    frame_ids: NDArray[np.intp],
    object_ids: NDArray[np.intp],
    points: NDArray[np.float64],
) -> _PreparedRows:
    """Keep first frame/ID rows and return frame-sorted compact arrays."""
    # Sorting integer columns and comparing adjacent keys avoids materializing
    # Python tuples or np.unique's returned structured-key array. Sorting the
    # selected source positions restores NVIDIA's file-order keep-first rule.
    key_order = np.lexsort((object_ids, frame_ids))
    sorted_frames = frame_ids[key_order]
    sorted_objects = object_ids[key_order]
    unique = np.ones(len(key_order), dtype=np.bool_)
    unique[1:] = (sorted_frames[1:] != sorted_frames[:-1]) | (sorted_objects[1:] != sorted_objects[:-1])
    first_indices = key_order[unique]
    first_indices.sort()
    frame_ids = frame_ids[first_indices] + 1
    object_ids = object_ids[first_indices]
    points = points[first_indices]

    order = np.argsort(frame_ids, kind="stable")
    return _PreparedRows(
        frame_ids=frame_ids[order],
        object_ids=object_ids[order],
        points=np.asarray(points[order], dtype=np.float64),
    )


def _load_prepared_multicamera_file(
    path: str | Path,
    *,
    file_format: str,
    camera_ids: Sequence[int],
    allow_empty: bool = False,
) -> _PreparedRows:
    """Load a file, keep only the preparation columns, then compact and deduplicate them."""
    compact = _load_multicamera_columns(
        path,
        file_format=file_format,
        camera_ids=camera_ids,
        columns=(_COL_FRAME, _COL_ID, _COL_XWORLD, _COL_YWORLD),
        allow_empty=allow_empty,
    )
    frame_ids = compact[:, 0].astype(np.intp)
    object_ids = compact[:, 1].astype(np.intp)
    points = np.round(compact[:, 2:4], 3)
    return _dedup_sort_prepared(frame_ids, object_ids, points)


def _group_by_frame(
    data: _PreparedRows,
) -> dict[int, tuple[NDArray[np.intp], NDArray[np.float64]]]:
    """Group compact, frame-sorted rows into slice views."""
    if len(data.frame_ids) == 0:
        return {}

    unique_frames, start_indices = np.unique(data.frame_ids, return_index=True)
    boundaries = np.append(start_indices, len(data.frame_ids))
    grouped: dict[int, tuple[NDArray[np.intp], NDArray[np.float64]]] = {}
    for frame_index, frame in enumerate(unique_frames.tolist()):
        start = boundaries[frame_index]
        stop = boundaries[frame_index + 1]
        grouped[int(frame)] = (data.object_ids[start:stop], data.points[start:stop])
    return grouped


def _build_id_mapping(data: _PreparedRows) -> dict[int, int]:
    """Map original object IDs to contiguous 0-indexed values."""
    unique_ids = np.unique(data.object_ids)
    return {int(original): index for index, original in enumerate(unique_ids)}


def _prepare_multicamera_files(
    gt_path: str | Path,
    pred_path: str | Path,
    *,
    file_format: str,
    camera_ids: Sequence[int],
    zero_distance: float,
) -> _MultiCameraSequenceData:
    """Load and compact each side sequentially to bound peak resident memory."""
    gt_prepared = _load_prepared_multicamera_file(
        gt_path,
        file_format=file_format,
        camera_ids=camera_ids,
    )
    pred_prepared = _load_prepared_multicamera_file(
        pred_path,
        file_format=file_format,
        camera_ids=camera_ids,
        allow_empty=True,
    )
    return _assemble_multicamera_sequence(gt_prepared, pred_prepared, zero_distance=zero_distance)


def _assemble_multicamera_sequence(
    gt_prepared: _PreparedRows,
    pred_prepared: _PreparedRows,
    *,
    zero_distance: float,
) -> _MultiCameraSequenceData:
    """Build dense sequence semantics from compact prepared rows.

    The sequence spans up to the last frame on either side, as in NVIDIA's evaluator, so prediction frames past the
    ground truth count as false positives rather than being rejected.
    """
    gt_grouped = _group_by_frame(gt_prepared)
    pred_grouped = _group_by_frame(pred_prepared)

    num_frames = max(
        max(gt_grouped.keys(), default=0),
        max(pred_grouped.keys(), default=0),
    )

    gt_id_map = _build_id_mapping(gt_prepared)
    tracker_id_map = _build_id_mapping(pred_prepared)

    empty_ids = np.empty(0, dtype=np.intp)
    empty_points = np.empty((0, 2), dtype=np.float64)
    empty_similarity = np.empty((0, 0), dtype=np.float64)
    per_frame_gt_ids: list[NDArray[np.intp]] = [empty_ids] * num_frames
    per_frame_tracker_ids: list[NDArray[np.intp]] = [empty_ids] * num_frames
    per_frame_similarity: list[NDArray[np.float64]] = [empty_similarity] * num_frames
    for frame in sorted(gt_grouped.keys() | pred_grouped.keys()):
        gt_original_ids, gt_points = gt_grouped.get(frame, (empty_ids, empty_points))
        pred_original_ids, pred_points = pred_grouped.get(frame, (empty_ids, empty_points))
        per_frame_gt_ids[frame - 1] = _remap_ids(gt_original_ids, gt_id_map)
        per_frame_tracker_ids[frame - 1] = _remap_ids(pred_original_ids, tracker_id_map)
        per_frame_similarity[frame - 1] = _euclidean_similarity(gt_points, pred_points, zero_distance)

    return _MultiCameraSequenceData(
        gt_ids=per_frame_gt_ids,
        tracker_ids=per_frame_tracker_ids,
        similarity_scores=per_frame_similarity,
    )
