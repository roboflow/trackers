# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""AI City 2024 multi-camera file I/O and world-plane HOTA preparation."""

from __future__ import annotations

import json
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Column layout for AI City 2024 9-column text files.
_COL_CAMERA = 0
_COL_ID = 1
_COL_FRAME = 2
_COL_X = 3
_COL_Y = 4
_COL_W = 5
_COL_H = 6
_COL_XWORLD = 7
_COL_YWORLD = 8
_NUM_COLUMNS = 9

_SUPPORTED_FILE_FORMATS = ("aicity-2024",)

# Measured across NVIDIA's official 30-scene AI City 2024 test inputs:
# every scene has 23,994 frames and 20-39 identities per side. The sequence
# bound is exact to the protocol fixture. Pair budgets retain large headroom for
# imperfect submissions while bounding the largest temporary allocations:
# 8x the observed identities per frame on each side (64x pairs), and 16x the
# observed identities globally on each side (256x pairs). At the global limit,
# HOTA's 19-plane float64 tensor is about 56 MiB.
_MAX_SEQUENCE_LENGTH = 23_994
_OBSERVED_MAX_IDENTITIES = 39
_OBSERVED_MAX_SCENE_DETECTIONS = 2_116_406
_MAX_FRAME_PAIR_COUNT = (_OBSERVED_MAX_IDENTITIES * 8) ** 2
_MAX_IDENTITY_PAIR_COUNT = (_OBSERVED_MAX_IDENTITIES * 16) ** 2
# Twice the measured largest official scene (scene_080 GT) permits imperfect
# submissions while stopping duplicate-heavy inputs before chunk concatenation.
_MAX_DETECTIONS_PER_SEQUENCE = _OBSERVED_MAX_SCENE_DETECTIONS * 2
_MAX_SAFE_INTEGER = 2**53 - 1

# Grow the numeric buffer in fixed-size chunks so peak RAM tracks kept rows,
# not a whole-file Python object graph.
_PARSE_CHUNK_ROWS = 65_536


@dataclass
class _MultiCameraSequenceData:
    """Prepared multi-camera sequence data ready for HOTA evaluation.

    Attributes:
        gt_ids: Ground-truth track IDs per frame, 0-indexed.
        tracker_ids: Tracker track IDs per frame, 0-indexed.
        similarity_scores: World-plane Euclidean similarity matrices per frame.
        num_frames: Sequence length (max frame across GT and predictions).
        num_gt_ids: Count of unique ground-truth track IDs.
        num_tracker_ids: Count of unique tracker track IDs.
        num_gt_dets: Total ground-truth detections after camera filter and dedup.
        num_tracker_dets: Total tracker detections after camera filter and dedup.
        gt_id_mapping: Mapping from original GT IDs to 0-indexed values.
        tracker_id_mapping: Mapping from original tracker IDs to 0-indexed values.
    """

    gt_ids: list[NDArray[np.intp]]
    tracker_ids: list[NDArray[np.intp]]
    similarity_scores: list[NDArray[np.float64]]
    num_frames: int
    num_gt_ids: int
    num_tracker_ids: int
    num_gt_dets: int
    num_tracker_dets: int
    gt_id_mapping: dict[int, int]
    tracker_id_mapping: dict[int, int]


@dataclass
class _PreparedRows:
    """Compact rows after filter, stable deduplication, rounding, and sorting."""

    frame_ids: NDArray[np.int64]
    object_ids: NDArray[np.int64]
    points: NDArray[np.float64]


def load_multicamera_file(
    path: str | Path,
    *,
    file_format: str = "aicity-2024",
    camera_ids: Sequence[int] | None = None,
) -> NDArray[np.float64]:
    """Load an AI City multi-camera tracking file into an ``(N, 9)`` array.

    Each row is ``camera_id obj_id frame_id x y w h xworld yworld``
    (space-delimited). Negative ``camera_id``, ``obj_id``, or ``frame_id``
    values raise for the whole file. Identifiers use unsigned decimal syntax
    and must not exceed ``2**53 - 1``, the largest integer stored exactly in
    the returned float64 array. World coordinates may be negative.

    Rows are streamed from disk into numeric chunks (never via whole-file
    ``read_text`` / Python ``list`` materialisation) so large AI City files
    remain memory-bounded.

    Args:
        path: Path to the text file.
        file_format: Edition of the on-disk format. Only ``\"aicity-2024\"`` is
            supported in this release. Future editions (JSON / 3-D box HOTA)
            will be additive values of this parameter.
        camera_ids: If provided, keep only rows whose camera ID is in this set.
            Applied identically to ground truth and predictions. An empty result
            after filtering raises ``ValueError``.

    Returns:
        Float array of shape ``(N, 9)`` in file order after optional camera
        filtering. Rounding, deduplication, and the frame-ID offset are applied
        later by `_prepare_multicamera_sequence`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``file_format`` is unsupported, the file is empty /
            malformed, identifiers are negative or not exactly representable,
            camera filters are ambiguous, or filtering removes all rows.
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
        raise FileNotFoundError(f"Multi-camera file not found: {path}")

    camera_set = _validate_camera_ids(camera_ids) if camera_ids is not None else None
    data = _stream_parse_aicity_2024(path, camera_ids=camera_set, columns=columns)
    if len(data) == 0:
        if camera_set is not None:
            raise ValueError(f"Multi-camera file is empty after camera filtering ({sorted(camera_set)}): {path}")
        raise ValueError(f"Multi-camera file is empty: {path}")
    return data


def load_scene_camera_map(path: str | Path) -> dict[str, list[int]]:
    """Load NVIDIA's ``scene_name_2_cam_id`` JSON mapping.

    Args:
        path: Path to a JSON list of
            ``{\"scene_name\": \"scene_061\", \"camera_ids\": [535, ...]}``
            objects.

    Returns:
        Mapping from scene name to camera ID list (order preserved).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the JSON schema is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scene camera map not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
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
        if scene_name in mapping:
            mapping[scene_name].extend(camera_ids)
        else:
            mapping[scene_name] = list(camera_ids)
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


def _truncate_multicamera_rows(
    path: str | Path,
    *,
    max_frame: int,
    file_format: str = "aicity-2024",
    assume_sorted_frames: bool = False,
) -> list[list[str]]:
    """Stream rows with ``frame_id < max_frame`` in file order.

    Real AI City per-scene ``ground_truth.txt`` files are often *not* globally
    sorted by ``frame_id`` (camera blocks interleave), so the default streams
    the whole file while keeping only in-window rows — peak memory tracks the
    truncated slice, not a whole-file ``read_text`` / float object graph.

    When ``assume_sorted_frames=True`` and the file is known to be
    non-decreasing in ``frame_id``, scanning stops at the first row with
    ``frame_id >= max_frame``.

    Args:
        path: AI City 2024 text file.
        max_frame: Exclusive upper bound on ``frame_id`` (0-based, as on disk).
        file_format: On-disk format edition.
        assume_sorted_frames: Enable early exit for sorted inputs only.

    Returns:
        Kept rows as token lists in file order.
    """
    if file_format not in _SUPPORTED_FILE_FORMATS:
        raise ValueError(f"Unsupported file_format={file_format!r}. Supported: {list(_SUPPORTED_FILE_FORMATS)}.")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Multi-camera file not found: {path}")

    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(f"Blank line at {path}:{line_number} is not allowed in AI City 2024 files.")
            tokens = line.split()
            _parse_row_tokens(tokens, path=path, line_number=line_number)
            frame = int(tokens[_COL_FRAME])
            if frame >= max_frame:
                if assume_sorted_frames:
                    break
                continue
            rows.append(list(tokens))
    return rows


def _parse_row_tokens(
    tokens: Sequence[str],
    *,
    path: Path,
    line_number: int,
) -> tuple[float, ...]:
    """Parse and validate one AI City 2024 row into nine floats."""
    if len(tokens) != _NUM_COLUMNS:
        raise ValueError(f"Expected {_NUM_COLUMNS} columns at {path}:{line_number}, got {len(tokens)}.")

    for column, name in (
        (_COL_CAMERA, "camera_id"),
        (_COL_ID, "obj_id"),
        (_COL_FRAME, "frame_id"),
    ):
        token = tokens[column]
        if token.startswith("-") and token[1:].isdecimal():
            raise ValueError(f"Invalid negative values found for CameraId, Id, or FrameId in {path}.")
        if not token.isdecimal():
            raise ValueError(
                f"Non-integer {name} token {token!r} at {path}:{line_number}; "
                "AI City 2024 identifiers require unsigned decimal integer syntax."
            )
        if int(token) > _MAX_SAFE_INTEGER:
            raise ValueError(f"{name}={token} at {path}:{line_number} exceeds the exact float64-safe bound 2**53 - 1.")

    values: list[float] = []
    for token_index, token in enumerate(tokens):
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Non-numeric token {token!r} at {path}:{line_number} column {token_index}.") from exc
        if not np.isfinite(value):
            raise ValueError(f"Non-finite value {token!r} at {path}:{line_number} column {token_index}.")
        values.append(value)

    return tuple(values)


def _stream_parse_aicity_2024(
    path: Path,
    *,
    camera_ids: set[int] | None,
    columns: tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    """Stream-parse a file into a full or projected float64 array."""
    output_columns = _NUM_COLUMNS if columns is None else len(columns)
    chunks: list[NDArray[np.float64]] = []
    buffer = np.empty((_PARSE_CHUNK_ROWS, output_columns), dtype=np.float64)
    fill = 0
    retained_rows = 0
    saw_content = False

    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = _validate_aicity_line(raw_line, path=path, line_number=line_number, saw_content=saw_content)
            if line is None:
                continue
            values = _parse_row_tokens(line.split(), path=path, line_number=line_number)
            saw_content = True
            if camera_ids is not None and int(values[_COL_CAMERA]) not in camera_ids:
                continue
            retained_rows += 1
            if retained_rows > _MAX_DETECTIONS_PER_SEQUENCE:
                raise ValueError(
                    f"Retained rows exceed MAX_DETECTIONS_PER_SEQUENCE={_MAX_DETECTIONS_PER_SEQUENCE}; "
                    "the largest official AI City 2024 scene contains 2,116,406 detections."
                )
            buffer[fill] = values if columns is None else tuple(values[column] for column in columns)
            fill += 1
            if fill == _PARSE_CHUNK_ROWS:
                chunks.append(buffer.copy())
                fill = 0

    if not saw_content:
        return np.empty((0, output_columns), dtype=np.float64)
    if fill:
        chunks.append(buffer[:fill].copy())
    if not chunks:
        return np.empty((0, output_columns), dtype=np.float64)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks, axis=0)


def _validate_aicity_line(
    raw_line: str,
    *,
    path: Path,
    line_number: int,
    saw_content: bool,
) -> str | None:
    """Normalize a parser line, allowing leading blanks only."""
    stripped_newlines = raw_line.strip("\n\r")
    line = stripped_newlines.strip()
    if not line:
        if not saw_content:
            return None
        raise ValueError(f"Blank line at {path}:{line_number} is not allowed in AI City 2024 files.")
    if line.startswith("#") or line.lower().startswith("camera"):
        raise ValueError(f"Header or comment line at {path}:{line_number} is not allowed: {stripped_newlines!r}")
    return line


def _euclidean_similarity(
    points1: NDArray[np.float64],
    points2: NDArray[np.float64],
    zero_distance: float,
) -> NDArray[np.float64]:
    """Convert pairwise Euclidean distances into similarities in ``[0, 1]``.

    ``sim = max(0, 1 - dist / zero_distance)``. Matches TrackEval's
    ``_calculate_euclidean_similarity`` (MOT15_3D / AI City 2024).
    """
    if not np.isfinite(zero_distance) or zero_distance <= 0:
        raise ValueError(f"zero_distance must be > 0, got {zero_distance}")
    if len(points1) == 0 or len(points2) == 0:
        return np.empty((len(points1), len(points2)), dtype=np.float64)
    _validate_pair_allocation(len(points1), len(points2), _MAX_FRAME_PAIR_COUNT)
    squared_distances = points1 @ points2.T
    squared_distances *= -2.0
    squared_distances += np.sum(points1 * points1, axis=1)[:, np.newaxis]
    squared_distances += np.sum(points2 * points2, axis=1)[np.newaxis, :]
    np.maximum(squared_distances, 0.0, out=squared_distances)
    np.sqrt(squared_distances, out=squared_distances)
    squared_distances /= zero_distance
    squared_distances *= -1.0
    squared_distances += 1.0
    np.maximum(squared_distances, 0.0, out=squared_distances)
    return squared_distances


def _validate_pair_allocation(left: int, right: int, limit: int) -> None:
    """Reject a Cartesian product before overflow or array allocation."""
    if left < 0 or right < 0 or limit < 0:
        raise ValueError("Pair-allocation dimensions and limit must be nonnegative.")
    if left and right > limit // left:
        if limit == _MAX_FRAME_PAIR_COUNT:
            limit_name = "MAX_FRAME_PAIR_COUNT"
        elif limit == _MAX_IDENTITY_PAIR_COUNT:
            limit_name = "MAX_IDENTITY_PAIR_COUNT"
        else:
            limit_name = "pair limit"
        raise ValueError(f"{left} x {right} exceeds {limit_name}={limit}; refusing dense allocation.")


def _filter_round_dedup_offset(
    data: NDArray[np.float64],
    camera_ids: Sequence[int] | None,
) -> _PreparedRows:
    """Apply camera filter, half-to-even rounding, keep-first dedup, and ``frame_id += 1``."""
    if len(data) == 0:
        raise ValueError("Multi-camera array is empty before preparation.")

    prepared = np.asarray(data, dtype=np.float64)
    selected_indices: NDArray[np.intp] | None = None
    if camera_ids is not None:
        camera_set = _validate_camera_ids(camera_ids)
        selected_indices = np.flatnonzero(np.isin(prepared[:, _COL_CAMERA], tuple(camera_set)))
        if len(selected_indices) == 0:
            raise ValueError(f"Multi-camera data is empty after camera filtering ({sorted(camera_set)}).")

    selected = prepared if selected_indices is None else prepared[selected_indices]
    frame_ids = selected[:, _COL_FRAME].astype(np.int64)
    object_ids = selected[:, _COL_ID].astype(np.int64)
    points = np.round(selected[:, [_COL_XWORLD, _COL_YWORLD]], 3)
    return _dedup_sort_prepared(frame_ids, object_ids, points)


def _dedup_sort_prepared(
    frame_ids: NDArray[np.int64],
    object_ids: NDArray[np.int64],
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

    max_frame = int(frame_ids.max())
    if max_frame > _MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence length {max_frame} exceeds MAX_SEQUENCE_LENGTH={_MAX_SEQUENCE_LENGTH}; "
            "the official AI City 2024 test scenes contain 23,994 frames."
        )
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
) -> _PreparedRows:
    """Stream only preparation columns, then compact and deduplicate them."""
    compact = _load_multicamera_columns(
        path,
        file_format=file_format,
        camera_ids=camera_ids,
        columns=(_COL_FRAME, _COL_ID, _COL_XWORLD, _COL_YWORLD),
    )
    frame_ids = compact[:, 0].astype(np.int64)
    object_ids = compact[:, 1].astype(np.int64)
    points = np.round(compact[:, 2:4], 3)
    return _dedup_sort_prepared(frame_ids, object_ids, points)


def _group_by_frame(
    data: _PreparedRows,
) -> dict[int, tuple[NDArray[np.int64], NDArray[np.float64]]]:
    """Group compact, frame-sorted rows into slice views."""
    if len(data.frame_ids) == 0:
        return {}

    unique_frames, start_indices = np.unique(data.frame_ids, return_index=True)
    boundaries = np.append(start_indices, len(data.frame_ids))
    grouped: dict[int, tuple[NDArray[np.int64], NDArray[np.float64]]] = {}
    for frame_index, frame in enumerate(unique_frames.tolist()):
        start = boundaries[frame_index]
        stop = boundaries[frame_index + 1]
        grouped[int(frame)] = (data.object_ids[start:stop], data.points[start:stop])
    return grouped


def _build_id_mapping(data: _PreparedRows) -> dict[int, int]:
    unique_ids = np.unique(data.object_ids)
    return {int(original): index for index, original in enumerate(unique_ids)}


def _remap_ids(ids: NDArray[np.intp], id_map: dict[int, int]) -> NDArray[np.intp]:
    if len(ids) == 0:
        return np.array([], dtype=np.intp)
    return np.array([id_map[int(original)] for original in ids], dtype=np.intp)


def _prepare_multicamera_sequence(
    gt_data: NDArray[np.float64],
    pred_data: NDArray[np.float64],
    *,
    camera_ids: Sequence[int] | None = None,
    zero_distance: float = 2.0,
) -> _MultiCameraSequenceData:
    """Prepare GT and prediction arrays for world-plane HOTA evaluation.

    Applies camera filtering (if ``camera_ids`` is set), half-to-even rounding of
    world coordinates to 3 decimals, ``(frame_id, obj_id)`` keep-first
    deduplication, ``frame_id += 1``, then builds per-frame ID lists and
    Euclidean similarity matrices.

    Args:
        gt_data: Ground-truth ``(N, 9)`` array from `load_multicamera_file`.
        pred_data: Prediction ``(M, 9)`` array from `load_multicamera_file`.
        camera_ids: Optional camera filter applied to both inputs. Prefer
            filtering at load time; this argument covers the filter-then-dedup
            composition when raw arrays are passed in.
        zero_distance: Distance (metres) at which similarity becomes zero.
            Defaults to ``2.0`` (AI City 2024 / MOT15_3D).

    Returns:
        `_MultiCameraSequenceData` ready for `compute_hota_metrics`.
    """
    if not np.isfinite(zero_distance) or zero_distance <= 0:
        raise ValueError(f"zero_distance must be > 0, got {zero_distance}")

    gt_prepared = _filter_round_dedup_offset(gt_data, camera_ids)
    pred_prepared = _filter_round_dedup_offset(pred_data, camera_ids)
    return _assemble_multicamera_sequence(gt_prepared, pred_prepared, zero_distance=zero_distance)


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
    )
    if not np.isfinite(zero_distance) or zero_distance <= 0:
        raise ValueError(f"zero_distance must be > 0, got {zero_distance}")
    return _assemble_multicamera_sequence(gt_prepared, pred_prepared, zero_distance=zero_distance)


def _assemble_multicamera_sequence(
    gt_prepared: _PreparedRows,
    pred_prepared: _PreparedRows,
    *,
    zero_distance: float,
) -> _MultiCameraSequenceData:
    """Build dense sequence semantics from compact prepared rows."""
    gt_grouped = _group_by_frame(gt_prepared)
    pred_grouped = _group_by_frame(pred_prepared)

    num_frames = max(
        max(gt_grouped.keys(), default=0),
        max(pred_grouped.keys(), default=0),
    )
    if num_frames <= 0:
        raise ValueError("Prepared multi-camera sequence has no frames.")

    gt_id_map = _build_id_mapping(gt_prepared)
    tracker_id_map = _build_id_mapping(pred_prepared)
    _validate_pair_allocation(len(gt_id_map), len(tracker_id_map), _MAX_IDENTITY_PAIR_COUNT)

    empty_ids = np.empty(0, dtype=np.intp)
    empty_similarity = np.empty((0, 0), dtype=np.float64)
    per_frame_gt_ids: list[NDArray[np.intp]] = [empty_ids] * num_frames
    per_frame_tracker_ids: list[NDArray[np.intp]] = [empty_ids] * num_frames
    per_frame_similarity: list[NDArray[np.float64]] = [empty_similarity] * num_frames
    total_gt = 0
    total_tracker = 0

    for frame in sorted(gt_grouped.keys() | pred_grouped.keys()):
        gt_rows = gt_grouped.get(frame)
        pred_rows = pred_grouped.get(frame)
        gt_original_ids, gt_points = gt_rows if gt_rows is not None else (empty_ids, np.empty((0, 2)))
        pred_original_ids, pred_points = pred_rows if pred_rows is not None else (empty_ids, np.empty((0, 2)))
        gt_ids = _remap_ids(np.asarray(gt_original_ids, dtype=np.intp), gt_id_map)
        tracker_ids = _remap_ids(np.asarray(pred_original_ids, dtype=np.intp), tracker_id_map)
        similarity = _euclidean_similarity(gt_points, pred_points, zero_distance)
        per_frame_gt_ids[frame - 1] = gt_ids
        per_frame_tracker_ids[frame - 1] = tracker_ids
        per_frame_similarity[frame - 1] = similarity
        total_gt += len(gt_ids)
        total_tracker += len(tracker_ids)

    return _MultiCameraSequenceData(
        gt_ids=per_frame_gt_ids,
        tracker_ids=per_frame_tracker_ids,
        similarity_scores=per_frame_similarity,
        num_frames=num_frames,
        num_gt_ids=len(gt_id_map),
        num_tracker_ids=len(tracker_id_map),
        num_gt_dets=total_gt,
        num_tracker_dets=total_tracker,
        gt_id_mapping=gt_id_map,
        tracker_id_mapping=tracker_id_map,
    )
