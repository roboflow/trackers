# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Appearance embedding helpers for tracker association."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import supervision as sv

from trackers.core.reid.encoder import ReIDEncoder
from trackers.io.frames import load_mot_frame_image, resolve_mot_frame_path
from trackers.io.mot import load_mot_file

_NORM_EPS = 1e-12


def _require_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return a finite float32 embedding matrix."""
    cleaned = np.asarray(embeddings, dtype=np.float32)
    if cleaned.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {cleaned.shape}")
    if cleaned.size > 0 and not np.all(np.isfinite(cleaned)):
        raise ValueError("embeddings must contain only finite values")
    return cleaned


def _l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """Return an L2-normalised 1-D vector."""
    flat = np.asarray(embedding, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("embedding must be non-empty")
    if not np.all(np.isfinite(flat)):
        raise ValueError("embedding must contain only finite values")
    norm = float(np.linalg.norm(flat))
    return (flat / max(norm, _NORM_EPS)).astype(np.float32)


def _l2_normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise each row in an embedding matrix."""
    if embeddings.size == 0:
        return embeddings
    mat = embeddings.astype(np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.maximum(norms, _NORM_EPS)).astype(np.float32)


def extract_detection_embeddings(
    model: ReIDEncoder,
    frame: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    """Extract appearance embeddings for detection boxes.

    Args:
        model: Encoder that returns one embedding per detection.
        frame: BGR image with shape ``(H, W, C)``.
        boxes: Detection boxes in ``xyxy`` format with shape ``(N, 4)``.

    Returns:
        Float32 embedding matrix with shape ``(N, D)``. Returns shape ``(0, 0)``
        when ``boxes`` is empty without calling ``model``.

    Raises:
        ValueError: If the encoder output is not a finite 2-D matrix or its row
            count does not match the number of boxes.

    Example:
        >>> class Encoder:
        ...     def extract_features(self, detections, frame):
        ...         return np.ones((len(detections), 2), dtype=np.float32)
        >>> frame = np.zeros((8, 8, 3), dtype=np.uint8)
        >>> boxes = np.array([[0.0, 0.0, 4.0, 4.0]], dtype=np.float32)
        >>> extract_detection_embeddings(Encoder(), frame, boxes)
        array([[0.70710677, 0.70710677]], dtype=float32)
    """
    if len(boxes) == 0:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = _require_embedding_matrix(model.extract_features(sv.Detections(xyxy=boxes), frame))
    if embeddings.shape[0] != len(boxes):
        raise ValueError(f"embedding rows ({embeddings.shape[0]}) must match detection boxes ({len(boxes)})")
    return _l2_normalize_rows(embeddings)


def extract_ground_truth_embeddings(
    model: ReIDEncoder,
    dataset_root: str | Path,
    *,
    sequences: Sequence[str] | None = None,
    keep_classes: Sequence[int] | None = None,
    frame_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Embed every ground-truth crop in a MOT-format dataset.

    Walks ``{dataset_root}/{sequence}/gt/gt.txt`` against the frames in
    ``{dataset_root}/{sequence}/img1``. Rows flagged ignore (confidence ``0``)
    are always dropped. Identities are renumbered across sequences, so the same
    track number in two videos stays two identities.

    The four returned arrays are what
    :func:`trackers.core.reid.sample_appearance_distances` expects, which is how
    an ``appearance_threshold`` gets calibrated on a new dataset.

    Args:
        model: Encoder to embed crops with.
        dataset_root: Directory holding one folder per sequence.
        sequences: Sequences to read. Defaults to every one found.
        keep_classes: MOT class ids to keep, e.g. ``(1,)`` for MOT17
            pedestrians. Defaults to every class, which suits single-class
            datasets such as SoccerNet.
        frame_stride: Embed every Nth frame. Raising it trims a dense dataset,
            at the cost of the smallest frame gaps.

    Returns:
        ``(embeddings, ids, frame_ids, sequence_ids)``, aligned row-wise.

    Raises:
        FileNotFoundError: If no sequence under ``dataset_root`` has a
            ``gt/gt.txt``, or a named sequence is missing one.
        ValueError: If no crop survived the filters.

    Examples:
        >>> from trackers.core.reid import extract_ground_truth_embeddings  # doctest: +SKIP
        >>>
        >>> crops = extract_ground_truth_embeddings(  # doctest: +SKIP
        ...     model, "mot17/val", keep_classes=(1,)
        ... )
    """
    root = Path(dataset_root)
    names = sorted(p.parent.parent.name for p in root.glob("*/gt/gt.txt")) if sequences is None else list(sequences)
    if not names:
        raise FileNotFoundError(f"no sequences with gt/gt.txt under {root}")

    embeddings: list[np.ndarray] = []
    ids: list[int] = []
    frame_ids: list[int] = []
    sequence_ids: list[int] = []
    identity_by_key: dict[str, int] = {}

    for sequence_id, name in enumerate(names):
        ground_truth = load_mot_file(root / name / "gt" / "gt.txt")
        frame_dir = root / name / "img1"
        for frame_id in range(1, max(ground_truth) + 1, frame_stride):
            rows = ground_truth.get(frame_id)
            if rows is None:
                continue
            keep = rows.confidences > 0
            if keep_classes is not None:
                keep &= np.isin(rows.classes, list(keep_classes))
            if not keep.any():
                continue
            frame_path = resolve_mot_frame_path(frame_dir, frame_id)
            if frame_path is None:
                continue
            boxes = sv.xywh_to_xyxy(rows.boxes[keep]).astype(np.float32)
            features = extract_detection_embeddings(model, load_mot_frame_image(frame_dir, frame_id), boxes)
            for feature, track_id in zip(features, rows.ids[keep], strict=True):
                key = f"{name}_{int(track_id)}"
                embeddings.append(feature)
                ids.append(identity_by_key.setdefault(key, len(identity_by_key)))
                frame_ids.append(frame_id)
                sequence_ids.append(sequence_id)

    if not embeddings:
        raise ValueError(f"no ground-truth crops under {root} survived the class and ignore-flag filters")
    return (
        np.stack(embeddings),
        np.asarray(ids, dtype=np.int64),
        np.asarray(frame_ids, dtype=np.int64),
        np.asarray(sequence_ids, dtype=np.int64),
    )


def appearance_similarity(
    track_features: Sequence[np.ndarray | None],
    det_embeddings: np.ndarray,
    *,
    det_embeddings_normalized: bool = False,
) -> np.ndarray:
    """Compute cosine similarities between track and detection embeddings.

    Args:
        track_features: Sequence of ``T`` track features, each with shape ``(D,)``.
            Entries may be ``None`` when a track has no appearance feature.
        det_embeddings: Detection embedding matrix with shape ``(N, D)``.
        det_embeddings_normalized: Whether detection rows are already validated
            unit embeddings from :func:`extract_detection_embeddings`.

    Returns:
        Float32 similarity matrix with shape ``(T, N)``. A ``None`` track feature
        produces an all-zero row.

    Raises:
        ValueError: If detection embeddings are not a finite 2-D matrix, or a
            track feature is empty, non-finite, or has the wrong dimension.

    Example:
        >>> tracks = [np.array([1.0, 0.0], dtype=np.float32), None]
        >>> detections = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        >>> appearance_similarity(tracks, detections)
        array([[1., 0.],
               [0., 0.]], dtype=float32)
    """
    n_tracks = len(track_features)
    if det_embeddings_normalized:
        det_embeddings = np.asarray(det_embeddings, dtype=np.float32)
    else:
        det_embeddings = _l2_normalize_rows(_require_embedding_matrix(det_embeddings))
    n_dets = det_embeddings.shape[0]
    similarity = np.zeros((n_tracks, n_dets), dtype=np.float32)

    if n_tracks == 0 or n_dets == 0:
        return similarity

    embed_dim = det_embeddings.shape[1]
    track_rows: list[np.ndarray] = []
    kept_indices: list[int] = []
    for track_idx, feature in enumerate(track_features):
        if feature is None:
            continue
        flat = np.asarray(feature, dtype=np.float32).reshape(-1)
        if flat.shape[0] != embed_dim:
            raise ValueError(
                f"track feature dim {flat.shape[0]} does not match detection "
                f"embedding dim {embed_dim} (track index {track_idx})"
            )
        track_rows.append(flat)
        kept_indices.append(track_idx)

    if not track_rows:
        return similarity

    normalized_track_rows = _l2_normalize_rows(_require_embedding_matrix(np.stack(track_rows)))
    cosine_similarities = (normalized_track_rows @ det_embeddings.T).astype(np.float32)
    similarity[kept_indices] = cosine_similarities

    return similarity
