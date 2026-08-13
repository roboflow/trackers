# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Appearance embedding helpers for tracker association."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import supervision as sv

from trackers.core.reid.encoder import ReIDEncoder

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
        array([[1., 1.]], dtype=float32)
    """
    if len(boxes) == 0:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = _require_embedding_matrix(model.extract_features(sv.Detections(xyxy=boxes), frame))
    if embeddings.shape[0] != len(boxes):
        raise ValueError(f"embedding rows ({embeddings.shape[0]}) must match detection boxes ({len(boxes)})")
    return _l2_normalize_rows(embeddings)


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
