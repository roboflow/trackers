# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Appearance helpers for BoT-SORT association."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import supervision as sv

_NORM_EPS = 1e-12


class ReIDEncoder(Protocol):
    """Encoder interface used by BoT-SORT appearance association."""

    def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        """Return appearance embeddings for each detection box."""
        ...


class FeatureBank:
    """Per-track EMA appearance embedding.

    Args:
        alpha: EMA momentum in ``[0, 1]``.
    """

    def __init__(self, alpha: float = 0.9) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha
        self._feature: np.ndarray | None = None

    @property
    def feature(self) -> np.ndarray | None:
        """Current stored embedding, or ``None`` if never updated."""
        return None if self._feature is None else self._feature.copy()

    @property
    def is_initialized(self) -> bool:
        """``True`` after the first update."""
        return self._feature is not None

    def update(self, embedding: np.ndarray) -> None:
        """Blend an embedding into the stored feature."""
        cleaned = _require_embedding(embedding)

        if self._feature is None:
            self._feature = cleaned.copy()
            return

        if self._feature.shape != cleaned.shape:
            raise ValueError(
                f"embedding shape {cleaned.shape} does not match stored feature shape {self._feature.shape}"
            )

        self._feature = (self._alpha * self._feature + (1.0 - self._alpha) * cleaned).astype(np.float32)

    def reset(self) -> None:
        """Clear the stored feature."""
        self._feature = None


def _require_embedding(embedding: np.ndarray) -> np.ndarray:
    """Return a finite 1-D float32 vector."""
    flat = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise ValueError("embedding must be non-empty")
    if not np.all(np.isfinite(flat)):
        raise ValueError("embedding must contain only finite values")
    return flat


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
    flat = _require_embedding(embedding).astype(np.float64)
    norm = float(np.linalg.norm(flat))
    return (flat / max(norm, _NORM_EPS)).astype(np.float32)


def _l2_normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise each row in an embedding matrix."""
    if embeddings.size == 0:
        return embeddings
    return np.stack([_l2_normalize(row) for row in embeddings])


def extract_detection_embeddings(
    model: ReIDEncoder,
    frame: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    """Extract appearance embeddings for detection boxes."""
    if len(boxes) == 0:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = _require_embedding_matrix(model.extract_features(sv.Detections(xyxy=boxes), frame))
    if embeddings.shape[0] != len(boxes):
        raise ValueError(f"embedding rows ({embeddings.shape[0]}) must match detection boxes ({len(boxes)})")
    return embeddings


def appearance_similarity(
    track_features: Sequence[np.ndarray | None],
    det_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between track and detection embeddings."""
    n_tracks = len(track_features)
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
        track_rows.append(_l2_normalize(flat))
        kept_indices.append(track_idx)

    if not track_rows:
        return similarity

    cosine_similarities = (np.stack(track_rows) @ det_embeddings.T).astype(np.float32)
    for local_idx, track_idx in enumerate(kept_indices):
        similarity[track_idx] = cosine_similarities[local_idx]

    return similarity
