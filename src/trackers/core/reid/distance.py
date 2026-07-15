# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Appearance similarity utilities for tracker association."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trackers.core.reid.feature_bank import FeatureBank


def sanitize_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Replace non-finite rows with zeros so cosine similarity stays safe."""
    if embeddings.size == 0:
        return embeddings
    cleaned = np.asarray(embeddings, dtype=np.float32)
    if cleaned.ndim != 2:
        raise ValueError(f"det_embeddings must be 2-D, got shape {cleaned.shape}")
    row_finite = np.isfinite(cleaned).all(axis=1)
    if not np.all(row_finite):
        cleaned = cleaned.copy()
        cleaned[~row_finite] = 0.0
    return cleaned


def appearance_similarity(
    track_features: Sequence[np.ndarray | None],
    det_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between track features and detection embeddings.

    Both inputs are expected to be L2-normalised. Tracks with ``None`` features
    receive similarity ``0.0``. Non-finite detection rows are zeroed before the
    dot product so they cannot poison assignment. Track rows whose embedding
    dimension does not match the detection matrix are treated as unavailable.

    Args:
        track_features: One embedding per track (``None`` = no feature yet).
        det_embeddings: Detection embeddings, shape ``(N, D)``.

    Returns:
        Similarity matrix of shape ``(T, N)``.
    """
    n_tracks = len(track_features)
    det_embeddings = sanitize_embedding_matrix(det_embeddings)
    n_dets = det_embeddings.shape[0]
    sim = np.zeros((n_tracks, n_dets), dtype=np.float32)

    if n_tracks == 0 or n_dets == 0:
        return sim

    embed_dim = det_embeddings.shape[1]
    track_rows: list[np.ndarray] = []
    kept_indices: list[int] = []
    for track_idx, feature in enumerate(track_features):
        if feature is None:
            continue
        normalized = FeatureBank.normalize_embedding(feature)
        if normalized is None or normalized.shape[0] != embed_dim:
            continue
        track_rows.append(normalized)
        kept_indices.append(track_idx)

    if not track_rows:
        return sim

    track_matrix = np.stack(track_rows)
    cos_sims = (track_matrix @ det_embeddings.T).astype(np.float32)

    for local_idx, track_idx in enumerate(kept_indices):
        sim[track_idx] = cos_sims[local_idx]

    return sim
