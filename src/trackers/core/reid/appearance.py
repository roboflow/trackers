# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Appearance embedding helpers for tracker association."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trackers.core.reid.feature_bank import FeatureBank


def _require_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return embeddings as float32 ``(N, D)``, raising on bad shape or values."""
    cleaned = np.asarray(embeddings, dtype=np.float32)
    if cleaned.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {cleaned.shape}")
    if cleaned.size > 0 and not np.all(np.isfinite(cleaned)):
        raise ValueError("embeddings must contain only finite values")
    return cleaned


def _l2_normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise each row (eps floor), preserving shape ``(N, D)``."""
    if embeddings.size == 0:
        return embeddings
    return np.stack([FeatureBank.normalize_embedding(row) for row in embeddings])


def appearance_similarity(
    track_features: Sequence[np.ndarray | None],
    det_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between track features and detection embeddings.

    Both sides are L2-normalised before the dot product (cosine owns
    normalisation). Tracks with ``None`` features receive similarity ``0.0``.
    Non-finite values or mismatched embedding dimensions raise ``ValueError``.

    Args:
        track_features: One embedding per track (``None`` = no feature yet).
        det_embeddings: Detection embeddings, shape ``(N, D)``.

    Returns:
        Similarity matrix of shape ``(T, N)``.
    """
    n_tracks = len(track_features)
    det_embeddings = _l2_normalize_rows(_require_embedding_matrix(det_embeddings))
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
        flat = np.asarray(feature, dtype=np.float32).reshape(-1)
        if flat.shape[0] != embed_dim:
            raise ValueError(
                f"track feature dim {flat.shape[0]} does not match detection "
                f"embedding dim {embed_dim} (track index {track_idx})"
            )
        track_rows.append(FeatureBank.normalize_embedding(flat))
        kept_indices.append(track_idx)

    if not track_rows:
        return sim

    track_matrix = np.stack(track_rows)
    cos_sims = (track_matrix @ det_embeddings.T).astype(np.float32)

    for local_idx, track_idx in enumerate(kept_indices):
        sim[track_idx] = cos_sims[local_idx]

    return sim
