# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Appearance similarity utilities for tracker association."""

from __future__ import annotations

import numpy as np


def appearance_similarity(
    track_features: list[np.ndarray | None],
    det_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between track features and detection embeddings.

    Both inputs are expected to be L2-normalised. Tracks with ``None`` features
    receive similarity ``0.0``.

    Args:
        track_features: One embedding per track (``None`` = no feature yet).
        det_embeddings: Detection embeddings, shape ``(N, D)``.

    Returns:
        Similarity matrix of shape ``(T, N)``.
    """
    n_tracks = len(track_features)
    n_dets = len(det_embeddings)
    sim = np.zeros((n_tracks, n_dets), dtype=np.float32)

    if n_tracks == 0 or n_dets == 0:
        return sim

    # Collect indices and embeddings of tracks that have a feature.
    valid_indices = [i for i, f in enumerate(track_features) if f is not None]
    if not valid_indices:
        return sim

    track_matrix = np.stack([track_features[i] for i in valid_indices])  # (K, D)
    cos_sims = (track_matrix @ det_embeddings.T).astype(np.float32)  # (K, N)

    for local_idx, track_idx in enumerate(valid_indices):
        sim[track_idx] = cos_sims[local_idx]

    return sim
