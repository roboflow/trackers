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

    Both inputs are expected to be **L2-normalised**, so cosine similarity
    reduces to the dot product and is fast to compute.  Tracks that have no
    stored feature yet (``None``) receive a similarity of ``0.0`` for every
    detection, making the fused cost fall back to IoU alone for those entries.

    Args:
        track_features: One embedding per track, in the same order as the
            rows of the IoU matrix being fused.  ``None`` entries are treated
            as "no appearance information available".
        det_embeddings: Detection embeddings, shape ``(N, D)``.  Must be
            L2-normalised.

    Returns:
        Similarity matrix of shape ``(T, N)`` with values in ``[-1, 1]``
        (practically ``[0, 1]`` for well-trained re-ID embeddings).

    Examples:
        >>> import numpy as np
        >>> det = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> feats = [np.array([1.0, 0.0]), None]
        >>> sim = appearance_similarity(feats, det)
        >>> sim.shape
        (2, 2)
        >>> float(sim[0, 0])  # identical vectors → similarity 1
        1.0
        >>> float(sim[1, 0])  # no feature → similarity 0
        0.0
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
    cos_sims = (track_matrix @ det_embeddings.T).astype(np.float32)       # (K, N)

    for local_idx, track_idx in enumerate(valid_indices):
        sim[track_idx] = cos_sims[local_idx]

    return sim
