# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections.abc import Sequence

import numpy as np

from trackers.core.botsort.tracklet import BoTSORTTracklet


def get_alive_tracklets(
    tracklets: Sequence[BoTSORTTracklet],
    minimum_consecutive_frames: int,
    maximum_frames_without_update: int,
) -> list[BoTSORTTracklet]:
    """
    Remove dead or immature lost tracklets and return alive ones.

    A tracklet is kept if it is within ``maximum_frames_without_update`` **and**
    it is either mature (enough successful updates) or was just updated this
    frame.

    Args:
        tracklets: List of BoTSORTTracklet objects.
        minimum_consecutive_frames: Number of successful updates that an object
            must have before it is considered a 'valid' track.
        maximum_frames_without_update: Maximum number of frames without update
            before a track is considered dead.

    Returns:
        List of alive tracklets.
    """
    alive_tracklets = []
    for tracker in tracklets:
        is_mature = tracker.number_of_successful_updates >= minimum_consecutive_frames
        is_active = tracker.time_since_update == 0
        if tracker.time_since_update < maximum_frames_without_update and (
            is_mature or is_active
        ):
            alive_tracklets.append(tracker)
    return alive_tracklets


def _fuse_score(iou_similarity: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fuse IoU similarity matrix with detection confidence scores.

    Following the original ByteTrack implementation, the IoU similarity is
    multiplied element-wise by the detection scores.  This biases the
    association toward higher-confidence detections.

    Args:
        iou_similarity: IoU similarity matrix of shape ``(n_tracks, n_dets)``.
        scores: Detection confidence scores of shape ``(n_dets,)``.

    Returns:
        Fused similarity matrix of the same shape.
    """
    if iou_similarity.size == 0:
        return iou_similarity
    return iou_similarity * scores[np.newaxis, :]
