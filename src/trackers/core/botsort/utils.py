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
    it satisfies at least one liveness condition:

    - it was updated this frame (``time_since_update == 0``), OR
    - it holds a real tracker ID (``tracker_id != -1``) — sticky maturity:
      a track instant-activated on the first frame is preserved on a miss
      rather than deleted, OR
    - it has accumulated at least ``minimum_consecutive_frames`` successful
      updates.

    Args:
        tracklets: List of BoTSORTTracklet objects.
        minimum_consecutive_frames: Number of successful updates required for
            maturity. Ignored for tracklets that already hold a real
            ``tracker_id`` (sticky maturity path).
        maximum_frames_without_update: Maximum number of frames without update
            before a track is considered dead.

    Returns:
        List of alive tracklets.
    """
    alive_tracklets = []
    for tracker in tracklets:
        # Maturity is sticky: number_of_successful_updates is cumulative and
        # never decremented, but instant first-frame activation assigns
        # tracker_id before nsu reaches minimum_consecutive_frames. The
        # tracker_id != -1 guard keeps those instant-activated tracks alive
        # on a miss without waiting for nsu to catch up.
        is_mature = tracker.tracker_id != -1 or tracker.number_of_successful_updates >= minimum_consecutive_frames
        is_active = tracker.time_since_update == 0
        if tracker.time_since_update < maximum_frames_without_update and (is_mature or is_active):
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
