# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
#
# Adapted from NirAharon/BoT-SORT (MIT)
# Copyright (c) 2022 Nir Aharon
# Source: https://github.com/NirAharon/BoT-SORT
# Reference: tracker/bot_sort.py (ReID appearance-IoU cost fusion)
#
# Adapted from GerardMaggiolino/Deep-OC-SORT (MIT)
# Copyright (c) 2023 Gerard Maggiolino
# Source: https://github.com/GerardMaggiolino/Deep-OC-SORT
# Reference: trackers/integrated_ocsort_embedding/association.py
#            (compute_aw_new_metric, adaptive appearance weighting)
# ------------------------------------------------------------------------

"""Appearance-IoU fusion methods for ReID association.

Fusion methods are numpy-only and take track-detection similarity matrices, so they are reusable across trackers rather
than tied to any single one.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ReidFusionMethod = Literal["botsort", "adaptive"]


def fuse_botsort_reid_association(
    association_similarity: np.ndarray,
    appearance_similarity: np.ndarray,
    *,
    reid_proximity_threshold: float,
    reid_appearance_threshold: float,
    proximity_iou_similarity: np.ndarray | None = None,
) -> np.ndarray:
    """Fuse IoU and appearance the way BoT-SORT ``bot_sort.py`` does.

    Computes ``min(association_cost, capped_appearance_cost)`` with proximity
    and appearance gates, then returns the corresponding similarity matrix
    (``1 - cost``).

    ``proximity_iou_similarity`` is the standard-IoU gate (defaults to
    ``association_similarity``). Pass it separately when association uses
    GIoU/DIoU/CIoU so proximity still uses plain IoU.

    Args:
        association_similarity: Geometry-based track-detection similarities with
            shape ``(T, N)``.
        appearance_similarity: Cosine similarities for the same pairs with shape
            ``(T, N)``.
        reid_proximity_threshold: Maximum standard-IoU distance at which appearance
            may lower the association cost.
        reid_appearance_threshold: Maximum appearance cost allowed to contribute to
            the fused association.
        proximity_iou_similarity: Standard-IoU similarities with shape ``(T, N)``.
            Defaults to ``association_similarity``.

    Returns:
        Fused track-detection similarities with shape ``(T, N)``, obtained from
        ``1 - min(d_iou, d_app)`` after applying both gates.
    """
    if proximity_iou_similarity is None:
        proximity_iou_similarity = association_similarity

    d_iou = 1.0 - association_similarity
    d_iou_proximity = 1.0 - proximity_iou_similarity
    d_app = 0.5 * (1.0 - appearance_similarity)
    d_app = np.where(d_app > reid_appearance_threshold, 1.0, d_app)
    d_app = np.where(d_iou_proximity > reid_proximity_threshold, 1.0, d_app)
    fused_cost = np.minimum(d_iou, d_app)
    return 1.0 - fused_cost


def _top_two_margin(similarity: np.ndarray, axis: int, cap: float) -> np.ndarray:
    """Margin between the best and second-best similarity along ``axis``, capped.

    A large margin means the best candidate stands clear of the rest, so appearance is discriminative for that row or
    column. Zero when there are fewer than two candidates to compare.
    """
    if similarity.shape[axis] < 2:
        return np.zeros(similarity.shape[1 - axis], dtype=float)
    partitioned = np.partition(similarity, -2, axis=axis)
    best = np.take(partitioned, -1, axis=axis)
    runner_up = np.take(partitioned, -2, axis=axis)
    return np.minimum(best - runner_up, cap)


def fuse_adaptive_reid_association(
    association_similarity: np.ndarray,
    appearance_similarity: np.ndarray,
    *,
    reid_appearance_weight: float,
    reid_adaptive_weight_cap: float,
    reid_proximity_threshold: float,
    proximity_iou_similarity: np.ndarray | None = None,
) -> np.ndarray:
    """Fuse IoU and appearance with Deep OC-SORT's adaptive weighting.

    Adds a weighted appearance term to the geometric similarity instead of
    taking a minimum. The weight grows when the best appearance match stands
    clear of the runner-up and falls back to ``reid_appearance_weight`` when
    the top candidates are hard to tell apart.

    Implements equations (4)-(6) of the Deep OC-SORT paper, matching
    ``compute_aw_new_metric`` in the authors' ``integrated_ocsort_embedding``
    tracker, the variant behind their published results.

    ``proximity_iou_similarity`` is the standard-IoU gate (defaults to
    ``association_similarity``). Pass it separately when association uses
    GIoU/DIoU/CIoU so proximity still uses plain IoU.

    Args:
        association_similarity: Geometry-based track-detection similarities with
            shape ``(T, N)``.
        appearance_similarity: Cosine similarities for the same pairs with shape
            ``(T, N)``.
        reid_appearance_weight: Base appearance weight, ``a_w`` in the paper.
            The authors report ``0.75`` for MOT17 and MOT20 and ``1.25`` for
            DanceTrack.
        reid_adaptive_weight_cap: Upper bound on the adaptive bonus, ``epsilon``
            in the paper. The authors report ``0.5`` for MOT17 and MOT20 and
            ``1.0`` for DanceTrack.
        reid_proximity_threshold: Maximum standard-IoU distance at which
            appearance may contribute.
        proximity_iou_similarity: Standard-IoU similarities with shape ``(T, N)``.
            Defaults to ``association_similarity``.

    Returns:
        Fused track-detection similarities with shape ``(T, N)``, spanning
        ``[0, 1 + reid_appearance_weight + reid_adaptive_weight_cap]``.
        Association thresholds tuned against the BoT-SORT fusion do not carry
        over and need retuning.

    Notes:
        Negative cosine similarities are clamped to zero, so appearance can
        only ever raise a similarity.

        Geometric plausibility is enforced before fusion rather than after
        matching. The reference fuses appearance ungated, runs the assignment,
        then discards any matched pair whose raw IoU is below its IoU
        threshold. BoT-SORT has no such post-match check: its association
        threshold is applied to the fused similarity, which a zero-IoU pair
        can clear on appearance alone. ``reid_proximity_threshold`` is where
        that check lives here; ``1.0`` disables it, which is what lets a
        track be recovered after leaving the frame, since its prediction no
        longer overlaps the returning detection. The gate applies only to the
        appearance term: the margin is measured over all candidates, as in
        the reference, so a candidate excluded on geometry cannot read as
        similarity ``0`` and widen the margin.

        The velocity-direction term of the full Deep OC-SORT cost is not
        included; it belongs to OC-SORT's motion model.
    """
    if proximity_iou_similarity is None:
        proximity_iou_similarity = association_similarity

    appearance = np.clip(appearance_similarity, 0.0, None)

    track_margin = _top_two_margin(appearance, axis=1, cap=reid_adaptive_weight_cap)
    detection_margin = _top_two_margin(appearance, axis=0, cap=reid_adaptive_weight_cap)
    adaptive_bonus = (track_margin[:, np.newaxis] + detection_margin[np.newaxis, :]) / 2.0

    out_of_range = (1.0 - proximity_iou_similarity) > reid_proximity_threshold
    gated_appearance = np.where(out_of_range, 0.0, appearance)

    return association_similarity + (reid_appearance_weight + adaptive_bonus) * gated_appearance
