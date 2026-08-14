# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Association-local appearance distance sampling for threshold selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from trackers.core.reid.appearance import _l2_normalize_rows, _require_embedding_matrix

if TYPE_CHECKING:
    from matplotlib.figure import Figure

DEFAULT_FRAME_GAP_BANDS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 5),
    (6, 15),
    (16, 30),
    (31, 60),
    (61, 120),
    (121, 240),
)

_SAME_ID_COLOR = "#3366CC"
_DIFFERENT_ID_COLOR = "#DC3912"

_THRESHOLD_STYLES = (("#111111", "--", 1.6), ("#666666", ":", 1.5))

ThresholdLines = Sequence[float] | Mapping[float, str]


class _NoPairsInBand(ValueError):
    """Signal that valid inputs contain no sampleable pairs in one gap band."""


@dataclass(frozen=True)
class AppearanceDistances:
    """Sampled appearance distances for one frame-gap band.

    Distances are ``0.5 * (1 - cosine_similarity)``, the term BoT-SORT gates on
    with ``reid_appearance_threshold``.

    Attributes:
        same_id: Distances between two crops of the same identity.
        different_id: Distances between crops of two different identities.
        minimum_frame_gap: Lower bound of the band the pairs were drawn from.
        maximum_frame_gap: Upper bound of the band the pairs were drawn from.
    """

    same_id: np.ndarray
    different_id: np.ndarray
    minimum_frame_gap: int
    maximum_frame_gap: int

    @property
    def label(self) -> str:
        """Gap band as an axis label, e.g. ``"1"`` or ``"6-15"``."""
        if self.minimum_frame_gap == self.maximum_frame_gap:
            return str(self.minimum_frame_gap)
        return f"{self.minimum_frame_gap}-{self.maximum_frame_gap}"

    @property
    def roc_auc(self) -> float:
        """Threshold-free separability of the two classes.

        See :func:`roc_auc`.
        """
        return roc_auc(self.same_id, self.different_id)

    def rates_at(self, threshold: float) -> tuple[float, float]:
        """Return the match rate of both classes at one candidate threshold.

        A pair counts as accepted at or below the threshold, matching the gate in
        :func:`trackers.core.reid.fusion.fuse_botsort_reid_association`, which discards
        appearance only once the distance *exceeds* ``reid_appearance_threshold``.

        Args:
            threshold: Candidate ``reid_appearance_threshold``.

        Returns:
            ``(same_id_rate, different_id_rate)``: the fraction of same-ID pairs
            accepted, to be maximised, and the fraction of different-ID pairs
            accepted, to be minimised.
        """
        return (
            float(np.mean(self.same_id <= threshold)),
            float(np.mean(self.different_id <= threshold)),
        )


def roc_auc(same_id: np.ndarray, different_id: np.ndarray) -> float:
    """Return the probability that a same-ID pair scores closer than a different-ID pair.

    Ties count as half. Equivalent to the area under the curve traced by sweeping
    the threshold from 0 to 1 and plotting the two rates from
    :meth:`AppearanceDistances.rates_at` against each other, which is why it
    summarises every threshold instead of one chosen operating point.

    Args:
        same_id: Distances between crops of the same identity.
        different_id: Distances between crops of different identities.

    Returns:
        ``1.0`` when every same-ID pair is closer than every different-ID pair,
        ``0.5`` when appearance carries no information, and ``0.0`` when the two
        classes are ordered the wrong way round.

    Raises:
        ValueError: If either distance array is empty.
    """
    if len(same_id) == 0 or len(different_id) == 0:
        raise ValueError("both distance arrays must be non-empty")
    # Compares every pair, so O(n*m); a rank-based form would scale better if pair counts ever grow.
    same = same_id[:, None]
    different = different_id[None, :]
    return float(np.mean(same < different) + 0.5 * np.mean(same == different))


def _window(frames: np.ndarray, anchor_frame: int, minimum_frame_gap: int, maximum_frame_gap: int) -> np.ndarray:
    """Positions into sorted ``frames`` lying in the gap band before or after ``anchor_frame``."""
    before = np.arange(
        np.searchsorted(frames, anchor_frame - maximum_frame_gap, side="left"),
        np.searchsorted(frames, anchor_frame - minimum_frame_gap, side="right"),
    )
    after = np.arange(
        np.searchsorted(frames, anchor_frame + minimum_frame_gap, side="left"),
        np.searchsorted(frames, anchor_frame + maximum_frame_gap, side="right"),
    )
    return np.concatenate((before, after))


class _SequenceIndex:
    """Frame-sorted crop index for one sequence, with binary-search gap lookup.

    A *slot* is a position in the frame-sorted order, not a crop index.

    Attributes:
        order: Crop indexes sorted by frame.
        frames: Frame number per slot.
        ids: Identity per slot.
        tracks: Slots per identity, frame-sorted because the sort is stable.
        track_of: Index into ``tracks`` per slot, so drawing never re-hashes a label.
    """

    def __init__(self, crop_indexes: np.ndarray, ids: np.ndarray, frame_ids: np.ndarray) -> None:
        order = np.argsort(frame_ids[crop_indexes], kind="stable")
        self.order = crop_indexes[order]
        self.frames = frame_ids[self.order]
        self.ids = ids[self.order]
        tracks: list[list[int]] = []
        track_by_id: dict[Hashable, int] = {}
        self.track_of = np.empty(len(self.ids), dtype=np.intp)
        for slot, identity in enumerate(self.ids):
            track = track_by_id.setdefault(identity, len(tracks))
            if track == len(tracks):
                tracks.append([])
            tracks[track].append(slot)
            self.track_of[slot] = track
        self.tracks = [np.asarray(slots) for slots in tracks]

    def get_candidates(
        self, anchor: int, minimum_frame_gap: int, maximum_frame_gap: int, *, same_id: bool
    ) -> np.ndarray:
        """Slots inside the gap band around ``anchor`` that may pair with it."""
        anchor_frame = int(self.frames[anchor])
        if same_id:
            slots = self.tracks[self.track_of[anchor]]
            return slots[_window(self.frames[slots], anchor_frame, minimum_frame_gap, maximum_frame_gap)]
        candidates = _window(self.frames, anchor_frame, minimum_frame_gap, maximum_frame_gap)
        return candidates[self.ids[candidates] != self.ids[anchor]]

    def get_anchor_groups(self, minimum_frame_gap: int, maximum_frame_gap: int, *, same_id: bool) -> list[np.ndarray]:
        """Anchors that have a candidate in the band, grouped so that every group is drawn equally often.

        Same-ID anchors group by identity, so a long track cannot dominate the sample. Different-ID anchors form a
        single group, i.e. uniform over crops.
        """
        grouped = self.tracks if same_id else [np.arange(len(self.order))]
        eligible = (
            np.asarray(
                [
                    s
                    for s in map(int, slots)
                    if len(self.get_candidates(s, minimum_frame_gap, maximum_frame_gap, same_id=same_id))
                ]
            )
            for slots in grouped
        )
        return [group for group in eligible if len(group)]


def _split_quota(total: int, bucket_count: int) -> list[int]:
    """Spread ``total`` draws as evenly as possible over ``bucket_count`` buckets."""
    base, remainder = divmod(total, bucket_count)
    return [base + (1 if index < remainder else 0) for index in range(bucket_count)]


def _draw_distances(
    rng: np.random.Generator,
    indexes: Mapping[Hashable, _SequenceIndex],
    embeddings: np.ndarray,
    *,
    same_id: bool,
    total_pairs: int,
    minimum_frame_gap: int,
    maximum_frame_gap: int,
) -> np.ndarray:
    """Draw pairs of one class, splitting the quota equally over the sequences that hold any."""
    active = [
        (index, groups)
        for index in indexes.values()
        if (groups := index.get_anchor_groups(minimum_frame_gap, maximum_frame_gap, same_id=same_id))
    ]
    if not active:
        return np.asarray([], dtype=np.float64)

    distances: list[float] = []
    for (index, groups), quota in zip(active, _split_quota(total_pairs, len(active)), strict=True):
        for _ in range(quota):
            group = groups[int(rng.integers(len(groups)))]
            anchor = int(group[int(rng.integers(len(group)))])
            candidates = index.get_candidates(anchor, minimum_frame_gap, maximum_frame_gap, same_id=same_id)
            candidate = int(candidates[int(rng.integers(len(candidates)))])
            first, second = index.order[anchor], index.order[candidate]
            distances.append(0.5 * (1.0 - float(embeddings[first] @ embeddings[second])))
    return np.asarray(distances, dtype=np.float64)


def sample_appearance_distances(
    embeddings: np.ndarray,
    ids: np.ndarray,
    frame_ids: np.ndarray,
    sequence_ids: np.ndarray,
    *,
    same_id_pairs: int = 5000,
    different_id_pairs: int = 5000,
    minimum_frame_gap: int = 1,
    maximum_frame_gap: int = 30,
    seed: int = 0,
) -> AppearanceDistances:
    """Sample same-ID and different-ID crop pairs inside one frame-gap band.

    Anchors are filtered to those that have a candidate in the active gap band
    before sampling, and every sequence holding any gets an equal quota. Within a
    sequence, same-ID pairs pick an identity uniformly so that long tracks cannot
    dominate, different-ID pairs pick an anchor crop uniformly, and both then pick
    a candidate uniformly inside the band.

    Args:
        embeddings: Appearance embeddings, shape ``(N, D)``. Normalised here, so
            either raw or unit-length input works.
        ids: Hashable scalar identity label per embedding, shape ``(N,)``.
        frame_ids: Frame number per embedding, shape ``(N,)``.
        sequence_ids: Hashable scalar sequence or video label per embedding, shape ``(N,)``.
        same_id_pairs: Same-ID pairs to draw across all sequences.
        different_id_pairs: Different-ID pairs to draw across all sequences.
        minimum_frame_gap: Smallest allowed frame gap. Must be at least 1: a gap
            of 0 lets a crop pair with itself, which is what puts the spike at
            distance 0 in the original BoT-SORT figure.
        maximum_frame_gap: Largest allowed frame gap, i.e. the association
            horizon being measured.
        seed: Seed for the pair-drawing generator.

    Returns:
        The sampled distances for this band.

    Raises:
        TypeError: If identity or sequence labels are not hashable.
        ValueError: If the arrays disagree in length, the gap band is invalid, or
            either class yielded no pairs at all.
    """
    if minimum_frame_gap < 1 or maximum_frame_gap < minimum_frame_gap:
        raise ValueError(
            f"invalid frame gap band [{minimum_frame_gap}, {maximum_frame_gap}], expected 1 <= minimum <= maximum"
        )
    embeddings = _require_embedding_matrix(embeddings)
    ids = np.asarray(ids)
    frame_ids = np.asarray(frame_ids)
    sequence_ids = np.asarray(sequence_ids)
    lengths = {len(embeddings), len(ids), len(frame_ids), len(sequence_ids)}
    if len(lengths) != 1:
        raise ValueError(f"embeddings, ids, frame_ids and sequence_ids must have equal length, got {sorted(lengths)}")
    if len(embeddings) == 0:
        raise ValueError("embeddings, ids, frame_ids and sequence_ids must contain at least one row")

    normalized = _l2_normalize_rows(embeddings)
    rng = np.random.default_rng(seed)
    crop_indexes_by_sequence: dict[Hashable, list[int]] = defaultdict(list)
    for crop_index, sequence in enumerate(sequence_ids):
        crop_indexes_by_sequence[sequence].append(crop_index)
    indexes = {
        sequence: _SequenceIndex(np.asarray(crop_indexes), ids, frame_ids)
        for sequence, crop_indexes in crop_indexes_by_sequence.items()
    }

    same_id = _draw_distances(
        rng,
        indexes,
        normalized,
        same_id=True,
        total_pairs=same_id_pairs,
        minimum_frame_gap=minimum_frame_gap,
        maximum_frame_gap=maximum_frame_gap,
    )
    different_id = _draw_distances(
        rng,
        indexes,
        normalized,
        same_id=False,
        total_pairs=different_id_pairs,
        minimum_frame_gap=minimum_frame_gap,
        maximum_frame_gap=maximum_frame_gap,
    )
    if len(same_id) == 0 or len(different_id) == 0:
        raise _NoPairsInBand(
            f"no association-local pairs in frame gap band [{minimum_frame_gap}, {maximum_frame_gap}]; "
            "widen the band or check that ids, frame_ids and sequence_ids line up with the embeddings"
        )
    return AppearanceDistances(
        same_id=same_id,
        different_id=different_id,
        minimum_frame_gap=minimum_frame_gap,
        maximum_frame_gap=maximum_frame_gap,
    )


def sweep_frame_gap(
    embeddings: np.ndarray,
    ids: np.ndarray,
    frame_ids: np.ndarray,
    sequence_ids: np.ndarray,
    *,
    gap_bands: Sequence[tuple[int, int]] = DEFAULT_FRAME_GAP_BANDS,
    pairs_per_class: int = 5000,
    seed: int = 0,
) -> list[AppearanceDistances]:
    """Measure separability inside each frame-gap band in turn.

    A threshold tuned on consecutive frames says nothing about re-finding a track
    after an occlusion, so this reports how far the same threshold carries as the
    gap widens. Bands that hold no pairs are skipped rather than raising, since a
    short dataset legitimately has no 240-frame gaps.

    Args:
        embeddings: Appearance embeddings, shape ``(N, D)``.
        ids: Identity label per embedding.
        frame_ids: Frame number per embedding.
        sequence_ids: Sequence or video label per embedding.
        gap_bands: ``(minimum, maximum)`` frame gaps to measure.
        pairs_per_class: Pairs drawn per class within each band.
        seed: Seed for the pair-drawing generator.

    Returns:
        One entry per band that yielded pairs, in ``gap_bands`` order.

    Raises:
        ValueError: If the input arrays or a requested gap band are invalid.
    """
    sweep: list[AppearanceDistances] = []
    for minimum_frame_gap, maximum_frame_gap in gap_bands:
        try:
            sweep.append(
                sample_appearance_distances(
                    embeddings,
                    ids,
                    frame_ids,
                    sequence_ids,
                    same_id_pairs=pairs_per_class,
                    different_id_pairs=pairs_per_class,
                    minimum_frame_gap=minimum_frame_gap,
                    maximum_frame_gap=maximum_frame_gap,
                    seed=seed,
                )
            )
        except _NoPairsInBand:
            continue
    return sweep


def _threshold_lines(thresholds: ThresholdLines) -> Iterator[tuple[float, dict[str, Any]]]:
    """Yield ``(value, line keyword arguments)`` for each reference line."""
    notes = thresholds if isinstance(thresholds, Mapping) else {}
    for index, value in enumerate(thresholds):
        note = notes.get(value)
        color, linestyle, linewidth = _THRESHOLD_STYLES[min(index, len(_THRESHOLD_STYLES) - 1)]
        label = f"θ = {value:.2f}" + (f" ({note})" if note else "")
        yield value, {"label": label, "color": color, "ls": linestyle, "lw": linewidth}


def plot_appearance_distances(
    distances: AppearanceDistances,
    *,
    thresholds: ThresholdLines = (0.25,),
    title: str | None = None,
) -> Figure:
    """Plot the two distance distributions with candidate thresholds marked.

    Where the two histograms overlap is where no threshold can separate them.

    Args:
        distances: Output of :func:`sample_appearance_distances`.
        thresholds: Candidate thresholds to draw as vertical reference lines. Pass a
            mapping to annotate them, e.g. ``{0.20: "selected", 0.25: "default"}``.
            The first is drawn dashed black and the rest recede into grey.
        title: Figure title. Defaults to naming the gap band that was sampled.

    Returns:
        The figure the distances were drawn on.
    """
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0.0, 1.0, 51).tolist()

    for values, color, name in (
        (distances.same_id, _SAME_ID_COLOR, "same ID"),
        (distances.different_id, _DIFFERENT_ID_COLOR, "different ID"),
    ):
        ax.hist(
            values,
            bins=bins,
            weights=np.full(len(values), 1.0 / len(values)),
            alpha=0.65,
            color=color,
            label=f"{name} (n={len(values)})",
        )
    for value, style in _threshold_lines(thresholds):
        ax.axvline(value, **style)

    gap = f"{distances.label} frame gap"
    ax.set(
        xlabel="appearance distance  (0.5 * (1 - cosine similarity))",
        ylabel="probability",
        title=title if title is not None else f"appearance distances, {gap}",
        xlim=(0.0, 1.0),
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    figure.tight_layout()
    return figure


def plot_frame_gap_sweep(
    sweep: Sequence[AppearanceDistances],
    *,
    thresholds: ThresholdLines = (0.25,),
    percentiles: tuple[int, int] = (10, 90),
    title: str | None = None,
) -> Figure:
    """Plot how separability degrades as the frame gap widens.

    The upper panel tracks each class's median and percentile band against the
    gap; the lower panel tracks :func:`roc_auc`, which answers the same question
    without committing to a threshold. Note that the two panels do not measure
    the same thing: the shaded bands can sit clear of each other while the AUC is
    still short of 1.0, because percentile ranges ignore where the mass sits.

    Args:
        sweep: Output of :func:`sweep_frame_gap`.
        thresholds: Candidate thresholds to draw as horizontal reference lines. Pass a
            mapping to annotate them, e.g. ``{0.20: "selected", 0.25: "default"}``.
        percentiles: ``(low, high)`` bounds of the band shaded around each class's
            median. Keep it symmetric so both classes are read the same way.
        title: Figure title.

    Returns:
        The figure the sweep was drawn on.

    Raises:
        ValueError: If ``sweep`` is empty.
    """
    if len(sweep) == 0:
        raise ValueError("sweep is empty; nothing to plot")
    import matplotlib.pyplot as plt

    low_percentile, high_percentile = percentiles
    positions = np.arange(len(sweep))

    figure, (ax_distance, ax_auc) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    for values, color, name in (
        ([band.same_id for band in sweep], _SAME_ID_COLOR, "same ID"),
        ([band.different_id for band in sweep], _DIFFERENT_ID_COLOR, "different ID"),
    ):
        quantiles = np.array([np.percentile(band, [low_percentile, 50, high_percentile]) for band in values])
        ax_distance.fill_between(positions, quantiles[:, 0], quantiles[:, 2], color=color, alpha=0.22)
        ax_distance.plot(positions, quantiles[:, 1], color=color, marker="o", lw=2, label=name)
    for value, style in _threshold_lines(thresholds):
        ax_distance.axhline(value, **style)

    ax_distance.set(ylabel="appearance distance")
    ax_distance.set_title(
        f"line = median, shaded = {low_percentile}th to {high_percentile}th percentile",
        fontsize=8.5,
        color="#333333",
        pad=4,
    )
    ax_distance.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.92, edgecolor="none")
    ax_distance.grid(True, alpha=0.25)

    auc_values = [band.roc_auc for band in sweep]
    ax_auc.plot(positions, auc_values, color="#111111", marker="s", lw=2)
    for position, auc in zip(positions, auc_values, strict=True):
        ax_auc.annotate(
            f"{auc:.3f}",
            (position, auc),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7.5,
            color="#111111",
        )
    ax_auc.axhline(0.5, color="#999999", ls=":", lw=1.2)
    ax_auc.set(
        xlabel="frames between the two crops",
        ylabel="separability",
        ylim=(0.42, 1.12),
        xticks=positions,
        xticklabels=[band.label for band in sweep],
    )
    ax_auc.set_title(
        "take one same-ID and one different-ID pair at random: how often is the same-ID one closer?"
        "\n1.0 = always, 0.5 = coin flip. Counts every sampled pair, not the shaded overlap above.",
        fontsize=8,
        color="#333333",
        pad=4,
    )
    ax_auc.grid(True, alpha=0.25)

    if title is not None:
        figure.suptitle(title, y=0.995)
    figure.tight_layout()
    return figure
