# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Association-local appearance distance sampling for threshold selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from trackers.core.reid.appearance import _l2_normalize_rows, _require_embedding_matrix

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Frame gaps to sweep by default, in frames. The first band is the consecutive-frame
# case the original BoT-SORT figure measured; the later ones cover re-finding a track
# after an occlusion, which is where a single threshold is most likely to break down.
DEFAULT_FRAME_GAP_BANDS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 5),
    (6, 15),
    (16, 30),
    (31, 60),
    (61, 120),
    (121, 240),
)

# Percentile band drawn around the median in the sweep plot. Symmetric so both
# classes are read the same way.
_BAND_PERCENTILES = (10, 90)
_SAME_ID_COLOR = "#3366CC"
_DIFFERENT_ID_COLOR = "#DC3912"
_MATPLOTLIB_HINT = "Plotting appearance distances requires matplotlib.\nInstall with: pip install 'trackers[reid]'"

# Reference lines: the first threshold is the one under consideration, the rest are
# there for comparison, so they are drawn to recede.
_THRESHOLD_STYLES = (("#111111", "--", 1.6), ("#666666", ":", 1.5))

# Thresholds to draw, optionally annotated: ``(0.20, 0.25)`` or
# ``{0.20: "selected", 0.25: "default"}``.
ThresholdLines = Sequence[float] | Mapping[float, str]

# Half-open slot ranges before and after an anchor frame, each ``(start, stop)``.
_SlotRanges = tuple[tuple[int, int], tuple[int, int]]
_SameIdCandidates = list[tuple[np.ndarray, np.ndarray]]
_CandidatesT = TypeVar("_CandidatesT")


def _scalar_key(value: object) -> Hashable:
    """Return a hashable Python scalar without changing the label's value."""
    scalar = value.item() if isinstance(value, np.generic) else value
    if not isinstance(scalar, Hashable):
        raise TypeError(f"labels must be hashable scalars, got {type(scalar).__name__}")
    return scalar


class _NoPairsInBand(ValueError):
    """Signal that valid inputs contain no sampleable pairs in one gap band."""


@dataclass(frozen=True)
class AppearanceDistances:
    """Sampled appearance distances for one frame-gap band.

    Distances are ``0.5 * (1 - cosine_similarity)``, the term BoT-SORT gates on
    with ``appearance_threshold``.

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
        ``fuse_botsort_reid_association``, which discards appearance only once the
        distance *exceeds* ``appearance_threshold``.

        Args:
            threshold: Candidate ``appearance_threshold``.

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
        ``1.0`` when the two distributions never cross, ``0.5`` when appearance
        carries no information.

    Raises:
        ValueError: If either distance array is empty.
    """
    if len(same_id) == 0 or len(different_id) == 0:
        raise ValueError("both distance arrays must be non-empty")
    ordered = np.sort(different_id)
    right = np.searchsorted(ordered, same_id, side="right")
    left = np.searchsorted(ordered, same_id, side="left")
    greater_count = len(ordered) - right
    equal_count = right - left
    return float(np.mean((greater_count + 0.5 * equal_count) / len(ordered)))


class _SequenceIndex:
    """Frame-sorted crop index for one sequence, with binary-search gap lookup.

    A *slot* is a position in the frame-sorted order, not a crop index.

    Attributes:
        order: Crop indexes sorted by frame.
        frames: Frame number per slot.
        ids: Identity per slot.
        slots_by_id: Slots per identity, frame-sorted because the sort is stable.
    """

    def __init__(self, crop_indexes: np.ndarray, ids: np.ndarray, frame_ids: np.ndarray) -> None:
        order = np.argsort(frame_ids[crop_indexes], kind="stable")
        self.order = crop_indexes[order]
        self.frames = frame_ids[self.order]
        self.ids = ids[self.order]
        slots_by_id: dict[Hashable, list[int]] = defaultdict(list)
        for slot, identity in enumerate(self.ids):
            slots_by_id[_scalar_key(identity)].append(slot)
        self.slots_by_id = {identity: np.asarray(slots) for identity, slots in slots_by_id.items()}

    def window(
        self,
        frames: np.ndarray,
        anchor_frame: int,
        minimum_frame_gap: int,
        maximum_frame_gap: int,
    ) -> _SlotRanges:
        """Half-open slot ranges of ``frames`` lying that far before and after ``anchor_frame``."""
        return (
            (
                int(np.searchsorted(frames, anchor_frame - maximum_frame_gap, side="left")),
                int(np.searchsorted(frames, anchor_frame - minimum_frame_gap, side="right")),
            ),
            (
                int(np.searchsorted(frames, anchor_frame + minimum_frame_gap, side="left")),
                int(np.searchsorted(frames, anchor_frame + maximum_frame_gap, side="right")),
            ),
        )


def _pick_in_window(rng: np.random.Generator, window: _SlotRanges) -> int | None:
    """Uniformly pick one slot across both ranges, or ``None`` if both are empty."""
    (before_start, before_stop), (after_start, after_stop) = window
    before_count = max(0, before_stop - before_start)
    after_count = max(0, after_stop - after_start)
    if before_count + after_count == 0:
        return None
    draw = int(rng.integers(before_count + after_count))
    return before_start + draw if draw < before_count else after_start + (draw - before_count)


def _window_size(window: _SlotRanges) -> int:
    """Return the total number of slots in both halves of a gap window."""
    return sum(max(0, stop - start) for start, stop in window)


def _same_id_candidates(
    index: _SequenceIndex,
    minimum_frame_gap: int,
    maximum_frame_gap: int,
) -> _SameIdCandidates | None:
    """Collect identities and anchors that have a same-ID partner in the active gap band."""
    candidates: _SameIdCandidates = []
    for slots in index.slots_by_id.values():
        frames = index.frames[slots]
        anchors = [
            int(slot)
            for slot in slots
            if _window_size(index.window(frames, int(index.frames[slot]), minimum_frame_gap, maximum_frame_gap)) > 0
        ]
        if anchors:
            candidates.append((slots, np.asarray(anchors)))
    return candidates or None


def _draw_same_id_pair(
    rng: np.random.Generator,
    index: _SequenceIndex,
    candidates: _SameIdCandidates,
    minimum_frame_gap: int,
    maximum_frame_gap: int,
) -> tuple[int, int]:
    """Pick a valid identity uniformly, then a valid anchor and in-band partner."""
    slots, anchors = candidates[int(rng.integers(len(candidates)))]
    anchor = int(anchors[int(rng.integers(len(anchors)))])
    window = index.window(index.frames[slots], int(index.frames[anchor]), minimum_frame_gap, maximum_frame_gap)
    partner_slot = _pick_in_window(rng, window)
    if partner_slot is None:
        raise RuntimeError("same-ID candidate has no partner in the active gap band")
    partner = int(slots[partner_slot])
    return int(index.order[anchor]), int(index.order[partner])


def _slots_in_window(window: _SlotRanges) -> np.ndarray:
    """Return every slot contained in a two-part gap window."""
    return np.concatenate([np.arange(start, stop) for start, stop in window])


def _different_id_partners(
    index: _SequenceIndex,
    anchor: int,
    minimum_frame_gap: int,
    maximum_frame_gap: int,
) -> np.ndarray:
    """Return in-band partner slots whose identity differs from the anchor."""
    window = index.window(index.frames, int(index.frames[anchor]), minimum_frame_gap, maximum_frame_gap)
    partners = _slots_in_window(window)
    return partners[index.ids[partners] != index.ids[anchor]]


def _different_id_candidates(
    index: _SequenceIndex,
    minimum_frame_gap: int,
    maximum_frame_gap: int,
) -> list[int] | None:
    """Collect anchors that have a different-ID partner in the active gap band."""
    anchors = [
        anchor
        for anchor in range(len(index.order))
        if len(_different_id_partners(index, anchor, minimum_frame_gap, maximum_frame_gap)) > 0
    ]
    return anchors or None


def _draw_different_id_pair(
    rng: np.random.Generator,
    index: _SequenceIndex,
    candidates: list[int],
    minimum_frame_gap: int,
    maximum_frame_gap: int,
) -> tuple[int, int]:
    """Pick a valid anchor uniformly, then a different-ID in-band partner."""
    anchor = candidates[int(rng.integers(len(candidates)))]
    partners = _different_id_partners(index, anchor, minimum_frame_gap, maximum_frame_gap)
    partner = int(partners[int(rng.integers(len(partners)))])
    return int(index.order[anchor]), int(index.order[partner])


def _split_quota(total: int, bucket_count: int) -> list[int]:
    """Spread ``total`` draws as evenly as possible over ``bucket_count`` buckets."""
    base, remainder = divmod(total, bucket_count)
    return [base + (1 if index < remainder else 0) for index in range(bucket_count)]


def _draw_distances(
    rng: np.random.Generator,
    indexes: Mapping[Hashable, _SequenceIndex],
    embeddings: np.ndarray,
    build_candidates: Callable[[_SequenceIndex, int, int], _CandidatesT | None],
    draw_pair: Callable[[np.random.Generator, _SequenceIndex, _CandidatesT, int, int], tuple[int, int]],
    *,
    total_pairs: int,
    minimum_frame_gap: int,
    maximum_frame_gap: int,
) -> np.ndarray:
    """Draw pairs after splitting the quota equally over active sequences."""
    active_indexes = [
        (index, candidates)
        for index in indexes.values()
        if (candidates := build_candidates(index, minimum_frame_gap, maximum_frame_gap)) is not None
    ]
    if not active_indexes:
        return np.asarray([], dtype=np.float64)

    distances: list[float] = []
    quotas = _split_quota(total_pairs, len(active_indexes))
    for (index, candidates), quota in zip(active_indexes, quotas, strict=True):
        for _ in range(quota):
            pair = draw_pair(rng, index, candidates, minimum_frame_gap, maximum_frame_gap)
            first, second = pair
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
    maximum_attempts_per_pair: int = 64,
) -> AppearanceDistances:
    """Sample same-ID and different-ID crop pairs inside one frame-gap band.

    Candidate identities and anchors are filtered to those with a partner in the
    active gap band before sampling. Every sequence with valid candidates gets an
    equal quota. Within a sequence, same-ID pairs pick a valid identity uniformly
    so that long tracks cannot dominate, and different-ID pairs pick a valid
    anchor crop uniformly, then a valid partner uniformly inside the band.

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
        maximum_attempts_per_pair: Retained for compatibility. Candidate
            prefiltering makes retries unnecessary, so this value has no effect.

    Returns:
        The sampled distances for this band.

    Raises:
        TypeError: If identity or sequence labels are not hashable scalars.
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
        crop_indexes_by_sequence[_scalar_key(sequence)].append(crop_index)
    indexes = {
        sequence: _SequenceIndex(np.asarray(crop_indexes), ids, frame_ids)
        for sequence, crop_indexes in crop_indexes_by_sequence.items()
    }

    same_id = _draw_distances(
        rng,
        indexes,
        normalized,
        _same_id_candidates,
        _draw_same_id_pair,
        total_pairs=same_id_pairs,
        minimum_frame_gap=minimum_frame_gap,
        maximum_frame_gap=maximum_frame_gap,
    )
    different_id = _draw_distances(
        rng,
        indexes,
        normalized,
        _different_id_candidates,
        _draw_different_id_pair,
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


def _threshold_lines(thresholds: ThresholdLines) -> Iterator[tuple[float, str, str, str, float]]:
    """Yield ``(value, label, color, linestyle, linewidth)`` for each reference line."""
    notes = thresholds if isinstance(thresholds, Mapping) else {}
    for index, value in enumerate(thresholds):
        note = notes.get(value)
        label = f"θ = {value:.2f}" + (f" ({note})" if note else "")
        color, linestyle, linewidth = _THRESHOLD_STYLES[min(index, len(_THRESHOLD_STYLES) - 1)]
        yield value, label, color, linestyle, linewidth


def _pyplot() -> ModuleType:
    """Import ``matplotlib.pyplot``, or explain how to install it."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(_MATPLOTLIB_HINT) from exc
    return plt


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

    Raises:
        ImportError: If matplotlib is not installed.
    """
    plt = _pyplot()
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
    for value, label, color, linestyle, linewidth in _threshold_lines(thresholds):
        ax.axvline(value, color=color, ls=linestyle, lw=linewidth, label=label)

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
        title: Figure title.

    Returns:
        The figure the sweep was drawn on.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If ``sweep`` is empty.
    """
    if len(sweep) == 0:
        raise ValueError("sweep is empty; nothing to plot")
    plt = _pyplot()
    low_percentile, high_percentile = _BAND_PERCENTILES
    positions = np.arange(len(sweep))

    figure, (ax_distance, ax_auc) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    for attribute, color, name in (
        ("same_id", _SAME_ID_COLOR, "same ID"),
        ("different_id", _DIFFERENT_ID_COLOR, "different ID"),
    ):
        quantiles = np.array(
            [np.percentile(getattr(band, attribute), [low_percentile, 50, high_percentile]) for band in sweep]
        )
        ax_distance.fill_between(positions, quantiles[:, 0], quantiles[:, 2], color=color, alpha=0.22)
        ax_distance.plot(positions, quantiles[:, 1], color=color, marker="o", lw=2, label=name)
    for value, label, color, linestyle, linewidth in _threshold_lines(thresholds):
        ax_distance.axhline(value, color=color, ls=linestyle, lw=linewidth, label=label)

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
