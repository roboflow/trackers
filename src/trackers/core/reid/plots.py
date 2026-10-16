# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Plots for choosing an appearance threshold, built on :mod:`trackers.core.reid.thresholds`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from trackers.core.reid.thresholds import AppearanceDistances

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_SAME_ID_COLOR = "#3366CC"
_DIFFERENT_ID_COLOR = "#DC3912"

_THRESHOLD_STYLES = (("#111111", "--", 1.6), ("#666666", ":", 1.5))

ThresholdLines = Sequence[float] | Mapping[float, str]


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
