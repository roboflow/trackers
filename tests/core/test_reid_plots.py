# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Appearance threshold plotting tests."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from trackers.core.reid.plots import plot_appearance_distances, plot_frame_gap_sweep
from trackers.core.reid.threshold_selection import AppearanceDistances, sweep_frame_gap

# Two sequences, two orthogonal identities, one crop of each in frames 1 to 6.
_EMBEDDINGS = np.array([[1.0, 0.0], [0.0, 1.0]] * 12, dtype=np.float32)
_IDS = np.array([0, 1] * 12)
_FRAME_IDS = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6] * 2)
_SEQUENCE_IDS = np.array([0] * 12 + [1] * 12)


def test_both_plots_build() -> None:
    """Both figures build, and the histogram displays its complete data range."""
    distances = AppearanceDistances(
        same_id=np.array([0.0, 0.2]),
        different_id=np.array([0.5, 1.0]),
        minimum_frame_gap=1,
        maximum_frame_gap=1,
    )
    sweep = sweep_frame_gap(_EMBEDDINGS, _IDS, _FRAME_IDS, _SEQUENCE_IDS, pairs_per_class=4)

    histogram = plot_appearance_distances(distances, thresholds={0.2: "selected"})
    lower_bound, upper_bound = histogram.axes[0].get_xlim()

    assert lower_bound <= min(np.min(distances.same_id), np.min(distances.different_id))
    assert upper_bound >= max(np.max(distances.same_id), np.max(distances.different_id))
    assert plot_frame_gap_sweep(sweep) is not None


def test_importing_the_tracker_does_not_pull_matplotlib() -> None:
    """Plotting is opt-in, so the tracking path must not pay for matplotlib."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import trackers.core.botsort.tracker; assert 'matplotlib' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
