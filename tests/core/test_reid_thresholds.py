# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Appearance threshold-selection sampling and metrics tests."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from trackers.core.reid.thresholds import (
    AppearanceDistances,
    plot_appearance_distances,
    plot_frame_gap_sweep,
    roc_auc,
    sample_appearance_distances,
    sweep_frame_gap,
)

# Two sequences, two identities, one crop of each identity in frames 1 to 6. The two
# identity vectors are orthogonal, so a same-ID pair is exactly 0 apart and a
# different-ID pair exactly 0.5, whichever frames the sampler happens to draw.
_FIRST_IDENTITY = [1.0, 0.0]
_SECOND_IDENTITY = [0.0, 1.0]
_EMBEDDINGS = np.array([_FIRST_IDENTITY, _SECOND_IDENTITY] * 12, dtype=np.float32)
_IDS = np.array([0, 1] * 12)
_FRAME_IDS = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6] * 2)
_SEQUENCE_IDS = np.array([0] * 12 + [1] * 12)
_DATASET = (_EMBEDDINGS, _IDS, _FRAME_IDS, _SEQUENCE_IDS)


class TestRocAuc:
    """Unit tests for the threshold-free separability metric."""

    def test_disjoint_distributions_score_one(self) -> None:
        assert roc_auc(np.array([0.0, 0.1]), np.array([0.5, 0.6])) == pytest.approx(1.0)

    def test_reversed_distributions_score_zero(self) -> None:
        assert roc_auc(np.array([0.5, 0.6]), np.array([0.0, 0.1])) == pytest.approx(0.0)

    def test_identical_distributions_are_a_coin_flip(self) -> None:
        """All ties, so every comparison counts as half."""
        values = np.array([0.2, 0.2, 0.2])
        assert roc_auc(values, values) == pytest.approx(0.5)

    def test_ties_count_as_half(self) -> None:
        # One same-ID value below both, one exactly equal to one of them.
        assert roc_auc(np.array([0.0, 0.5]), np.array([0.5, 0.9])) == pytest.approx(0.875)


class TestSampleAppearanceDistances:
    """Unit tests for association-local pair sampling."""

    def test_draws_the_requested_pairs_from_each_class(self) -> None:
        distances = sample_appearance_distances(*_DATASET, same_id_pairs=8, different_id_pairs=10)

        assert len(distances.same_id) == 8
        assert len(distances.different_id) == 10
        np.testing.assert_allclose(distances.same_id, 0.0, atol=1e-6)
        np.testing.assert_allclose(distances.different_id, 0.5, atol=1e-6)
        assert distances.roc_auc == pytest.approx(1.0)

    def test_zero_frame_gap_is_rejected(self) -> None:
        """A gap of 0 would let a crop pair with itself and fake a spike at distance 0."""
        with pytest.raises(ValueError, match="invalid frame gap band"):
            sample_appearance_distances(*_DATASET, minimum_frame_gap=0)

    def test_empty_dataset_is_rejected(self) -> None:
        embeddings = np.empty((0, 2), dtype=np.float32)
        empty_labels = np.array([], dtype=int)

        with pytest.raises(ValueError, match="at least one row"):
            sample_appearance_distances(embeddings, empty_labels, empty_labels, empty_labels)

    def test_string_sequence_and_identity_labels_are_supported(self) -> None:
        embeddings = np.array([_FIRST_IDENTITY, _SECOND_IDENTITY] * 2, dtype=np.float32)

        distances = sample_appearance_distances(
            embeddings,
            np.array(["person-a", "person-b"] * 2),
            np.array([1, 1, 2, 2]),
            np.array(["camera-a"] * 4),
            same_id_pairs=4,
            different_id_pairs=4,
            maximum_frame_gap=1,
        )

        np.testing.assert_allclose(distances.same_id, 0.0, atol=1e-6)
        np.testing.assert_allclose(distances.different_id, 0.5, atol=1e-6)

    def test_distinct_non_integer_identity_labels_are_not_merged(self) -> None:
        embeddings = np.array([_FIRST_IDENTITY, _SECOND_IDENTITY] * 2, dtype=np.float32)

        distances = sample_appearance_distances(
            embeddings,
            np.array([1.2, 1.8] * 2),
            np.array([1, 1, 2, 2]),
            np.zeros(4),
            same_id_pairs=32,
            different_id_pairs=4,
            maximum_frame_gap=1,
        )

        np.testing.assert_allclose(distances.same_id, 0.0, atol=1e-6)
        np.testing.assert_allclose(distances.different_id, 0.5, atol=1e-6)

    def test_every_sequence_gets_an_equal_quota(self) -> None:
        """The per-sequence split is what stops one crowded sequence deciding the answer.

        Both identities in the second sequence are given the same embedding, so a different-ID pair drawn there measures
        0 while one from the first sequence measures 0.5. Asking for two pairs must produce one of each.
        """
        embeddings = _EMBEDDINGS.copy()
        embeddings[12:] = _FIRST_IDENTITY

        distances = sample_appearance_distances(
            embeddings,
            _IDS,
            _FRAME_IDS,
            _SEQUENCE_IDS,
            same_id_pairs=2,
            different_id_pairs=2,
        )

        np.testing.assert_allclose(sorted(distances.different_id), [0.0, 0.5], atol=1e-6)

    def test_pair_quotas_are_redistributed_over_sequences_valid_for_the_gap_band(self) -> None:
        embeddings = np.array(
            [_FIRST_IDENTITY, _SECOND_IDENTITY, _FIRST_IDENTITY, _SECOND_IDENTITY] * 2,
            dtype=np.float32,
        )

        distances = sample_appearance_distances(
            embeddings,
            np.array([0, 1, 0, 1] * 2),
            np.array([1, 1, 100, 100, 1, 1, 2, 2]),
            np.array(["unpairable"] * 4 + ["pairable"] * 4),
            same_id_pairs=8,
            different_id_pairs=8,
            maximum_frame_gap=1,
        )

        assert len(distances.same_id) == 8
        assert len(distances.different_id) == 8

    def test_same_id_sampling_is_uniform_over_identities_valid_for_the_gap_band(self) -> None:
        isolated_frames = np.arange(100, 1100, 10)
        embeddings = np.array(
            [_FIRST_IDENTITY, _FIRST_IDENTITY, _SECOND_IDENTITY]
            + [_FIRST_IDENTITY, _SECOND_IDENTITY]
            + [_FIRST_IDENTITY] * len(isolated_frames),
            dtype=np.float32,
        )
        ids = np.array(["dense", "dense", "single"] + ["sparse"] * (2 + len(isolated_frames)))
        frame_ids = np.concatenate(([1, 2, 1, 10, 11], isolated_frames))

        distances = sample_appearance_distances(
            embeddings,
            ids,
            frame_ids,
            np.zeros(len(ids)),
            same_id_pairs=2000,
            different_id_pairs=4,
            maximum_frame_gap=1,
            seed=7,
        )

        assert len(distances.same_id) == 2000
        assert np.mean(distances.same_id) == pytest.approx(0.25, abs=0.03)


class TestAppearanceDistances:
    """Unit tests for what a sampled band reports about itself."""

    @pytest.mark.parametrize(
        ("threshold", "expected"),
        [
            (0.05, (0.0, 0.0)),
            # On the boundary: the tracker keeps appearance at exactly the threshold.
            (0.1, (0.5, 0.0)),
            (0.15, (0.5, 0.0)),
            (0.45, (1.0, 0.5)),
            (1.0, (1.0, 1.0)),
        ],
    )
    def test_rates_at_counts_both_classes(self, threshold: float, expected: tuple[float, float]) -> None:
        distances = AppearanceDistances(
            same_id=np.array([0.1, 0.3]),
            different_id=np.array([0.4, 0.6]),
            minimum_frame_gap=1,
            maximum_frame_gap=1,
        )
        assert distances.rates_at(threshold) == expected

    @pytest.mark.parametrize(("band", "expected"), [((1, 1), "1"), ((6, 15), "6-15")])
    def test_label_describes_the_gap_band(self, band: tuple[int, int], expected: str) -> None:
        distances = AppearanceDistances(np.array([0.1]), np.array([0.5]), *band)
        assert distances.label == expected


def test_sweep_skips_bands_the_data_cannot_fill() -> None:
    """Six frames per sequence hold no gap wider than five, so the later bands drop out."""
    sweep = sweep_frame_gap(*_DATASET, pairs_per_class=4)

    assert [band.label for band in sweep] == ["1", "2-5"]


def test_sweep_rejects_mismatched_dataset_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        sweep_frame_gap(_EMBEDDINGS[:-1], _IDS, _FRAME_IDS, _SEQUENCE_IDS)


def test_both_plots_build() -> None:
    """Both figures build, and the histogram displays its complete data range."""
    distances = AppearanceDistances(
        same_id=np.array([0.0, 0.2]),
        different_id=np.array([0.5, 1.0]),
        minimum_frame_gap=1,
        maximum_frame_gap=1,
    )
    sweep = sweep_frame_gap(*_DATASET, pairs_per_class=4)

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
