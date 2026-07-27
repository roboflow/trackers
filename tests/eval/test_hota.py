# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trackers.eval.hota import (
    ALPHA_THRESHOLDS,
    aggregate_hota_metrics,
    compute_hota_metrics,
)


def _sequential_hota_reference(
    gt_ids: list[np.ndarray],
    tracker_ids: list[np.ndarray],
    similarity_scores: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Pre-vectorization HOTA reference: per-alpha Python loop + dict id-remapping.

    Mirrors the pre-PR second-pass logic (commit 2ca10bd^) for differential testing.
    Used only by test_output_matches_sequential_reference to guard the hot path.
    """
    from scipy.optimize import linear_sum_assignment

    from trackers.eval.constants import EPS

    na = len(ALPHA_THRESHOLDS)
    n_gt = sum(len(x) for x in gt_ids)
    n_tr = sum(len(x) for x in tracker_ids)
    tp = np.zeros(na)
    fn = np.zeros(na)
    fp = np.zeros(na)
    la = np.zeros(na)

    if n_tr == 0:
        fn[:] = n_gt
        return {
            "HOTA_TP_array": tp,
            "HOTA_FN_array": fn,
            "HOTA_FP_array": fp,
            "LocA_array": np.ones(na),
            "AssA_array": np.zeros(na),
        }
    if n_gt == 0:
        fp[:] = n_tr
        return {
            "HOTA_TP_array": tp,
            "HOTA_FN_array": fn,
            "HOTA_FP_array": fp,
            "LocA_array": np.ones(na),
            "AssA_array": np.zeros(na),
        }

    ugt = np.unique(np.concatenate(gt_ids))
    utr = np.unique(np.concatenate(tracker_ids))
    gmap = {int(v): i for i, v in enumerate(ugt)}
    tmap = {int(v): i for i, v in enumerate(utr)}
    ng, nt = len(ugt), len(utr)
    pot = np.zeros((ng, nt))
    gc = np.zeros((ng, 1))
    tc = np.zeros((1, nt))

    for t_i, (gids, tids) in enumerate(zip(gt_ids, tracker_ids)):
        if len(gids) == 0 or len(tids) == 0:
            if len(gids):
                gc[[gmap[int(v)] for v in gids]] += 1
            if len(tids):
                tc[0, [tmap[int(v)] for v in tids]] += 1
            continue
        gi = np.array([gmap[int(v)] for v in gids])
        ti = np.array([tmap[int(v)] for v in tids])
        s = similarity_scores[t_i]
        denom = s.sum(0)[None, :] + s.sum(1)[:, None] - s
        siou = np.where(denom > EPS, s / np.where(denom > 0, denom, 1.0), 0.0)
        pot[gi[:, None], ti[None, :]] += siou
        gc[gi] += 1
        tc[0, ti] += 1

    ga = pot / (gc + tc - pot)
    mcs = [np.zeros((ng, nt)) for _ in range(na)]

    for t_i, (gids, tids) in enumerate(zip(gt_ids, tracker_ids)):
        if len(gids) == 0:
            fp += len(tids)
            continue
        if len(tids) == 0:
            fn += len(gids)
            continue
        gi = np.array([gmap[int(v)] for v in gids])
        ti = np.array([tmap[int(v)] for v in tids])
        s = similarity_scores[t_i]
        mr, mc = linear_sum_assignment(-(ga[gi[:, None], ti[None, :]] * s))
        for a, alpha in enumerate(ALPHA_THRESHOLDS):
            amask = s[mr, mc] >= alpha - EPS
            ar, ac = mr[amask], mc[amask]
            n = int(amask.sum())
            tp[a] += n
            fn[a] += len(gids) - n
            fp[a] += len(tids) - n
            if n:
                la[a] += s[ar, ac].sum()
                mcs[a][gi[ar], ti[ac]] += 1

    ass_a = np.zeros(na)
    for a in range(na):
        m = mcs[a]
        ass_a[a] = (m * (m / np.maximum(1, gc + tc - m))).sum() / max(1.0, tp[a])

    la = np.maximum(1e-10, la) / np.maximum(1e-10, tp)
    return {"HOTA_TP_array": tp, "HOTA_FN_array": fn, "HOTA_FP_array": fp, "LocA_array": la, "AssA_array": ass_a}


class TestComputeHOTAMetrics:
    """Per-sequence HOTA metric computation (DetA, AssA, LocA, HOTA)."""

    @pytest.mark.parametrize(
        (
            "gt_ids",
            "tracker_ids",
            "similarity_scores",
            "expected",
        ),
        [
            # Empty GT and tracker
            (
                [np.array([])],
                [np.array([])],
                [np.zeros((0, 0))],
                {"HOTA_TP": 0, "HOTA_FN": 0, "HOTA_FP": 0},
            ),
            # No tracker detections - all FN
            (
                [np.array([0, 1]), np.array([0, 1])],
                [np.array([]), np.array([])],
                [np.zeros((2, 0)), np.zeros((2, 0))],
                {
                    "HOTA": 0.0,
                    "DetA": 0.0,
                    "HOTA_FN": 4 * 19,
                    "HOTA_FP": 0,
                    "HOTA_TP": 0,
                },
            ),
            # No GT detections - all FP
            (
                [np.array([]), np.array([])],
                [np.array([10, 20]), np.array([10, 20])],
                [np.zeros((0, 2)), np.zeros((0, 2))],
                {
                    "HOTA": 0.0,
                    "DetA": 0.0,
                    "HOTA_FP": 4 * 19,
                    "HOTA_FN": 0,
                    "HOTA_TP": 0,
                },
            ),
            # Perfect tracking with high IoU
            (
                [np.array([0, 1]), np.array([0, 1])],
                [np.array([10, 20]), np.array([10, 20])],
                [
                    np.array([[0.9, 0.0], [0.0, 0.9]]),
                    np.array([[0.9, 0.0], [0.0, 0.9]]),
                ],
                {"HOTA_min": 0.8, "DetA_min": 0.8, "AssA_min": 0.9},
            ),
            # ID switch reduces AssA
            (
                [np.array([0, 1]), np.array([0, 1])],
                [np.array([10, 20]), np.array([10, 20])],
                [
                    np.array([[0.8, 0.1], [0.1, 0.8]]),  # Normal matching
                    np.array([[0.1, 0.8], [0.8, 0.1]]),  # Swapped!
                ],
                {"DetA_min": 0.5, "AssA_max": 0.8},
            ),
            # Low IoU passes fewer alpha thresholds
            (
                [np.array([0])],
                [np.array([10])],
                [np.array([[0.3]])],
                {"HOTA_min": 0.0, "HOTA_max": 0.5, "HOTA_TP_min": 1},
            ),
            # Multiple objects partial match
            (
                [np.array([0, 1, 2])],
                [np.array([10, 20])],  # Only 2 trackers for 3 GTs
                [
                    np.array(
                        [
                            [0.8, 0.0],
                            [0.0, 0.8],
                            [0.0, 0.0],  # GT2 has no match
                        ]
                    )
                ],
                {"HOTA_FN_min": 19, "DetRe_max": 1.0, "DetPr_max": 1.0},
            ),
            # Single frame perfect matching
            (
                [np.array([0])],
                [np.array([10])],
                [np.array([[0.8]])],
                {"HOTA_TP_min": 1, "LocA_min": 0.8},
            ),
        ],
    )
    def test_scenarios(
        self,
        gt_ids: list[np.ndarray[Any, np.dtype[Any]]],
        tracker_ids: list[np.ndarray[Any, np.dtype[Any]]],
        similarity_scores: list[np.ndarray[Any, np.dtype[Any]]],
        expected: dict[str, Any],
    ) -> None:
        """compute_hota_metrics produces expected HOTA fields across scenarios."""
        result = compute_hota_metrics(gt_ids, tracker_ids, similarity_scores)

        for field in [
            "HOTA",
            "DetA",
            "AssA",
            "DetRe",
            "DetPr",
            "AssRe",
            "AssPr",
            "LocA",
        ]:
            assert field in result
            assert isinstance(result[field], float)

        for field in ["HOTA_TP", "HOTA_FN", "HOTA_FP"]:
            assert field in result
            assert isinstance(result[field], int)

        for key, value in expected.items():
            if key.endswith("_min"):
                actual_key = key[:-4]
                assert result[actual_key] >= value, f"{actual_key} should be >= {value}, got {result[actual_key]}"
            elif key.endswith("_max"):
                actual_key = key[:-4]
                assert result[actual_key] <= value, f"{actual_key} should be <= {value}, got {result[actual_key]}"
            elif key.endswith("_approx"):
                actual_key = key[:-7]
                assert result[actual_key] == pytest.approx(value, rel=0.01), (
                    f"{actual_key} should be ~{value}, got {result[actual_key]}"
                )
            elif isinstance(value, float):
                assert result[key] == pytest.approx(value), f"Mismatch for {key}: {result[key]} != {value}"
            else:
                assert result[key] == value, f"Mismatch for {key}: {result[key]} != {value}"

    def test_result_structure(self) -> None:
        """Result contains all float summary, integer count, and array fields."""
        result = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.8]])],
        )

        for field in [
            "HOTA",
            "DetA",
            "AssA",
            "DetRe",
            "DetPr",
            "AssRe",
            "AssPr",
            "LocA",
            "OWTA",
        ]:
            assert field in result
            assert isinstance(result[field], float)
            assert 0 <= result[field] <= 1

        for field in ["HOTA_TP", "HOTA_FN", "HOTA_FP"]:
            assert field in result
            assert isinstance(result[field], int)
            assert result[field] >= 0

        for field in [
            "HOTA_TP_array",
            "HOTA_FN_array",
            "HOTA_FP_array",
            "AssA_array",
            "AssRe_array",
            "AssPr_array",
            "LocA_array",
        ]:
            assert field in result
            assert isinstance(result[field], np.ndarray)
            assert len(result[field]) == len(ALPHA_THRESHOLDS)

    def test_metrics_invariant_to_id_relabeling(self) -> None:
        """HOTA depends only on association structure, not on the integer id values.

        Relabeling ground-truth and tracker ids with a consistent, non-monotonic
        bijection (including ids that are unsorted within a frame) must leave every
        metric unchanged. This guards the internal id-to-index remapping.
        """
        gt_ids = [
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            np.array([0, 1]),
        ]
        tracker_ids = [
            np.array([10, 11, 12]),
            np.array([10, 11, 12]),
            np.array([10, 12]),  # tracker 11 drops out...
            np.array([10, 11, 12]),  # ...and reappears, exercising association
        ]
        similarity_scores = [
            np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.7]]),
            np.array([[0.85, 0.0, 0.1], [0.1, 0.75, 0.0], [0.0, 0.1, 0.7]]),
            np.array([[0.8, 0.1], [0.1, 0.0], [0.0, 0.7]]),
            np.array([[0.8, 0.1, 0.0], [0.1, 0.7, 0.2]]),
        ]

        baseline = compute_hota_metrics(gt_ids, tracker_ids, similarity_scores)

        # Non-monotonic bijection so the sorted-unique reindexing differs from identity
        # and relabeled ids are unsorted within a frame.
        gt_remap = {0: 1007, 1: 1003, 2: 1009}
        tracker_remap = {10: 5002, 11: 5008, 12: 5001}
        relabeled_gt = [np.array([gt_remap[i] for i in frame]) for frame in gt_ids]
        relabeled_tracker = [np.array([tracker_remap[i] for i in frame]) for frame in tracker_ids]

        relabeled = compute_hota_metrics(relabeled_gt, relabeled_tracker, similarity_scores)

        for key in ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA"]:
            assert relabeled[key] == pytest.approx(baseline[key]), f"{key} changed under id relabeling"
        for key in [
            "HOTA_TP_array",
            "HOTA_FN_array",
            "HOTA_FP_array",
            "AssA_array",
            "AssRe_array",
            "AssPr_array",
            "LocA_array",
        ]:
            np.testing.assert_allclose(relabeled[key], baseline[key], err_msg=f"{key} changed under id relabeling")

    @pytest.mark.parametrize(
        ("gt_ids", "tracker_ids", "similarity_scores"),
        [
            pytest.param(
                [np.array([0, 1]), np.array([0, 1]), np.array([0, 1])],
                [np.array([10, 20]), np.array([10, 20]), np.array([10, 20])],
                [
                    np.array([[0.9, 0.0], [0.0, 0.8]]),
                    np.array([[0.85, 0.1], [0.05, 0.75]]),
                    np.array([[0.8, 0.0], [0.0, 0.7]]),
                ],
                id="normal-contiguous-ids",
            ),
            pytest.param(
                [np.array([0, 1]), np.array([0, 1])],
                [np.array([10, 20]), np.array([10, 20])],
                [np.array([[0.8, 0.1], [0.1, 0.8]]), np.array([[0.1, 0.8], [0.8, 0.1]])],
                id="id-switch",
            ),
            pytest.param(
                [np.array([1007, 1003, 1009]), np.array([1003, 1009])],
                [np.array([5002, 5008, 5001]), np.array([5001, 5008])],
                [
                    np.array([[0.9, 0.0, 0.1], [0.0, 0.8, 0.0], [0.1, 0.0, 0.7]]),
                    np.array([[0.0, 0.8], [0.7, 0.0]]),
                ],
                id="non-monotonic-ids",
            ),
            pytest.param(
                [np.array([0, 1, 2]), np.array([0, 1])],
                [np.array([10, 20]), np.array([10, 20, 30])],
                [
                    np.array([[0.8, 0.0], [0.0, 0.75], [0.0, 0.0]]),
                    np.array([[0.85, 0.05, 0.0], [0.05, 0.8, 0.1]]),
                ],
                id="partial-match-asymmetric",
            ),
        ],
    )
    def test_output_matches_sequential_reference(
        self,
        gt_ids: list[np.ndarray],
        tracker_ids: list[np.ndarray],
        similarity_scores: list[np.ndarray],
    ) -> None:
        """Vectorized implementation is bit-identical to pre-vectorization sequential version.

        Guards the hot path against future silent drift. Compares per-alpha arrays
        between the vectorized code and _sequential_hota_reference (dict map +
        per-alpha Python loop, matching commit 2ca10bd^).
        """
        new_result = compute_hota_metrics(gt_ids, tracker_ids, similarity_scores)
        ref_result = _sequential_hota_reference(gt_ids, tracker_ids, similarity_scores)

        for key in ref_result:
            np.testing.assert_array_equal(
                new_result[key],
                ref_result[key],
                err_msg=f"{key}: vectorized result differs from sequential reference",
            )


class TestAggregateHOTAMetrics:
    """Multi-sequence aggregation of HOTA metrics."""

    def test_empty(self) -> None:
        """Empty sequence list returns all-zero HOTA metrics."""
        result = aggregate_hota_metrics([])
        assert result["HOTA"] == pytest.approx(0.0)
        assert result["HOTA_TP"] == 0
        assert result["HOTA_FN"] == 0
        assert result["HOTA_FP"] == 0

    def test_single_sequence(self) -> None:
        """Single-sequence aggregation reproduces per-sequence HOTA scores."""
        seq_result = compute_hota_metrics(
            gt_ids=[np.array([0, 1])],
            tracker_ids=[np.array([10, 20])],
            similarity_scores=[np.array([[0.8, 0.0], [0.0, 0.8]])],
        )

        agg_result = aggregate_hota_metrics([seq_result])

        assert agg_result["HOTA"] == pytest.approx(seq_result["HOTA"], rel=1e-4)
        assert agg_result["DetA"] == pytest.approx(seq_result["DetA"], rel=1e-4)
        assert agg_result["AssA"] == pytest.approx(seq_result["AssA"], rel=1e-4)

    def test_multiple_sequences(self) -> None:
        """Identical sequences aggregate to the same HOTA; TP counts are summed."""
        seq1 = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.9]])],
        )
        seq2 = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.9]])],
        )

        agg_result = aggregate_hota_metrics([seq1, seq2])

        assert agg_result["HOTA_TP"] == seq1["HOTA_TP"] + seq2["HOTA_TP"]
        assert agg_result["HOTA"] == pytest.approx(seq1["HOTA"], rel=0.01)

    def test_weighted_by_tp(self) -> None:
        """Aggregated HOTA skews toward the higher-TP sequence."""
        high_quality = compute_hota_metrics(
            gt_ids=[np.array([0, 1, 2, 3])],
            tracker_ids=[np.array([10, 20, 30, 40])],
            similarity_scores=[np.diag([0.9, 0.9, 0.9, 0.9])],
        )
        low_quality = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.3]])],
        )

        agg_result = aggregate_hota_metrics([high_quality, low_quality])

        assert agg_result["HOTA"] > low_quality["HOTA"]
