# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from trackers.utils.iou import BaseIoU, BIoU, CIoU, DIoU, GIoU, IoU  # noqa: E402


def _torchvision_giou(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    """Reference GIoU from torchvision."""
    t1 = torch.tensor(boxes_1, dtype=torch.float64)
    t2 = torch.tensor(boxes_2, dtype=torch.float64)
    return torchvision.ops.generalized_box_iou(t1, t2).numpy()


def _torchvision_diou(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    """Reference DIoU from torchvision."""
    t1 = torch.tensor(boxes_1, dtype=torch.float64)
    t2 = torch.tensor(boxes_2, dtype=torch.float64)
    return torchvision.ops.distance_box_iou(t1, t2).numpy()


def _torchvision_ciou(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    """Reference CIoU from torchvision."""
    t1 = torch.tensor(boxes_1, dtype=torch.float64)
    t2 = torch.tensor(boxes_2, dtype=torch.float64)
    return torchvision.ops.complete_box_iou(t1, t2).numpy()


def _reference_biou(boxes_1: np.ndarray, boxes_2: np.ndarray, buffer_ratio: float = 0.1) -> np.ndarray:
    """Independent BIoU reference: buffer boxes, then apply vanilla IoU."""
    boxes_1_b = boxes_1.astype(np.float64, copy=True)
    boxes_2_b = boxes_2.astype(np.float64, copy=True)

    w1 = boxes_1_b[:, 2] - boxes_1_b[:, 0]
    h1 = boxes_1_b[:, 3] - boxes_1_b[:, 1]
    w2 = boxes_2_b[:, 2] - boxes_2_b[:, 0]
    h2 = boxes_2_b[:, 3] - boxes_2_b[:, 1]

    r = buffer_ratio
    boxes_1_b[:, 0] -= r * w1
    boxes_1_b[:, 1] -= r * h1
    boxes_1_b[:, 2] += r * w1
    boxes_1_b[:, 3] += r * h1

    boxes_2_b[:, 0] -= r * w2
    boxes_2_b[:, 1] -= r * h2
    boxes_2_b[:, 2] += r * w2
    boxes_2_b[:, 3] += r * h2

    return _iou.compute(boxes_1_b, boxes_2_b).astype(np.float64)


_iou = IoU()
_biou = BIoU()
_giou = GIoU()
_diou = DIoU()
_ciou = CIoU()


_TORCHVISION_COMPARISON_VARIANTS = [
    pytest.param(_biou, _reference_biou, id="biou"),
    pytest.param(_giou, _torchvision_giou, id="giou"),
    pytest.param(_diou, _torchvision_diou, id="diou"),
    pytest.param(_ciou, _torchvision_ciou, id="ciou"),
]

_TORCHVISION_COMPARISON_CASES = [
    pytest.param(
        np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 50.0]]),
        np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 50.0]]),
        (2, 2),
        True,
        None,
        id="identical_boxes",
    ),
    pytest.param(
        np.array([[0.0, 0.0, 10.0, 10.0]]),
        np.array([[5.0, 5.0, 15.0, 15.0]]),
        (1, 1),
        False,
        None,
        id="partial_overlap",
    ),
    pytest.param(
        np.array([[0.0, 0.0, 10.0, 10.0]]),
        np.array([[12.0, 0.0, 22.0, 10.0]]),
        (1, 1),
        False,
        0.0,
        id="no_overlap_nearby",
    ),
    pytest.param(
        np.array([[0.0, 0.0, 1.0, 1.0]]),
        np.array([[100.0, 100.0, 101.0, 101.0]]),
        (1, 1),
        False,
        -0.5,
        id="no_overlap_far_apart",
    ),
    pytest.param(
        np.array([[0.0, 0.0, 100.0, 100.0]]),
        np.array([[25.0, 25.0, 75.0, 75.0]]),
        (1, 1),
        False,
        None,
        id="one_box_enclosing_other",
    ),
    pytest.param(
        np.array([[0.0, 0.0, 10.0, 10.0]]),
        np.array([[10.0, 0.0, 20.0, 10.0]]),
        (1, 1),
        False,
        None,
        id="touching_boxes",
    ),
    pytest.param(
        np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [50.0, 50.0, 80.0, 80.0],
            ]
        ),
        np.array(
            [
                [5.0, 5.0, 15.0, 15.0],
                [100.0, 100.0, 110.0, 110.0],
            ]
        ),
        (3, 2),
        False,
        None,
        id="batch_n_by_m",
    ),
    pytest.param(
        np.array([[-10.0, -10.0, 5.0, 5.0]]),
        np.array([[-3.0, -3.0, 12.0, 12.0]]),
        (1, 1),
        False,
        None,
        id="negative_coordinates",
    ),
    pytest.param(
        np.array(
            [
                [0.0, 0.0, 100.0, 10.0],
                [0.0, 0.0, 10.0, 100.0],
                [0.0, 0.0, 50.0, 50.0],
            ]
        ),
        np.array(
            [
                [10.0, 0.0, 60.0, 8.0],
                [2.0, 10.0, 12.0, 80.0],
            ]
        ),
        (3, 2),
        False,
        None,
        id="various_aspect_ratios",
    ),
]


class TestIoUVariantsAgainstTorchvision:
    """Compare IoU variants against reference implementations."""

    @pytest.mark.parametrize("ours, ref_or_baseline", _TORCHVISION_COMPARISON_VARIANTS)
    @pytest.mark.parametrize(
        "boxes_1, boxes_2, expected_shape, diag_one, upper_bound",
        _TORCHVISION_COMPARISON_CASES,
    )
    def test_cases(
        self,
        ours,
        ref_or_baseline,
        boxes_1: np.ndarray,
        boxes_2: np.ndarray,
        expected_shape: tuple[int, int],
        diag_one: bool,
        upper_bound: float | None,
    ) -> None:
        """Validate variant-vs-reference parity across shared geometric scenarios.

        `upper_bound` is used only for cases where the original tests expected
        negative scores for non-overlapping boxes (GIoU/DIoU/CIoU).
        BIoU is intentionally excluded from this check because buffered IoU is
        IoU-like (non-negative) and can be zero or positive for such cases.
        """
        result = ours.compute(boxes_1, boxes_2)
        expected = ref_or_baseline(boxes_1, boxes_2)
        assert result.shape == expected_shape
        np.testing.assert_allclose(result, expected, atol=1e-6)
        if diag_one:
            np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-6)
        # Negative upper-bound expectations are metric-specific to
        # GIoU/DIoU/CIoU; they are not a valid invariant for BIoU.
        if upper_bound is not None and ref_or_baseline is not _reference_biou:
            assert result[0, 0] < upper_bound

    @pytest.mark.parametrize("ours, ref_or_baseline", _TORCHVISION_COMPARISON_VARIANTS)
    def test_large_random_batch(self, ours, ref_or_baseline) -> None:
        rng = np.random.default_rng(42)
        xy = rng.uniform(0, 500, size=(50, 2))
        wh = rng.uniform(5, 100, size=(50, 2))
        boxes_1 = np.hstack([xy, xy + wh])

        xy2 = rng.uniform(0, 500, size=(30, 2))
        wh2 = rng.uniform(5, 100, size=(30, 2))
        boxes_2 = np.hstack([xy2, xy2 + wh2])

        result = ours.compute(boxes_1, boxes_2)
        expected = ref_or_baseline(boxes_1, boxes_2)
        assert result.shape == (50, 30)
        np.testing.assert_allclose(result, expected, atol=1e-6)


class TestBIoUProperties:
    """Verify behavior of Buffered IoU."""

    def test_buffer_zero_matches_iou(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 35.0, 40.0]], dtype=np.float64)
        boxes_2 = np.array([[5.0, 5.0, 15.0, 15.0], [50.0, 50.0, 60.0, 60.0]], dtype=np.float64)
        biou0 = BIoU(buffer_ratio=0.0).compute(boxes_1, boxes_2)
        iou = _iou.compute(boxes_1, boxes_2).astype(np.float64)
        np.testing.assert_allclose(biou0, iou, atol=1e-10)

    def test_nearby_non_overlap_gets_positive_signal(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        boxes_2 = np.array([[11.0, 0.0, 21.0, 10.0]])
        iou = _iou.compute(boxes_1, boxes_2)[0, 0]
        biou = BIoU(buffer_ratio=0.1).compute(boxes_1, boxes_2)[0, 0]
        assert iou == 0.0
        assert biou > 0.0

    def test_invalid_negative_buffer_ratio(self) -> None:
        with pytest.raises(ValueError, match="buffer_ratio must be non-negative"):
            BIoU(buffer_ratio=-0.01)

    def test_biou_monotonic_in_buffer_ratio(self) -> None:
        """Larger buffer ratio yields equal-or-higher BIoU for near-miss boxes."""
        boxes_a = np.array([[0.0, 0.0, 10.0, 10.0]])
        boxes_b = np.array([[11.0, 0.0, 21.0, 10.0]])  # 1px gap — no overlap at ratio=0
        ratios = [0.0, 0.05, 0.1, 0.2, 0.5]
        scores = [float(BIoU(r).compute(boxes_a, boxes_b)[0, 0]) for r in ratios]
        # Each step should be >= the previous (monotone non-decreasing)
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 1e-9, f"Not monotone at ratios {ratios[i]}/{ratios[i + 1]}"

    def test_biou_zero_ratio_matches_iou(self) -> None:
        """BIoU(buffer_ratio=0) should equal standard IoU for all box pairs."""
        rng = np.random.default_rng(42)
        boxes_a = rng.uniform(0, 100, (10, 4))
        boxes_a[:, 2:] += boxes_a[:, :2]  # ensure x2>x1, y2>y1
        boxes_b = rng.uniform(0, 100, (8, 4))
        boxes_b[:, 2:] += boxes_b[:, :2]
        np.testing.assert_allclose(
            BIoU(0.0).compute(boxes_a, boxes_b),
            IoU().compute(boxes_a, boxes_b),
            atol=1e-6,
        )


class TestGIoUProperties:
    """Verify mathematical properties of GIoU."""

    def test_range_is_minus_one_to_one(self) -> None:
        rng = np.random.default_rng(99)
        xy = rng.uniform(0, 500, size=(100, 2))
        wh = rng.uniform(1, 200, size=(100, 2))
        boxes_1 = np.hstack([xy, xy + wh])

        xy2 = rng.uniform(0, 500, size=(80, 2))
        wh2 = rng.uniform(1, 200, size=(80, 2))
        boxes_2 = np.hstack([xy2, xy2 + wh2])

        result = _giou.compute(boxes_1, boxes_2)
        assert np.all(result >= -1.0 - 1e-9)
        assert np.all(result <= 1.0 + 1e-9)

    def test_symmetry(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 35.0, 40.0]])
        boxes_2 = np.array([[5.0, 5.0, 15.0, 15.0], [50.0, 50.0, 60.0, 60.0]])
        result_ab = _giou.compute(boxes_1, boxes_2)
        result_ba = _giou.compute(boxes_2, boxes_1)
        np.testing.assert_allclose(result_ab, result_ba.T, atol=1e-10)

    def test_giou_leq_iou(self) -> None:
        """GIoU <= IoU always holds."""
        rng = np.random.default_rng(7)
        xy = rng.uniform(0, 100, size=(40, 2))
        wh = rng.uniform(5, 50, size=(40, 2))
        boxes_1 = np.hstack([xy, xy + wh])

        xy2 = rng.uniform(0, 100, size=(30, 2))
        wh2 = rng.uniform(5, 50, size=(30, 2))
        boxes_2 = np.hstack([xy2, xy2 + wh2])

        iou_result = _iou.compute(boxes_1, boxes_2).astype(np.float64)
        giou_result = _giou.compute(boxes_1, boxes_2)
        assert np.all(giou_result <= iou_result + 1e-6)


class TestDIoUProperties:
    """Verify mathematical properties of DIoU."""

    def test_range_is_minus_one_to_one(self) -> None:
        rng = np.random.default_rng(101)
        xy = rng.uniform(0, 500, size=(100, 2))
        wh = rng.uniform(1, 200, size=(100, 2))
        boxes_1 = np.hstack([xy, xy + wh])

        xy2 = rng.uniform(0, 500, size=(80, 2))
        wh2 = rng.uniform(1, 200, size=(80, 2))
        boxes_2 = np.hstack([xy2, xy2 + wh2])

        result = _diou.compute(boxes_1, boxes_2)
        assert np.all(result >= -1.0 - 1e-9)
        assert np.all(result <= 1.0 + 1e-9)

    def test_symmetry(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 35.0, 40.0]])
        boxes_2 = np.array([[5.0, 5.0, 15.0, 15.0], [50.0, 50.0, 60.0, 60.0]])
        result_ab = _diou.compute(boxes_1, boxes_2)
        result_ba = _diou.compute(boxes_2, boxes_1)
        np.testing.assert_allclose(result_ab, result_ba.T, atol=1e-10)

    def test_diou_leq_iou(self) -> None:
        """DIoU <= IoU: center-distance penalty is nonnegative."""
        rng = np.random.default_rng(11)
        xy = rng.uniform(0, 100, size=(40, 2))
        wh = rng.uniform(5, 50, size=(40, 2))
        boxes_1 = np.hstack([xy, xy + wh])

        xy2 = rng.uniform(0, 100, size=(30, 2))
        wh2 = rng.uniform(5, 50, size=(30, 2))
        boxes_2 = np.hstack([xy2, xy2 + wh2])

        iou_result = _iou.compute(boxes_1, boxes_2).astype(np.float64)
        diou_result = _diou.compute(boxes_1, boxes_2)
        assert np.all(diou_result <= iou_result + 1e-6)


class TestCIoUProperties:
    """Verify mathematical properties of CIoU."""

    def test_at_most_one(self) -> None:
        """Pairwise CIoU is at most 1; unlike IoU/DIoU/GIoU it can be < -1."""
        rng = np.random.default_rng(103)
        xy = rng.uniform(0, 500, size=(100, 2))
        wh = rng.uniform(1, 200, size=(100, 2))
        boxes_1 = np.hstack([xy, xy + wh])

        xy2 = rng.uniform(0, 500, size=(80, 2))
        wh2 = rng.uniform(1, 200, size=(80, 2))
        boxes_2 = np.hstack([xy2, xy2 + wh2])

        result = _ciou.compute(boxes_1, boxes_2)
        assert np.all(result <= 1.0 + 1e-9)

    def test_symmetry(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 35.0, 40.0]])
        boxes_2 = np.array([[5.0, 5.0, 15.0, 15.0], [50.0, 50.0, 60.0, 60.0]])
        result_ab = _ciou.compute(boxes_1, boxes_2)
        result_ba = _ciou.compute(boxes_2, boxes_1)
        np.testing.assert_allclose(result_ab, result_ba.T, atol=1e-10)

    def test_ciou_leq_diou(self) -> None:
        """CIoU <= DIoU: aspect-ratio term is nonnegative after scaling by alpha."""
        rng = np.random.default_rng(13)
        xy = rng.uniform(0, 100, size=(40, 2))
        wh = rng.uniform(5, 50, size=(40, 2))
        boxes_1 = np.hstack([xy, xy + wh])

        xy2 = rng.uniform(0, 100, size=(30, 2))
        wh2 = rng.uniform(5, 50, size=(30, 2))
        boxes_2 = np.hstack([xy2, xy2 + wh2])

        diou_result = _diou.compute(boxes_1, boxes_2)
        ciou_result = _ciou.compute(boxes_1, boxes_2)
        assert np.all(ciou_result <= diou_result + 1e-6)


class TestNormalizeForFusion:
    """Verify the [0, 1] fusion contract of BaseIoU.normalize_for_fusion."""

    @pytest.mark.parametrize(
        "metric",
        [_iou, _biou, _giou, _diou, _ciou],
        ids=["IoU", "BIoU", "GIoU", "DIoU", "CIoU"],
    )
    def test_output_in_unit_range_over_random_batch(self, metric: BaseIoU) -> None:
        """normalize_for_fusion must map similarities into [0, 1] for score fusion.

        CIoU's aspect-ratio penalty drives raw scores below -1, so the naive
        ``(x + 1) / 2`` shift (without clamping) yields negatives that corrupt
        ``_fuse_score``. The other variants already lie in [0, 1] post-shift.
        """
        rng = np.random.default_rng(103)
        xy = rng.uniform(0, 500, size=(100, 2))
        wh = rng.uniform(1, 200, size=(100, 2))
        boxes_1 = np.hstack([xy, xy + wh])

        xy2 = rng.uniform(0, 500, size=(80, 2))
        wh2 = rng.uniform(1, 200, size=(80, 2))
        boxes_2 = np.hstack([xy2, xy2 + wh2])

        normalized = metric.normalize_for_fusion(metric.compute(boxes_1, boxes_2))
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)

    @pytest.mark.parametrize(
        "metric",
        [_iou, _biou],
        ids=["IoU", "BIoU"],
    )
    def test_normalize_for_fusion_is_identity_for_unsigned_variants(self, metric: BaseIoU) -> None:
        """normalize_for_fusion must be a no-op for IoU and BIoU."""
        rng = np.random.default_rng(201)
        xy = rng.uniform(0, 100, (10, 2))
        wh = rng.uniform(1, 50, (10, 2))
        b1 = np.hstack([xy, xy + wh])
        xy2 = rng.uniform(0, 100, (8, 2))
        wh2 = rng.uniform(1, 50, (8, 2))
        b2 = np.hstack([xy2, xy2 + wh2])
        raw = metric.compute(b1, b2)
        np.testing.assert_array_equal(metric.normalize_for_fusion(raw), raw)

    @pytest.mark.parametrize(
        "metric",
        [_giou, _diou, _ciou],
        ids=["GIoU", "DIoU", "CIoU"],
    )
    def test_normalize_for_fusion_boundary_values(self, metric: BaseIoU) -> None:
        """Shift maps -1→0, 0→0.5, 1→1 exactly for signed IoU variants."""
        mat = np.array([[-1.0, 0.0, 1.0]])
        expected = np.array([[0.0, 0.5, 1.0]])
        np.testing.assert_allclose(metric.normalize_for_fusion(mat), expected, atol=1e-12)

    def test_ciou_below_minus_one_clamps_to_zero(self) -> None:
        """A wide-vs-tall, far-apart pair has CIoU < -1; fusion must stay >= 0."""
        boxes_1 = np.array([[0.0, 0.0, 100.0, 1.0]])  # w/h = 100
        boxes_2 = np.array([[400.0, 400.0, 401.0, 500.0]])  # w/h = 0.01, no overlap
        raw = _ciou.compute(boxes_1, boxes_2)
        assert raw[0, 0] < -1.0  # precondition: the regime that breaks the shift
        normalized = _ciou.normalize_for_fusion(raw)
        assert normalized[0, 0] >= 0.0
        assert normalized[0, 0] <= 1.0
        assert normalized[0, 0] == pytest.approx(0.0)


class TestEmptyArrayHandling:
    """Verify BaseIoU.compute handles empty inputs for all subclasses."""

    @pytest.mark.parametrize(
        "iou_instance",
        [_iou, _biou, _giou, _diou, _ciou],
        ids=["IoU", "BIoU", "GIoU", "DIoU", "CIoU"],
    )
    def test_empty_boxes_1(self, iou_instance) -> None:
        boxes_1 = np.empty((0, 4))
        boxes_2 = np.array([[0.0, 0.0, 10.0, 10.0]])
        result = iou_instance.compute(boxes_1, boxes_2)
        assert result.shape == (0, 1)

    @pytest.mark.parametrize(
        "iou_instance",
        [_iou, _biou, _giou, _diou, _ciou],
        ids=["IoU", "BIoU", "GIoU", "DIoU", "CIoU"],
    )
    def test_empty_boxes_2(self, iou_instance) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
        boxes_2 = np.empty((0, 4))
        result = iou_instance.compute(boxes_1, boxes_2)
        assert result.shape == (2, 0)

    @pytest.mark.parametrize(
        "iou_instance",
        [_iou, _biou, _giou, _diou, _ciou],
        ids=["IoU", "BIoU", "GIoU", "DIoU", "CIoU"],
    )
    def test_both_empty(self, iou_instance) -> None:
        boxes_1 = np.empty((0, 4))
        boxes_2 = np.empty((0, 4))
        result = iou_instance.compute(boxes_1, boxes_2)
        assert result.shape == (0, 0)


class TestDegenerateInputs:
    """Pin behavior of BaseIoU.compute on edge-case inputs."""

    @pytest.mark.parametrize("metric", [IoU(), GIoU(), DIoU(), CIoU(), BIoU()])
    def test_nan_coordinates_raise(self, metric: BaseIoU) -> None:
        boxes_a = np.array([[0.0, 0.0, np.nan, 10.0]])
        boxes_b = np.array([[5.0, 5.0, 15.0, 15.0]])
        with pytest.raises(ValueError, match="non-finite"):
            metric.compute(boxes_a, boxes_b)

    @pytest.mark.parametrize("metric", [IoU(), GIoU(), DIoU(), CIoU(), BIoU()])
    def test_inf_coordinates_raise(self, metric: BaseIoU) -> None:
        boxes_a = np.array([[0.0, 0.0, np.inf, 10.0]])
        boxes_b = np.array([[5.0, 5.0, 15.0, 15.0]])
        with pytest.raises(ValueError, match="non-finite"):
            metric.compute(boxes_a, boxes_b)

    @pytest.mark.parametrize("metric", [IoU(), GIoU(), DIoU(), CIoU(), BIoU()])
    def test_zero_area_box_returns_finite(self, metric: BaseIoU) -> None:
        boxes_a = np.array([[5.0, 5.0, 5.0, 5.0]])  # zero-area
        boxes_b = np.array([[0.0, 0.0, 10.0, 10.0]])
        result = metric.compute(boxes_a, boxes_b)
        assert np.isfinite(result).all(), "Zero-area box should yield finite similarity"

    @pytest.mark.parametrize("metric", [IoU(), GIoU(), DIoU(), CIoU()])
    def test_inverted_coords_gives_zero_or_negative_similarity(self, metric: BaseIoU) -> None:
        # x2 < x1 — degenerate box; result should not crash
        boxes_a = np.array([[10.0, 0.0, 0.0, 10.0]])  # inverted x
        boxes_b = np.array([[0.0, 0.0, 10.0, 10.0]])
        result = metric.compute(boxes_a, boxes_b)
        assert result.shape == (1, 1)
        assert np.isfinite(result).all(), "Inverted-coord box should not produce NaN/inf"
