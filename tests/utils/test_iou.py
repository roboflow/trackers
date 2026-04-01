# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchvision

from trackers.utils.iou import BIoU, CIoU, DIoU, GIoU, IoU


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


_iou = IoU()
_biou = BIoU()
_giou = GIoU()
_diou = DIoU()
_ciou = CIoU()


class TestGIoUAgainstTorchvision:
    """Compare our GIoU against torchvision.ops.generalized_box_iou."""

    def test_identical_boxes(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 50.0]])
        result = _giou.compute(boxes, boxes)
        expected = _torchvision_giou(boxes, boxes)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-6)

    def test_partial_overlap(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        boxes_2 = np.array([[5.0, 5.0, 15.0, 15.0]])
        result = _giou.compute(boxes_1, boxes_2)
        expected = _torchvision_giou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_no_overlap_nearby(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        boxes_2 = np.array([[12.0, 0.0, 22.0, 10.0]])
        result = _giou.compute(boxes_1, boxes_2)
        expected = _torchvision_giou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        assert result[0, 0] < 0, "GIoU should be negative for non-overlapping boxes"

    def test_no_overlap_far_apart(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 1.0, 1.0]])
        boxes_2 = np.array([[100.0, 100.0, 101.0, 101.0]])
        result = _giou.compute(boxes_1, boxes_2)
        expected = _torchvision_giou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        assert result[0, 0] < -0.5, "GIoU should be very negative for distant boxes"

    def test_one_box_enclosing_other(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 100.0, 100.0]])
        boxes_2 = np.array([[25.0, 25.0, 75.0, 75.0]])
        result = _giou.compute(boxes_1, boxes_2)
        expected = _torchvision_giou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_touching_boxes(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        boxes_2 = np.array([[10.0, 0.0, 20.0, 10.0]])
        result = _giou.compute(boxes_1, boxes_2)
        expected = _torchvision_giou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_batch_n_by_m(self) -> None:
        boxes_1 = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [50.0, 50.0, 80.0, 80.0],
            ]
        )
        boxes_2 = np.array(
            [
                [5.0, 5.0, 15.0, 15.0],
                [100.0, 100.0, 110.0, 110.0],
            ]
        )
        result = _giou.compute(boxes_1, boxes_2)
        expected = _torchvision_giou(boxes_1, boxes_2)
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_negative_coordinates(self) -> None:
        boxes_1 = np.array([[-10.0, -10.0, 5.0, 5.0]])
        boxes_2 = np.array([[-3.0, -3.0, 12.0, 12.0]])
        result = _giou.compute(boxes_1, boxes_2)
        expected = _torchvision_giou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_various_aspect_ratios(self) -> None:
        boxes_1 = np.array(
            [
                [0.0, 0.0, 100.0, 10.0],  # wide
                [0.0, 0.0, 10.0, 100.0],  # tall
                [0.0, 0.0, 50.0, 50.0],  # square
            ]
        )
        boxes_2 = np.array(
            [
                [10.0, 0.0, 60.0, 8.0],  # wide, offset
                [2.0, 10.0, 12.0, 80.0],  # tall, offset
            ]
        )
        result = _giou.compute(boxes_1, boxes_2)
        expected = _torchvision_giou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_large_random_batch(self) -> None:
        rng = np.random.default_rng(42)
        xy = rng.uniform(0, 500, size=(50, 2))
        wh = rng.uniform(5, 100, size=(50, 2))
        boxes_1 = np.hstack([xy, xy + wh])

        xy2 = rng.uniform(0, 500, size=(30, 2))
        wh2 = rng.uniform(5, 100, size=(30, 2))
        boxes_2 = np.hstack([xy2, xy2 + wh2])

        result = _giou.compute(boxes_1, boxes_2)
        expected = _torchvision_giou(boxes_1, boxes_2)
        assert result.shape == (50, 30)
        np.testing.assert_allclose(result, expected, atol=1e-6)


class TestBIoUProperties:
    """Verify behavior of Buffered IoU."""

    def test_buffer_zero_matches_iou(self) -> None:
        boxes_1 = np.array(
            [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 35.0, 40.0]], dtype=np.float64
        )
        boxes_2 = np.array(
            [[5.0, 5.0, 15.0, 15.0], [50.0, 50.0, 60.0, 60.0]], dtype=np.float64
        )
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


class TestDIoUAgainstTorchvision:
    """Compare our DIoU against torchvision.ops.distance_box_iou."""

    def test_identical_boxes(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 50.0]])
        result = _diou.compute(boxes, boxes)
        expected = _torchvision_diou(boxes, boxes)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-6)

    def test_partial_overlap(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        boxes_2 = np.array([[5.0, 5.0, 15.0, 15.0]])
        result = _diou.compute(boxes_1, boxes_2)
        expected = _torchvision_diou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_no_overlap_nearby(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        boxes_2 = np.array([[12.0, 0.0, 22.0, 10.0]])
        result = _diou.compute(boxes_1, boxes_2)
        expected = _torchvision_diou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        assert result[0, 0] < 0, "DIoU should be negative for this non-overlap"

    def test_no_overlap_far_apart(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 1.0, 1.0]])
        boxes_2 = np.array([[100.0, 100.0, 101.0, 101.0]])
        result = _diou.compute(boxes_1, boxes_2)
        expected = _torchvision_diou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        assert result[0, 0] < -0.5, "DIoU should be very negative for distant boxes"

    def test_one_box_enclosing_other(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 100.0, 100.0]])
        boxes_2 = np.array([[25.0, 25.0, 75.0, 75.0]])
        result = _diou.compute(boxes_1, boxes_2)
        expected = _torchvision_diou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_touching_boxes(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        boxes_2 = np.array([[10.0, 0.0, 20.0, 10.0]])
        result = _diou.compute(boxes_1, boxes_2)
        expected = _torchvision_diou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_batch_n_by_m(self) -> None:
        boxes_1 = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [50.0, 50.0, 80.0, 80.0],
            ]
        )
        boxes_2 = np.array(
            [
                [5.0, 5.0, 15.0, 15.0],
                [100.0, 100.0, 110.0, 110.0],
            ]
        )
        result = _diou.compute(boxes_1, boxes_2)
        expected = _torchvision_diou(boxes_1, boxes_2)
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_negative_coordinates(self) -> None:
        boxes_1 = np.array([[-10.0, -10.0, 5.0, 5.0]])
        boxes_2 = np.array([[-3.0, -3.0, 12.0, 12.0]])
        result = _diou.compute(boxes_1, boxes_2)
        expected = _torchvision_diou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_various_aspect_ratios(self) -> None:
        boxes_1 = np.array(
            [
                [0.0, 0.0, 100.0, 10.0],  # wide
                [0.0, 0.0, 10.0, 100.0],  # tall
                [0.0, 0.0, 50.0, 50.0],  # square
            ]
        )
        boxes_2 = np.array(
            [
                [10.0, 0.0, 60.0, 8.0],  # wide, offset
                [2.0, 10.0, 12.0, 80.0],  # tall, offset
            ]
        )
        result = _diou.compute(boxes_1, boxes_2)
        expected = _torchvision_diou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_large_random_batch(self) -> None:
        rng = np.random.default_rng(42)
        xy = rng.uniform(0, 500, size=(50, 2))
        wh = rng.uniform(5, 100, size=(50, 2))
        boxes_1 = np.hstack([xy, xy + wh])

        xy2 = rng.uniform(0, 500, size=(30, 2))
        wh2 = rng.uniform(5, 100, size=(30, 2))
        boxes_2 = np.hstack([xy2, xy2 + wh2])

        result = _diou.compute(boxes_1, boxes_2)
        expected = _torchvision_diou(boxes_1, boxes_2)
        assert result.shape == (50, 30)
        np.testing.assert_allclose(result, expected, atol=1e-6)


class TestCIoUAgainstTorchvision:
    """Compare our CIoU against torchvision.ops.complete_box_iou."""

    def test_identical_boxes(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 50.0]])
        result = _ciou.compute(boxes, boxes)
        expected = _torchvision_ciou(boxes, boxes)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-6)

    def test_partial_overlap(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        boxes_2 = np.array([[5.0, 5.0, 15.0, 15.0]])
        result = _ciou.compute(boxes_1, boxes_2)
        expected = _torchvision_ciou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_no_overlap_nearby(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        boxes_2 = np.array([[12.0, 0.0, 22.0, 10.0]])
        result = _ciou.compute(boxes_1, boxes_2)
        expected = _torchvision_ciou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        assert result[0, 0] < 0, "CIoU should be negative for this non-overlap"

    def test_no_overlap_far_apart(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 1.0, 1.0]])
        boxes_2 = np.array([[100.0, 100.0, 101.0, 101.0]])
        result = _ciou.compute(boxes_1, boxes_2)
        expected = _torchvision_ciou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        assert result[0, 0] < -0.5, "CIoU should be very negative for distant boxes"

    def test_one_box_enclosing_other(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 100.0, 100.0]])
        boxes_2 = np.array([[25.0, 25.0, 75.0, 75.0]])
        result = _ciou.compute(boxes_1, boxes_2)
        expected = _torchvision_ciou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_touching_boxes(self) -> None:
        boxes_1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        boxes_2 = np.array([[10.0, 0.0, 20.0, 10.0]])
        result = _ciou.compute(boxes_1, boxes_2)
        expected = _torchvision_ciou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_batch_n_by_m(self) -> None:
        boxes_1 = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [50.0, 50.0, 80.0, 80.0],
            ]
        )
        boxes_2 = np.array(
            [
                [5.0, 5.0, 15.0, 15.0],
                [100.0, 100.0, 110.0, 110.0],
            ]
        )
        result = _ciou.compute(boxes_1, boxes_2)
        expected = _torchvision_ciou(boxes_1, boxes_2)
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_negative_coordinates(self) -> None:
        boxes_1 = np.array([[-10.0, -10.0, 5.0, 5.0]])
        boxes_2 = np.array([[-3.0, -3.0, 12.0, 12.0]])
        result = _ciou.compute(boxes_1, boxes_2)
        expected = _torchvision_ciou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_various_aspect_ratios(self) -> None:
        boxes_1 = np.array(
            [
                [0.0, 0.0, 100.0, 10.0],  # wide
                [0.0, 0.0, 10.0, 100.0],  # tall
                [0.0, 0.0, 50.0, 50.0],  # square
            ]
        )
        boxes_2 = np.array(
            [
                [10.0, 0.0, 60.0, 8.0],  # wide, offset
                [2.0, 10.0, 12.0, 80.0],  # tall, offset
            ]
        )
        result = _ciou.compute(boxes_1, boxes_2)
        expected = _torchvision_ciou(boxes_1, boxes_2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_large_random_batch(self) -> None:
        rng = np.random.default_rng(42)
        xy = rng.uniform(0, 500, size=(50, 2))
        wh = rng.uniform(5, 100, size=(50, 2))
        boxes_1 = np.hstack([xy, xy + wh])

        xy2 = rng.uniform(0, 500, size=(30, 2))
        wh2 = rng.uniform(5, 100, size=(30, 2))
        boxes_2 = np.hstack([xy2, xy2 + wh2])

        result = _ciou.compute(boxes_1, boxes_2)
        expected = _torchvision_ciou(boxes_1, boxes_2)
        assert result.shape == (50, 30)
        np.testing.assert_allclose(result, expected, atol=1e-6)


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
