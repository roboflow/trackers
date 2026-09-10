# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from trackers.eval.box import EPS, BoxFormat, box_ioa, box_iou


class TestBoxIoU:
    """Pairwise intersection-over-union computation."""

    @pytest.mark.parametrize(
        ("boxes1", "boxes2", "box_format", "expected_iou"),
        [
            (
                np.array([[0, 0, 10, 10]]),
                np.array([[0, 0, 10, 10]]),
                "xyxy",
                np.array([[1.0]]),
            ),  # identical boxes, perfect overlap
            (
                np.array([[0, 0, 10, 10]]),
                np.array([[20, 20, 30, 30]]),
                "xyxy",
                np.array([[0.0]]),
            ),  # disjoint boxes, no overlap
            (
                np.array([[0, 0, 10, 10]]),
                np.array([[5, 0, 15, 10]]),
                "xyxy",
                np.array([[1 / 3]]),
            ),  # partial overlap, intersection=50, union=150
            (
                np.array([[0, 0, 20, 20]]),
                np.array([[5, 5, 15, 15]]),
                "xyxy",
                np.array([[0.25]]),
            ),  # contained box, intersection=100, union=400
            (
                np.array([[0, 0, 10, 10]]),
                np.array([[10, 0, 20, 10]]),
                "xyxy",
                np.array([[0.0]]),
            ),  # boxes touching at edge
            (
                np.array([[0, 0, 10, 10]]),
                np.array([[10, 10, 20, 20]]),
                "xyxy",
                np.array([[0.0]]),
            ),  # boxes touching at corner
            (
                np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
                np.array([[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 110, 110]]),
                "xyxy",
                np.array([[1.0, 1 / 3, 0.0], [0.0, 0.0, 0.0]]),
            ),  # multiple boxes batch
            (
                np.array([[0, 0, 10, 10]]),
                np.array([[5, 0, 10, 10]]),
                "xywh",
                np.array([[1 / 3]]),
            ),  # xywh format
            (
                np.empty((0, 4)),
                np.array([[0, 0, 10, 10]]),
                "xyxy",
                np.empty((0, 1)),
            ),  # empty boxes1
            (
                np.array([[0, 0, 10, 10]]),
                np.empty((0, 4)),
                "xyxy",
                np.empty((1, 0)),
            ),  # empty boxes2
            (
                np.empty((0, 4)),
                np.empty((0, 4)),
                "xyxy",
                np.empty((0, 0)),
            ),  # both empty
            (
                np.array([[5, 5, 5, 5]]),
                np.array([[0, 0, 10, 10]]),
                "xyxy",
                np.array([[0.0]]),
            ),  # zero-area box
            (
                np.array([[1e6, 1e6, 1e6 + 10, 1e6 + 10]]),
                np.array([[1e6, 1e6, 1e6 + 10, 1e6 + 10]]),
                "xyxy",
                np.array([[1.0]]),
            ),  # large coordinates
        ],
    )
    def test_values(
        self,
        boxes1: np.ndarray[Any, np.dtype[Any]],
        boxes2: np.ndarray[Any, np.dtype[Any]],
        box_format: BoxFormat,
        expected_iou: np.ndarray[Any, np.dtype[Any]],
    ) -> None:
        """box_iou returns the expected pairwise IoU matrix for assorted box layouts."""
        result = box_iou(boxes1, boxes2, box_format=box_format)
        assert result.shape == expected_iou.shape
        assert np.allclose(result, expected_iou, rtol=1e-6, atol=1e-12)

    def test_invalid_format(self) -> None:
        """box_iou raises ValueError when box_format is not 'xyxy' or 'xywh'."""
        boxes = np.array([[0, 0, 10, 10]])
        with pytest.raises(ValueError, match="box_format must be"):
            box_iou(boxes, boxes, box_format="invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("boxes1", "boxes2", "expected_iou"),
        [
            (
                np.array([[0.5, 0.5, 10.5, 10.5]]),
                np.array([[0.5, 0.5, 10.5, 10.5]]),
                np.array([[1.0]]),
            ),  # floating point coords, identical boxes
            (
                np.array([[0.0, 0.0, 1.0, 1.0]]),
                np.array([[0.5, 0.0, 1.5, 1.0]]),
                np.array([[1 / 3]]),
            ),  # unit boxes with 50% horizontal overlap
            (
                np.array([[0.0, 0.0, 0.1, 0.1]]),
                np.array([[0.0, 0.0, 0.1, 0.1]]),
                np.array([[1.0]]),
            ),  # very small boxes (area=0.01)
            (
                np.array([[0.0, 0.0, 1e-6, 1e-6]]),
                np.array([[0.0, 0.0, 1e-6, 1e-6]]),
                np.array([[1.0]]),
            ),  # near-epsilon sized boxes
            (
                np.array([[0.0, 0.0, 100.0, 100.0]]),
                np.array([[99.9, 99.9, 100.0, 100.0]]),
                np.array([[0.01 / (10000 + 0.01 - 0.01)]]),
            ),  # tiny overlap (0.1 x 0.1 = 0.01)
            (
                np.array([[0.123456789, 0.987654321, 10.111111111, 10.222222222]]),
                np.array([[0.123456789, 0.987654321, 10.111111111, 10.222222222]]),
                np.array([[1.0]]),
            ),  # many decimal places, identical
            (
                np.array([[1e-10, 1e-10, 1.0 + 1e-10, 1.0 + 1e-10]]),
                np.array([[0.0, 0.0, 1.0, 1.0]]),
                np.array([[1.0]]),
            ),  # near-identical with tiny offset
        ],
    )
    def test_floating_point(
        self,
        boxes1: np.ndarray[Any, np.dtype[Any]],
        boxes2: np.ndarray[Any, np.dtype[Any]],
        expected_iou: np.ndarray[Any, np.dtype[Any]],
    ) -> None:
        """box_iou is numerically stable across sub-pixel and large-coordinate boxes."""
        result = box_iou(boxes1, boxes2, box_format="xyxy")
        assert result.shape == expected_iou.shape
        assert np.allclose(result, expected_iou, rtol=1e-5, atol=1e-10)

    @pytest.mark.parametrize(
        ("num_boxes1", "num_boxes2"),
        [
            (5, 5),
            (5, 10),
            (10, 5),
            (50, 50),
        ],
    )
    def test_valid_range(self, num_boxes1: int, num_boxes2: int) -> None:
        """box_iou results stay within [0, 1] (modulo EPS) for random boxes."""
        rng = np.random.default_rng(42)
        boxes1 = rng.random((num_boxes1, 4)) * 100
        boxes2 = rng.random((num_boxes2, 4)) * 100

        # Ensure valid xyxy format (x1 > x0, y1 > y0)
        boxes1[:, 2:] = boxes1[:, :2] + np.abs(boxes1[:, 2:])
        boxes2[:, 2:] = boxes2[:, :2] + np.abs(boxes2[:, 2:])

        ious = box_iou(boxes1, boxes2, box_format="xyxy")

        assert ious.shape == (num_boxes1, num_boxes2)
        assert (ious >= 0 - EPS).all()
        assert (ious <= 1 + EPS).all()


class TestBoxIoA:
    """Intersection-over-area computation."""

    @pytest.mark.parametrize(
        ("boxes1", "boxes2", "box_format", "expected_ioa"),
        [
            (
                np.array([[0, 0, 10, 10]]),
                np.array([[0, 0, 10, 10]]),
                "xyxy",
                np.array([[1.0]]),
            ),  # identical boxes
            (
                np.array([[5, 5, 15, 15]]),
                np.array([[0, 0, 20, 20]]),
                "xyxy",
                np.array([[1.0]]),
            ),  # detection fully inside ignore region
            (
                np.array([[0, 0, 10, 10]]),
                np.array([[5, 0, 15, 10]]),
                "xyxy",
                np.array([[0.5]]),
            ),  # partial overlap, intersection=50, area1=100
            (
                np.array([[0, 0, 10, 10]]),
                np.array([[20, 20, 30, 30]]),
                "xyxy",
                np.array([[0.0]]),
            ),  # no overlap
            (
                np.array([[5, 5, 5, 5]]),
                np.array([[0, 0, 10, 10]]),
                "xyxy",
                np.array([[0.0]]),
            ),  # zero-area box
            (
                np.array([[0, 0, 10, 10]]),
                np.array([[5, 0, 10, 10]]),
                "xywh",
                np.array([[0.5]]),
            ),  # xywh format
            (
                np.array([[0, 0, 10, 10], [0, 0, 20, 20]]),
                np.array([[5, 5, 15, 15], [0, 0, 10, 10]]),
                "xyxy",
                np.array([[0.25, 1.0], [0.25, 0.25]]),
            ),  # square N=2, M=2 batch: row-wise division by each boxes1 area
            (
                np.array([[0, 0, 10, 10], [10, 10, 30, 30]]),
                np.array([[0, 0, 5, 5], [5, 5, 15, 15], [100, 100, 110, 110]]),
                "xyxy",
                np.array([[0.25, 0.25, 0.0], [0.0, 0.0625, 0.0]]),
            ),  # asymmetric N=2, M=3 batch (N != M)
        ],
    )
    def test_values(
        self,
        boxes1: np.ndarray[Any, np.dtype[Any]],
        boxes2: np.ndarray[Any, np.dtype[Any]],
        box_format: BoxFormat,
        expected_ioa: np.ndarray[Any, np.dtype[Any]],
    ) -> None:
        """box_ioa returns the expected intersection-over-area matrix."""
        result = box_ioa(boxes1, boxes2, box_format=box_format)
        assert result.shape == expected_ioa.shape
        assert np.allclose(result, expected_ioa, rtol=1e-6, atol=1e-12)

    def test_invalid_format(self) -> None:
        """box_ioa raises ValueError when box_format is not 'xyxy' or 'xywh'."""
        boxes = np.array([[0, 0, 10, 10]])
        with pytest.raises(ValueError, match="box_format must be"):
            box_ioa(boxes, boxes, box_format="invalid")  # type: ignore[arg-type]


def test_epsilon_matches_trackeval() -> None:
    """EPS constant equals numpy float epsilon, matching the TrackEval reference."""
    assert EPS == np.finfo("float").eps


def _naive_pairwise_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Reference IoU for a single box pair, computed without any broadcasting."""
    inter_x0 = max(box1[0], box2[0])
    inter_y0 = max(box1[1], box2[1])
    inter_x1 = min(box1[2], box2[2])
    inter_y1 = min(box1[3], box2[3])
    intersection = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union <= EPS:
        return 0.0
    return intersection / union


def _naive_pairwise_ioa(box1: np.ndarray, box2: np.ndarray) -> float:
    """Reference IoA for a single box pair, computed without any broadcasting."""
    inter_x0 = max(box1[0], box2[0])
    inter_y0 = max(box1[1], box2[1])
    inter_x1 = min(box1[2], box2[2])
    inter_y1 = min(box1[3], box2[3])
    intersection = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    if area1 <= EPS:
        return 0.0
    return intersection / area1


def _naive_matrix(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    pairwise_fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Build an (N, M) matrix by applying a single-pair reference fn to every pair."""
    return np.array([[pairwise_fn(b1, b2) for b2 in boxes2] for b1 in boxes1])


def _random_valid_boxes(rng: np.random.Generator, count: int) -> np.ndarray:
    """Generate `count` random, valid xyxy boxes (x1 > x0, y1 > y0).

    Mirrors the box-generation approach used in `TestBoxIoU.test_valid_range`.
    """
    boxes = rng.random((count, 4)) * 100
    boxes[:, 2:] = boxes[:, :2] + np.abs(boxes[:, 2:])
    return boxes


def test_box_iou_matches_naive_reference_implementation() -> None:
    """box_iou agrees with an independent, double-loop reference IoU implementation.

    Every existing box_iou assertion compares against a hand-computed constant, so a broadcast-axis regression that
    happens to still satisfy those specific constants would go undetected. This test instead checks agreement, over
    randomized boxes, with a reference implementation computed without any array broadcasting.
    """
    rng = np.random.default_rng(1234)
    boxes1 = _random_valid_boxes(rng, 6)
    boxes2 = _random_valid_boxes(rng, 9)

    result = box_iou(boxes1, boxes2, box_format="xyxy")

    expected = _naive_matrix(boxes1, boxes2, _naive_pairwise_iou)
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-9)


def test_box_ioa_matches_naive_reference_implementation() -> None:
    """box_ioa agrees with an independent, double-loop reference IoA implementation.

    Mirrors `test_box_iou_matches_naive_reference_implementation` for box_ioa, whose
    row-wise division by area1 is a distinct code path with its own susceptibility to
    broadcast-axis regressions that hand-computed constants alone would not catch.
    """
    rng = np.random.default_rng(5678)
    boxes1 = _random_valid_boxes(rng, 6)
    boxes2 = _random_valid_boxes(rng, 9)

    result = box_ioa(boxes1, boxes2, box_format="xyxy")

    expected = _naive_matrix(boxes1, boxes2, _naive_pairwise_ioa)
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-9)


def test_box_iou_raises_on_ragged_columns() -> None:
    """box_iou rejects box arrays whose column counts disagree.

    Broadcasting all four coordinate planes rejected a column mismatch implicitly. Reading only columns 0-3 hides it, so
    pairing plain xyxy boxes with boxes that carry an extra score column would return a matrix instead of raising.
    """
    boxes1 = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [1, 1, 2, 2]])
    boxes2 = np.array([[0, 0, 10, 10, 0.9], [5, 5, 15, 15, 0.8]])

    with pytest.raises(ValueError, match="matching trailing dimensions"):
        box_iou(boxes1, boxes2)
