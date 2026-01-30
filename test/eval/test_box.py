# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trackers.eval.box import EPS, BoxFormat, box_ioa, box_iou


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
def test_box_iou(
    boxes1: np.ndarray[Any, np.dtype[Any]],
    boxes2: np.ndarray[Any, np.dtype[Any]],
    box_format: BoxFormat,
    expected_iou: np.ndarray[Any, np.dtype[Any]],
) -> None:
    result = box_iou(boxes1, boxes2, box_format=box_format)
    assert result.shape == expected_iou.shape
    assert np.allclose(result, expected_iou, rtol=1e-6, atol=1e-12)


def test_box_iou_invalid_format() -> None:
    boxes = np.array([[0, 0, 10, 10]])
    with pytest.raises(ValueError, match="box_format must be"):
        box_iou(boxes, boxes, box_format="invalid")  # type: ignore[arg-type]


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
    ],
)
def test_box_ioa(
    boxes1: np.ndarray[Any, np.dtype[Any]],
    boxes2: np.ndarray[Any, np.dtype[Any]],
    box_format: BoxFormat,
    expected_ioa: np.ndarray[Any, np.dtype[Any]],
) -> None:
    result = box_ioa(boxes1, boxes2, box_format=box_format)
    assert result.shape == expected_ioa.shape
    assert np.allclose(result, expected_ioa, rtol=1e-6, atol=1e-12)


def test_box_ioa_invalid_format() -> None:
    boxes = np.array([[0, 0, 10, 10]])
    with pytest.raises(ValueError, match="box_format must be"):
        box_ioa(boxes, boxes, box_format="invalid")  # type: ignore[arg-type]


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
def test_box_iou_floating_point(
    boxes1: np.ndarray[Any, np.dtype[Any]],
    boxes2: np.ndarray[Any, np.dtype[Any]],
    expected_iou: np.ndarray[Any, np.dtype[Any]],
) -> None:
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
def test_box_iou_valid_range(num_boxes1: int, num_boxes2: int) -> None:
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


def test_epsilon_matches_trackeval() -> None:
    assert EPS == np.finfo("float").eps
