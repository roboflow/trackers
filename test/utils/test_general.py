# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import pytest

from trackers.datasets.manifest import DatasetAsset, DatasetSplit
from trackers.utils.general import _normalize_list


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("train", ["train"]),
        (DatasetSplit.TRAIN, ["train"]),
        (DatasetAsset.FRAMES, ["frames"]),
        (["train", "val"], ["train", "val"]),
        (
            [DatasetSplit.TRAIN, DatasetSplit.VAL],
            ["train", "val"],
        ),
        (
            [DatasetSplit.TRAIN, "val"],
            ["train", "val"],
        ),
        (
            [DatasetAsset.ANNOTATIONS, "frames"],
            ["annotations", "frames"],
        ),
    ],
)
def test_normalize_list(
    value: str | DatasetSplit | DatasetAsset | list | None,
    expected: list | None,
) -> None:
    assert _normalize_list(value) == expected
