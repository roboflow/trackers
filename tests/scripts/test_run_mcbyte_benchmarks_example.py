# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import sys

import pytest
from pytest import MonkeyPatch

from trackers.scripts.run_mcbyte_benchmarks_example import parse_args


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [pytest.param([], 6, id="default"), pytest.param(["--cmc-downscale", "2"], 2, id="override")],
)
def test_mcbyte_benchmark_cli_cmc_downscale(
    monkeypatch: MonkeyPatch,
    arguments: list[str],
    expected: int,
) -> None:
    """The benchmark follows McByte's default and retains explicit overrides."""
    monkeypatch.setattr(sys, "argv", ["run_mcbyte_benchmarks_example", *arguments])

    args = parse_args()

    assert args.cmc_downscale == expected
