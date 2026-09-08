# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for evaluation result rendering."""

from __future__ import annotations

from trackers.eval.results import _format_metric_rows


class TestFormatMetricRows:
    """Fixed-width table renderer shared by the MOT and multicamera tables."""

    def test_benchmark_layout_matches_golden(self) -> None:
        rows = [
            ("MOT17-02", {"HOTA": 0.623, "IDSW": 42}),
            ("a-very-long-sequence-name-exceeding-30-chars", {"HOTA": 1.0, "IDSW": 0}),
            ("COMBINED", {"HOTA": 0.8115, "IDSW": 42}),
        ]
        expected = "\n".join(
            [
                "Sequence                         HOTA  IDSW",
                "-------------------------------------------",
                "MOT17-02                       62.300    42",
                "a-very-long-sequence-name-exceeding-30-chars100.000     0",
                "-------------------------------------------",
                "COMBINED                       81.150    42",
            ]
        )

        assert _format_metric_rows(rows, ["HOTA", "IDSW"], rule_before_last=True) == expected
