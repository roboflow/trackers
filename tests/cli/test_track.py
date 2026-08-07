# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import fields
from typing import ClassVar

import numpy as np
import pytest
import supervision as sv

from trackers.cli.track import (
    ShowOptions,
    TrackerOptions,
    _abbreviate_parameter_name,
    _expand_parameter_name,
    _format_labels,
    _init_annotators,
    _init_tracker,
    _resolve_class_filter,
    _resolve_track_id_filter,
)
from trackers.core.base import BaseTracker


class TestInitAnnotators:
    @pytest.mark.parametrize(
        "flags,expected_types,has_label_annotator",
        [
            (
                {"boxes": True, "masks": False, "ids": False},
                [sv.BoxAnnotator],
                False,
            ),
            (
                {"boxes": False, "masks": True, "ids": False},
                [sv.MaskAnnotator],
                False,
            ),
            (
                {"boxes": False, "masks": False, "ids": True},
                [],
                True,
            ),
            (
                {"boxes": True, "masks": True, "ids": True},
                [sv.BoxAnnotator, sv.MaskAnnotator],
                True,
            ),
        ],
    )
    def test_creates_annotators_based_on_flags(
        self,
        flags: dict,
        expected_types: list,
        has_label_annotator: bool,
    ) -> None:
        annotators, label_annotator = _init_annotators(ShowOptions(**flags))

        assert len(annotators) == len(expected_types)
        for annotator, expected_type in zip(annotators, expected_types):
            assert isinstance(annotator, expected_type)

        if has_label_annotator:
            assert isinstance(label_annotator, sv.LabelAnnotator)
        else:
            assert label_annotator is None


class TestFormatLabels:
    @pytest.mark.parametrize(
        "detections_kwargs,class_names,label_flags,expected",
        [
            pytest.param(
                {
                    "xyxy": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
                    "class_id": np.array([0, 1]),
                },
                ["person", "car"],
                {"labels": True},
                ["person", "car"],
                id="class_names_from_list",
            ),
            pytest.param(
                {
                    "xyxy": np.array([[0, 0, 10, 10]]),
                    "class_id": np.array([5]),
                },
                ["person", "car"],
                {"labels": True},
                ["5"],
                id="fallback_to_class_id_when_out_of_range",
            ),
            pytest.param(
                {
                    "xyxy": np.array([[0, 0, 10, 10]]),
                    "tracker_id": np.array([42]),
                },
                [],
                {"ids": True},
                ["#42"],
                id="tracker_ids_only",
            ),
            pytest.param(
                {
                    "xyxy": np.array([[0, 0, 10, 10]]),
                    "class_id": np.array([0]),
                    "confidence": np.array([0.95]),
                    "tracker_id": np.array([1]),
                },
                ["person"],
                {"ids": True, "labels": True, "confidence": True},
                ["#1 person 0.95"],
                id="combined_id_class_confidence",
            ),
        ],
    )
    def test_generates_labels(
        self,
        detections_kwargs: dict,
        class_names: list[str],
        label_flags: dict,
        expected: list[str],
    ) -> None:
        detections = sv.Detections(**detections_kwargs)
        labels = _format_labels(detections, class_names, ShowOptions(**label_flags))
        assert labels == expected


class TestResolveClassFilter:
    CLASS_NAMES: ClassVar[list[str]] = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
    ]

    @pytest.mark.parametrize(
        "classes_arg,expected",
        [
            pytest.param(None, None, id="none_returns_none"),
            pytest.param([], None, id="empty_returns_none"),
            pytest.param([0, 2], [0, 2], id="integer_ids"),
            pytest.param(["0", "2"], [0, 2], id="quoted_integer_ids"),
            pytest.param(["person", "car"], [0, 2], id="class_names"),
            pytest.param(["person", 2, "motorcycle"], [0, 2, 3], id="mixed_names_and_ids"),
            pytest.param([" person ", " car "], [0, 2], id="whitespace_stripped"),
            pytest.param([99], [99], id="out_of_range_id_kept"),
        ],
    )
    def test_resolves_classes(
        self,
        classes_arg: list[str | int] | None,
        expected: list[int] | None,
    ) -> None:
        result = _resolve_class_filter(classes_arg, self.CLASS_NAMES)
        assert result == expected

    def test_unknown_name_warns_and_skips(self, capsys: pytest.CaptureFixture) -> None:
        result = _resolve_class_filter(["person", "unicorn", "car"], self.CLASS_NAMES)
        assert result == [0, 2]
        assert "unicorn" in capsys.readouterr().err

    def test_all_unknown_names_returns_none(self, capsys: pytest.CaptureFixture) -> None:
        result = _resolve_class_filter(["unicorn", "dragon"], self.CLASS_NAMES)
        assert result is None
        assert "unicorn" in capsys.readouterr().err


class TestResolveTrackIdFilter:
    @pytest.mark.parametrize(
        "track_ids_arg,expected",
        [
            pytest.param(None, None, id="none_returns_none"),
            pytest.param([], None, id="empty_returns_none"),
            pytest.param([0, 2], [0, 2], id="integer_ids"),
            pytest.param(["0", "2"], [0, 2], id="quoted_integer_ids"),
            pytest.param(["person", "car"], None, id="words_returns_none"),
            pytest.param(["person", 2, "motorcycle"], [2], id="mixed_names_and_ids"),
            pytest.param([" 1 ", " 3 "], [1, 3], id="whitespace_stripped"),
            pytest.param([99], [99], id="out_of_range_id_kept"),
        ],
    )
    def test_resolves_track_ids(
        self,
        track_ids_arg: list[str | int] | None,
        expected: list[int] | None,
    ) -> None:
        result = _resolve_track_id_filter(track_ids_arg)
        assert result == expected

    def test_non_integer_warns_and_skips(self, capsys: pytest.CaptureFixture) -> None:
        result = _resolve_track_id_filter([1, "abc", 3])
        assert result == [1, 3]
        assert "abc" in capsys.readouterr().err

    def test_all_non_integer_returns_none(self, capsys: pytest.CaptureFixture) -> None:
        result = _resolve_track_id_filter(["abc", "def"])
        assert result is None
        assert "abc" in capsys.readouterr().err


class TestTrackerParameterAbbreviations:
    """Abbreviated CLI parameter names must survive the trip to the constructor.

    ``_init_tracker`` forwards only keys matching the tracker ``__init__``
    signature, so a CLI-side rename with no matching alias is dropped without
    an error and the tracker silently keeps its own default. These tests make
    that failure mode visible.
    """

    @pytest.mark.parametrize(
        ("name", "abbreviated"),
        [
            pytest.param("minimum_iou_threshold", "min_iou_threshold", id="minimum_prefix"),
            pytest.param("maximum_age", "max_age", id="maximum_prefix"),
            pytest.param("lost_track_buffer", "lost_track_buffer", id="no_standard_prefix"),
            pytest.param("minimum_iou_threshold_first_assoc", "min_iou_threshold_first_assoc", id="suffix_preserved"),
        ],
    )
    def test_only_standard_leading_tokens_are_abbreviated(self, name: str, abbreviated: str) -> None:
        """Leading minimum_/maximum_ shorten while domain words stay spelled out."""
        assert _abbreviate_parameter_name(name) == abbreviated
        assert _expand_parameter_name(abbreviated) == name

    @pytest.mark.parametrize(
        ("tracker_id", "option_field", "keyword", "value"),
        [
            pytest.param("bytetrack", "min_iou_threshold", "minimum_iou_threshold", 0.42, id="bytetrack_iou"),
            pytest.param("sort", "min_consecutive_frames", "minimum_consecutive_frames", 7, id="sort_frames"),
            pytest.param("ocsort", "min_iou_threshold", "minimum_iou_threshold", 0.33, id="ocsort_iou"),
            pytest.param(
                "botsort",
                "min_iou_threshold_first_assoc",
                "minimum_iou_threshold_first_assoc",
                0.11,
                id="botsort_first_assoc",
            ),
        ],
    )
    def test_short_cli_name_reaches_the_tracker(
        self,
        tracker_id: str,
        option_field: str,
        keyword: str,
        value: float,
    ) -> None:
        """A short CLI parameter is forwarded under its long constructor keyword."""
        options = TrackerOptions(name=tracker_id)
        setattr(options, option_field, value)

        tracker = _init_tracker(options)

        assert getattr(tracker, keyword) == pytest.approx(value)

    def test_every_option_field_maps_to_a_known_tracker_parameter(self) -> None:
        """No TrackerOptions field can be dropped silently by _init_tracker."""
        # ``name`` selects the tracker and ``iou_variant`` aliases ``iou``;
        # neither is forwarded to the constructor under its own field name.
        accepted = {"name", "iou_variant"}
        for tracker_id in BaseTracker._registered_trackers():
            info = BaseTracker._lookup_tracker(tracker_id)
            assert info is not None
            accepted.update(info.parameters)

        unmatched = [
            field.name
            for field in fields(TrackerOptions)
            if field.name not in accepted and _expand_parameter_name(field.name) not in accepted
        ]

        assert unmatched == []
