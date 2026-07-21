# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import supervision as sv

from trackers.scripts.track import (
    _apply_reid_tracker_params,
    _format_labels,
    _init_annotators,
    _reid_requested,
    _resolve_class_filter,
    _resolve_track_id_filter,
    add_track_subparser,
)


class TestInitAnnotators:
    @pytest.mark.parametrize(
        "flags,expected_types,has_label_annotator",
        [
            (
                {"show_boxes": True, "show_masks": False, "show_ids": False},
                [sv.BoxAnnotator],
                False,
            ),
            (
                {"show_boxes": False, "show_masks": True, "show_ids": False},
                [sv.MaskAnnotator],
                False,
            ),
            (
                {"show_boxes": False, "show_masks": False, "show_ids": True},
                [],
                True,
            ),
            (
                {"show_boxes": True, "show_masks": True, "show_ids": True},
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
        annotators, label_annotator = _init_annotators(**flags)

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
                {"show_labels": True},
                ["person", "car"],
                id="class_names_from_list",
            ),
            pytest.param(
                {
                    "xyxy": np.array([[0, 0, 10, 10]]),
                    "class_id": np.array([5]),
                },
                ["person", "car"],
                {"show_labels": True},
                ["5"],
                id="fallback_to_class_id_when_out_of_range",
            ),
            pytest.param(
                {
                    "xyxy": np.array([[0, 0, 10, 10]]),
                    "tracker_id": np.array([42]),
                },
                [],
                {"show_ids": True},
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
                {"show_ids": True, "show_labels": True, "show_confidence": True},
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
        labels = _format_labels(detections, class_names, **label_flags)
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
            pytest.param("", None, id="empty_returns_none"),
            pytest.param("0,2", [0, 2], id="integer_ids"),
            pytest.param("person,car", [0, 2], id="class_names"),
            pytest.param("person,2,motorcycle", [0, 2, 3], id="mixed_names_and_ids"),
            pytest.param(" person , car ", [0, 2], id="whitespace_stripped"),
            pytest.param("99", [99], id="out_of_range_id_kept"),
        ],
    )
    def test_resolves_classes(
        self,
        classes_arg: str | None,
        expected: list[int] | None,
    ) -> None:
        result = _resolve_class_filter(classes_arg, self.CLASS_NAMES)
        assert result == expected

    def test_unknown_name_warns_and_skips(self, capsys: pytest.CaptureFixture) -> None:
        result = _resolve_class_filter("person,unicorn,car", self.CLASS_NAMES)
        assert result == [0, 2]
        assert "unicorn" in capsys.readouterr().err

    def test_all_unknown_names_returns_none(self, capsys: pytest.CaptureFixture) -> None:
        result = _resolve_class_filter("unicorn,dragon", self.CLASS_NAMES)
        assert result is None
        assert "unicorn" in capsys.readouterr().err


class TestResolveTrackIdFilter:
    @pytest.mark.parametrize(
        "track_ids_arg,expected",
        [
            pytest.param(None, None, id="none_returns_none"),
            pytest.param("", None, id="empty_returns_none"),
            pytest.param("0,2", [0, 2], id="integer_ids"),
            pytest.param("person,car", None, id="words_returns_none"),
            pytest.param("person,2,motorcycle", [2], id="mixed_names_and_ids"),
            pytest.param(" 1 , 3 ", [1, 3], id="whitespace_stripped"),
            pytest.param("99", [99], id="out_of_range_id_kept"),
        ],
    )
    def test_resolves_track_ids(
        self,
        track_ids_arg: str | None,
        expected: list[int] | None,
    ) -> None:
        result = _resolve_track_id_filter(track_ids_arg)
        assert result == expected

    def test_non_integer_warns_and_skips(self, capsys: pytest.CaptureFixture) -> None:
        result = _resolve_track_id_filter("1,abc,3")
        assert result == [1, 3]
        assert "abc" in capsys.readouterr().err

    def test_all_non_integer_returns_none(self, capsys: pytest.CaptureFixture) -> None:
        result = _resolve_track_id_filter("abc,def")
        assert result is None
        assert "abc" in capsys.readouterr().err


class _FakeReIDModel:
    last_kwargs: ClassVar[dict | None] = None

    @classmethod
    def from_pretrained(cls, **kwargs: object) -> _FakeReIDModel:
        cls.last_kwargs = dict(kwargs)
        return cls()


class TestReidTrackCli:
    def test_model_source_implies_enable(self) -> None:
        args = argparse.Namespace(
            tracker_reid_enable=False,
            tracker_reid_model="osnet_x1_0_msmt17_combineall",
        )
        assert _reid_requested(args)

    def test_requires_source_before_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "trackers.scripts.track.reid_provider.import_reid_model",
            lambda: pytest.fail("model provider should not be called"),
        )
        args = argparse.Namespace(
            tracker_reid_enable=True,
            tracker_reid_model=None,
            tracker_reid_device="cpu",
            tracker_reid_architecture=None,
            source=None,
        )
        params, err = _apply_reid_tracker_params("botsort", args, {})
        assert err is not None and "--source" in err
        assert params == {}

    def test_passes_architecture(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            "trackers.scripts.track.reid_provider.import_reid_model",
            lambda: _FakeReIDModel,
        )
        weights = tmp_path / "weights.pth"
        weights.touch()
        args = argparse.Namespace(
            tracker_reid_enable=False,
            tracker_reid_model=str(weights),
            tracker_reid_device="cpu",
            tracker_reid_architecture="osnet_x1_0",
            source="video.mp4",
        )
        params, err = _apply_reid_tracker_params("botsort", args, {})
        assert err is None
        assert _FakeReIDModel.last_kwargs is not None
        assert _FakeReIDModel.last_kwargs["architecture"] == "osnet_x1_0"
        assert "reid_model" in params

    def test_rejects_non_botsort_tracker(self) -> None:
        args = argparse.Namespace(
            tracker_reid_enable=True,
            tracker_reid_model=None,
            tracker_reid_device="cpu",
            tracker_reid_architecture=None,
            source="video.mp4",
        )
        _, error = _apply_reid_tracker_params("bytetrack", args, {})
        assert error is not None and "botsort" in error

    def test_help_lists_reid_flags(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)
        help_text = subparsers.choices["track"].format_help()
        assert "--tracker.reid.enable" in help_text
        assert "--tracker.reid.architecture" in help_text
        assert "--tracker.appearance_threshold" in help_text
