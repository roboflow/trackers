# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest
import supervision as sv

from trackers.cli.track import (
    _EXCLUDED_TRACKER_PARAMETERS,
    DetectionOptions,
    OutputOptions,
    ShowOptions,
    TrackerOptions,
    _abbreviate_parameter_name,
    _abbreviated_tracker_parameters,
    _expand_parameter_name,
    _format_labels,
    _init_annotators,
    _init_tracker,
    _resolve_class_filter,
    _resolve_track_id_filter,
    _resolve_tracker_kwargs,
    _tracker_options_as_dict,
    track_command,
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

    ``_init_tracker`` forwards only keys matching the tracker ``__init__`` signature, so a CLI-side rename with no
    matching alias is dropped without an error and the tracker silently keeps its own default. These tests make that
    failure mode visible.
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

    def test_registry_parameters_map_bidirectionally_to_option_fields(self) -> None:
        """TrackerOptions fields and the registry union agree in both directions.

        The old version of this test only checked that every ``TrackerOptions``
        field mapped to a known registry parameter — sufficient for a
        hand-maintained dataclass, but vacuous once the fields are generated
        from that same registry. The reverse direction is what actually
        guards against drift: a newly registered tracker parameter that never
        made it into the generated dataclass.
        """
        # ``name`` selects the tracker and ``iou_variant`` aliases ``iou``;
        # neither is a registry parameter under its own field name. Walking
        # ``.items()`` rather than the dict directly matters: it is the
        # override that drops ``iou`` (see ``TrackerParameters.items``),
        # matching what ``_tracker_parameter_union`` itself iterates over.
        accepted = {"name", "iou_variant"}
        registry_names: set[str] = set()
        for tracker_id in BaseTracker._registered_trackers():
            info = BaseTracker._lookup_tracker(tracker_id)
            assert info is not None
            registry_names.update(name for name, _ in info.parameters.items())
        accepted.update(registry_names)

        option_fields = {field.name for field in fields(TrackerOptions)}

        unmatched_fields = [
            name for name in option_fields if name not in accepted and _expand_parameter_name(name) not in accepted
        ]
        unmatched_registry_names = [
            name
            for name in registry_names
            if name not in _EXCLUDED_TRACKER_PARAMETERS and _abbreviate_parameter_name(name) not in option_fields
        ]

        assert unmatched_fields == []
        assert unmatched_registry_names == []

    def test_abbreviation_round_trips_for_every_registry_parameter(self) -> None:
        """Every registry parameter's CLI name expands back to its Python name."""
        for tracker_id in BaseTracker._registered_trackers():
            info = BaseTracker._lookup_tracker(tracker_id)
            assert info is not None
            for python_name, _ in info.parameters.items():
                if python_name in _EXCLUDED_TRACKER_PARAMETERS:
                    continue
                cli_name = _abbreviate_parameter_name(python_name)
                assert _expand_parameter_name(cli_name) == python_name

    def test_mask_rename_is_not_listed_as_a_deprecation(self) -> None:
        """``mask_config`` never shipped under its Python name, so it is not deprecated.

        Pins the deliberate exclusion in ``_abbreviated_tracker_parameters``: warning that ``--tracker.mask_config`` is
        deprecated would be a lie, since no released CLI ever exposed that spelling.
        """
        assert "mask_config" not in _abbreviated_tracker_parameters()

    def test_mask_field_reaches_the_tracker_as_a_dataclass_not_a_dict(self) -> None:
        """``mask`` is forwarded intact, not flattened by ``dataclasses.asdict``.

        Regression guard for the bug ``_init_tracker`` had before it stopped using ``asdict``: that call recurses into
        nested dataclass fields, so ``mask`` would have reached ``McByteTracker`` as a plain ``dict`` instead of the
        ``McByteMaskConfig`` instance it requires.
        """
        from trackers.core.mcbyte.tracker import McByteMaskConfig

        mask_config = McByteMaskConfig(cutie_mem_every=7)
        options = TrackerOptions(name="mcbyte", mask=mask_config)

        raw = _tracker_options_as_dict(options)
        assert raw["mask"] is mask_config

        info = BaseTracker._lookup_tracker("mcbyte")
        assert info is not None
        kwargs, dropped = _resolve_tracker_kwargs(
            {k: v for k, v in raw.items() if k not in ("name", "iou_variant")},
            set(info.parameters),
        )

        assert dropped == []
        assert kwargs["mask_config"] is mask_config

    def test_unsupported_override_is_dropped_with_a_warning(self) -> None:
        """A field the selected tracker does not accept warns instead of vanishing."""
        options = TrackerOptions(name="sort", min_mask_coverage=0.3)

        with pytest.warns(UserWarning, match=r"sort.*--tracker\.min_mask_coverage"):
            tracker = _init_tracker(options)

        assert not hasattr(tracker, "minimum_mask_coverage")


class TestInitTrackerDetectedFrameRate:
    """``detected_frame_rate`` should seed ``frame_rate`` unless the CLI already overrode it.

    Regression coverage for H-FRAME-RATE-NEVER-PROPAGATED: every tracker's ``frame_rate`` defaults to a hardcoded
    30.0 (see e.g. ``BaseTracker._compute_maximum_frames_without_update``, ``trackers.utils.motion_models``), and
    without this parameter ``_init_tracker`` had no way to fill it in from the source's real fps.
    """

    def test_detected_frame_rate_reaches_the_tracker(self) -> None:
        """A source fps with no explicit ``--tracker.frame_rate`` override reaches the constructor."""
        options = TrackerOptions(name="bytetrack")

        tracker = _init_tracker(options, detected_frame_rate=24.0)

        assert tracker._frame_rate == pytest.approx(24.0)

    def test_explicit_override_wins_over_detected_frame_rate(self) -> None:
        """``--tracker.frame_rate`` set explicitly must not be clobbered by the detected value."""
        options = TrackerOptions(name="bytetrack", frame_rate=60.0)

        tracker = _init_tracker(options, detected_frame_rate=24.0)

        assert tracker._frame_rate == pytest.approx(60.0)

    def test_no_detected_frame_rate_keeps_the_constructor_default(self) -> None:
        """``detected_frame_rate=None`` (unknown source fps, e.g. a webcam) changes nothing."""
        options = TrackerOptions(name="bytetrack")

        tracker = _init_tracker(options, detected_frame_rate=None)

        assert tracker._frame_rate == pytest.approx(30.0)

    @pytest.mark.parametrize("bad_value", [0.0, -24.0, float("nan"), float("inf")])
    def test_non_positive_or_non_finite_detected_frame_rate_is_ignored(self, bad_value: float) -> None:
        """A degenerate probed fps (``<= 0``, non-finite, e.g. a corrupt container) falls back to the tracker default.

        Without the ``math.isfinite`` guard, ``inf``/``nan`` pass the old ``> 0`` check (``inf > 0`` is ``True``) and
        reach the tracker constructor, which raises ``ValueError`` inside
        ``BaseTracker._compute_maximum_frames_without_update`` instead of falling back cleanly.
        """
        options = TrackerOptions(name="bytetrack")

        tracker = _init_tracker(options, detected_frame_rate=bad_value)

        assert tracker._frame_rate == pytest.approx(30.0)

    def test_every_registered_tracker_accepts_frame_rate(self) -> None:
        """Every tracker in the registry accepts ``frame_rate`` today.

        Backs the ``"frame_rate" in accepted`` guard in ``_init_tracker``: that guard exists so a detected value is
        silently skipped for a tracker that has no such parameter, but no registered tracker currently exercises that
        branch — this pins the current registry shape so the guard's dead branch is visible rather than silently
        unreachable forever.
        """
        for tracker_id in BaseTracker._registered_trackers():
            info = BaseTracker._lookup_tracker(tracker_id)
            assert info is not None
            assert "frame_rate" in info.parameters


class TestTrackCommandDetectsSourceFrameRate:
    """``track_command`` must read the source's real fps before building the tracker.

    ``_init_tracker`` used to be called before ``source_info`` (which holds the real fps) even
    existed in this function's scope — the CLI entry point had no way to reach the fix in
    ``TestInitTrackerDetectedFrameRate`` above without first reordering ``track_command`` itself.
    This is the end-to-end wiring check for that reorder: a real (tiny) video written at a
    non-default fps, replayed through ``--detection.mot_file`` so no detection model is needed.
    """

    @staticmethod
    def _write_video(path: Path, fps: float, frame_count: int = 3) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (4, 4))
        try:
            for _ in range(frame_count):
                writer.write(np.zeros((4, 4, 3), dtype=np.uint8))
        finally:
            writer.release()

    def test_video_fps_reaches_the_tracker(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The video's real (non-30) fps is what the tracker is built with."""
        video_path = tmp_path / "clip.mp4"
        self._write_video(video_path, fps=24.0)

        detection_file = tmp_path / "dets.txt"
        detection_file.write_text("1,1,1,1,2,2,0.9,-1,-1,-1\n2,1,1,1,2,2,0.9,-1,-1,-1\n")

        real_init_tracker = _init_tracker
        captured: dict[str, float | None] = {}

        def spy_init_tracker(
            params: TrackerOptions | None,  # type: ignore[valid-type]
            *,
            detected_frame_rate: float | None = None,
        ) -> BaseTracker:
            captured["detected_frame_rate"] = detected_frame_rate
            return real_init_tracker(params, detected_frame_rate=detected_frame_rate)

        monkeypatch.setattr("trackers.cli.track._init_tracker", spy_init_tracker)

        exit_code = track_command(
            source=str(video_path),
            detection=DetectionOptions(mot_file=detection_file),
            tracker=TrackerOptions(name="bytetrack"),
            output=OutputOptions(mot_results=tmp_path / "out.txt"),
        )

        assert exit_code == 0
        assert captured["detected_frame_rate"] == pytest.approx(24.0)

    def test_mot_file_only_run_has_no_detected_frame_rate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No video source at all (pure ``--detection.mot_file`` replay): nothing to detect from."""
        detection_file = tmp_path / "dets.txt"
        detection_file.write_text("1,1,1,1,2,2,0.9,-1,-1,-1\n2,1,1,1,2,2,0.9,-1,-1,-1\n")

        real_init_tracker = _init_tracker
        captured: dict[str, float | None] = {}

        def spy_init_tracker(
            params: TrackerOptions | None,  # type: ignore[valid-type]
            *,
            detected_frame_rate: float | None = None,
        ) -> BaseTracker:
            captured["detected_frame_rate"] = detected_frame_rate
            return real_init_tracker(params, detected_frame_rate=detected_frame_rate)

        monkeypatch.setattr("trackers.cli.track._init_tracker", spy_init_tracker)

        exit_code = track_command(
            detection=DetectionOptions(mot_file=detection_file),
            tracker=TrackerOptions(name="bytetrack"),
            output=OutputOptions(mot_results=tmp_path / "out.txt"),
        )

        assert exit_code == 0
        assert captured["detected_frame_rate"] is None
