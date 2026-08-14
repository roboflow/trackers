# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock

import numpy as np
import pytest
import supervision as sv

from trackers.cli.track import (
    _EXCLUDED_TRACKER_PARAMETERS,
    DetectionOptions,
    ReIDOptions,
    ShowOptions,
    TrackerOptions,
    _abbreviate_parameter_name,
    _abbreviated_tracker_parameters,
    _expand_parameter_name,
    _format_labels,
    _init_annotators,
    _init_tracker,
    _load_reid_model,
    _reid_requested,
    _resolve_class_filter,
    _resolve_track_id_filter,
    _resolve_tracker_kwargs,
    _tracker_options_as_dict,
    track_command,
)
from trackers.core.base import BaseTracker
from trackers.core.botsort.tracker import BoTSORTTracker


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


class _FakeReIDModel:
    """Stand-in for ``reid.ReIDModel`` that records its loading kwargs."""

    last_kwargs: ClassVar[dict | None] = None

    @classmethod
    def from_pretrained(cls, **kwargs: object) -> _FakeReIDModel:
        cls.last_kwargs = dict(kwargs)
        return cls()

    def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        return np.zeros((len(detections), 8), dtype=np.float32)


@pytest.fixture
def fake_reid_module(monkeypatch: pytest.MonkeyPatch) -> type[_FakeReIDModel]:
    """Install a fake ``reid`` module so loading needs no checkpoint download."""
    import sys
    from types import ModuleType

    module = ModuleType("reid")
    module.ReIDModel = _FakeReIDModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "reid", module)
    _FakeReIDModel.last_kwargs = None
    return _FakeReIDModel


class TestReIDOptions:
    """CLI wiring for optional appearance-encoder loading."""

    def test_model_source_implies_enable(self) -> None:
        """Naming a checkpoint is enough; --reid.enable is not also required."""
        assert _reid_requested(ReIDOptions(model="osnet_x1_0_msmt17_combineall"))

    @pytest.mark.parametrize(
        "options",
        [
            pytest.param(ReIDOptions(architecture="osnet_x1_0"), id="architecture"),
            pytest.param(ReIDOptions(device="cpu"), id="device"),
        ],
    )
    def test_non_default_options_imply_enable(self, options: ReIDOptions) -> None:
        assert _reid_requested(options)

    def test_explicit_disable_conflicts_with_model(self) -> None:
        with pytest.raises(ValueError, match="cannot be combined"):
            _reid_requested(ReIDOptions(enable=False, model="osnet_x1_0_msmt17_combineall"))

    def test_absent_by_default(self) -> None:
        """Default options leave ReID off, so geometry-only tracking is unchanged."""
        assert not _reid_requested(ReIDOptions())

    def test_track_command_rejects_reid_without_source(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The command entry point rejects ReID when MOT input supplies no frames."""
        exit_code = track_command(
            detection=DetectionOptions(mot_file=Path("detections.txt")),
            reid=ReIDOptions(enable=True),
        )

        assert exit_code == 1
        assert capsys.readouterr().err == (
            "Error: ReID requires --source (video/webcam/images) so appearance embeddings "
            "can be extracted from frames.\n"
        )

    def test_missing_optional_extra_reports_install_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing ReID dependency is translated into actionable CLI guidance."""
        import sys
        from types import ModuleType

        monkeypatch.setitem(sys.modules, "reid", ModuleType("reid"))

        with pytest.raises(ValueError) as exc_info:
            _load_reid_model(ReIDOptions(enable=True))

        assert str(exc_info.value) == (
            "ReID tracking requires the optional `trackers[reid]` extra.\nInstall with: pip install 'trackers[reid]'"
        )
        assert isinstance(exc_info.value.__cause__, ImportError)

    @pytest.mark.parametrize(
        "load_error",
        [
            pytest.param(OSError("checkpoint unavailable"), id="os_error"),
            pytest.param(ValueError("checkpoint unavailable"), id="value_error"),
            pytest.param(RuntimeError("checkpoint unavailable"), id="runtime_error"),
        ],
    )
    def test_model_load_errors_include_reid_context(
        self,
        load_error: Exception,
        fake_reid_module: type[_FakeReIDModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Expected loader failures identify the ReID checkpoint operation."""
        monkeypatch.setattr(fake_reid_module, "from_pretrained", Mock(side_effect=load_error))

        with pytest.raises(ValueError, match="Failed to load ReID model: checkpoint unavailable") as exc_info:
            _load_reid_model(ReIDOptions(enable=True))

        assert exc_info.value.__cause__ is load_error

    def test_encoder_reaches_the_tracker(self, fake_reid_module: type[_FakeReIDModel]) -> None:
        """A requested encoder is injected as the tracker's reid_model."""
        tracker = _init_tracker(TrackerOptions(name="botsort"), ReIDOptions(enable=True, device="cpu"))

        assert isinstance(tracker, BoTSORTTracker)
        assert isinstance(tracker.reid_model, _FakeReIDModel)
        assert fake_reid_module.last_kwargs == {"device": "cpu"}

    def test_architecture_is_forwarded(self, fake_reid_module: type[_FakeReIDModel], tmp_path: Path) -> None:
        """Bare weights pass both source and architecture through to the loader."""
        weights = tmp_path / "weights.pth"
        weights.touch()

        _init_tracker(
            TrackerOptions(name="botsort"),
            ReIDOptions(model=str(weights), device="cpu", architecture="osnet_x1_0"),
        )

        assert fake_reid_module.last_kwargs is not None
        assert fake_reid_module.last_kwargs["architecture"] == "osnet_x1_0"
        assert fake_reid_module.last_kwargs["source"] == str(weights)

    def test_architecture_requires_model(self, fake_reid_module: type[_FakeReIDModel]) -> None:
        """An architecture with no checkpoint to apply it to is rejected."""
        with pytest.raises(ValueError, match=r"--reid\.architecture requires --reid\.model"):
            _init_tracker(TrackerOptions(name="botsort"), ReIDOptions(enable=True, architecture="osnet_x0_25"))

    def test_tracker_without_encoder_support_is_rejected(self, fake_reid_module: type[_FakeReIDModel]) -> None:
        """Requesting ReID on a geometry-only tracker fails loudly."""
        with pytest.raises(ValueError, match=r"--reid\.\* options apply only to a tracker that accepts an encoder"):
            _init_tracker(TrackerOptions(name="bytetrack"), ReIDOptions(enable=True))

    def test_transitive_reid_import_error_is_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original_import = builtins.__import__

        def fail_reid_import(
            name: str,
            global_vars: Mapping[str, object] | None = None,
            local_vars: Mapping[str, object] | None = None,
            fromlist: Sequence[str] | None = None,
            level: int = 0,
        ) -> object:
            if name == "reid":
                raise ImportError("No module named 'broken_dependency'", name="broken_dependency")
            return original_import(name, global_vars, local_vars, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fail_reid_import)

        with pytest.raises(ImportError, match="broken_dependency") as exc_info:
            _load_reid_model(ReIDOptions(enable=True))

        assert exc_info.value.name == "broken_dependency"

    def test_reid_model_is_not_a_cli_flag(self) -> None:
        """The encoder is built from ReIDOptions, so it must not become a --tracker flag."""
        assert "reid_model" in _EXCLUDED_TRACKER_PARAMETERS
        assert "reid_model" not in {field.name for field in fields(TrackerOptions)}

    def test_appearance_parameters_stay_cli_reachable(self) -> None:
        """The three scalar ReID knobs are still generated from the registry."""
        option_fields = {field.name for field in fields(TrackerOptions)}
        assert {
            "reid_appearance_threshold",
            "reid_ema_alpha",
            "reid_proximity_threshold",
        } <= option_fields
        assert {"appearance_threshold", "proximity_threshold"}.isdisjoint(option_fields)
