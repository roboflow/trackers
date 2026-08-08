# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the ``trackers inspect`` component group.

Every test here stops at the parsing and validation surface. None of the
components can run without SAM and Cutie weights, so the contract under test is
that a wrong invocation is rejected before any model is touched.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from trackers.cli._legacy import _translate_legacy_args
from trackers.cli._parser import _SUBCOMMANDS
from trackers.cli.inspect import INSPECT_COMPONENTS
from trackers.cli.inspect._common import (
    IMAGE_EXTENSIONS,
    INSPECT_OUTPUT_ROOT,
    list_selected_frame_paths,
    parse_xyxy_box,
    timestamped_run_dir,
)
from trackers.cli.inspect.mask_manager import (
    _GT_ONLY_OPTIONS,
    _MANUAL_ONLY_OPTIONS,
    MaskManagerMode,
    _raise_for_mode_option_conflict,
    mask_manager_command,
)


class TestInspectRegistration:
    """The component table and its wiring into the shared CLI machinery."""

    def test_every_component_names_what_it_inspects(self) -> None:
        """Component keys name the inspected thing, not the tracker that uses it."""
        assert set(INSPECT_COMPONENTS) == {"sam", "cutie", "mask-manager", "mcbyte"}

    def test_components_are_callables(self) -> None:
        """Each entry dispatches to a command function."""
        assert all(callable(command) for command in INSPECT_COMPONENTS.values())

    def test_inspect_is_a_registered_subcommand(self) -> None:
        """An absent entry makes the legacy translator skip the hyphen sweep."""
        assert "inspect" in _SUBCOMMANDS

    def test_benchmark_replaced_mcbyte_at_the_top_level(self) -> None:
        """The top level holds verbs only; ``mcbyte`` is now an argument to one."""
        assert "benchmark" in _SUBCOMMANDS
        assert "mcbyte" not in _SUBCOMMANDS


class TestInspectArgumentNormalisation:
    """Hyphen and underscore spellings reach the components identically."""

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            pytest.param(
                ["inspect", "sam", "--image-path", "a.jpg"],
                ["inspect", "sam", "--image_path", "a.jpg"],
                id="sam-image-path",
            ),
            pytest.param(
                ["inspect", "mask-manager", "--gt-file", "gt.txt"],
                ["inspect", "mask-manager", "--gt_file", "gt.txt"],
                id="mask-manager-gt-file",
            ),
            pytest.param(
                ["inspect", "mcbyte", "--sequence.image-dir", "frames"],
                ["inspect", "mcbyte", "--sequence.image_dir", "frames"],
                id="mcbyte-dotted-group",
            ),
        ],
    )
    def test_hyphenated_options_normalise(self, argv: list[str], expected: list[str]) -> None:
        """Option names normalise while the component token is left alone."""
        assert _translate_legacy_args(argv) == expected

    def test_hyphenated_component_token_survives(self) -> None:
        """``mask-manager`` is a positional, so the hyphen sweep must not touch it."""
        translated = _translate_legacy_args(["inspect", "mask-manager", "--mode", "gt"])
        assert translated[1] == "mask-manager"


class TestMaskManagerModeValidation:
    """``--mode`` selects one option set, and the other is rejected, not ignored."""

    def test_manual_and_gt_option_sets_are_disjoint(self) -> None:
        """The two tables must not overlap, or an option would be foreign to both modes."""
        assert not set(_MANUAL_ONLY_OPTIONS) & set(_GT_ONLY_OPTIONS)

    @pytest.mark.parametrize(
        ("mode", "supplied", "expected"),
        [
            pytest.param(
                "manual",
                {"start_file": "a.jpg", "end_file": "b.jpg", "box": [(1.0, 2.0, 3.0, 4.0)], "gt_file": "gt.txt"},
                "--gt_file is a --mode gt option",
                id="gt-option-under-manual",
            ),
            pytest.param(
                "gt",
                {"gt_file": "gt.txt", "start_frame": 1, "end_frame": 2, "box": [(1.0, 2.0, 3.0, 4.0)]},
                "--box is a --mode manual option",
                id="manual-option-under-gt",
            ),
        ],
    )
    def test_foreign_option_is_rejected(self, mode: MaskManagerMode, supplied: dict, expected: str) -> None:
        """An option belonging to the other mode names itself in the error."""
        with pytest.raises(ValueError, match=expected):
            _raise_for_mode_option_conflict(mode, supplied)

    @pytest.mark.parametrize(
        ("mode", "supplied", "expected"),
        [
            pytest.param("manual", {}, "--mode manual requires", id="manual-missing-all"),
            pytest.param("gt", {}, "--mode gt requires", id="gt-missing-all"),
            pytest.param(
                "gt",
                {"gt_file": "gt.txt", "start_frame": 1},
                "--end_frame",
                id="gt-missing-one",
            ),
        ],
    )
    def test_missing_required_option_is_reported(self, mode: MaskManagerMode, supplied: dict, expected: str) -> None:
        """Each mode's own required options are enforced by name."""
        with pytest.raises(ValueError, match=expected):
            _raise_for_mode_option_conflict(mode, supplied)

    @pytest.mark.parametrize(
        ("mode", "supplied"),
        [
            pytest.param(
                "manual",
                {
                    "start_file": "a.jpg",
                    "end_file": "b.jpg",
                    "box": [(1.0, 2.0, 3.0, 4.0)],
                    "add_at": ["a.jpg:1,2,3,4"],
                },
                id="manual-complete",
            ),
            pytest.param(
                "gt",
                {"gt_file": "gt.txt", "start_frame": 1, "end_frame": 9, "tracklet_id": [3]},
                id="gt-complete",
            ),
        ],
    )
    def test_valid_invocation_passes(self, mode: MaskManagerMode, supplied: dict) -> None:
        """A complete, single-mode invocation raises nothing."""
        _raise_for_mode_option_conflict(mode, supplied)

    def test_conflict_is_reported_before_any_model_loads(self, capsys: pytest.CaptureFixture) -> None:
        """The command returns non-zero without importing SAM or Cutie.

        The image directory does not exist either. Reaching the filesystem check
        first would mean the mode check ran too late to be the guard it claims.
        """
        exit_code = mask_manager_command(Path("does-not-exist"), mode="manual", gt_file=Path("gt.txt"))

        assert exit_code == 1
        assert "--gt_file is a --mode gt option" in capsys.readouterr().err


class TestInspectCommonHelpers:
    """Helpers shared by the components, extracted only where implementations matched."""

    def test_parse_xyxy_box_returns_four_floats(self) -> None:
        """A well-formed box parses to a float 4-tuple."""
        assert parse_xyxy_box("10,20,110,220") == (10.0, 20.0, 110.0, 220.0)

    @pytest.mark.parametrize(
        "box",
        [
            pytest.param("10,20,110", id="too-few"),
            pytest.param("10,20,110,220,330", id="too-many"),
        ],
    )
    def test_parse_xyxy_box_rejects_wrong_arity(self, box: str) -> None:
        """A box that is not exactly four values is an error."""
        with pytest.raises(ValueError, match="exactly 4 comma-separated values"):
            parse_xyxy_box(box)

    def test_output_root_is_relative_to_the_working_directory(self) -> None:
        """Outputs land where the caller ran the command, never in the source tree."""
        assert not INSPECT_OUTPUT_ROOT.is_absolute()
        assert INSPECT_OUTPUT_ROOT.parts[0] == "outputs"

    def test_timestamped_run_dir_creates_a_fresh_directory(self, tmp_path: Path) -> None:
        """Each run gets its own directory under the given root."""
        run_dir = timestamped_run_dir(tmp_path)

        assert run_dir.is_dir()
        assert run_dir.parent == tmp_path

    def test_list_selected_frame_paths_is_inclusive(self, tmp_path: Path) -> None:
        """Both endpoints are included in the returned range."""
        for index in range(1, 5):
            (tmp_path / f"{index:06d}.jpg").write_bytes(b"")

        selected = list_selected_frame_paths(tmp_path, "000002.jpg", "000004.jpg")

        assert [path.name for path in selected] == ["000002.jpg", "000003.jpg", "000004.jpg"]

    def test_list_selected_frame_paths_ignores_non_images(self, tmp_path: Path) -> None:
        """Files whose suffix is not a frame extension are not part of the range."""
        (tmp_path / "000001.jpg").write_bytes(b"")
        (tmp_path / "000002.jpg").write_bytes(b"")
        (tmp_path / "notes.txt").write_bytes(b"")

        selected = list_selected_frame_paths(tmp_path, "000001.jpg", "000002.jpg")

        assert [path.name for path in selected] == ["000001.jpg", "000002.jpg"]
        assert ".txt" not in IMAGE_EXTENSIONS

    def test_list_selected_frame_paths_rejects_reversed_range(self, tmp_path: Path) -> None:
        """An end frame before the start frame is an error, not an empty result."""
        for index in range(1, 3):
            (tmp_path / f"{index:06d}.jpg").write_bytes(b"")

        with pytest.raises(ValueError, match="must not come before"):
            list_selected_frame_paths(tmp_path, "000002.jpg", "000001.jpg")

    def test_list_selected_frame_paths_reports_a_missing_endpoint(self, tmp_path: Path) -> None:
        """A start file that is not in the directory names itself in the error."""
        (tmp_path / "000001.jpg").write_bytes(b"")

        with pytest.raises(FileNotFoundError, match=re.escape("000009.jpg")):
            list_selected_frame_paths(tmp_path, "000009.jpg", "000001.jpg")
