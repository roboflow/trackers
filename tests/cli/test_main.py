# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for trackers.cli.__main__ — jsonargparse CLI integration."""

from __future__ import annotations

import re
from argparse import ArgumentError
from importlib.metadata import version
from pathlib import Path

import pytest
import yaml
from jsonargparse import ArgumentParser

from trackers.cli.__main__ import _CLIParser, _translate_legacy_args
from trackers.cli.eval import eval_cmd
from trackers.cli.track import track


@pytest.fixture()
def track_parser() -> ArgumentParser:
    """ArgumentParser built from the track() signature with --config support."""
    parser = ArgumentParser(exit_on_error=False)
    parser.add_function_arguments(track)
    parser.add_argument("--config", action="config")
    return parser


class TestConfigFileSupport:
    """Verify jsonargparse --config flag behaviour for the track subcommand."""

    def test_config_value_applied_to_tracker(self, track_parser: ArgumentParser, tmp_path: Path) -> None:
        """YAML --config value is parsed into the track() namespace."""
        cfg = tmp_path / "run.yaml"
        cfg.write_text(yaml.dump({"tracker": "sort"}))

        ns = track_parser.parse_args(["--config", str(cfg)])

        assert ns.tracker == "sort"

    def test_cli_arg_overrides_config_value(self, track_parser: ArgumentParser, tmp_path: Path) -> None:
        """Explicit CLI arg takes precedence over the --config file value."""
        cfg = tmp_path / "run.yaml"
        cfg.write_text(yaml.dump({"tracker": "sort"}))

        ns = track_parser.parse_args(["--config", str(cfg), "--tracker", "bytetrack"])

        assert ns.tracker == "bytetrack"

    def test_nested_dataclass_field_in_config(self, track_parser: ArgumentParser, tmp_path: Path) -> None:
        """Nested DetectionOptions fields can be set via --config."""
        cfg = tmp_path / "run.yaml"
        cfg.write_text(yaml.dump({"detection": {"confidence": 0.3}}))

        ns = track_parser.parse_args(["--config", str(cfg)])

        assert ns.detection.confidence == pytest.approx(0.3)


class TestCliMigration:
    """Verify legacy CLI arguments transition to dotted dataclass paths."""

    def test_semantic_output_and_mot_file_paths_are_available(self) -> None:
        """CLI names identify the output artifact and precomputed MOT input file."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track)

        parsed = parser.instantiate_classes(
            parser.parse_args(
                [
                    "--detection.mot_file",
                    "detections.txt",
                    "--output.video",
                    "tracked.mp4",
                ]
            )
        )

        assert parsed.detection.mot_file == Path("detections.txt")
        assert parsed.output.video == Path("tracked.mp4")

    def test_overwrite_is_nested_and_display_remains_flat(self) -> None:
        """Output write policy is nested while live preview remains a flat action."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track)

        parsed = parser.instantiate_classes(
            parser.parse_args(
                [
                    "--output.overwrite",
                    "--display",
                    "--show.boxes",
                    "false",
                    "--show.ids",
                    "false",
                ]
            )
        )

        assert parsed.output.overwrite is True
        assert parsed.display is True
        assert parsed.show.boxes is False
        assert parsed.show.ids is False

    def test_hyphenated_new_paths_normalize_to_underscores(self) -> None:
        """Hyphenated spellings work for current options that use underscores."""
        args = _translate_legacy_args(
            [
                "track",
                "--detection.mot-file",
                "detections.txt",
                "--output.mot-results",
                "tracks.txt",
            ]
        )

        assert args == [
            "track",
            "--detection.mot_file",
            "detections.txt",
            "--output.mot_results",
            "tracks.txt",
        ]

    @pytest.mark.parametrize(
        "option",
        ["--no-boxes", "--no-show.boxes", "--no-ids", "--no-show.ids"],
    )
    def test_negative_show_aliases_are_not_available(self, option: str) -> None:
        """Boxes and IDs use explicit boolean values instead of negative flags."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track)

        with pytest.raises(ArgumentError, match=option):
            parser.parse_args([option])

    def test_pr_era_paths_map_to_semantic_names(self) -> None:
        """Temporary jsonargparse paths remain deprecated aliases."""
        with pytest.warns(FutureWarning, match=r"--detection\.detections.*--detection\.mot_file"):
            args = _translate_legacy_args(
                [
                    "track",
                    "--detection.detections",
                    "detections.txt",
                    "--out.output",
                    "tracked.mp4",
                ]
            )

        assert args == [
            "track",
            "--detection.mot_file",
            "detections.txt",
            "--output.video",
            "tracked.mp4",
        ]

    def test_pr_era_overwrite_path_maps_to_output_group(self) -> None:
        """The temporary output overwrite path targets the canonical output field."""
        with pytest.warns(FutureWarning, match=r"--out\.overwrite.*--output\.overwrite"):
            args = _translate_legacy_args(["track", "--out.overwrite"])

        assert args == ["track", "--output.overwrite"]

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            (["track", "--detections", "detections.txt"], ["track", "--detection.mot_file", "detections.txt"]),
            (
                ["track", "--detection.detections", "detections.txt"],
                ["track", "--detection.mot_file", "detections.txt"],
            ),
            (["track", "--vis.display"], ["track", "--display"]),
            (["track", "--overwrite"], ["track", "--output.overwrite"]),
            (["track", "--out.overwrite"], ["track", "--output.overwrite"]),
        ],
    )
    def test_evidenced_track_aliases_map_to_intended_cli(
        self,
        source: list[str],
        target: list[str],
    ) -> None:
        """Develop and PR-era aliases map to the intended semantic CLI."""
        with pytest.warns(FutureWarning):
            assert _translate_legacy_args(source) == target

    @pytest.mark.parametrize(
        "args",
        [
            ["track", "--detection.model", "rfdetr-base", "--detection.mot_file", "detections.txt"],
            ["track", "--model", "rfdetr-base", "--detections", "detections.txt"],
            ["track", "--model", "rfdetr-base", "--detection.detections", "detections.txt"],
        ],
    )
    def test_explicit_model_and_mot_file_are_mutually_exclusive(self, args: list[str]) -> None:
        """Preserve develop's exclusive detector-source contract for CLI inputs."""
        with pytest.raises(ValueError, match=r"--detection\.model.*--detection\.mot_file"):
            _translate_legacy_args(args)

    def test_track_parser_accepts_dotted_dataclass_arguments(self) -> None:
        """Dotted CLI options instantiate nested options before calling track()."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track)

        parsed = parser.instantiate_classes(
            parser.parse_args(
                [
                    "--detection.model",
                    "rfdetr-base",
                    "--detection.confidence",
                    "0.3",
                    "--filters.classes",
                    "person,car",
                    "--output.video",
                    "tracked.mp4",
                    "--show.boxes",
                    "false",
                ]
            )
        )

        assert parsed.detection.model == "rfdetr-base"
        assert parsed.detection.confidence == pytest.approx(0.3)
        assert parsed.filters.classes == "person,car"
        assert parsed.output.video == Path("tracked.mp4")
        assert parsed.show.boxes is False

    def test_track_legacy_arguments_map_to_nested_paths(self) -> None:
        """Legacy track options warn and preserve the intended parsed values."""
        with pytest.warns(FutureWarning, match=r"--model.*--detection\.model"):
            args = _translate_legacy_args(
                [
                    "track",
                    "--model=rfdetr-base",
                    "--model.confidence",
                    "0.3",
                    "--tracker.lost_track_buffer",
                    "40",
                    "--tracker.enable_cmc",
                    "--mot-output",
                    "tracks.txt",
                    "--show.boxes",
                    "false",
                ]
            )

        assert args == [
            "track",
            "--detection.model=rfdetr-base",
            "--detection.confidence",
            "0.3",
            "--tracker_params.lost_track_buffer",
            "40",
            "--tracker_params.enable_cmc=false",
            "--output.mot_results",
            "tracks.txt",
            "--show.boxes",
            "false",
        ]
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track)
        parsed = parser.instantiate_classes(parser.parse_args(args[1:]))

        assert parsed.detection.model == "rfdetr-base"
        assert parsed.detection.confidence == pytest.approx(0.3)
        assert parsed.tracker_params.lost_track_buffer == 40
        assert parsed.tracker_params.enable_cmc is False
        assert parsed.output.mot_results == Path("tracks.txt")
        assert parsed.show.boxes is False

    def test_legacy_and_new_argument_for_same_target_fails(self) -> None:
        """Mixed spellings cannot silently choose a value for one target."""
        with pytest.raises(ValueError, match=r"--model.*--detection\.model"):
            _translate_legacy_args(["track", "--model", "rfdetr-base", "--detection.model", "rfdetr-nano"])

    def test_download_positional_dataset_maps_to_named_argument(self) -> None:
        """Legacy download DATASET syntax remains available during transition."""
        with pytest.warns(FutureWarning, match=r"positional dataset.*--dataset"):
            args = _translate_legacy_args(["download", "mot17", "--cache-dir", "cache"])

        assert args == ["download", "--dataset", "mot17", "--cache_dir", "cache"]

    def test_legacy_space_separated_lists_map_to_jsonargparse_lists(self) -> None:
        """Legacy metrics and columns lists remain usable during the transition."""
        with pytest.warns(FutureWarning, match=r"space-separated --metrics"):
            args = _translate_legacy_args(
                [
                    "eval",
                    "--metrics",
                    "CLEAR",
                    "HOTA",
                    "--columns",
                    "MOTA",
                    "HOTA",
                    "-o",
                    "results.json",
                ]
            )

        assert args == [
            "eval",
            "--metrics",
            '["CLEAR", "HOTA"]',
            "--columns",
            '["MOTA", "HOTA"]',
            "--output",
            "results.json",
        ]
        parser = ArgumentParser(exit_on_error=False)
        parser.add_function_arguments(eval_cmd)
        parsed = parser.parse_args(args[1:])

        assert parsed.metrics == ["CLEAR", "HOTA"]
        assert parsed.columns == ["MOTA", "HOTA"]
        assert parsed.output == Path("results.json")

    def test_current_json_list_values_are_not_rewritten(self) -> None:
        """Current jsonargparse list syntax remains the canonical input form."""
        args = ["eval", "--metrics", "[CLEAR,HOTA]", "--columns=[MOTA,HOTA]"]

        assert _translate_legacy_args(args) == args

    @pytest.mark.parametrize(
        "args",
        [
            ["track", "--model", "rfdetr-base"],
            ["eval", "--metrics", "CLEAR", "HOTA"],
            ["download", "mot17"],
        ],
    )
    def test_legacy_warnings_state_the_scheduled_removal_release(self, args: list[str]) -> None:
        """Every legacy CLI transition names the release in which it is removed."""
        major, minor, *_ = version("trackers").split(".")
        removal_version = f"{major}.{int(minor) + 3}.0"

        with pytest.warns(FutureWarning, match=re.escape(f"removed in {removal_version}")):
            _translate_legacy_args(args)

    @pytest.mark.parametrize(
        ("hyphenated", "canonical"),
        [
            (
                ["eval", "--gt-dir", "gt", "--tracker-dir=predictions"],
                ["eval", "--gt_dir", "gt", "--tracker_dir=predictions"],
            ),
            (
                ["tune", "--detections-dir", "detections", "--n-trials=5"],
                ["tune", "--detections_dir", "detections", "--n_trials=5"],
            ),
            (["tune", "--no-enqueue-defaults"], ["tune", "--no-enqueue_defaults"]),
        ],
    )
    def test_hyphenated_non_track_arguments_map_to_canonical_spellings(
        self,
        hyphenated: list[str],
        canonical: list[str],
    ) -> None:
        """Hyphens and underscores are interchangeable in current option names."""
        assert _translate_legacy_args(hyphenated) == canonical
