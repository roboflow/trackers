# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for trackers.cli.__main__ — jsonargparse CLI integration."""

from __future__ import annotations

import re
import subprocess
import sys
import warnings
from argparse import ArgumentError
from pathlib import Path

import pytest
import yaml
from jsonargparse import ActionConfigFile, ArgumentParser

from trackers.cli.__main__ import (
    _CLIParser,
    _translate_legacy_args,
)
from trackers.cli._legacy import _LEGACY_ARGUMENTS
from trackers.cli._parser import _SUBCOMMANDS
from trackers.cli.download import download_command
from trackers.cli.eval import eval_command
from trackers.cli.track import DEFAULT_TRACKER, track_command
from trackers.cli.tune import tune_command
from trackers.core.base import BaseTracker


@pytest.fixture()
def track_parser() -> ArgumentParser:
    """ArgumentParser built from the track_command() signature with --config support."""
    parser = ArgumentParser(exit_on_error=False)
    parser.add_function_arguments(track_command)
    parser.add_argument("--config", action="config")
    return parser


class TestConfigFileSupport:
    """Verify jsonargparse --config flag behaviour for the track subcommand."""

    def test_config_value_applied_to_tracker(self, track_parser: ArgumentParser, tmp_path: Path) -> None:
        """YAML --config value is parsed into the track_command() namespace."""
        cfg = tmp_path / "run.yaml"
        cfg.write_text(yaml.dump({"tracker": {"name": "sort"}}))

        ns = track_parser.parse_args(["--config", str(cfg)])

        assert ns.tracker.name == "sort"

    def test_cli_arg_overrides_config_value(self, track_parser: ArgumentParser, tmp_path: Path) -> None:
        """Explicit CLI arg takes precedence over the --config file value."""
        cfg = tmp_path / "run.yaml"
        cfg.write_text(yaml.dump({"tracker": {"name": "sort"}}))

        ns = track_parser.parse_args(["--config", str(cfg), "--tracker.name", "bytetrack"])

        assert ns.tracker.name == "bytetrack"

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
        parser.add_function_arguments(track_command)

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
        parser.add_function_arguments(track_command)

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
        ("hyphenated", "canonical"),
        [
            pytest.param("--show.no-ids", "--show.no_ids", id="grouped_negation"),
            pytest.param("--output.no-overwrite", "--output.no_overwrite", id="other_group_negation"),
            pytest.param("--no-display", "--no_display", id="ungrouped_negation"),
            pytest.param("--tracker.min-iou-threshold=0.3", "--tracker.min_iou_threshold=0.3", id="inline_value"),
            pytest.param("--tracker.lost-track-buffer", "--tracker.lost_track_buffer", id="multiple_hyphens"),
        ],
    )
    def test_every_hyphenated_option_shape_normalizes(self, hyphenated: str, canonical: str) -> None:
        """Only the name is rewritten, however many hyphens it carries."""
        assert _translate_legacy_args(["track", hyphenated]) == ["track", canonical]

    @pytest.mark.parametrize(
        "arguments",
        [
            pytest.param(["--detection.model", "rfdetr-base"], id="separate_value"),
            pytest.param(["--detection.model=rfdetr-base"], id="inline_value"),
            pytest.param(["--source", "my-dir/some-file.mp4"], id="path_value"),
            pytest.param(["--filters.classes", "[traffic-light,car]"], id="list_value"),
            pytest.param(["--tracker.name", "sort", "--", "extra-arg"], id="after_separator"),
        ],
    )
    def test_hyphenated_values_survive_normalization(self, arguments: list[str]) -> None:
        """A hyphen inside a value is data, not a name separator."""
        assert _translate_legacy_args(["track", *arguments]) == ["track", *arguments]

    @pytest.mark.parametrize(
        ("hyphenated", "underscored", "replacement"),
        [
            pytest.param("--no-boxes", "--no_boxes", "--show.no_boxes", id="develop_negation"),
            pytest.param("--show-boxes", "--show_boxes", "--show.boxes", id="develop_flag"),
            pytest.param("--mot-output", "--mot_output", "--output.mot_results", id="develop_output"),
            pytest.param("--track-ids", "--track_ids", "--filters.track_ids", id="develop_filter"),
        ],
    )
    def test_deprecated_spellings_resolve_under_either_separator(
        self,
        hyphenated: str,
        underscored: str,
        replacement: str,
    ) -> None:
        """A deprecated option resolves whichever separator the user reached for."""
        for spelling in (hyphenated, underscored):
            with pytest.warns(FutureWarning, match=re.escape(replacement)):
                assert _translate_legacy_args(["track", spelling])[1].partition("=")[0] == replacement

    @pytest.mark.parametrize(
        "option",
        ["--no-show.boxes", "--no-show.ids"],
    )
    def test_group_prefixed_negations_are_not_available(self, option: str) -> None:
        """Negating the group was this branch's spelling; the field carries it now."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track_command)

        with pytest.raises(ArgumentError, match=option):
            parser.parse_args([option])

    @pytest.mark.parametrize(
        ("option", "replacement"),
        [
            pytest.param("--no-boxes", "--show.no_boxes", id="boxes"),
            pytest.param("--no-ids", "--show.no_ids", id="ids"),
        ],
    )
    def test_develop_negative_flags_map_to_the_field_negation(self, option: str, replacement: str) -> None:
        """Develop's negative flags land on the negative half of the option pair."""
        with pytest.warns(FutureWarning, match=re.escape(option)):
            args = _translate_legacy_args(["track", option])

        assert args == ["track", replacement]
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track_command)
        parsed = parser.parse_args(args[1:])

        assert parsed.show.boxes is (option != "--no-boxes")
        assert parsed.show.ids is (option != "--no-ids")

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            (["track", "--detections", "detections.txt"], ["track", "--detection.mot_file", "detections.txt"]),
            (["track", "--overwrite"], ["track", "--output.overwrite"]),
        ],
    )
    def test_evidenced_track_aliases_map_to_intended_cli(
        self,
        source: list[str],
        target: list[str],
    ) -> None:
        """Develop aliases map to the intended semantic CLI."""
        with pytest.warns(FutureWarning):
            assert _translate_legacy_args(source) == target

    @pytest.mark.parametrize(
        "args",
        [
            ["track", "--detection.model", "rfdetr-base", "--detection.mot_file", "detections.txt"],
            ["track", "--model", "rfdetr-base", "--detections", "detections.txt"],
        ],
    )
    def test_explicit_model_and_mot_file_are_mutually_exclusive(self, args: list[str]) -> None:
        """Preserve develop's exclusive detector-source contract for CLI inputs."""
        with pytest.raises(ValueError, match=r"--detection\.model.*--detection\.mot_file"):
            _translate_legacy_args(args)

    def test_track_parser_accepts_dotted_dataclass_arguments(self) -> None:
        """Dotted CLI options instantiate nested options before calling track_command()."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track_command)

        parsed = parser.instantiate_classes(
            parser.parse_args(
                [
                    "--detection.model",
                    "rfdetr-base",
                    "--detection.confidence",
                    "0.3",
                    "--filters.classes",
                    "[person,car]",
                    "--filters.track_ids",
                    "[1,3,5]",
                    "--output.video",
                    "tracked.mp4",
                    "--show.boxes",
                    "false",
                ]
            )
        )

        assert parsed.detection.model == "rfdetr-base"
        assert parsed.detection.confidence == pytest.approx(0.3)
        assert parsed.filters.classes == ["person", "car"]
        assert parsed.filters.track_ids == [1, 3, 5]
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
            "--tracker.lost_track_buffer",
            "40",
            "--tracker.enable_cmc=false",
            "--output.mot_results",
            "tracks.txt",
            "--show.boxes",
            "false",
        ]
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track_command)
        parsed = parser.instantiate_classes(parser.parse_args(args[1:]))

        assert parsed.detection.model == "rfdetr-base"
        assert parsed.detection.confidence == pytest.approx(0.3)
        assert parsed.tracker.lost_track_buffer == 40
        assert parsed.tracker.enable_cmc is False
        assert parsed.output.mot_results == Path("tracks.txt")
        assert parsed.show.boxes is False

    def test_legacy_and_new_argument_for_same_target_fails(self) -> None:
        """Mixed spellings cannot silently choose a value for one target."""
        with pytest.raises(ValueError, match=r"--model.*--detection\.model"):
            _translate_legacy_args(["track", "--model", "rfdetr-base", "--detection.model", "rfdetr-nano"])

    def test_download_positional_dataset_maps_to_named_argument(self) -> None:
        """Legacy download DATASET syntax remains available during transition."""
        with pytest.warns(FutureWarning, match=r"positional dataset.*--name"):
            args = _translate_legacy_args(["download", "mot17", "--cache-dir", "cache"])

        assert args == ["download", "--name", "mot17", "--cache_dir", "cache"]

    def test_download_dataset_option_maps_to_name(self) -> None:
        """The superseded --dataset spelling still selects the dataset."""
        with pytest.warns(FutureWarning, match=r"--dataset is deprecated; use --name"):
            args = _translate_legacy_args(["download", "--dataset", "mot17"])

        assert args == ["download", "--name", "mot17"]

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
        parser.add_function_arguments(eval_command)
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
        """Every legacy CLI transition names the release in which it is removed.

        The release is asserted as a literal rather than recomputed from the
        installed version. Deriving it here would mirror the implementation and
        keep passing even if the deadline started moving with each release.
        """
        with pytest.warns(FutureWarning, match=re.escape("removed in 2.10.0")):
            _translate_legacy_args(args)

    @pytest.mark.parametrize(
        ("hyphenated", "canonical"),
        [
            (
                ["eval", "--gt-dir", "gt", "--predictions-dir=results"],
                ["eval", "--gt_dir", "gt", "--predictions_dir=results"],
            ),
            (
                ["tune", "--detections-dir", "detections", "--n-trials=5"],
                ["tune", "--detections_dir", "detections", "--n_trials=5"],
            ),
            (["tune", "--no-enqueue-defaults"], ["tune", "--no_enqueue_defaults"]),
        ],
    )
    def test_hyphenated_non_track_arguments_map_to_canonical_spellings(
        self,
        hyphenated: list[str],
        canonical: list[str],
    ) -> None:
        """Hyphens and underscores are interchangeable in current option names."""
        assert _translate_legacy_args(hyphenated) == canonical

    @pytest.mark.parametrize(
        ("deprecated", "replacement"),
        [
            pytest.param("--tracker", "--predictions", id="single_sequence"),
            pytest.param("--tracker-dir", "--predictions_dir", id="benchmark"),
            pytest.param("--tracker_dir", "--predictions_dir", id="benchmark_underscored"),
        ],
    )
    def test_eval_prediction_inputs_are_renamed_off_tracker(
        self,
        deprecated: str,
        replacement: str,
    ) -> None:
        """Eval's prediction inputs no longer share a name with the algorithm option."""
        with pytest.warns(FutureWarning, match=re.escape(replacement)):
            args = _translate_legacy_args(["eval", deprecated, "results"])

        assert args == ["eval", replacement, "results"]
        parser = ArgumentParser(exit_on_error=False)
        parser.add_function_arguments(eval_command)

        assert parser.parse_args(args[1:])[replacement.removeprefix("--")] == Path("results")

    def test_a_subcommand_without_legacy_spellings_still_normalises(self) -> None:
        """A new subcommand reaches the hyphen sweep and its empty legacy table.

        Both halves fail silently or loudly if a registration site is missed:
        an absent ``_SUBCOMMANDS`` entry returns argv unnormalised, so half the
        spellings of every option stop parsing, and an absent
        ``_LEGACY_ARGUMENTS`` entry raises ``KeyError`` from an unguarded
        subscript before any option is examined.
        """
        assert _translate_legacy_args(["mcbyte", "--cmc-downscale", "2"]) == ["mcbyte", "--cmc_downscale", "2"]

    def test_eval_rejects_mixing_the_deprecated_and_current_spelling(self) -> None:
        """Supplying both spellings of one input is an error, not a silent winner."""
        with pytest.raises(ValueError, match="--predictions_dir"):
            _translate_legacy_args(["eval", "--tracker_dir", "old", "--predictions_dir", "new"])


class TestListValuedTrackFilters:
    """Track filters take lists, matching the eval and tune list-valued options."""

    @pytest.mark.parametrize(
        ("legacy", "canonical"),
        [
            pytest.param(
                ["track", "--filters.classes", "person,car"],
                ["track", "--filters.classes", '["person", "car"]'],
                id="separate_value",
            ),
            pytest.param(
                ["track", "--filters.classes=person,car"],
                ["track", "--filters.classes", '["person", "car"]'],
                id="inline_value",
            ),
            pytest.param(
                ["track", "--filters.track_ids", "1,3,5"],
                ["track", "--filters.track_ids", '["1", "3", "5"]'],
                id="track_ids",
            ),
            pytest.param(
                ["track", "--filters.classes", "person"],
                ["track", "--filters.classes", '["person"]'],
                id="single_value",
            ),
            pytest.param(
                ["track", "--filters.classes", " person , car "],
                ["track", "--filters.classes", '["person", "car"]'],
                id="whitespace_stripped",
            ),
        ],
    )
    def test_comma_separated_filter_values_remain_usable(
        self,
        legacy: list[str],
        canonical: list[str],
    ) -> None:
        """The comma-separated filter spelling transitions to a JSON list."""
        with pytest.warns(FutureWarning, match=r"comma-separated --filters\.\w+ values are deprecated"):
            assert _translate_legacy_args(legacy) == canonical

    def test_develop_classes_alias_still_accepts_a_comma_separated_value(self) -> None:
        """The develop --classes spelling and its comma value both transition."""
        with pytest.warns(FutureWarning):
            args = _translate_legacy_args(["track", "--classes", "person,car"])

        assert args == ["track", "--filters.classes", '["person", "car"]']

    @pytest.mark.parametrize(
        "args",
        [
            pytest.param(["track", "--filters.classes", "[person,car]"], id="bracket_shorthand"),
            pytest.param(["track", "--filters.classes=[person,car]"], id="inline_bracket_shorthand"),
            pytest.param(["track", "--filters.track_ids", "[1,3,5]"], id="numeric_bracket_shorthand"),
            pytest.param(["track", "--", "--filters.classes", "person,car"], id="passthrough_untouched"),
            pytest.param(["track", "--filters.classes", "--display"], id="missing_value_left_to_parser"),
        ],
    )
    def test_current_list_values_are_not_rewritten(self, args: list[str]) -> None:
        """Bracket shorthand and passthrough arguments stay untouched."""
        assert _translate_legacy_args(args) == args

    def test_comma_separated_value_reaches_the_parser_as_a_list(self) -> None:
        """A translated legacy value parses into the FilterOptions list field."""
        with pytest.warns(FutureWarning):
            args = _translate_legacy_args(["track", "--filters.classes", "person,car"])
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track_command)

        parsed = parser.instantiate_classes(parser.parse_args(args[1:]))

        assert parsed.filters.classes == ["person", "car"]

    def test_bracket_shorthand_keeps_class_names_and_ids_distinguishable(self) -> None:
        """Mixed class names and IDs survive parsing, as the resolver expects."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track_command)

        parsed = parser.instantiate_classes(parser.parse_args(["--filters.classes=[person,2,car]"]))

        assert parsed.filters.classes == ["person", 2, "car"]


class TestTrackerParameterAbbreviations:
    """Tracker parameters abbreviate only their standard leading token."""

    @pytest.mark.parametrize(
        ("unabbreviated", "abbreviated"),
        [
            pytest.param(
                "--tracker.minimum_consecutive_frames",
                "--tracker.min_consecutive_frames",
                id="consecutive_frames",
            ),
            pytest.param(
                "--tracker.minimum_iou_threshold",
                "--tracker.min_iou_threshold",
                id="iou_threshold",
            ),
            pytest.param(
                "--tracker.minimum_iou_threshold_first_assoc",
                "--tracker.min_iou_threshold_first_assoc",
                id="first_assoc",
            ),
            pytest.param(
                "--tracker.minimum_iou_threshold_second_assoc",
                "--tracker.min_iou_threshold_second_assoc",
                id="second_assoc",
            ),
            pytest.param(
                "--tracker.minimum_iou_threshold_unconfirmed_assoc",
                "--tracker.min_iou_threshold_unconfirmed_assoc",
                id="unconfirmed_assoc",
            ),
        ],
    )
    def test_unabbreviated_parameters_transition_to_short_names(
        self,
        unabbreviated: str,
        abbreviated: str,
    ) -> None:
        """Every renamed tracker parameter keeps a warning-emitting alias."""
        with pytest.warns(FutureWarning, match=r"is deprecated; use --tracker\.min_"):
            assert _translate_legacy_args(["track", unabbreviated, "0.3"]) == ["track", abbreviated, "0.3"]

    def test_current_short_names_do_not_warn(self) -> None:
        """The canonical short spelling is not treated as a legacy argument."""
        args = ["track", "--tracker.min_iou_threshold", "0.3"]

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert _translate_legacy_args(args) == args

    def test_develop_tracker_prefix_maps_to_the_short_name(self) -> None:
        """The develop --tracker.<name> path lands on the abbreviated CLI name."""
        with pytest.warns(FutureWarning, match=r"--tracker\.minimum_consecutive_frames"):
            args = _translate_legacy_args(["track", "--tracker.minimum_consecutive_frames", "5"])

        assert args == ["track", "--tracker.min_consecutive_frames", "5"]

    def test_develop_iou_parameter_maps_to_the_variant_field(self) -> None:
        """``--tracker.iou`` is renamed rather than left to argparse prefix matching."""
        with pytest.warns(FutureWarning, match=r"--tracker\.iou.*--tracker\.iou_variant"):
            args = _translate_legacy_args(["track", "--tracker.iou", "giou"])

        assert args == ["track", "--tracker.iou_variant", "giou"]

    def test_develop_bare_boolean_flag_becomes_an_explicit_false(self) -> None:
        """A develop ``store_false`` flag keeps meaning "turn this off"."""
        with pytest.warns(FutureWarning, match=r"--tracker\.enable_cmc"):
            args = _translate_legacy_args(["track", "--tracker.enable_cmc"])

        assert args == ["track", "--tracker.enable_cmc=false"]

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            pytest.param(["--tracker.enable_cmc=true"], True, id="inline_true"),
            pytest.param(["--tracker.enable_cmc=false"], False, id="inline_false"),
            pytest.param(["--tracker.enable_cmc", "false"], False, id="separate_false"),
        ],
    )
    def test_explicit_boolean_values_pass_through_and_parse(
        self,
        arguments: list[str],
        expected: bool,
    ) -> None:
        """The current explicit-value syntax must not collect a second value."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track_command)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            args = _translate_legacy_args(["track", *arguments])

        assert args == ["track", *arguments]
        assert parser.parse_args(args[1:]).tracker.enable_cmc is expected

    def test_unchanged_parameter_names_are_not_deprecated(self) -> None:
        """A develop parameter whose spelling never changed stays warning-free."""
        args = ["track", "--tracker.lost_track_buffer", "40"]

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert _translate_legacy_args(args) == args

    def test_unabbreviated_and_short_names_cannot_be_combined(self) -> None:
        """Mixed spellings cannot silently choose a value for one parameter."""
        with pytest.raises(ValueError, match=r"minimum_iou_threshold.*min_iou_threshold"):
            _translate_legacy_args(
                [
                    "track",
                    "--tracker.minimum_iou_threshold",
                    "0.3",
                    "--tracker.min_iou_threshold",
                    "0.4",
                ]
            )

    def test_short_name_parses_into_tracker_options(self) -> None:
        """The abbreviated CLI path populates the TrackerOptions field."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track_command)

        parsed = parser.instantiate_classes(parser.parse_args(["--tracker.min_iou_threshold", "0.42"]))

        assert parsed.tracker.min_iou_threshold == pytest.approx(0.42)


class TestBooleanOptionSyntax:
    """Every boolean option offers a bare pair and an explicit value."""

    @pytest.fixture
    def parser(self) -> _CLIParser:
        """Track parser with all option groups registered."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track_command)
        return parser

    def test_negation_nests_inside_the_group(self, parser: _CLIParser) -> None:
        """The prefix lands on the field, so the group name is never negated."""
        options = {option for action in parser._actions for option in action.option_strings}

        assert "--show.no_ids" in options
        assert "--no_show.ids" not in options

    @pytest.mark.parametrize(
        ("command", "arguments", "attribute"),
        [
            pytest.param(
                tune_command,
                ["--no_enqueue_defaults", "--tracker", "sort", "--gt_dir", "gt", "--detections_dir", "det"],
                "enqueue_defaults",
                id="tune",
            ),
            pytest.param(download_command, ["--no_list_available"], "list_available", id="download"),
        ],
    )
    def test_ungrouped_negation_reaches_false(
        self,
        command: object,
        arguments: list[str],
        attribute: str,
    ) -> None:
        """An option with no group has no dot to split on, which must not skip the prefix test."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(command)

        assert getattr(parser.parse_args(arguments), attribute) is False

    def test_display_gained_a_negative_half(self, parser: _CLIParser) -> None:
        """The flat preview flag is a boolean like any other, pair included."""
        assert parser.parse_args(["--display"]).display is True
        assert parser.parse_args(["--no_display"]).display is False

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            pytest.param([], True, id="default"),
            pytest.param(["--show.ids"], True, id="bare_positive"),
            pytest.param(["--show.no_ids"], False, id="bare_negative"),
            pytest.param(["--show.ids=false"], False, id="inline_false"),
            pytest.param(["--show.ids", "false"], False, id="separate_false"),
            pytest.param(["--show.no_ids=false"], True, id="double_negative"),
            pytest.param(["--show.ids", "--show.no_ids"], False, id="last_wins_negative"),
            pytest.param(["--show.no_ids", "--show.ids"], True, id="last_wins_positive"),
        ],
    )
    def test_every_boolean_spelling_resolves(
        self,
        parser: _CLIParser,
        arguments: list[str],
        expected: bool,
    ) -> None:
        """Bare pair, explicit value and repeats all land on one field."""
        assert parser.parse_args(arguments).show.ids is expected

    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            pytest.param("--show.masks", True, id="default_false_positive"),
            pytest.param("--show.no_masks", False, id="default_false_negative"),
            pytest.param("--output.no_overwrite", False, id="other_group"),
        ],
    )
    def test_pair_is_offered_regardless_of_default(
        self,
        parser: _CLIParser,
        option: str,
        expected: bool,
    ) -> None:
        """The default no longer decides which syntax a boolean field gets."""
        parsed = parser.instantiate_classes(parser.parse_args([option]))
        group, _, field_name = option.removeprefix("--").partition(".")

        assert getattr(getattr(parsed, group), field_name.removeprefix("no_")) is expected

    def test_config_value_is_overridden_from_the_command_line(self, tmp_path: Path) -> None:
        """A negation on the command line must beat a ``true`` in the config file."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_argument("--config", action=ActionConfigFile)
        parser.add_function_arguments(track_command)
        cfg = tmp_path / "run.yaml"
        cfg.write_text(yaml.dump({"show": {"ids": True}}))

        parsed = parser.parse_args(["--config", str(cfg), "--show.no_ids"])

        assert parsed.show.ids is False


class TestTrackerChoices:
    """The tracker registry is the accept list for --tracker.name."""

    @pytest.mark.parametrize("tracker_id", BaseTracker._registered_trackers())
    def test_every_registered_tracker_is_accepted(self, tracker_id: str) -> None:
        """Choices come from the registry, so no registered tracker is rejected."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track_command)

        parsed = parser.parse_args(["--tracker.name", tracker_id])

        assert parsed.tracker.name == tracker_id

    def test_unknown_tracker_is_rejected_while_parsing(self) -> None:
        """A mistyped tracker fails before track_command can load a detection model."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track_command)

        with pytest.raises(ArgumentError, match=r"invalid choice: 'nosuchtracker'"):
            parser.parse_args(["--tracker.name", "nosuchtracker"])

    def test_default_tracker_is_sourced_from_the_track_module(self) -> None:
        """The CLI default mirrors track.DEFAULT_TRACKER rather than a duplicated literal."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track_command)

        parsed = parser.parse_args([])

        assert parsed.tracker.name == DEFAULT_TRACKER


class TestTrackerShorthand:
    """``--tracker <id>`` stays the short spelling of ``--tracker.name <id>``."""

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            pytest.param(["--tracker", "sort"], ["--tracker.name", "sort"], id="separate_value"),
            pytest.param(["--tracker=ocsort"], ["--tracker.name=ocsort"], id="inline_value"),
            pytest.param(
                ["--tracker.min_iou_threshold", "0.4"],
                ["--tracker.min_iou_threshold", "0.4"],
                id="parameter_path_untouched",
            ),
            pytest.param(
                ['--tracker={"name": "sort"}'],
                ['--tracker={"name": "sort"}'],
                id="json_group_untouched",
            ),
        ],
    )
    def test_shorthand_expands_to_the_name_field(self, arguments: list[str], expected: list[str]) -> None:
        """The shorthand targets the name field without disturbing other spellings."""
        assert _translate_legacy_args(["track", *arguments]) == ["track", *expected]

    def test_shorthand_does_not_warn(self) -> None:
        """``--tracker`` is a supported spelling, not a deprecated one."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert _translate_legacy_args(["track", "--tracker", "sort"]) == ["track", "--tracker.name", "sort"]

    def test_shorthand_parses_into_tracker_options(self) -> None:
        """The expanded shorthand populates the TrackerOptions selector."""
        parser = _CLIParser(exit_on_error=False)
        parser.add_function_arguments(track_command)
        args = _translate_legacy_args(["track", "--tracker", "ocsort"])

        parsed = parser.instantiate_classes(parser.parse_args(args[1:]))

        assert parsed.tracker.name == "ocsort"


class TestEntryPointCentralisation:
    """Every CLI under ``src/`` is reachable through the one ``trackers`` command."""

    def test_every_subcommand_is_dispatchable_and_translatable(self) -> None:
        """The dispatch table and the two per-subcommand tables cannot drift apart.

        ``_SUBCOMMANDS`` gates argv normalisation and ``_LEGACY_ARGUMENTS`` is
        subscripted unguarded, so a command present in one table and missing
        from another half-works rather than failing cleanly.
        """
        dispatched = {"track", "eval", "tune", "download", "mcbyte"}

        assert set(_SUBCOMMANDS) == dispatched
        assert set(_LEGACY_ARGUMENTS) == dispatched

    def test_the_entry_point_imports_without_torch(self) -> None:
        """Importing the CLI must not drag in the optional ``mask`` extra.

        ``torch`` ships only in the ``mask`` extra, so a default install would
        stop having a CLI at all if a subcommand imported it at module level.
        """
        # A subprocess, because this test session has already imported torch.
        script = "import sys; import trackers.cli.__main__; print('torch' in sys.modules)"

        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)  # noqa: S603

        assert result.stdout.strip() == "False"
