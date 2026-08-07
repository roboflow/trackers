# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the ``trackers mcbyte`` benchmark subcommand."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import get_args

import pytest

from trackers.cli.__main__ import _CLIParser, _normalise_option
from trackers.cli.mcbyte import (
    DATASETS,
    DatasetName,
    DatasetPaths,
    _runtime_error,
    _unknown_datasets_error,
    benchmark_command,
    resolve_datasets,
    run_dataset,
)


@pytest.fixture
def benchmark_parser() -> _CLIParser:
    """Parser exposing the benchmark options, matching the script's own wiring."""
    parser = _CLIParser(exit_on_error=False)
    parser.add_function_arguments(benchmark_command)
    return parser


@pytest.fixture
def config_parser() -> _CLIParser:
    """Benchmark parser that also accepts ``--config``, as the real CLI does."""
    parser = _CLIParser(exit_on_error=False)
    parser.add_argument("--config", action="config")
    parser.add_function_arguments(benchmark_command)
    return parser


class TestBenchmarkOptions:
    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            pytest.param([], 6, id="default"),
            pytest.param(["--cmc_downscale", "2"], 2, id="underscore_override"),
            pytest.param(["--cmc-downscale", "2"], 2, id="hyphen_override"),
            pytest.param(["--cmc_downscale=2"], 2, id="inline_value"),
        ],
    )
    def test_cmc_downscale(self, benchmark_parser: _CLIParser, arguments: list[str], expected: int) -> None:
        """The benchmark follows McByte's default and retains explicit overrides."""
        namespace = benchmark_parser.parse_args([_normalise_option(arg) for arg in arguments])

        assert namespace.cmc_downscale == expected

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            pytest.param([], True, id="enabled_by_default"),
            pytest.param(["--no_enable_cmc"], False, id="negated"),
            pytest.param(["--no-enable-cmc"], False, id="negated_hyphenated"),
            pytest.param(["--enable_cmc=false"], False, id="explicit_false"),
        ],
    )
    def test_enable_cmc(self, benchmark_parser: _CLIParser, arguments: list[str], expected: bool) -> None:
        """Camera-motion compensation is on unless a negative spelling turns it off."""
        namespace = benchmark_parser.parse_args([_normalise_option(arg) for arg in arguments])

        assert namespace.enable_cmc is expected

    def test_dataset_accepts_a_list(self, benchmark_parser: _CLIParser) -> None:
        """Datasets are selected as one list rather than a repeated option."""
        namespace = benchmark_parser.parse_args(["--dataset=[mot17,soccernet]"])

        assert namespace.dataset == ["mot17", "soccernet"]

    def test_unknown_dataset_is_rejected(self, benchmark_parser: _CLIParser) -> None:
        """An unregistered dataset name fails while parsing."""
        with pytest.raises(Exception, match="dataset"):
            benchmark_parser.parse_args(["--dataset=[nonexistent]"])

    def test_selectable_names_match_the_dataset_table(self) -> None:
        """No dataset can be configured without also being selectable."""
        assert set(get_args(DatasetName)) == set(DATASETS)


class TestRuntimeError:
    @pytest.mark.parametrize(
        ("device", "cmc_downscale", "expected"),
        [
            pytest.param("cpu", 6, "", id="usable"),
            pytest.param("cpu", 0, "cmc_downscale must be positive.", id="zero_downscale"),
            pytest.param("cpu", -1, "cmc_downscale must be positive.", id="negative_downscale"),
        ],
    )
    def test_reports_the_first_problem(self, device: str, cmc_downscale: int, expected: str) -> None:
        """Runtime arguments are validated before any run directory is created."""
        assert _runtime_error(device, cmc_downscale) == expected

    def test_invalid_arguments_exit_non_zero(self, capsys: pytest.CaptureFixture) -> None:
        """The command reports the problem on stderr instead of raising."""
        code = benchmark_command(device="cpu", cmc_downscale=0)

        assert code == 1
        assert "cmc_downscale must be positive." in capsys.readouterr().err


class TestDatasetRoots:
    def test_inline_json_supplies_both_roots(self, benchmark_parser: _CLIParser) -> None:
        """A dataset's roots are configurable on the command line as one JSON mapping."""
        namespace = benchmark_parser.parse_args(
            ['--datasets={"mot17": {"detection_root": "/data/dets", "image_root": "/data/imgs"}}']
        )
        instantiated = benchmark_parser.instantiate_classes(namespace)

        assert instantiated.datasets == {"mot17": DatasetPaths(Path("/data/dets"), Path("/data/imgs"))}

    def test_config_file_supplies_both_roots(self, config_parser: _CLIParser, tmp_path: Path) -> None:
        """The same mapping reaches the command from a ``--config`` file."""
        config = tmp_path / "run.yaml"
        config.write_text(
            "dataset: [mot17]\ndatasets:\n  mot17:\n    detection_root: /data/dets\n    image_root: /data/imgs\n"
        )

        namespace = config_parser.parse_args([f"--config={config}"])
        instantiated = config_parser.instantiate_classes(namespace)

        assert instantiated.dataset == ["mot17"]
        assert instantiated.datasets == {"mot17": DatasetPaths(Path("/data/dets"), Path("/data/imgs"))}

    def test_both_roots_are_required_together(self, benchmark_parser: _CLIParser) -> None:
        """Half an entry is rejected while parsing rather than failing mid-run."""
        with pytest.raises(Exception, match=r"detection_root|image_root"):
            benchmark_parser.parse_args(['--datasets={"mot17": {"detection_root": "/data/dets"}}'])

    def test_defaults_leave_both_roots_unset(self) -> None:
        """No root has a built-in value, so an unconfigured run cannot half-work."""
        assert all(config.detection_root is None and config.image_root is None for config in DATASETS.values())

    def test_merge_keeps_dataset_specific_behaviour(self) -> None:
        """Only the roots are user-supplied; parsing behaviour stays table-driven."""
        merged = resolve_datasets({"soccernet": DatasetPaths(Path("dets"), Path("imgs"))})

        assert merged["soccernet"].detection_root == Path("dets")
        assert merged["soccernet"].image_root == Path("imgs")
        assert merged["soccernet"].detection_format == "mot"
        assert merged["soccernet"].soccernet_filename is True
        assert merged["soccernet"].confidence_override == 1.0

    def test_merge_leaves_unnamed_datasets_alone(self) -> None:
        """Configuring one dataset does not disturb the others."""
        merged = resolve_datasets({"mot17": DatasetPaths(Path("dets"), Path("imgs"))})

        assert merged["dancetrack"].detection_root is None

    @pytest.mark.parametrize(
        ("datasets", "expected"),
        [
            pytest.param(None, "", id="unsupplied"),
            pytest.param({}, "", id="empty"),
            pytest.param({"mot17": None}, "", id="known"),
            pytest.param({"mot18": None}, "Unknown --datasets entry 'mot18'.", id="typo"),
        ],
    )
    def test_unknown_entries_are_reported(self, datasets: dict | None, expected: str) -> None:
        """A key naming no dataset is caught rather than silently ignored."""
        assert _unknown_datasets_error(datasets).startswith(expected)

    def test_unknown_entry_exits_non_zero(self, capsys: pytest.CaptureFixture) -> None:
        """A mistyped dataset name stops the run before a run directory is created."""
        code = benchmark_command(device="cpu", datasets={"mot18": DatasetPaths(Path("a"), Path("b"))})

        assert code == 1
        assert "Unknown --datasets entry 'mot18'." in capsys.readouterr().err


class TestUnconfiguredDataset:
    def test_run_dataset_asks_for_the_missing_roots(self) -> None:
        """A dataset neither the source nor the run supplied roots for says so."""
        with pytest.raises(ValueError, match=r"Please configure DATASETS\['mot17'\]"):
            run_dataset(
                config=DATASETS["mot17"],
                output_dir=Path("unused"),
                device="cpu",
                enable_isolated_mask_matching=False,
                enable_cmc=True,
                cmc_method="sparseOptFlow",
                cmc_downscale=6,
                skip_existing=False,
                keep_partial_results=False,
                logger=logging.getLogger("test_mcbyte"),
            )

    def test_supplied_roots_reach_the_filesystem(self, tmp_path: Path) -> None:
        """Roots from the run are what the dataset is actually looked for under."""
        detection_root = tmp_path / "dets"
        image_root = tmp_path / "frames"
        detection_root.mkdir()
        image_root.mkdir()
        resolved = resolve_datasets({"mot17": DatasetPaths(detection_root, image_root)})

        with pytest.raises(FileNotFoundError, match=str(detection_root)):
            run_dataset(
                config=resolved["mot17"],
                output_dir=tmp_path / "out",
                device="cpu",
                enable_isolated_mask_matching=False,
                enable_cmc=True,
                cmc_method="sparseOptFlow",
                cmc_downscale=6,
                skip_existing=False,
                keep_partial_results=False,
                logger=logging.getLogger("test_mcbyte"),
            )

    def test_the_error_names_the_command_line_route(self) -> None:
        """The message stays actionable now that editing the source is not the only fix."""
        with pytest.raises(ValueError, match="--datasets"):
            run_dataset(
                config=DATASETS["mot17"],
                output_dir=Path("unused"),
                device="cpu",
                enable_isolated_mask_matching=False,
                enable_cmc=True,
                cmc_method="sparseOptFlow",
                cmc_downscale=6,
                skip_existing=False,
                keep_partial_results=False,
                logger=logging.getLogger("test_mcbyte"),
            )
