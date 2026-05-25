# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for trackers.cli.__main__ — jsonargparse CLI integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from jsonargparse import ArgumentParser

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
