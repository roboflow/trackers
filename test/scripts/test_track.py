# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the track command."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from trackers.scripts.track import (
    _format_labels,
    _init_annotators,
    _init_tracker,
    add_track_subparser,
)


class TestAddTrackSubparser:
    """Tests for add_track_subparser function."""

    def test_default_values(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        args = parser.parse_args(
            ["track", "--source", "test.mp4", "--model", "rfdetr-nano"]
        )

        assert args.model == "rfdetr-nano"
        assert args.model_confidence == 0.5
        assert args.model_device == "auto"
        assert args.tracker == "bytetrack"
        assert args.show_boxes is True
        assert args.display is False

    def test_model_and_detections_mutually_exclusive(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "track",
                    "--source",
                    "test.mp4",
                    "--model",
                    "rfdetr-large",
                    "--detections",
                    "det.txt",
                ]
            )

    def test_tracker_choice(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        args = parser.parse_args(
            [
                "track",
                "--source",
                "test.mp4",
                "--model",
                "rfdetr-nano",
                "--tracker",
                "sort",
            ]
        )
        assert args.tracker == "sort"

    def test_visualization_options(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        args = parser.parse_args(
            [
                "track",
                "--source",
                "test.mp4",
                "--model",
                "rfdetr-nano",
                "--display",
                "--show-labels",
                "--show-ids",
                "--show-confidence",
                "--show-trajectories",
                "--show-masks",
            ]
        )

        assert args.display is True
        assert args.show_labels is True
        assert args.show_ids is True
        assert args.show_confidence is True
        assert args.show_trajectories is True
        assert args.show_masks is True

    def test_no_boxes_flag(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        args = parser.parse_args(
            ["track", "--source", "test.mp4", "--model", "rfdetr-nano", "--no-boxes"]
        )
        assert args.show_boxes is False

    def test_output_paths(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        args = parser.parse_args(
            [
                "track",
                "--source",
                "test.mp4",
                "--model",
                "rfdetr-nano",
                "-o",
                "out.mp4",
                "--mot-output",
                "results.txt",
                "--overwrite",
            ]
        )

        assert args.output == Path("out.mp4")
        assert args.mot_output == Path("results.txt")
        assert args.overwrite is True

    def test_namespaced_model_args(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        args = parser.parse_args(
            [
                "track",
                "--source",
                "test.mp4",
                "--model",
                "rfdetr-nano",
                "--model.confidence",
                "0.7",
            ]
        )

        assert args.model_confidence == 0.7


class TestInitTracker:
    """Tests for _init_tracker function."""

    @pytest.mark.parametrize(
        "tracker_name,tracker_class_name",
        [
            ("bytetrack", "ByteTrackTracker"),
            ("sort", "SORTTracker"),
        ],
    )
    def test_creates_registered_tracker(
        self, tracker_name: str, tracker_class_name: str
    ) -> None:
        import trackers

        tracker = _init_tracker(tracker_name)
        expected_class = getattr(trackers, tracker_class_name)
        assert isinstance(tracker, expected_class)

    def test_unknown_tracker_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tracker"):
            _init_tracker("unknown_tracker")

    def test_passes_custom_params(self) -> None:
        tracker = _init_tracker(
            "bytetrack",
            lost_track_buffer=60,
            frame_rate=60.0,
        )
        # ByteTrack calculates: maximum_frames_without_update = 60/30 * 60 = 120
        assert tracker.maximum_frames_without_update == 120  # type: ignore[attr-defined]


class TestInitAnnotators:
    """Tests for _init_annotators function."""

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
    """Tests for _format_labels function."""

    @pytest.mark.parametrize(
        "detections_kwargs,class_names,label_flags,expected",
        [
            # Class names from list
            (
                {"xyxy": [[0, 0, 10, 10], [20, 20, 30, 30]], "class_id": [0, 1]},
                ["person", "car"],
                {"show_labels": True},
                ["person", "car"],
            ),
            # Fallback to class ID when out of range
            (
                {"xyxy": [[0, 0, 10, 10]], "class_id": [5]},
                ["person", "car"],
                {"show_labels": True},
                ["5"],
            ),
            # Tracker IDs only
            (
                {"xyxy": [[0, 0, 10, 10]], "tracker_id": [42]},
                [],
                {"show_ids": True},
                ["#42"],
            ),
            # Combined: ID + class + confidence
            (
                {
                    "xyxy": [[0, 0, 10, 10]],
                    "class_id": [0],
                    "confidence": [0.95],
                    "tracker_id": [1],
                },
                ["person"],
                {"show_ids": True, "show_labels": True, "show_confidence": True},
                ["#1 person 0.95"],
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
        # Convert lists to numpy arrays
        for key, val in detections_kwargs.items():
            detections_kwargs[key] = np.array(val)

        detections = sv.Detections(**detections_kwargs)
        labels = _format_labels(detections, class_names, **label_flags)
        assert labels == expected
