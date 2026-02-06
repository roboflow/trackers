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
    _create_annotators,
    _create_tracker,
    _generate_labels,
    _get_frame_detections,
    _validate_output_paths,
    _write_mot_frame,
    add_track_subparser,
    create_frame_generator,
)


class TestAddTrackSubparser:
    """Tests for add_track_subparser function."""

    def test_adds_track_subcommand(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        # Should parse track command
        args = parser.parse_args(["track", "--source", "video.mp4"])
        assert args.source == "video.mp4"

    def test_default_values(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        args = parser.parse_args(["track", "--source", "test.mp4"])

        assert args.model == "rfdetr-nano"
        assert args.model_confidence == 0.5
        assert args.device == "auto"
        assert args.tracker == "bytetrack"
        assert args.show_boxes is True
        assert args.display is False

    def test_model_and_detections_mutually_exclusive(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        with pytest.raises(SystemExit):
            parser.parse_args([
                "track",
                "--source", "test.mp4",
                "--model", "rfdetr-large",
                "--detections", "det.txt",
            ])

    def test_tracker_choice(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        args = parser.parse_args(["track", "--source", "test.mp4", "--tracker", "sort"])
        assert args.tracker == "sort"

    def test_visualization_options(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        args = parser.parse_args([
            "track",
            "--source", "test.mp4",
            "--display",
            "--show-labels",
            "--show-ids",
            "--show-confidence",
            "--show-trajectories",
            "--show-masks",
        ])

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

        args = parser.parse_args(["track", "--source", "test.mp4", "--no-boxes"])
        assert args.show_boxes is False

    def test_output_paths(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        args = parser.parse_args([
            "track",
            "--source", "test.mp4",
            "-o", "out.mp4",
            "--mot-output", "results.txt",
            "--overwrite",
        ])

        assert args.output == Path("out.mp4")
        assert args.mot_output == Path("results.txt")
        assert args.overwrite is True

    def test_namespaced_model_args(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)

        args = parser.parse_args([
            "track",
            "--source", "test.mp4",
            "--model.confidence", "0.7",
        ])

        assert args.model_confidence == 0.7


class TestValidateOutputPaths:
    """Tests for _validate_output_paths function."""

    def test_raises_when_output_exists(self, tmp_path: Path) -> None:
        existing_file = tmp_path / "existing.mp4"
        existing_file.touch()

        args = argparse.Namespace(
            output=existing_file,
            mot_output=None,
            overwrite=False,
        )

        with pytest.raises(FileExistsError, match="already exists"):
            _validate_output_paths(args)

    def test_allows_overwrite(self, tmp_path: Path) -> None:
        existing_file = tmp_path / "existing.mp4"
        existing_file.touch()

        args = argparse.Namespace(
            output=existing_file,
            mot_output=None,
            overwrite=True,
        )

        # Should not raise
        _validate_output_paths(args)

    def test_allows_nonexistent_path(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            output=tmp_path / "new.mp4",
            mot_output=None,
            overwrite=False,
        )

        # Should not raise
        _validate_output_paths(args)


class TestCreateTracker:
    """Tests for _create_tracker function."""

    def test_creates_bytetrack(self) -> None:
        from trackers import ByteTrackTracker

        args = argparse.Namespace(tracker="bytetrack")
        # Add tracker params that might be present
        for param in ["lost_track_buffer", "frame_rate", "track_activation_threshold",
                      "minimum_consecutive_frames", "minimum_iou_threshold"]:
            setattr(args, f"tracker_{param}", None)

        tracker = _create_tracker(args)
        assert isinstance(tracker, ByteTrackTracker)

    def test_creates_sort(self) -> None:
        from trackers import SORTTracker

        args = argparse.Namespace(tracker="sort")
        for param in ["lost_track_buffer", "frame_rate", "track_activation_threshold",
                      "minimum_consecutive_frames", "minimum_iou_threshold"]:
            setattr(args, f"tracker_{param}", None)

        tracker = _create_tracker(args)
        assert isinstance(tracker, SORTTracker)

    def test_unknown_tracker_raises(self) -> None:
        args = argparse.Namespace(tracker="unknown_tracker")

        with pytest.raises(ValueError, match="Unknown tracker"):
            _create_tracker(args)

    def test_passes_custom_params(self) -> None:
        args = argparse.Namespace(
            tracker="bytetrack",
            tracker_lost_track_buffer=60,
            tracker_frame_rate=60.0,
        )
        for param in ["track_activation_threshold", "minimum_consecutive_frames",
                      "minimum_iou_threshold"]:
            setattr(args, f"tracker_{param}", None)

        tracker = _create_tracker(args)
        # ByteTrack calculates: maximum_frames_without_update = 60/30 * 60 = 120
        assert tracker.maximum_frames_without_update == 120  # type: ignore[attr-defined]


class TestGetFrameDetections:
    """Tests for _get_frame_detections function."""

    def test_converts_mot_to_detections(self) -> None:
        from trackers.eval.io import MOTFrameData

        # Create mock MOT data with xywh format
        frame_data = MOTFrameData(
            ids=np.array([1, 2]),
            boxes=np.array([[100, 100, 50, 80], [200, 150, 60, 90]]),  # xywh
            confidences=np.array([0.9, 0.8]),
            classes=np.array([0, 1]),
        )
        detections_data = {1: frame_data}

        result = _get_frame_detections(detections_data, 1)

        assert len(result) == 2
        # Check xyxy conversion
        np.testing.assert_array_almost_equal(
            result.xyxy,
            np.array([[100, 100, 150, 180], [200, 150, 260, 240]])  # xyxy
        )

    def test_returns_empty_for_missing_frame(self) -> None:
        result = _get_frame_detections({}, 99)
        assert len(result) == 0


class TestWriteMotFrame:
    """Tests for _write_mot_frame function."""

    def test_writes_mot_format(self, tmp_path: Path) -> None:
        output_file = tmp_path / "output.txt"

        detections = sv.Detections(
            xyxy=np.array([[100.0, 100.0, 150.0, 180.0]]),
            confidence=np.array([0.95]),
            tracker_id=np.array([1]),
        )

        with open(output_file, "w") as f:
            _write_mot_frame(f, 1, detections)

        content = output_file.read_text()
        # Format: frame,id,x,y,w,h,conf,-1,-1,-1
        assert content.startswith("1,1,100.00,100.00,50.00,80.00,0.9500")

    def test_handles_empty_detections(self, tmp_path: Path) -> None:
        output_file = tmp_path / "output.txt"

        with open(output_file, "w") as f:
            _write_mot_frame(f, 1, sv.Detections.empty())

        content = output_file.read_text()
        assert content == ""


class TestCreateAnnotators:
    """Tests for _create_annotators function."""

    def test_creates_box_annotator_by_default(self) -> None:
        args = argparse.Namespace(
            show_boxes=True,
            show_masks=False,
            show_labels=False,
            show_ids=False,
            show_confidence=False,
        )

        annotators, label_annotator = _create_annotators(args)
        assert len(annotators) == 1
        assert isinstance(annotators[0], sv.BoxAnnotator)
        assert label_annotator is None

    def test_creates_mask_annotator(self) -> None:
        args = argparse.Namespace(
            show_boxes=False,
            show_masks=True,
            show_labels=False,
            show_ids=False,
            show_confidence=False,
        )

        annotators, label_annotator = _create_annotators(args)
        assert any(isinstance(a, sv.MaskAnnotator) for a in annotators)
        assert label_annotator is None

    def test_creates_label_annotator_for_ids(self) -> None:
        args = argparse.Namespace(
            show_boxes=False,
            show_masks=False,
            show_labels=False,
            show_ids=True,
            show_confidence=False,
        )

        annotators, label_annotator = _create_annotators(args)
        assert label_annotator is not None
        assert isinstance(label_annotator, sv.LabelAnnotator)


class TestGenerateLabels:
    """Tests for _generate_labels function."""

    def test_generates_class_names(self) -> None:
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            class_id=np.array([0, 1]),
        )
        class_names = ["person", "car"]
        args = argparse.Namespace(
            show_labels=True,
            show_ids=False,
            show_confidence=False,
        )

        labels = _generate_labels(detections, class_names, args)

        assert labels == ["person", "car"]

    def test_falls_back_to_class_id(self) -> None:
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10]]),
            class_id=np.array([5]),
        )
        class_names = ["person", "car"]  # No index 5
        args = argparse.Namespace(
            show_labels=True,
            show_ids=False,
            show_confidence=False,
        )

        labels = _generate_labels(detections, class_names, args)

        assert labels == ["5"]

    def test_generates_tracker_ids(self) -> None:
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10]]),
            tracker_id=np.array([42]),
        )
        args = argparse.Namespace(
            show_labels=False,
            show_ids=True,
            show_confidence=False,
        )

        labels = _generate_labels(detections, [], args)

        assert labels == ["#42"]

    def test_generates_combined_labels(self) -> None:
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10]]),
            class_id=np.array([0]),
            confidence=np.array([0.95]),
            tracker_id=np.array([1]),
        )
        class_names = ["person"]
        args = argparse.Namespace(
            show_labels=True,
            show_ids=True,
            show_confidence=True,
        )

        labels = _generate_labels(detections, class_names, args)

        assert labels == ["#1 person 0.95"]


class TestCreateFrameGenerator:
    """Tests for create_frame_generator function."""

    def test_generates_from_image_directory(self, tmp_path: Path) -> None:
        # Create test images
        import cv2

        for i in range(3):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"frame_{i:04d}.jpg"), img)

        frames = list(create_frame_generator(str(tmp_path)))

        assert len(frames) == 3
        assert frames[0][0] == 1  # First frame index is 1
        assert frames[0][1].shape == (100, 100, 3)

    def test_raises_for_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No image files found"):
            list(create_frame_generator(str(tmp_path)))

    def test_raises_for_invalid_source(self) -> None:
        with pytest.raises(ValueError, match="Cannot open video source"):
            list(create_frame_generator("/nonexistent/video.mp4"))
