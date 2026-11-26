# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from trackers.cli._detections import load_rgb_frame
from trackers.cli.inspect._common import load_rgb_image, save_rgb_image
from trackers.io._image import _read_image, _write_image
from trackers.io.frames import load_mot_frame_image
from trackers.io.video import frames_from_source

NON_ASCII_FOLDERS = ["séquence", "動画", "видео"]


class TestNonAsciiImagePaths:
    def test_cli_reads_and_writes_rgb_png_dotfile(self, tmp_path: Path) -> None:
        """Preserve RGB pixels at the requested Unicode path through all three CLI image helpers."""
        path = tmp_path / "動画" / ".png"
        rgb = np.array([[[17, 83, 224]]], dtype=np.uint8)

        save_rgb_image(rgb, path)

        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        np.testing.assert_array_equal(cv2.imdecode(encoded, cv2.IMREAD_COLOR), rgb[..., ::-1])
        np.testing.assert_array_equal(load_rgb_image(path), rgb)
        np.testing.assert_array_equal(load_rgb_frame(path), rgb)

    @pytest.mark.parametrize("folder_name", NON_ASCII_FOLDERS)
    def test_write_then_read_round_trips(self, tmp_path: Path, folder_name: str) -> None:
        image = np.random.default_rng(0).integers(0, 256, (4, 6, 3), dtype=np.uint8)
        path = tmp_path / folder_name / "frame.png"
        path.parent.mkdir()

        assert _write_image(path, image)

        loaded = _read_image(path)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, image)

    def test_write_keeps_non_ascii_file_name(self, tmp_path: Path) -> None:
        assert _write_image(tmp_path / "café.png", np.zeros((4, 6, 3), dtype=np.uint8))

        assert [path.name for path in tmp_path.iterdir()] == ["café.png"]

    def test_read_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        assert _read_image(tmp_path / "café.png") is None

    @pytest.mark.parametrize("folder_name", NON_ASCII_FOLDERS)
    def test_frame_sources_read_non_ascii_folders(self, tmp_path: Path, folder_name: str) -> None:
        frame_dir = tmp_path / folder_name / "img1"
        frame_dir.mkdir(parents=True)
        image = np.full((4, 6, 3), 90, dtype=np.uint8)
        for frame_idx in (1, 2):
            assert _write_image(frame_dir / f"{frame_idx:06d}.png", image)

        frames = list(frames_from_source(str(frame_dir)))

        assert [frame_id for frame_id, _ in frames] == [1, 2]
        np.testing.assert_array_equal(load_mot_frame_image(frame_dir, 2), image)
