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

from trackers.io.frames import load_mot_frame_image, mot_frame_stems, resolve_mot_frame_path


class TestMotFrameStems:
    def test_includes_six_and_eight_digit_stems(self) -> None:
        assert mot_frame_stems(1) == ("000001", "00000001", "1")

    def test_large_frame_index(self) -> None:
        assert mot_frame_stems(123) == ("000123", "00000123", "123")


class TestResolveMotFramePath:
    def test_finds_six_digit_mot_challenge_name(self, tmp_path: Path) -> None:
        path = tmp_path / "000042.jpg"
        cv2.imwrite(str(path), np.zeros((4, 4, 3), dtype=np.uint8))

        assert resolve_mot_frame_path(tmp_path, 42) == path

    def test_finds_eight_digit_dancetrack_name(self, tmp_path: Path) -> None:
        path = tmp_path / "00000042.jpg"
        cv2.imwrite(str(path), np.zeros((4, 4, 3), dtype=np.uint8))

        assert resolve_mot_frame_path(tmp_path, 42) == path

    def test_prefers_six_digit_when_both_exist(self, tmp_path: Path) -> None:
        six = tmp_path / "000001.jpg"
        eight = tmp_path / "00000001.jpg"
        cv2.imwrite(str(six), np.zeros((4, 4, 3), dtype=np.uint8))
        cv2.imwrite(str(eight), np.ones((4, 4, 3), dtype=np.uint8))

        assert resolve_mot_frame_path(tmp_path, 1) == six

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert resolve_mot_frame_path(tmp_path, 7) is None


class TestLoadMotFrameImage:
    def test_loads_bgr_array(self, tmp_path: Path) -> None:
        image_path = tmp_path / "00000001.jpg"
        cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))

        frame = load_mot_frame_image(tmp_path, 1)

        assert frame.shape == (8, 8, 3)

    def test_raises_when_frame_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"frame 3"):
            load_mot_frame_image(tmp_path, 3)

    def test_raises_when_file_cannot_be_decoded(self, tmp_path: Path) -> None:
        bad = tmp_path / "000001.jpg"
        bad.write_bytes(b"not-an-image")

        with pytest.raises(OSError, match=r"Failed to decode"):
            load_mot_frame_image(tmp_path, 1)
