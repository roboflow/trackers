# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

from trackers.utils import downloader
from trackers.utils.downloader import _extract_zip, _safe_zip_member_parts


def _write_zip(zip_path: Path, members: list[tuple[str | zipfile.ZipInfo, str]]) -> None:
    """Write a ZIP archive containing the provided text members."""
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for member, contents in members:
            zip_file.writestr(member, contents)


def _raw_zip_info(filename: str) -> zipfile.ZipInfo:
    """Create a ZIP member name without platform separator normalization."""
    zip_info = zipfile.ZipInfo("placeholder")
    zip_info.filename = filename
    return zip_info


def test_extract_zip_extracts_safe_members(tmp_path: Path) -> None:
    """Safe members are extracted beneath the requested output directory."""
    zip_path = tmp_path / "archive.zip"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_zip(
        zip_path,
        [
            ("nested/file.txt", "payload"),
            ("root.txt", "top"),
        ],
    )

    _extract_zip(zip_path, output_dir)

    assert (output_dir / "nested" / "file.txt").read_text() == "payload"
    assert (output_dir / "root.txt").read_text() == "top"


@pytest.mark.parametrize(
    ("member_name", "message"),
    [
        ("../escape.txt", "escapes output directory"),
        ("/escape.txt", "escapes output directory"),
        ("nested/../../escape.txt", "escapes output directory"),
        pytest.param(
            _raw_zip_info("nested\\escape.txt"),
            "escapes output directory",
            marks=pytest.mark.skipif(
                os.name == "nt",
                reason="zipfile normalizes backslash member names while reading on Windows",
            ),
        ),
        ("C:escape.txt", "escapes output directory"),
        ("C:/escape.txt", "escapes output directory"),
        (zipfile.ZipInfo(""), "empty member name"),
    ],
)
def test_extract_zip_rejects_unsafe_members(
    tmp_path: Path,
    member_name: str | zipfile.ZipInfo,
    message: str,
) -> None:
    """Unsafe member names fail before any files are extracted."""
    zip_path = tmp_path / "archive.zip"
    output_dir = tmp_path / "output"
    outside_target = tmp_path / "escape.txt"
    output_dir.mkdir()
    outside_target.write_text("sentinel")
    _write_zip(
        zip_path,
        [
            ("safe.txt", "safe"),
            (member_name, "evil"),
        ],
    )

    with pytest.raises(ValueError, match=message):
        _extract_zip(zip_path, output_dir)

    assert list(output_dir.iterdir()) == []
    assert outside_target.read_text() == "sentinel"


def test_safe_zip_member_parts_rejects_raw_backslash() -> None:
    """Raw ZIP member validation rejects backslash separators."""
    with pytest.raises(ValueError, match="escapes output directory"):
        _safe_zip_member_parts("nested\\escape.txt")


@pytest.mark.skipif(os.name == "nt", reason="fd-anchored swap probe is Unix-only")
def test_extract_zip_survives_output_dir_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A path swap after opening `output_dir` does not redirect writes."""
    zip_path = tmp_path / "archive.zip"
    output_dir = tmp_path / "output"
    backup_dir = tmp_path / "output-real"
    outside_dir = tmp_path / "outside"
    output_dir.mkdir()
    outside_dir.mkdir()
    _write_zip(zip_path, [("payload.txt", "payload")])

    original_open = downloader.os.open
    swapped = False

    def open_wrapper(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if dir_fd is None:
            fd = original_open(path, flags, mode)
        else:
            fd = original_open(path, flags, mode, dir_fd=dir_fd)

        if not swapped and dir_fd is None:
            output_dir.rename(backup_dir)
            output_dir.symlink_to(outside_dir, target_is_directory=True)
            swapped = True

        return fd

    monkeypatch.setattr(downloader.os, "open", open_wrapper)

    _extract_zip(zip_path, output_dir)

    assert (backup_dir / "payload.txt").read_text() == "payload"
    assert not (outside_dir / "payload.txt").exists()
