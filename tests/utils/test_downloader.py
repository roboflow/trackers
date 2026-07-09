# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from trackers.utils.downloader import _extract_zip


def _write_zip(zip_path: Path, members: dict[str, str]) -> None:
    """Write a ZIP archive containing the provided text members."""
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for name, contents in members.items():
            zip_file.writestr(name, contents)


def test_extract_zip_extracts_safe_members(tmp_path: Path) -> None:
    """Safe members are extracted beneath the requested output directory."""
    zip_path = tmp_path / "archive.zip"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_zip(
        zip_path,
        {
            "nested/file.txt": "payload",
            "root.txt": "top",
        },
    )

    _extract_zip(zip_path, output_dir)

    assert (output_dir / "nested" / "file.txt").read_text() == "payload"
    assert (output_dir / "root.txt").read_text() == "top"


@pytest.mark.parametrize(
    "member_name",
    [
        "../escape.txt",
        "/escape.txt",
        "nested/../../escape.txt",
    ],
)
def test_extract_zip_rejects_unsafe_members(tmp_path: Path, member_name: str) -> None:
    """Unsafe member names fail before any files are extracted."""
    zip_path = tmp_path / "archive.zip"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_zip(
        zip_path,
        {
            "safe.txt": "safe",
            member_name: "evil",
        },
    )

    with pytest.raises(ValueError, match="escapes output directory"):
        _extract_zip(zip_path, output_dir)

    assert list(output_dir.iterdir()) == []
