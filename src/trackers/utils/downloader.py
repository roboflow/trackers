# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import zipfile
from contextlib import suppress
from pathlib import Path, PurePosixPath, PureWindowsPath

import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TransferSpeedColumn,
)

_CHUNK_SIZE_BYTES = 8192
_DEFAULT_TIMEOUT_SECONDS = 30


def _compute_md5(file_path: Path) -> str:
    """Compute MD5 hex digest of a file."""
    hash_md5 = hashlib.md5(usedforsecurity=False)
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(_CHUNK_SIZE_BYTES), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _download_file(
    url: str,
    destination: Path,
    *,
    md5: str | None = None,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Download a file with a progress bar and optional MD5 verification.

    If `destination` already exists and `md5` is provided, the existing
    file's checksum is verified. The download is skipped when the
    checksum matches.

    Returns:
        `True` if the file was downloaded, `False` if a cached version
        was used.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and md5 is not None:
        if _compute_md5(destination) == md5:
            return False
        destination.unlink()

    temp_path = destination.with_suffix(".tmp")

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
    )
    with open(temp_path, "wb") as file_handle, progress:
        task = progress.add_task(destination.name, total=total or None)
        for chunk in response.iter_content(chunk_size=_CHUNK_SIZE_BYTES):
            if chunk:
                file_handle.write(chunk)
                progress.update(task, advance=len(chunk))

    temp_path.rename(destination)

    if md5 is not None:
        actual = _compute_md5(destination)
        if actual != md5:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"MD5 checksum mismatch for {destination.name}: expected {md5}, got {actual}")

    return True


def _safe_zip_member_parts(member_name: str) -> tuple[str, ...]:
    """Return sanitized ZIP path parts for a member.

    ZIP member names are treated as forward-slash paths. Backslashes and
    Windows drive-rooted forms are rejected so the extraction contract is
    consistent across platforms.
    """
    if not member_name:
        raise ValueError("ZIP archive contains an empty member name")

    if "\\" in member_name:
        raise ValueError(f"ZIP member path escapes output directory: {member_name}")

    posix_path = PurePosixPath(member_name)
    windows_path = PureWindowsPath(member_name)
    if posix_path.is_absolute() or windows_path.drive or windows_path.root:
        raise ValueError(f"ZIP member path escapes output directory: {member_name}")

    parts = posix_path.parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"ZIP member path escapes output directory: {member_name}")

    return parts


def _open_child_directory(parent_fd: int, directory_name: str) -> int:
    """Open a child directory below `parent_fd`, creating it if needed."""
    with suppress(FileExistsError):
        os.mkdir(directory_name, mode=0o777, dir_fd=parent_fd)

    directory_fd = os.open(
        directory_name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    try:
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise NotADirectoryError(directory_name)
    except Exception:
        os.close(directory_fd)
        raise
    return directory_fd


def _open_directory_path(root_fd: int, directory_parts: tuple[str, ...]) -> int:
    """Open a directory path below `root_fd`, creating missing levels."""
    directory_fd = os.dup(root_fd)
    try:
        for directory_name in directory_parts:
            next_fd = _open_child_directory(directory_fd, directory_name)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except Exception:
        os.close(directory_fd)
        raise


def _extract_zip_member(zip_file: zipfile.ZipFile, zip_info: zipfile.ZipInfo, root_fd: int) -> None:
    """Extract a single ZIP member relative to `root_fd`."""
    member_parts = _safe_zip_member_parts(zip_info.filename)

    if zip_info.is_dir():
        directory_fd = _open_directory_path(root_fd, member_parts)
        os.close(directory_fd)
        return

    parent_fd = _open_directory_path(root_fd, member_parts[:-1])
    try:
        file_fd = os.open(
            member_parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o666,
            dir_fd=parent_fd,
        )
        with os.fdopen(file_fd, "wb") as target, zip_file.open(zip_info, "r") as source:
            shutil.copyfileobj(source, target, length=_CHUNK_SIZE_BYTES)
    finally:
        os.close(parent_fd)


def _extract_zip_member_by_path(
    zip_file: zipfile.ZipFile,
    zip_info: zipfile.ZipInfo,
    output_dir: Path,
) -> None:
    """Extract a single ZIP member using validated paths.

    Windows does not expose the dir-fd primitives used by the Unix path, so
    the fallback keeps the same member validation and writes to resolved
    absolute paths.
    """
    member_parts = _safe_zip_member_parts(zip_info.filename)
    member_path = output_dir.joinpath(*member_parts)

    if zip_info.is_dir():
        member_path.mkdir(parents=True, exist_ok=True)
        return

    member_path.parent.mkdir(parents=True, exist_ok=True)
    with zip_file.open(zip_info, "r") as source, open(member_path, "wb") as target:
        shutil.copyfileobj(source, target, length=_CHUNK_SIZE_BYTES)


def _extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract a ZIP archive into `output_dir`.

    On Unix, extraction is anchored to the directory opened before members are
    written, so a path swap after validation cannot redirect writes outside
    `output_dir`. On Windows, the fallback keeps the member validation but uses
    resolved absolute paths because dir-fd containment is unavailable.

    Raises:
        ValueError: If any member would extract outside `output_dir`.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        member_infos = zip_file.infolist()
        for zip_info in member_infos:
            _safe_zip_member_parts(zip_info.filename)

        if os.name == "nt":
            for zip_info in member_infos:
                _extract_zip_member_by_path(zip_file, zip_info, output_dir)
            return

        root_fd = os.open(
            output_dir,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            for zip_info in member_infos:
                _extract_zip_member(zip_file, zip_info, root_fd)
        finally:
            os.close(root_fd)
