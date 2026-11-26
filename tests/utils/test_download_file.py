# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import builtins
import hashlib
import io
import os
import stat
import zipfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Barrier, Event, Thread
from typing import BinaryIO, cast
from unittest.mock import Mock

import pytest
import requests

from trackers.datasets.download import download_dataset
from trackers.datasets.manifest import _DATASETS, Dataset
from trackers.utils import downloader


@pytest.fixture
def response(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Provide a streamed response without contacting an external host."""
    response = Mock(spec=requests.Response)
    response.headers = {"content-length": "3"}
    response.iter_content.return_value = iter([b"", b"new"])
    monkeypatch.setattr(downloader.requests, "get", Mock(return_value=response))
    return response


def test_matching_cache_skips_download(tmp_path: Path, response: Mock) -> None:
    """A matching cached file is returned without requesting or rewriting it."""
    target = tmp_path / "model.pth"
    target.write_bytes(b"old")
    checksum = hashlib.md5(b"old", usedforsecurity=False).hexdigest()
    assert downloader._download_file("https://example.com/model", target, md5=checksum) is False
    downloader.requests.get.assert_not_called()
    response.iter_content.assert_not_called()
    assert target.read_bytes() == b"old"


@pytest.mark.parametrize("verify", [False, True])
def test_download_replaces_only_after_verification(tmp_path: Path, response: Mock, verify: bool) -> None:
    """A successful download can replace an existing file with or without MD5."""
    target = tmp_path / "model.pth"
    target.write_bytes(b"old")
    checksum = hashlib.md5(b"new", usedforsecurity=False).hexdigest() if verify else None
    assert downloader._download_file("https://example.com/model", target, md5=checksum) is True
    assert target.read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [target]
    response.close.assert_called_once()


def test_download_honors_file_creation_mode(tmp_path: Path, response: Mock) -> None:
    """A new download keeps the ordinary file creation mode under the current umask."""
    reference = tmp_path / "reference"
    reference.write_bytes(b"reference")
    target = tmp_path / "model.pth"
    downloader._download_file("https://example.com/model", target)
    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)


@pytest.mark.skipif(os.name != "posix", reason="POSIX umask and directory permissions are required")
@pytest.mark.parametrize("mask", [pytest.param(0o111, id="no-owner-search"), pytest.param(0o222, id="no-owner-write")])
def test_download_with_restrictive_umask(tmp_path: Path, response: Mock, mask: int) -> None:
    """Stage downloads under restrictive umasks while preserving the file creation mode."""
    previous_mask = os.umask(mask)
    try:
        reference = tmp_path / "reference"
        reference.write_bytes(b"reference")
        target = tmp_path / "model.pth"
        downloader._download_file("https://example.com/model", target)
        assert target.read_bytes() == b"new"
        assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)
        assert sorted(path.name for path in tmp_path.iterdir()) == ["model.pth", "reference"]
    finally:
        os.umask(previous_mask)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt, SystemExit])
def test_stream_failure_preserves_target_and_cleans_temporary_files(
    tmp_path: Path, response: Mock, existing: bool, error_type: type[BaseException]
) -> None:
    """Failure and cancellation remove only the current download's temporary data."""
    target = tmp_path / "model.pth"
    if existing:
        target.write_bytes(b"old")
    expected = hashlib.md5(b"new", usedforsecurity=False).hexdigest()

    def chunks(chunk_size: int) -> Iterator[bytes]:
        """Fail after transferring part of the response body."""
        yield b"partial"
        raise error_type("stopped")

    response.iter_content.side_effect = chunks
    with pytest.raises(error_type, match="stopped"):
        downloader._download_file("https://example.com/model", target, md5=expected)
    assert sorted(tmp_path.iterdir()) == ([target] if existing else [])
    if existing:
        assert target.read_bytes() == b"old"
    response.close.assert_called_once()


@pytest.mark.parametrize("existing", [False, True])
def test_checksum_failure_does_not_publish_payload(tmp_path: Path, response: Mock, existing: bool) -> None:
    """A bad response checksum cannot remove or replace a prior destination."""
    target = tmp_path / "model.pth"
    if existing:
        target.write_bytes(b"old")
    with pytest.raises(RuntimeError, match="MD5 checksum mismatch"):
        downloader._download_file("https://example.com/model", target, md5="0" * 32)
    assert list(tmp_path.iterdir()) == ([target] if existing else [])
    if existing:
        assert target.read_bytes() == b"old"
    response.close.assert_called_once()


def test_http_error_closes_response_and_preserves_file(tmp_path: Path, response: Mock) -> None:
    """An unsuccessful HTTP status leaves the previous file untouched."""
    target = tmp_path / "model.pth"
    target.write_bytes(b"old")
    response.raise_for_status.side_effect = requests.HTTPError("unavailable")
    with pytest.raises(requests.HTTPError, match="unavailable"):
        downloader._download_file("https://example.com/model", target, md5="0" * 32)
    assert target.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]
    response.close.assert_called_once()


@pytest.mark.parametrize("failure_stage", ["checksum", "replace"])
def test_post_transfer_failure_cleans_up(
    tmp_path: Path, response: Mock, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    """Read and publication failures preserve the prior destination and propagate."""
    target = tmp_path / "model.pth"
    target.write_bytes(b"old")
    checksum = hashlib.md5(b"new", usedforsecurity=False).hexdigest()
    if failure_stage == "checksum":
        original_compute = downloader._compute_md5

        def compute(file_path: Path) -> str:
            """Fail while reading only the downloaded temporary file."""
            if file_path != target:
                raise OSError("checksum read failed")
            return original_compute(file_path)

        monkeypatch.setattr(downloader, "_compute_md5", compute)
    else:
        monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("replace failed")))
    with pytest.raises(OSError, match="failed"):
        downloader._download_file("https://example.com/model", target, md5=checksum)
    assert target.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]
    response.close.assert_called_once()


@pytest.fixture
def archive_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, bytes]]:
    """Serve a small real ZIP over loopback and close the owned server afterward."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("mot17/train/fixture/gt/gt.txt", "1,1,10,20,30,40,1,1,1\n")
    payload = stream.getvalue()

    class Handler(BaseHTTPRequestHandler):
        """Return the fixture archive without external networking."""

        def log_message(self, message_format: str, *args: object) -> None:
            """Suppress the HTTP server's default stderr output."""

        def do_GET(self) -> None:
            """Send the complete archive with its actual content length."""
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/archive.zip", payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_concurrent_downloads_to_same_cache(
    tmp_path: Path, archive_server: tuple[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two writers may overlap without sharing or deleting each other's temporary file."""
    url, payload = archive_server
    target = tmp_path / "archive.zip"
    barrier = Barrier(2, timeout=5)

    def synchronized_open(file: Path, mode: str) -> BinaryIO:
        """Hold both real writable files open before either writer proceeds."""
        handle = cast(BinaryIO, builtins.open(file, mode))
        if mode == "wb":
            try:
                barrier.wait()
            except BaseException:
                handle.close()
                raise
        return handle

    monkeypatch.setattr(downloader, "open", synchronized_open, raising=False)
    checksum = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(downloader._download_file, url, target, md5=checksum, timeout=3) for _ in range(2)]
        assert [future.result(timeout=10) for future in futures] == [True, True]
    assert target.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [target]


def test_failed_download_cannot_remove_another_callers_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad transfer completing last cannot replace or unlink a verified concurrent result."""
    target = tmp_path / "model.pth"
    expected = hashlib.md5(b"good", usedforsecurity=False).hexdigest()
    started = Event()
    release = Event()
    good = Mock(spec=requests.Response)
    bad = Mock(spec=requests.Response)
    good.headers = bad.headers = {"content-length": "4"}
    good.iter_content.return_value = iter([b"good"])

    def bad_chunks(chunk_size: int) -> Iterator[bytes]:
        """Hold the invalid response until the other call publishes its result."""
        yield b"bad!"
        started.set()
        assert release.wait(timeout=5)

    bad.iter_content.side_effect = bad_chunks
    monkeypatch.setattr(downloader.requests, "get", Mock(side_effect=[bad, good]))
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(downloader._download_file, "https://example.com/bad", target, md5=expected)
        try:
            assert started.wait(timeout=5)
            valid = pool.submit(downloader._download_file, "https://example.com/good", target, md5=expected)
            assert valid.result(timeout=5) is True
            assert target.read_bytes() == b"good"
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="MD5 checksum mismatch"):
            failed.result(timeout=5)
    assert target.read_bytes() == b"good"
    assert list(tmp_path.iterdir()) == [target]
    good.close.assert_called_once()
    bad.close.assert_called_once()


def test_dataset_download_and_cache_reuse(
    tmp_path: Path, archive_server: tuple[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public dataset path downloads, extracts and reuses a verified archive."""
    url, payload = archive_server
    monkeypatch.setitem(
        _DATASETS,
        Dataset.MOT17,
        {
            "splits": {
                "train": {"annotations": {"url": url, "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest()}}
            }
        },
    )
    get = Mock(wraps=requests.get)
    monkeypatch.setattr(downloader.requests, "get", get)
    cache = tmp_path / "cache"
    for name in ("first", "second"):
        output = tmp_path / name
        download_dataset(name="mot17", split="train", asset="annotations", output=str(output), cache_dir=str(cache))
        assert (output / "mot17/train/fixture/gt/gt.txt").read_text() == "1,1,10,20,30,40,1,1,1\n"
    assert get.call_count == 1
    assert (cache / "archive.zip").read_bytes() == payload
    assert list(cache.iterdir()) == [cache / "archive.zip"]
