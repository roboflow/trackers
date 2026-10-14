# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Exercise EdgeTAM replay and shared Cutie event timing without model weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trackers.cli.__main__ import main
from trackers.cli.inspect._common import load_rgb_image, save_rgb_image
from trackers.cli.inspect.cutie import cutie_inspection
from trackers.cli.inspect.edgetam import edgetam_inspection
from trackers.core.masks.base import MaskOutput
from trackers.core.masks.dummy import DummyBoxMaskGenerator, DummyIdentityMaskPropagator


class FakeBackend(DummyBoxMaskGenerator, DummyIdentityMaskPropagator):
    """Use shared dummy masks while exposing the inspection metadata interface."""

    device = "cpu"
    model_revision = "fake-revision"

    def __init__(self, **kwargs: Any) -> None:
        """Keep deterministic counters without requiring optional dependencies."""
        DummyIdentityMaskPropagator.__init__(self)
        self.session_index = 0
        self.session_frame_count = 1


@pytest.mark.parametrize("component", ["cutie", "edgetam"])
def test_replay_preserves_event_timing_and_saves_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    """Both commands apply multiple additions and removals before the correct image."""
    images = tmp_path / "images"
    names = ("000001.png", "000008.png", "000050.png", "000099.png")
    for index, name in enumerate(names):
        save_rgb_image(np.full((100, 120, 3), index, dtype=np.uint8), images / name)
    backend = FakeBackend()
    if component == "cutie":
        backend.device = pytest.importorskip("torch").device("cpu")
    propagated = []
    added_on = []
    original_propagate = backend.propagate
    original_add = backend.add_masks

    def record_propagation(frame: np.ndarray) -> MaskOutput | None:
        """Observe which identities actually reach each selected image."""
        output = original_propagate(frame)
        assert output is not None
        propagated.append((int(frame[0, 0, 0]), sorted(output.tracklet_mask_dict)))
        return output

    def record_addition(frame: np.ndarray, mask_output: MaskOutput) -> None:
        """Ensure each new mask belongs to the preceding image, including gaps."""
        added_on.append((int(frame[0, 0, 0]), sorted(mask_output.tracklet_mask_dict)))
        original_add(frame, mask_output)

    monkeypatch.setattr(backend, "propagate", record_propagation)
    monkeypatch.setattr(backend, "add_masks", record_addition)
    monkeypatch.setattr("trackers.core.masks.edgetam.EdgeTAMMaskBackend", lambda **kwargs: backend)
    monkeypatch.setitem(
        sys.modules, "trackers.core.masks.sam", SimpleNamespace(SAMBoxMaskGenerator=lambda **kwargs: backend)
    )
    monkeypatch.setitem(
        sys.modules, "trackers.core.masks.cutie", SimpleNamespace(CutieMaskPropagator=lambda **kwargs: backend)
    )
    output_root = tmp_path / "outputs"
    command = cutie_inspection if component == "cutie" else edgetam_inspection
    result = command(
        images,
        "000001.png",
        "000099.png",
        [(1, 1, 5, 6)],
        add_at=["000008.png:2,2,6,8", "000008.png:8,2,12,8"],
        remove_at=["000050.png:1", "000050.png:3"],
        output_root=output_root,
        device="cpu",
    )
    assert result == 0
    # Pixel values identify the selected frames despite gaps in their filenames.
    # Boxes on 000008 (value 1) seed additions before propagation to 000050 (value 2).
    expected_additions = [(1, [2, 3])]
    expected_propagation = [(1, [1]), (2, [2]), (3, [2])]
    assert added_on == expected_additions
    assert propagated == expected_propagation
    run_dir = next(output_root.iterdir())
    assert sorted(path.name for path in run_dir.glob("*.png")) == list(names)
    for name in names:
        assert load_rgb_image(run_dir / name).shape == (100, 120, 3)
    if component == "cutie":
        return
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["mode"] == "manual-mask-replay"
    assert manifest["frames"][2]["active_ids"] == [2]
    assert manifest["frames"][2]["added_ids"] == [2, 3]
    assert manifest["frames"][2]["removed_ids"] == [1, 3]
    assert len(manifest["frames"][0]["sha256"]) == 64
    assert len(list(run_dir.glob("*.png"))) == 4
    config = json.loads((run_dir / "config.json").read_text())
    assert config["revision"] == "fake-revision"
    assert config["max_session_frames"] == 200


@pytest.mark.parametrize("max_session_frames", [200, None])
def test_generated_config_replays_through_cli_with_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_session_frames: int | None
) -> None:
    """Replay a generated config through the real CLI, preserving events and applying overrides."""
    images = tmp_path / "images"
    names = ("000001.png", "000008.png", "000050.png", "000099.png")
    for index, name in enumerate(names):
        save_rgb_image(np.full((100, 120, 3), index, dtype=np.uint8), images / name)
    backend_options: list[dict[str, Any]] = []

    def make_backend(**kwargs: Any) -> FakeBackend:
        """Record the effective model settings while avoiding optional inference dependencies."""
        backend_options.append(kwargs)
        return FakeBackend(**kwargs)

    monkeypatch.setattr("trackers.core.masks.edgetam.EdgeTAMMaskBackend", make_backend)
    output_root = tmp_path / "original"
    assert (
        edgetam_inspection(
            images,
            names[0],
            names[-1],
            [(1, 1, 5, 6)],
            add_at=["000008.png:2,2,6,8", "000008.png:8,2,12,8"],
            remove_at=["000050.png:1", "000050.png:3"],
            output_root=output_root,
            device="cpu",
            max_session_frames=max_session_frames,
        )
        == 0
    )
    original_dir = next(output_root.iterdir())
    config_path = original_dir / "config.json"
    config = json.loads(config_path.read_text())
    assert config["max_session_frames"] == max_session_frames
    replay_root = tmp_path / "replay"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trackers",
            "inspect",
            "edgetam",
            "--config",
            str(config_path),
            "--max_session_frames",
            "50",
            "--output_root",
            str(replay_root),
        ],
    )
    with pytest.warns(UserWarning, match="The trackers CLI is in beta"):
        assert main() == 0

    assert len(backend_options) == 2
    assert backend_options[1]["max_session_frames"] == 50
    assert backend_options[1]["revision"] == "fake-revision"
    replay_dir = next(replay_root.iterdir())
    replay_config = json.loads((replay_dir / "config.json").read_text())
    assert replay_config == {**config, "max_session_frames": 50, "output_root": str(replay_root.resolve())}
    original_manifest = json.loads((original_dir / "run.json").read_text())
    replay_manifest = json.loads((replay_dir / "run.json").read_text())
    assert replay_manifest["status"] == "completed"
    assert replay_manifest["frames"] == original_manifest["frames"]
    assert sorted(path.name for path in replay_dir.glob("*.png")) == list(names)
    for name in names:
        assert np.array_equal(load_rgb_image(replay_dir / name), load_rgb_image(original_dir / name))


def test_invalid_events_are_rejected_before_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid lifecycle input should never download or construct a model."""
    for name in ("a.png", "b.png"):
        save_rgb_image(np.zeros((8, 12, 3), dtype=np.uint8), tmp_path / name)

    def fail_load(**kwargs: Any) -> None:
        raise AssertionError("Model construction must not be reached")

    monkeypatch.setattr("trackers.core.masks.edgetam.EdgeTAMMaskBackend", fail_load)
    assert edgetam_inspection(tmp_path, "a.png", "b.png", [], add_at=["b.png:1,1,4,5"]) == 1


def test_manifest_write_failure_still_resets_and_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unwritable final output must not bypass cleanup or escape as a traceback."""
    for name in ("a.png", "b.png"):
        save_rgb_image(np.zeros((100, 120, 3), dtype=np.uint8), tmp_path / name)
    backend = FakeBackend()
    resets = []
    monkeypatch.setattr(backend, "reset", lambda: resets.append(True))
    monkeypatch.setattr("trackers.core.masks.edgetam.EdgeTAMMaskBackend", lambda **kwargs: backend)
    original_write = Path.write_text

    def fail_manifest(path: Path, *args: Any, **kwargs: Any) -> int:
        """Fail only the final manifest, leaving input and config writes usable."""
        if path.name == "run.json":
            raise OSError("Simulated full disk")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_manifest)
    result = edgetam_inspection(tmp_path, "a.png", "b.png", [(1, 1, 5, 6)], output_root=tmp_path / "output")
    assert result == 1
    assert resets
    assert "Simulated full disk" in capsys.readouterr().err
