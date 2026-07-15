# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trackers.scripts.track import (
    _apply_reid_tracker_params,
    _reid_requested,
    _validate_reid_cli_prerequisites,
    add_track_subparser,
    run_track,
)


class _FakeReIDModel:
    last_kwargs: dict | None = None

    @classmethod
    def from_pretrained(cls, **kwargs: object) -> _FakeReIDModel:
        cls.last_kwargs = dict(kwargs)
        return cls()


def test_reid_model_source_implies_enable() -> None:
    args = argparse.Namespace(tracker_reid_enable=False, tracker_reid_model="osnet_x1_0_msmt17_combineall")
    assert _reid_requested(args)


def test_validate_reid_requires_source_before_load(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeReIDModel.last_kwargs = None
    monkeypatch.setattr(
        "trackers.scripts.track.load_reid_model_class",
        lambda: _FakeReIDModel,
        raising=False,
    )
    monkeypatch.setattr(
        "trackers.core.reid._lazy.load_reid_model_class",
        lambda: _FakeReIDModel,
    )

    args = argparse.Namespace(
        tracker="botsort",
        tracker_reid_enable=True,
        tracker_reid_model=None,
        tracker_reid_device="cpu",
        tracker_reid_architecture=None,
        source=None,
        detections=None,
        output=None,
        display=False,
        overwrite=False,
        model="rfdetr-nano",
        model_confidence=0.5,
        model_device="cpu",
        model_api_key=None,
        classes=None,
        track_ids=None,
        mot_output=None,
        show_boxes=True,
        show_masks=False,
        show_labels=False,
        show_ids=True,
        show_confidence=False,
        show_trajectories=False,
    )

    err = _validate_reid_cli_prerequisites(args)
    assert err is not None and "--source" in err

    params, load_err = _apply_reid_tracker_params("botsort", args, {})
    assert load_err is not None and "--source" in load_err
    assert params == {}


def test_apply_reid_passes_architecture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("trackers.core.reid._lazy.load_reid_model_class", lambda: _FakeReIDModel)
    weights = tmp_path / "weights.pth"
    weights.touch()
    args = argparse.Namespace(
        tracker_reid_enable=False,
        tracker_reid_model=str(weights),
        tracker_reid_device="cpu",
        tracker_reid_architecture="osnet_x1_0",
        source="video.mp4",
    )
    params, err = _apply_reid_tracker_params("botsort", args, {})
    assert err is None
    assert _FakeReIDModel.last_kwargs is not None
    assert _FakeReIDModel.last_kwargs["source"] == str(weights)
    assert _FakeReIDModel.last_kwargs["architecture"] == "osnet_x1_0"
    assert "reid_model" in params


def test_apply_reid_model_load_error_is_concise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FailingModel:
        @classmethod
        def from_pretrained(cls, **kwargs: object) -> None:
            raise ValueError("architecture is required")

    monkeypatch.setattr("trackers.core.reid._lazy.load_reid_model_class", lambda: _FailingModel)
    bare_weights = tmp_path / "bare.pth"
    bare_weights.touch()
    args = argparse.Namespace(
        tracker_reid_enable=True,
        tracker_reid_model=str(bare_weights),
        tracker_reid_device="cpu",
        tracker_reid_architecture=None,
        source="video.mp4",
    )
    _, err = _apply_reid_tracker_params("botsort", args, {})
    assert err is not None and "Failed to load ReID model" in err
    assert "architecture is required" in err


def test_run_track_checks_source_before_reid_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    load_called = False

    def _fail_load(**kwargs: object) -> object:
        nonlocal load_called
        load_called = True
        raise AssertionError("from_pretrained should not run without --source")

    monkeypatch.setattr("trackers.core.reid._lazy.load_reid_model_class", lambda: _FakeReIDModel)
    monkeypatch.setattr(_FakeReIDModel, "from_pretrained", _fail_load)

    mot_file = tmp_path / "dets.txt"
    mot_file.write_text("1,1,10,10,20,20,1,-1,-1,-1\n")

    args = argparse.Namespace(
        tracker="botsort",
        tracker_reid_enable=True,
        tracker_reid_model=None,
        tracker_reid_device="cpu",
        tracker_reid_architecture=None,
        source=None,
        detections=mot_file,
        output=None,
        display=False,
        overwrite=False,
        model="rfdetr-nano",
        model_confidence=0.5,
        model_device="cpu",
        model_api_key=None,
        classes=None,
        track_ids=None,
        mot_output=None,
        show_boxes=True,
        show_masks=False,
        show_labels=False,
        show_ids=True,
        show_confidence=False,
        show_trajectories=False,
    )
    for key, value in {
        "tracker_lost_track_buffer": 30,
        "tracker_frame_rate": 30.0,
        "tracker_track_activation_threshold": 0.7,
        "tracker_minimum_consecutive_frames": 2,
        "tracker_minimum_iou_threshold_first_assoc": 0.2,
        "tracker_minimum_iou_threshold_second_assoc": 0.5,
        "tracker_minimum_iou_threshold_unconfirmed_assoc": 0.3,
        "tracker_high_conf_det_threshold": 0.6,
        "tracker_enable_cmc": True,
        "tracker_cmc_method": "sparseOptFlow",
        "tracker_cmc_downscale": 2,
        "tracker_instant_first_frame_activation": True,
        "tracker_reid_ema_alpha": 0.9,
        "tracker_appearance_threshold": 0.25,
        "tracker_proximity_threshold": 0.5,
    }.items():
        setattr(args, key, value)

    exit_code = run_track(args)
    assert exit_code == 1
    assert load_called is False


def test_cli_exposes_reid_architecture_and_tracker_thresholds() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    add_track_subparser(subparsers)
    track_parser = subparsers.choices["track"]
    help_text = track_parser.format_help()
    assert "--tracker.reid.architecture" in help_text
    assert "--tracker.appearance_threshold" in help_text
    assert "--tracker.proximity_threshold" in help_text
    assert "--tracker.reid_ema_alpha" in help_text


def test_run_track_propagates_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    class _InterruptModel:
        @classmethod
        def from_pretrained(cls, **kwargs: object) -> object:
            raise KeyboardInterrupt

    monkeypatch.setattr("trackers.core.reid._lazy.load_reid_model_class", lambda: _InterruptModel)
    monkeypatch.setattr("trackers.scripts.track._init_model", lambda *a, **k: MagicMock(class_names=[]))

    args = argparse.Namespace(
        tracker="botsort",
        tracker_reid_enable=True,
        tracker_reid_model=None,
        tracker_reid_device="cpu",
        tracker_reid_architecture=None,
        source="video.mp4",
        detections=None,
        output=None,
        display=False,
        overwrite=False,
        model="rfdetr-nano",
        model_confidence=0.5,
        model_device="cpu",
        model_api_key=None,
        classes=None,
        track_ids=None,
        mot_output=None,
        show_boxes=True,
        show_masks=False,
        show_labels=False,
        show_ids=True,
        show_confidence=False,
        show_trajectories=False,
    )
    for key, value in {
        "tracker_lost_track_buffer": 30,
        "tracker_frame_rate": 30.0,
        "tracker_track_activation_threshold": 0.7,
        "tracker_minimum_consecutive_frames": 2,
        "tracker_minimum_iou_threshold_first_assoc": 0.2,
        "tracker_minimum_iou_threshold_second_assoc": 0.5,
        "tracker_minimum_iou_threshold_unconfirmed_assoc": 0.3,
        "tracker_high_conf_det_threshold": 0.6,
        "tracker_enable_cmc": True,
        "tracker_cmc_method": "sparseOptFlow",
        "tracker_cmc_downscale": 2,
        "tracker_instant_first_frame_activation": True,
        "tracker_reid_ema_alpha": 0.9,
        "tracker_appearance_threshold": 0.25,
        "tracker_proximity_threshold": 0.5,
    }.items():
        setattr(args, key, value)

    with pytest.raises(KeyboardInterrupt):
        run_track(args)
