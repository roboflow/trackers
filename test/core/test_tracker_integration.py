# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import supervision as sv

from trackers.core.base import BaseTracker
from trackers.eval import evaluate_mot_sequences
from trackers.io.mot import _load_mot_file, _mot_frame_to_detections, _MOTOutput

_TRACKER_IDS = ["sort", "bytetrack", "ocsort"]
_METRICS = ["CLEAR", "HOTA", "Identity"]
_TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_expected(dataset: str) -> dict[str, Any]:
    """Load TrackEval-derived reference metrics for a dataset."""
    path = _TEST_DATA_DIR / f"tracker_expected_{dataset}.json"
    with open(path) as f:
        return json.load(f)


def _run_tracker_on_flat_dataset(
    tracker_id: str,
    data_path: Path,
    output_dir: Path,
    seqmap_path: Path,
) -> None:
    """Run a tracker on GT-derived detections and save flat MOT output files."""
    import trackers as _trackers  # noqa: F401 - triggers registration

    info = BaseTracker._lookup_tracker(tracker_id)
    assert info is not None, f"Unknown tracker: {tracker_id}"

    output_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = data_path / "gt"

    with open(seqmap_path) as f:
        sequences = [
            line.strip() for line in f if line.strip() and line.strip() != "name"
        ]

    for seq_name in sequences:
        gt_file = gt_dir / f"{seq_name}.txt"
        if not gt_file.exists():
            continue

        gt_data = _load_mot_file(gt_file)
        max_frame = max(gt_data.keys()) if gt_data else 0

        tracker = info.tracker_class()
        mot_path = output_dir / f"{seq_name}.txt"

        with _MOTOutput(mot_path) as mot:
            for frame_idx in range(1, max_frame + 1):
                if frame_idx in gt_data:
                    detections = _mot_frame_to_detections(gt_data[frame_idx])
                else:
                    detections = sv.Detections.empty()

                tracked = tracker.update(detections)
                if tracked.tracker_id is not None:
                    mature = tracked[tracked.tracker_id != -1]
                    mot.write(frame_idx, mature)
                else:
                    mot.write(frame_idx, tracked)


@pytest.mark.integration
@pytest.mark.parametrize("tracker_id", _TRACKER_IDS)
@pytest.mark.parametrize(
    "dataset, fixture_name",
    [
        ("sportsmot", "sportsmot_flat_data"),
        ("dancetrack", "dancetrack_flat_data"),
    ],
)
def test_tracker_regression(
    tracker_id: str,
    dataset: str,
    fixture_name: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    data_path, _ = request.getfixturevalue(fixture_name)
    expected = _load_expected(dataset)[tracker_id]
    tracker_output_dir = tmp_path / "tracker_output"

    _run_tracker_on_flat_dataset(
        tracker_id=tracker_id,
        data_path=data_path,
        output_dir=tracker_output_dir,
        seqmap_path=data_path / "seqmap.txt",
    )

    result = evaluate_mot_sequences(
        gt_dir=data_path / "gt",
        tracker_dir=tracker_output_dir,
        seqmap=data_path / "seqmap.txt",
        metrics=_METRICS,
    )

    aggregate = result.aggregate
    assert aggregate.HOTA.HOTA * 100 == pytest.approx(expected["HOTA"], abs=0.001)
    assert aggregate.CLEAR.MOTA * 100 == pytest.approx(expected["MOTA"], abs=0.001)
    assert aggregate.Identity.IDF1 * 100 == pytest.approx(expected["IDF1"], abs=0.001)
    assert aggregate.CLEAR.IDSW == expected["IDSW"]
