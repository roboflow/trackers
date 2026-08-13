#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Check multicamera parity against NVIDIA's evaluator.

Supply ``MTMC_Tracking_2024/eval`` from PhysicalAI-SmartSpaces revision ``1eebcf0f74a510994fe4c886f4fa77fbc6724ea8``.
The fixture always runs; ``--sample-dir`` also checks NVIDIA's full sample.

Run with NVIDIA's pandas dependency available::

    uv run --with pandas python scripts/verify_multicamera_eval.py \
        --nvidia-eval-dir /path/to/MTMC_Tracking_2024/eval
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
FIXTURE_DIR = REPO_ROOT / "tests" / "data" / "multicamera"
HEADLINE_FIELDS = ("HOTA", "DetA", "AssA", "LocA")
REL_TOLERANCE = 1e-4
ABS_TOLERANCE = 1e-4
AICITY_COLUMNS = 9
ParityPaths = tuple[Path, Path, Path, Path, Path]


def split_multicamera_file_by_scene(
    path: Path,
    scene_camera_map: Mapping[str, Sequence[int]],
    output_dir: Path,
    *,
    ground_truth_layout: bool = False,
) -> None:
    """Split a monolithic sample by camera while preserving file order."""
    camera_to_scenes: dict[int, list[str]] = {}
    handles: dict[str, TextIO] = {}
    try:
        for scene_name, camera_ids in scene_camera_map.items():
            if not scene_name or Path(scene_name).name != scene_name:
                raise ValueError(f"Invalid scene name: {scene_name!r}")
            destination = (
                output_dir / scene_name / "ground_truth.txt"
                if ground_truth_layout
                else output_dir / f"{scene_name}.txt"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            handles[scene_name] = destination.open("w", encoding="utf-8")
            for camera_id in camera_ids:
                camera_to_scenes.setdefault(camera_id, []).append(scene_name)

        with path.open("r", encoding="utf-8", newline="") as source:
            for line_number, raw_line in enumerate(source, start=1):
                line = raw_line.rstrip("\r\n")
                columns = line.split()
                if len(columns) != AICITY_COLUMNS or not columns[0].isdecimal():
                    raise ValueError(f"Malformed AI City row at {path}:{line_number}")
                camera_id = int(columns[0])
                for scene_name in camera_to_scenes.get(camera_id, ()):
                    handles[scene_name].write(line + "\n")
    finally:
        for handle in handles.values():
            handle.close()


def _load_nvidia_module(eval_dir: Path) -> Any:
    main_path = eval_dir / "main.py"
    if not main_path.exists():
        raise SystemExit(f"NVIDIA evaluator missing: {main_path}")
    # NVIDIA imports ``trackeval`` and ``utils`` relative to the eval directory.
    sys.path.insert(0, str(eval_dir))
    spec = importlib.util.spec_from_file_location("nvidia_mtmc_main", main_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to import NVIDIA evaluator from {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_nvidia_oracle(
    eval_dir: Path,
    *,
    prediction_file: Path,
    ground_truth_file: Path,
    scene_map_file: Path,
    num_cores: int = 1,
) -> dict[str, dict[str, float]]:
    """Run NVIDIA's evaluator and return per-scene and scene-mean scores."""
    module = _load_nvidia_module(eval_dir)
    with tempfile.TemporaryDirectory(prefix="nvidia-mtmc-") as tmp:
        result = module.computes_mot_metrics(
            str(prediction_file),
            str(ground_truth_file),
            tmp,
            num_cores,
            str(scene_map_file),
        )

    scores: dict[str, dict[str, float]] = {}
    data = result[0]["MotChallenge3DLocation"]["data"]
    for scene_name, scene_result in data.items():
        if scene_name == "COMBINED_SEQ":
            continue
        hota = scene_result["pedestrian"]["HOTA"]
        scores[scene_name] = {metric: float(np.mean(hota[metric])) for metric in HEADLINE_FIELDS}
    scores["SCENE_MEAN"] = {
        metric: float(np.mean([scene[metric] for scene in scores.values()])) for metric in HEADLINE_FIELDS
    }
    return scores


def _compare_results(
    label: str,
    result: Any,
    nvidia_scores: dict[str, dict[str, float]],
) -> None:
    scores: dict[str, dict[str, float]] = {}
    for scene_name, scene in result.scenes.items():
        if scene.HOTA is None:
            raise SystemExit(f"{label}: trackers returned no HOTA result for {scene_name}")
        scores[scene_name] = {metric: float(getattr(scene.HOTA, metric)) for metric in HEADLINE_FIELDS}
    scores["SCENE_MEAN"] = {metric: float(getattr(result.aggregate, metric)) for metric in HEADLINE_FIELDS}
    if scores.keys() != nvidia_scores.keys():
        raise SystemExit(f"{label}: scene mismatch; trackers={sorted(scores)}, NVIDIA={sorted(nvidia_scores)}")
    for scene_name in sorted(scores):
        for metric in HEADLINE_FIELDS:
            got, expected = scores[scene_name][metric], nvidia_scores[scene_name][metric]
            if abs(got - expected) > max(ABS_TOLERANCE, REL_TOLERANCE * abs(expected)):
                raise SystemExit(f"{label} {scene_name}.{metric}: trackers={got!r}, NVIDIA={expected!r}")
        summary = ", ".join(f"{metric}={scores[scene_name][metric]:.6f}" for metric in HEADLINE_FIELDS)
        print(f"  {scene_name}: {summary}")
    print(f"{label}: parity OK")


def _run_parity(
    label: str,
    eval_dir: Path,
    paths: ParityPaths,
    num_cores: int,
) -> None:
    from trackers.eval import evaluate_multicamera_scenes

    gt_dir, tracker_dir, ground_truth_file, prediction_file, scene_map_file = paths
    trackers_result = evaluate_multicamera_scenes(
        gt_dir=gt_dir,
        tracker_dir=tracker_dir,
        scene_camera_map=scene_map_file,
    )
    nvidia_result = run_nvidia_oracle(
        eval_dir,
        prediction_file=prediction_file,
        ground_truth_file=ground_truth_file,
        scene_map_file=scene_map_file,
        num_cores=num_cores,
    )
    _compare_results(label, trackers_result, nvidia_result)


def _merge_scenes(sources: Sequence[Path], destination: Path) -> Path:
    """Concatenate per-scene files into the monolithic layout NVIDIA expects."""
    rows = []
    for source in sources:
        rows.extend(line for line in source.read_text().splitlines() if line.strip() and not line.startswith("#"))
    destination.write_text("\n".join(rows) + "\n")
    return destination


def _run_fixture(eval_dir: Path, *, num_cores: int) -> None:
    from trackers.io.multicamera import load_scene_camera_map

    scene_map_file = FIXTURE_DIR / "scene_camera_map.json"
    scenes = load_scene_camera_map(scene_map_file)
    with tempfile.TemporaryDirectory(prefix="multicamera-parity-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        ground_truth_file = _merge_scenes(
            [FIXTURE_DIR / "gt" / scene / "ground_truth.txt" for scene in scenes],
            temporary_root / "ground_truth.txt",
        )
        prediction_file = _merge_scenes(
            [FIXTURE_DIR / "pred" / f"{scene}.txt" for scene in scenes],
            temporary_root / "pred.txt",
        )
        _run_parity(
            "fixture",
            eval_dir,
            (FIXTURE_DIR / "gt", FIXTURE_DIR / "pred", ground_truth_file, prediction_file, scene_map_file),
            num_cores,
        )


def _run_full_sample(eval_dir: Path, sample_dir: Path, *, num_cores: int) -> None:
    from trackers.io.multicamera import load_scene_camera_map

    prediction_file = sample_dir / "pred.txt"
    ground_truth_file = sample_dir / "ground_truth_test_full.txt"
    scene_map_file = sample_dir / "scene_name_2_cam_id_full.json"
    for path in (prediction_file, ground_truth_file, scene_map_file):
        if not path.is_file():
            raise SystemExit(f"Full sample input missing: {path}")

    scene_camera_map = load_scene_camera_map(scene_map_file)
    with tempfile.TemporaryDirectory(prefix="multicamera-parity-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        gt_dir = temporary_root / "gt"
        tracker_dir = temporary_root / "pred"
        split_multicamera_file_by_scene(ground_truth_file, scene_camera_map, gt_dir, ground_truth_layout=True)
        split_multicamera_file_by_scene(prediction_file, scene_camera_map, tracker_dir)
        _run_parity(
            "full sample",
            eval_dir,
            (gt_dir, tracker_dir, ground_truth_file, prediction_file, scene_map_file),
            num_cores,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare trackers with NVIDIA's pinned evaluator; always checks the committed fixture."
    )
    parser.add_argument(
        "--nvidia-eval-dir",
        type=Path,
        required=True,
        help="Path to NVIDIA's MTMC_Tracking_2024/eval directory at the pinned revision.",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        help="Also check pred.txt, ground_truth_test_full.txt, and scene_name_2_cam_id_full.json in this directory.",
    )
    parser.add_argument("--num-cores", type=int, default=1, help="NVIDIA TrackEval worker count (default: 1).")
    args = parser.parse_args()
    if args.num_cores < 1:
        parser.error("--num-cores must be at least 1")
    if not (args.nvidia_eval_dir / "main.py").is_file():
        parser.error(f"--nvidia-eval-dir does not contain main.py: {args.nvidia_eval_dir}")

    print(f"NVIDIA evaluator: {args.nvidia_eval_dir}")
    _run_fixture(args.nvidia_eval_dir, num_cores=args.num_cores)
    if args.sample_dir is not None:
        _run_full_sample(args.nvidia_eval_dir, args.sample_dir, num_cores=args.num_cores)


if __name__ == "__main__":
    main()
