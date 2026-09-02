# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""AI City 2024 multicamera evaluation on the world plane."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from trackers.eval.hota import compute_hota_metrics
from trackers.eval.results import (
    HOTAMetrics,
    SequenceResult,
    _format_metric_rows,
    _sequence_row,
)
from trackers.io.multicamera import _prepare_multicamera_files, _validate_zero_distance, load_scene_camera_map

_SCENE_MEAN_FIELDS = ("HOTA", "DetA", "AssA", "LocA")
_AGGREGATE_LABEL = "SCENE_MEAN"


@dataclass
class SceneMeanHOTA:
    """Benchmark-level HOTA under the AI City 2024 unweighted scene mean.

    The protocol averages four per-scene fields across scenes and defines no
    benchmark-level recall, precision, or detection counts, so those fields
    are absent rather than null.

    Attributes:
        HOTA: Mean per-scene Higher Order Tracking Accuracy.
        DetA: Mean per-scene detection accuracy.
        AssA: Mean per-scene association accuracy.
        LocA: Mean per-scene localization accuracy.
    """

    HOTA: float
    DetA: float
    AssA: float
    LocA: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneMeanHOTA:
        """Create `SceneMeanHOTA` from a dictionary.

        Args:
            data: Dictionary with the four scene-mean fields.

        Returns:
            `SceneMeanHOTA` instance.
        """
        return cls(**{field: float(data[field]) for field in _SCENE_MEAN_FIELDS})

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with the four scene-mean fields.
        """
        return {field: float(getattr(self, field)) for field in _SCENE_MEAN_FIELDS}


@dataclass
class MulticameraBenchmarkResult:
    """Result for an AI City 2024 multicamera benchmark run.

    Attributes:
        scenes: Per-scene results, each holding full HOTA metrics.
        aggregate: Unweighted mean of per-scene HOTA, DetA, AssA, and LocA.

    Examples:
        >>> from trackers.eval import MulticameraBenchmarkResult  # doctest: +SKIP
        >>>
        >>> result = MulticameraBenchmarkResult.load("results.json")  # doctest: +SKIP
        >>> result.aggregate.HOTA  # doctest: +SKIP
        0.7887
    """

    scenes: dict[str, SequenceResult]
    aggregate: SceneMeanHOTA

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MulticameraBenchmarkResult:
        """Create `MulticameraBenchmarkResult` from a dictionary.

        Args:
            data: Dictionary with per-scene results and the scene mean.

        Returns:
            `MulticameraBenchmarkResult` instance.

        Raises:
            ValueError: If ``data`` is a MOT benchmark payload.
        """
        if "sequences" in data:
            raise ValueError("Payload holds MOT sequences, not multicamera scenes; use BenchmarkResult.from_dict().")
        return cls(
            scenes={name: SequenceResult.from_dict(scene) for name, scene in data["scenes"].items()},
            aggregate=SceneMeanHOTA.from_dict(data["aggregate"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with per-scene results and the scene mean.
        """
        return {
            "scenes": {name: scene.to_dict() for name, scene in self.scenes.items()},
            "aggregate": self.aggregate.to_dict(),
        }

    def json(self, indent: int = 2) -> str:
        """Serialize to JSON string.

        Args:
            indent: Indentation level for formatting. Defaults to `2`.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def table(self) -> str:
        """Format per-scene rows and the scene mean as a table.

        Returns:
            Formatted table string with the four official fields.
        """
        rows: list[tuple[str, dict[str, float | int]]] = [
            (name, _sequence_row(self.scenes[name], _SCENE_MEAN_FIELDS)) for name in sorted(self.scenes)
        ]
        aggregate: dict[str, float | int] = dict(self.aggregate.to_dict())
        rows.append((_AGGREGATE_LABEL, aggregate))
        return _format_metric_rows(rows, _SCENE_MEAN_FIELDS, label_header="Scene", rule_before_last=True)

    def save(self, path: str | Path) -> None:
        """Save to a JSON file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.json())

    @classmethod
    def load(cls, path: str | Path) -> MulticameraBenchmarkResult:
        """Load from a JSON file.

        Args:
            path: Source file path.

        Returns:
            `MulticameraBenchmarkResult` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file holds a MOT benchmark payload.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        return cls.from_dict(json.loads(path.read_text()))


def evaluate_multicamera_scene(
    gt_path: str | Path,
    tracker_path: str | Path,
    camera_ids: Sequence[int],
    *,
    scene: str,
    file_format: Literal["aicity-2024"] = "aicity-2024",
    zero_distance: float = 2.0,
) -> SequenceResult:
    """Evaluate one multicamera scene with AI City 2024 world-plane HOTA.

    Collapses all cameras in the scene into a single world-plane sequence:
    rows are filtered to ``camera_ids``, world coordinates are rounded to 3
    decimals (half-to-even), duplicate ``(frame_id, obj_id)`` rows keep the
    first occurrence, and ``frame_id`` is shifted by ``+1``. Similarity is
    ``max(0, 1 - euclidean_2d / zero_distance)``.

    A prediction file that is empty, or has no rows left after camera
    filtering, scores HOTA, DetA, and AssA as zero; LocA is ``1.0`` by
    TrackEval convention. The same condition on ground truth raises, whereas
    NVIDIA's evaluator scores it silently.

    Args:
        gt_path: Path to the scene ground-truth file.
        tracker_path: Path to the scene prediction file.
        camera_ids: Camera IDs belonging to this scene. Applied to both ground
            truth and predictions.
        scene: Scene name used as `SequenceResult.sequence` (for example
            ``"scene_061"``). Required because every scene's ground-truth file
            is named ``ground_truth.txt``.
        file_format: On-disk format edition. Only ``"aicity-2024"`` is
            supported.
        zero_distance: Meters at which similarity reaches zero. Defaults to
            ``2.0``.

    Returns:
        `SequenceResult` with HOTA metrics for the scene.

    Raises:
        FileNotFoundError: If ``gt_path`` or ``tracker_path`` does not exist.
        TypeError: If ``camera_ids`` contains a non-integer.
        ValueError: If ``zero_distance`` is not finite and positive, a file
            is malformed, or ground truth is empty after camera filtering.

    Examples:
        >>> from trackers.eval import evaluate_multicamera_scene  # doctest: +SKIP
        >>>
        >>> result = evaluate_multicamera_scene(  # doctest: +SKIP
        ...     gt_path="tests/data/multicamera/gt/scene_a/ground_truth.txt",
        ...     tracker_path="tests/data/multicamera/pred/scene_a.txt",
        ...     camera_ids=[1, 2],
        ...     scene="scene_a",
        ... )
        >>> result.HOTA.HOTA  # doctest: +SKIP
        0.5773
    """
    _validate_zero_distance(zero_distance)
    sequence_data = _prepare_multicamera_files(
        gt_path,
        tracker_path,
        file_format=file_format,
        camera_ids=camera_ids,
        zero_distance=zero_distance,
    )
    hota = compute_hota_metrics(
        sequence_data.gt_ids,
        sequence_data.tracker_ids,
        sequence_data.similarity_scores,
    )
    return SequenceResult(sequence=scene, HOTA=HOTAMetrics.from_dict(hota))


def evaluate_multicamera_scenes(
    gt_dir: str | Path,
    tracker_dir: str | Path,
    scene_camera_map: str | Path | Mapping[str, Sequence[int]],
    *,
    file_format: Literal["aicity-2024"] = "aicity-2024",
    zero_distance: float = 2.0,
    scenes: Sequence[str] | None = None,
) -> MulticameraBenchmarkResult:
    """Evaluate multiple multicamera scenes and average HOTA unweighted.

    Ground truth is read from ``{gt_dir}/{scene}/ground_truth.txt`` and
    predictions from ``{tracker_dir}/{scene}.txt``. A missing prediction file
    raises because silently skipping a scene would inflate the unweighted mean.
    An existing but empty prediction file scores the scene as described in
    `evaluate_multicamera_scene`.

    Args:
        gt_dir: Parent directory of per-scene folders containing
            ``ground_truth.txt``.
        tracker_dir: Directory of ``{scene}.txt`` prediction files.
        scene_camera_map: Path to NVIDIA's ``scene_name_2_cam_id`` JSON, or an
            in-memory ``{scene_name: [camera_ids...]}`` mapping.
        file_format: On-disk format edition. Only ``"aicity-2024"`` is
            supported.
        zero_distance: Meters at which similarity reaches zero.
        scenes: Optional subset of scene names, without duplicates. Defaults
            to every scene in the camera map; a subset changes the headline.

    Returns:
        `MulticameraBenchmarkResult` with per-scene HOTA and the scene mean.

    Raises:
        FileNotFoundError: If a required ground-truth or prediction file is
            missing.
        TypeError: If a camera ID in the map is not an integer.
        ValueError: If ``zero_distance`` is not finite and positive, no scenes
            remain, ``scenes`` contains duplicates, or a selected scene is
            absent from the camera map.

    Examples:
        >>> from trackers.eval import evaluate_multicamera_scenes  # doctest: +SKIP
        >>>
        >>> result = evaluate_multicamera_scenes(  # doctest: +SKIP
        ...     gt_dir="tests/data/multicamera/gt",
        ...     tracker_dir="tests/data/multicamera/pred",
        ...     scene_camera_map="tests/data/multicamera/scene_camera_map.json",
        ... )
        >>>
        >>> print(result.table())  # doctest: +SKIP
        Scene                            HOTA     DetA     AssA     LocA
        ----------------------------------------------------------------
        scene_a                        57.735   42.857   77.778  100.000
        scene_b                       100.000  100.000  100.000  100.000
        ----------------------------------------------------------------
        SCENE_MEAN                     78.868   71.429   88.889  100.000
    """
    _validate_zero_distance(zero_distance)
    gt_dir = Path(gt_dir)
    tracker_dir = Path(tracker_dir)

    if isinstance(scene_camera_map, (str, Path)):
        camera_map = load_scene_camera_map(scene_camera_map)
    else:
        camera_map = {name: list(ids) for name, ids in scene_camera_map.items()}

    scene_names = list(scenes) if scenes is not None else list(camera_map)
    if not scene_names:
        raise ValueError("No scenes to evaluate.")
    if len(scene_names) != len(set(scene_names)):
        raise ValueError("Scene selection must not contain duplicates.")

    scene_results: dict[str, SequenceResult] = {}
    for scene_name in scene_names:
        if scene_name not in camera_map:
            raise ValueError(f"Scene {scene_name!r} is not present in the camera map.")

        scene_results[scene_name] = evaluate_multicamera_scene(
            gt_path=gt_dir / scene_name / "ground_truth.txt",
            tracker_path=tracker_dir / f"{scene_name}.txt",
            camera_ids=tuple(camera_map[scene_name]),
            scene=scene_name,
            file_format=file_format,
            zero_distance=zero_distance,
        )

    return MulticameraBenchmarkResult(scenes=scene_results, aggregate=_scene_mean(scene_results))


def _scene_mean(scene_results: dict[str, SequenceResult]) -> SceneMeanHOTA:
    """Average the four official HOTA fields across scenes without weighting."""
    hota_results = [scene.HOTA for scene in scene_results.values()]
    return SceneMeanHOTA(
        **{field: float(np.mean([getattr(hota, field) for hota in hota_results])) for field in _SCENE_MEAN_FIELDS}
    )
