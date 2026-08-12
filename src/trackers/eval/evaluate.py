# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np

from trackers.eval.clear import aggregate_clear_metrics, compute_clear_metrics
from trackers.eval.hota import aggregate_hota_metrics, compute_hota_metrics
from trackers.eval.identity import aggregate_identity_metrics, compute_identity_metrics
from trackers.eval.results import (
    AggregationIncompatibleError,
    BenchmarkCoverage,
    BenchmarkResult,
    CLEARMetrics,
    HOTAMetrics,
    IdentityMetrics,
    SequenceResult,
)
from trackers.io.mot import _prepare_mot_sequence, load_mot_file
from trackers.io.multicamera import (
    _prepare_multicamera_files,
    load_scene_camera_map,
)

logger = logging.getLogger(__name__)

SUPPORTED_METRICS = ["CLEAR", "HOTA", "Identity"]

# Official AI City 2024 headline fields; other HOTA components are undefined under
# an unweighted scene mean and serialize as JSON null on the aggregate row.
_SCENE_MEAN_HOTA_FIELDS = ("HOTA", "DetA", "AssA", "LocA")
_OFFICIAL_MULTICAMERA_SCENES = tuple(f"scene_{index:03d}" for index in range(61, 91))
_OFFICIAL_SCENE_CAMERA_MAP_SEMANTIC_SHA256 = "a3b7e28d5a5181cb7525c11bc06c828b66846c5a51ebef118ae1c4dc73e3ed00"
_SCENE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _require_tp_weighted_hota_arrays(sequence_metrics: list[dict[str, object]]) -> None:
    """Raise a typed error when per-alpha arrays needed for TP-weighted HOTA are missing."""
    for index, metrics in enumerate(sequence_metrics):
        if "HOTA_TP_array" not in metrics:
            raise AggregationIncompatibleError(
                "TP-weighted HOTA aggregation requires per-alpha arrays "
                f"(missing HOTA_TP_array in sequence_metrics[{index}]). "
                "Use BenchmarkResult.tp_weighted_hota() on a tp_weighted result, "
                "or keep aggregation='scene_mean' aggregates as-is."
            )


def _aggregate_hota_from_sequences(
    sequence_results: dict[str, SequenceResult] | list[SequenceResult],
) -> HOTAMetrics:
    """Internal TP-weighted HOTA aggregate used by MOT combine and `tp_weighted_hota`."""
    if isinstance(sequence_results, dict):
        sequences = list(sequence_results.values())
    else:
        sequences = list(sequence_results)

    hota_seq_metrics = [
        seq.HOTA.to_dict(include_arrays=True, arrays_as_list=False) for seq in sequences if seq.HOTA is not None
    ]
    if not hota_seq_metrics:
        raise ValueError("No HOTA results available to aggregate.")
    _require_tp_weighted_hota_arrays(hota_seq_metrics)
    return HOTAMetrics.from_dict(aggregate_hota_metrics(hota_seq_metrics))


def evaluate_mot_sequence(
    gt_path: str | Path,
    tracker_path: str | Path,
    metrics: list[str] | None = None,
    threshold: float = 0.5,
) -> SequenceResult:
    """Evaluate a single multi-object tracking result against ground truth. Computes
    standard multi-object tracking metrics (CLEAR MOT, HOTA, Identity) for one sequence
    by matching predicted tracks to ground-truth tracks using per-frame IoU
    (Intersection over Union).

    !!! tip "TrackEval parity"

        This evaluation code is intentionally designed to match the core matching logic
        and metric calculations of
        [TrackEval](https://github.com/JonathonLuiten/TrackEval).

    Args:
        gt_path: Path to the ground-truth MOT file.
        tracker_path: Path to the tracker MOT file.
        metrics: Metric families to compute. Supported values are
            `["CLEAR", "HOTA", "Identity"]`. Defaults to `["CLEAR"]`.
        threshold: IoU threshold for `CLEAR` and `Identity` matching. Defaults
            to `0.5`. `HOTA` evaluates across multiple thresholds internally.

    Returns:
        `SequenceResult` with `CLEAR`, `HOTA`, and/or `Identity` populated based
            on `metrics`.

    Raises:
        FileNotFoundError: If `gt_path` or `tracker_path` does not exist.
        ValueError: If an unsupported metric family is requested.

    Examples:
        >>> from trackers.eval import evaluate_mot_sequence  # doctest: +SKIP
        >>>
        >>> result = evaluate_mot_sequence(  # doctest: +SKIP
        ...     gt_path="data/gt/MOT17-02/gt.txt",
        ...     tracker_path="data/trackers/MOT17-02.txt",
        ...     metrics=["CLEAR", "HOTA", "Identity"],
        ... )
        >>>
        >>> result.CLEAR.MOTA  # doctest: +SKIP
        0.756
        >>>
        >>> result.table(columns=["MOTA", "HOTA", "IDF1", "IDSW"])  # doctest: +SKIP
        Sequence    MOTA    HOTA    IDF1  IDSW
        --------------------------------------
        MOT17-02  75.600  62.300  72.100    42
    """
    if metrics is None:
        metrics = ["CLEAR"]

    # Validate metrics
    for metric in metrics:
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric: {metric}. Supported metrics: {SUPPORTED_METRICS}")

    gt_path = Path(gt_path)
    tracker_path = Path(tracker_path)

    # Load data
    gt_data = load_mot_file(gt_path)
    tracker_data = load_mot_file(tracker_path)

    # Prepare sequence (compute IoU, remap IDs)
    seq_data = _prepare_mot_sequence(gt_data, tracker_data)

    # Compute metrics
    clear_metrics: CLEARMetrics | None = None
    hota_metrics: HOTAMetrics | None = None
    identity_metrics: IdentityMetrics | None = None

    if "CLEAR" in metrics:
        clear_metrics_dict = compute_clear_metrics(
            seq_data.gt_ids,
            seq_data.tracker_ids,
            seq_data.similarity_scores,
            threshold=threshold,
        )
        clear_metrics = CLEARMetrics.from_dict(clear_metrics_dict)

    if "HOTA" in metrics:
        hota_dict = compute_hota_metrics(
            seq_data.gt_ids,
            seq_data.tracker_ids,
            seq_data.similarity_scores,
        )
        hota_metrics = HOTAMetrics.from_dict(hota_dict)

    if "Identity" in metrics:
        identity_dict = compute_identity_metrics(
            seq_data.gt_ids,
            seq_data.tracker_ids,
            seq_data.similarity_scores,
            threshold=threshold,
        )
        identity_metrics = IdentityMetrics.from_dict(identity_dict)

    # Build result
    return SequenceResult(
        sequence=gt_path.stem,
        CLEAR=clear_metrics,
        HOTA=hota_metrics,
        Identity=identity_metrics,
    )


def evaluate_mot_sequences(
    gt_dir: str | Path,
    tracker_dir: str | Path,
    seqmap: str | Path | None = None,
    metrics: list[str] | None = None,
    threshold: float = 0.5,
) -> BenchmarkResult:
    """Evaluate multiple multi-object tracking results against ground truth. Computes
    standard multi-object tracking metrics (CLEAR MOT, HOTA, Identity) across one or
    more sequences by matching predicted tracks to ground-truth tracks using
    per-frame IoU (Intersection over Union). Returns both per-sequence and aggregated
    (combined) results.

    !!! tip "TrackEval parity"

        This evaluation code is intentionally designed to match the core matching logic
        and metric calculations of
        [TrackEval](https://github.com/JonathonLuiten/TrackEval).

    !!! tip "Supported dataset layouts"

        Both `gt_dir` and `tracker_dir` should point directly at the parent directory
        of the sequences. The evaluator auto-detects which layout you're using.

        === "MOT layout"

            ```
            gt_dir/
            ├── MOT17-02-FRCNN/
            │   └── gt/gt.txt
            ├── MOT17-04-FRCNN/
            │   └── gt/gt.txt
            └── MOT17-05-FRCNN/
                └── gt/gt.txt

            tracker_dir/
            ├── MOT17-02-FRCNN.txt
            ├── MOT17-04-FRCNN.txt
            └── MOT17-05-FRCNN.txt
            ```

        === "Flat layout"

            ```
            gt_dir/
            ├── MOT17-02.txt
            ├── MOT17-04.txt
            ├── MOT17-05.txt
            └── ...

            tracker_dir/
            ├── MOT17-02.txt
            ├── MOT17-04.txt
            ├── MOT17-05.txt
            └── ...
            ```

    Args:
        gt_dir: Directory containing ground-truth data. Should be the direct
            parent of the sequence folders (MOT layout) or sequence files
            (flat layout).
        tracker_dir: Directory containing tracker prediction files (`{seq}.txt`).
        seqmap: Optional sequence map. If provided, only those sequences are
            evaluated.
        metrics: Metric families to compute. Supported values are
            `["CLEAR", "HOTA", "Identity"]`. Defaults to `["CLEAR"]`.
        threshold: IoU threshold for `CLEAR` and `Identity`. Defaults to `0.5`.

    Returns:
        `BenchmarkResult` with per-sequence results and a `COMBINED` aggregate.

    Raises:
        FileNotFoundError: If `gt_dir` or `tracker_dir` does not exist.
        ValueError: If no sequences are found.

    Examples:
        Auto-detect layout and evaluate all sequences:

        >>> from trackers.eval import evaluate_mot_sequences  # doctest: +SKIP
        >>>
        >>> result = evaluate_mot_sequences(  # doctest: +SKIP
        ...     gt_dir="data/gt/",
        ...     tracker_dir="data/trackers/",
        ...     metrics=["CLEAR", "HOTA", "Identity"],
        ... )
        >>>
        >>> result.table(columns=["MOTA", "HOTA", "IDF1", "IDSW"])  # doctest: +SKIP
        Sequence     MOTA    HOTA    IDF1  IDSW
        ---------------------------------------
        sequence1  74.800  60.900  71.200    37
        sequence2  76.100  63.200  72.500    45
        ---------------------------------------
        COMBINED   75.450  62.050  71.850    82
    """
    if metrics is None:
        metrics = ["CLEAR"]

    gt_dir = Path(gt_dir)
    tracker_dir = Path(tracker_dir)

    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    if not tracker_dir.exists():
        raise FileNotFoundError(f"Tracker directory not found: {tracker_dir}")

    # Detect directory format
    data_format = _detect_format(gt_dir)
    logger.info("Detected format: %s", data_format)

    # Get sequence list
    if seqmap is not None:
        sequences = _parse_seqmap(seqmap)
    else:
        sequences = _discover_sequences(gt_dir, data_format)

    if not sequences:
        raise ValueError(f"No sequences found in {gt_dir}")

    logger.info("Evaluating %d sequences...", len(sequences))

    # Evaluate each sequence
    sequence_results: dict[str, SequenceResult] = {}

    for seq_name in sequences:
        gt_path, tracker_path = _get_paths(
            gt_dir=gt_dir,
            tracker_dir=tracker_dir,
            seq_name=seq_name,
            data_format=data_format,
        )

        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        if not tracker_path.exists():
            raise FileNotFoundError(f"Tracker file not found: {tracker_path}")

        seq_result = evaluate_mot_sequence(
            gt_path=gt_path,
            tracker_path=tracker_path,
            metrics=metrics,
            threshold=threshold,
        )
        # Fix sequence name (evaluate_mot_sequence uses file stem)
        sequence_results[seq_name] = SequenceResult(
            sequence=seq_name,
            CLEAR=seq_result.CLEAR,
            HOTA=seq_result.HOTA,
            Identity=seq_result.Identity,
        )

    # Compute aggregate metrics
    aggregate = _aggregate_metrics(sequence_results, metrics)

    return BenchmarkResult(
        sequences=sequence_results,
        aggregate=aggregate,
        aggregation="tp_weighted",
    )


def evaluate_multicamera_scene(
    scene: str,
    gt_path: str | Path,
    tracker_path: str | Path,
    camera_ids: list[int] | tuple[int, ...],
    *,
    file_format: Literal["aicity-2024"] = "aicity-2024",
    zero_distance: float = 2.0,
) -> SequenceResult:
    """Evaluate one multi-camera scene with AI City 2024 world-plane HOTA.

    Collapses all cameras in the scene into a single world-plane sequence:
    rows are filtered to ``camera_ids``, world coordinates are rounded to 3
    decimals (half-to-even), duplicate ``(frame_id, obj_id)`` rows keep the
    first occurrence, and ``frame_id`` is shifted by ``+1``. Similarity is
    ``max(0, 1 - euclidean_2d / zero_distance)``.

    Args:
        scene: Scene name used as `SequenceResult.sequence` (for example
            ``\"scene_061\"``). Must be passed explicitly because every scene's
            ground-truth file is named ``ground_truth.txt``.
        gt_path: Path to the scene ground-truth file.
        tracker_path: Path to the scene prediction file.
        camera_ids: Camera IDs belonging to this scene. Applied to both ground
            truth and predictions.
        file_format: On-disk format edition. Only ``\"aicity-2024\"`` is
            supported.
        zero_distance: Metres at which similarity reaches zero. Defaults to
            ``2.0``.

    Returns:
        `SequenceResult` with HOTA metrics for the scene.

    Raises:
        FileNotFoundError: If ``gt_path`` or ``tracker_path`` does not exist.
        ValueError: If inputs are malformed, empty after filtering, exceed
            protocol allocation bounds, or ``zero_distance`` is not finite and
            positive.
    """
    if not np.isfinite(zero_distance) or zero_distance <= 0:
        raise ValueError(f"zero_distance must be finite and > 0, got {zero_distance}")
    _validate_scene_name(scene)
    gt_path = Path(gt_path)
    tracker_path = Path(tracker_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker file not found: {tracker_path}")

    seq_data = _prepare_multicamera_files(
        gt_path,
        tracker_path,
        file_format=file_format,
        camera_ids=camera_ids,
        zero_distance=zero_distance,
    )
    hota_dict = compute_hota_metrics(
        seq_data.gt_ids,
        seq_data.tracker_ids,
        seq_data.similarity_scores,
    )
    return SequenceResult(
        sequence=scene,
        CLEAR=None,
        HOTA=HOTAMetrics.from_dict(hota_dict),
        Identity=None,
    )


def _scene_camera_map_sha256(camera_map: dict[str, list[int]]) -> str:
    """Hash normalized map semantics independently of JSON formatting."""
    encoded = json.dumps(camera_map, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_scene_name(scene_name: str) -> None:
    """Reject names that could escape benchmark input roots."""
    if not isinstance(scene_name, str) or not _SCENE_NAME_PATTERN.fullmatch(scene_name):
        raise ValueError(f"Invalid scene name {scene_name!r}; path separators and traversal are forbidden.")


def _resolve_scene_path(root: Path, relative_path: Path) -> Path:
    """Resolve a scene path and enforce containment within its supplied root."""
    resolved_root = root.resolve()
    resolved_path = (root / relative_path).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Scene path escapes supplied root: {relative_path}")
    return resolved_path


def evaluate_multicamera_scenes(
    gt_dir: str | Path,
    tracker_dir: str | Path,
    scene_camera_map: str | Path | dict[str, list[int]],
    *,
    file_format: Literal["aicity-2024"] = "aicity-2024",
    zero_distance: float = 2.0,
    scenes: list[str] | None = None,
    allow_partial: bool = False,
) -> BenchmarkResult:
    """Evaluate multiple multi-camera scenes and average HOTA unweighted.

    Ground truth is read from ``{gt_dir}/{scene}/ground_truth.txt`` and
    predictions from ``{tracker_dir}/{scene}.txt``. A missing prediction file
    raises because silently skipping a scene would inflate the unweighted mean.

    The aggregate row is the arithmetic mean of per-scene HOTA, DetA, AssA, and
    LocA (``aggregation=\"scene_mean\"``). DetRe/DetPr/AssRe/AssPr/OWTA and the
    TP/FN/FP counts are ``None`` (JSON ``null``) because they are undefined at
    scene-mean level by the protocol. `aggregate_hota_metrics` is not used;
    calling `BenchmarkResult.tp_weighted_hota` on the result raises
    `AggregationIncompatibleError`.

    Args:
        gt_dir: Parent directory of per-scene folders containing
            ``ground_truth.txt``.
        tracker_dir: Directory of ``{scene}.txt`` prediction files.
        scene_camera_map: Path to NVIDIA's ``scene_name_2_cam_id`` JSON, or an
            in-memory ``{scene_name: [camera_ids...]}`` mapping.
        file_format: On-disk format edition. Only ``\"aicity-2024\"`` is
            supported.
        zero_distance: Metres at which similarity reaches zero.
        scenes: Optional subset of scene names. Defaults to every scene in the
            camera map.
        allow_partial: Permit exploratory subsets or noncanonical camera maps.
            Partial results are labeled ``PARTIAL_SCENE_MEAN`` and serialize
            explicit expected/evaluated/missing coverage.

    Returns:
        `BenchmarkResult` with ``aggregation=\"scene_mean\"``.

    Raises:
        FileNotFoundError: If a required ground-truth or prediction file is
            missing.
        ValueError: If no scenes remain, or coverage is partial/noncanonical
            without ``allow_partial=True``.
    """
    if not np.isfinite(zero_distance) or zero_distance <= 0:
        raise ValueError(f"zero_distance must be finite and > 0, got {zero_distance}")
    gt_dir = Path(gt_dir)
    tracker_dir = Path(tracker_dir)

    if isinstance(scene_camera_map, (str, Path)):
        camera_map = load_scene_camera_map(scene_camera_map)
    else:
        camera_map = {name: list(ids) for name, ids in scene_camera_map.items()}

    map_sha256 = _scene_camera_map_sha256(camera_map)
    canonical_map = map_sha256 == _OFFICIAL_SCENE_CAMERA_MAP_SEMANTIC_SHA256
    scene_names = (
        list(scenes)
        if scenes is not None
        else list(_OFFICIAL_MULTICAMERA_SCENES)
        if canonical_map
        else list(camera_map.keys())
    )
    if not scene_names:
        raise ValueError("No scenes to evaluate.")
    if len(scene_names) != len(set(scene_names)):
        raise ValueError("Scene selection must not contain duplicates.")
    complete = (
        canonical_map
        and set(camera_map) == set(_OFFICIAL_MULTICAMERA_SCENES)
        and set(scene_names) == set(_OFFICIAL_MULTICAMERA_SCENES)
    )
    if not complete and not allow_partial:
        raise ValueError(
            "Official AI City Challenge 2024 results require the complete canonical scene-camera map and all "
            "scene_061..scene_090 scenes. Pass allow_partial=True for an explicitly labeled exploratory result."
        )

    sequence_results: dict[str, SequenceResult] = {}
    for scene_name in scene_names:
        _validate_scene_name(scene_name)
        if scene_name not in camera_map:
            raise ValueError(f"Scene {scene_name!r} is not present in the camera map.")

        gt_path = _resolve_scene_path(gt_dir, Path(scene_name) / "ground_truth.txt")
        tracker_path = _resolve_scene_path(tracker_dir, Path(f"{scene_name}.txt"))
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        if not tracker_path.exists():
            raise FileNotFoundError(f"Tracker file not found: {tracker_path}")

        sequence_results[scene_name] = evaluate_multicamera_scene(
            scene=scene_name,
            gt_path=gt_path,
            tracker_path=tracker_path,
            camera_ids=tuple(camera_map[scene_name]),
            file_format=file_format,
            zero_distance=zero_distance,
        )

    aggregate_label = "SCENE_MEAN" if complete else "PARTIAL_SCENE_MEAN"
    aggregate = _aggregate_scene_mean_hota(sequence_results, sequence=aggregate_label)
    evaluated_scenes = list(sequence_results)
    coverage = BenchmarkCoverage(
        benchmark="AI City Challenge 2024",
        split="test",
        protocol="MTMC_Tracking_2024",
        file_format=file_format,
        canonical_scene_camera_map_sha256=_OFFICIAL_SCENE_CAMERA_MAP_SEMANTIC_SHA256,
        scene_camera_map_sha256=map_sha256,
        expected_scenes=list(_OFFICIAL_MULTICAMERA_SCENES),
        evaluated_scenes=evaluated_scenes,
        missing_scenes=[scene for scene in _OFFICIAL_MULTICAMERA_SCENES if scene not in sequence_results],
        complete=complete,
    )
    return BenchmarkResult(
        sequences=sequence_results,
        aggregate=aggregate,
        aggregation="scene_mean",
        coverage=coverage,
    )


def _aggregate_scene_mean_hota(
    sequence_results: dict[str, SequenceResult],
    *,
    sequence: str,
) -> SequenceResult:
    """Unweighted mean of per-scene HOTA/DetA/AssA/LocA."""
    hota_results = [seq.HOTA for seq in sequence_results.values() if seq.HOTA is not None]
    if not hota_results:
        raise ValueError("No HOTA results available to average.")

    means = {
        field: float(np.mean([getattr(hota, field) for hota in hota_results])) for field in _SCENE_MEAN_HOTA_FIELDS
    }
    return SequenceResult(
        sequence=sequence,
        CLEAR=None,
        HOTA=HOTAMetrics(
            HOTA=means["HOTA"],
            DetA=means["DetA"],
            AssA=means["AssA"],
            DetRe=None,
            DetPr=None,
            AssRe=None,
            AssPr=None,
            LocA=means["LocA"],
            OWTA=None,
            HOTA_TP=None,
            HOTA_FN=None,
            HOTA_FP=None,
        ),
        Identity=None,
    )


def _detect_format(gt_dir: Path) -> Literal["flat", "mot"]:
    """Auto-detect directory format.

    Args:
        gt_dir: Ground truth directory (direct parent of sequences).

    Returns:
        "flat" if .txt files exist in gt_dir, "mot" if sequence
        subdirectories with gt/gt.txt exist.
    """
    # Check for flat format (*.txt files directly in gt_dir)
    txt_files = list(gt_dir.glob("*.txt"))
    if txt_files:
        return "flat"

    # Check for MOT format (*/gt/gt.txt pattern - sequences directly in gt_dir)
    mot_files = list(gt_dir.glob("*/gt/gt.txt"))
    if mot_files:
        return "mot"

    # Default to flat
    return "flat"


def _discover_sequences(
    gt_dir: Path,
    data_format: Literal["flat", "mot"],
) -> list[str]:
    """Discover sequence names from directory structure.

    Args:
        gt_dir: Ground truth directory (direct parent of sequences).
        data_format: Directory format.

    Returns:
        List of sequence names.
    """
    if data_format == "flat":
        return sorted(p.stem for p in gt_dir.glob("*.txt") if not p.name.startswith("."))
    else:
        # MOT format: gt_dir/{seq}/gt/gt.txt
        return sorted(p.parent.parent.name for p in gt_dir.glob("*/gt/gt.txt") if not p.name.startswith("."))


def _get_paths(
    gt_dir: Path,
    tracker_dir: Path,
    seq_name: str,
    data_format: Literal["flat", "mot"],
) -> tuple[Path, Path]:
    """Get GT and tracker file paths for a sequence.

    Args:
        gt_dir: Ground truth directory (direct parent of sequences).
        tracker_dir: Tracker directory (contains {seq}.txt files).
        seq_name: Sequence name.
        data_format: Directory format.

    Returns:
        Tuple of (gt_path, tracker_path).
    """
    if data_format == "flat":
        gt_path = gt_dir / f"{seq_name}.txt"
    else:
        # MOT format: gt_dir/{seq}/gt/gt.txt
        gt_path = gt_dir / seq_name / "gt" / "gt.txt"

    # Tracker files are always flat: tracker_dir/{seq}.txt
    tracker_path = tracker_dir / f"{seq_name}.txt"

    return gt_path, tracker_path


def _parse_seqmap(seqmap_path: str | Path) -> list[str]:
    """Parse a sequence map file to get list of sequence names."""
    seqmap_path = Path(seqmap_path)
    if not seqmap_path.exists():
        raise FileNotFoundError(f"Sequence map file not found: {seqmap_path}")

    sequences = []
    with open(seqmap_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, and header
            if not line or line.startswith("#") or line.lower() == "name":
                continue
            sequences.append(line)

    return sequences


def _aggregate_metrics(
    sequence_results: dict[str, SequenceResult],
    metrics: list[str],
) -> SequenceResult:
    """Aggregate metrics across sequences."""
    clear_agg: CLEARMetrics | None = None
    hota_agg: HOTAMetrics | None = None
    identity_agg: IdentityMetrics | None = None

    if "CLEAR" in metrics:
        clear_seq_metrics = [seq.CLEAR.to_dict() for seq in sequence_results.values() if seq.CLEAR is not None]
        if clear_seq_metrics:
            clear_agg = CLEARMetrics.from_dict(aggregate_clear_metrics(clear_seq_metrics))

    if "HOTA" in metrics:
        hota_present = any(seq.HOTA is not None for seq in sequence_results.values())
        if hota_present:
            hota_agg = _aggregate_hota_from_sequences(sequence_results)

    if "Identity" in metrics:
        identity_seq_metrics = [seq.Identity.to_dict() for seq in sequence_results.values() if seq.Identity is not None]
        if identity_seq_metrics:
            identity_agg = IdentityMetrics.from_dict(aggregate_identity_metrics(identity_seq_metrics))

    return SequenceResult(
        sequence="COMBINED",
        CLEAR=clear_agg,
        HOTA=hota_agg,
        Identity=identity_agg,
    )
