# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Run McByte on complete benchmark test sets and save MOTChallenge-format results.

Usage
-----

::

    trackers mcbyte --dataset=[mot17,soccernet] --device=cuda

Options come from the :func:`benchmark_command` signature, the way every
``trackers`` subcommand is built, so the shared conventions hold here too:
``--cmc-downscale`` and ``--cmc_downscale`` are the same option, each boolean has
a ``--no_`` half, and ``--config run.yaml`` supplies the same keys from a file.
Datasets are selected as one list; the repeated spelling is ``--dataset+ mot17
--dataset+ soccernet``, since a bare repeated ``--dataset`` overwrites.

Dataset roots are supplied through ``--datasets``, keyed by the same names
``--dataset`` selects. A config file is the readable spelling::

    # run.yaml
    dataset: [mot17]
    datasets:
      mot17:
        detection_root: /data/detections/MOT17/test
        image_root: /data/datasets/MOT17/test

    trackers mcbyte --config run.yaml

Supported datasets
------------------

- MOT17
- DanceTrack
- SportsMOT
- SoccerNet-tracking

Expected directory layout
-------------------------

NOTE: every dataset needs a ``detection_root`` and an ``image_root`` matching
your files on a disk or server. Neither has a built-in value, so both are
supplied per run through ``--datasets`` or a ``--config`` file. See the
instructions below for more details.

The layout below is what those two roots are expected to point at.

Detection files
^^^^^^^^^^^^^^^

MOT17, DanceTrack and SportsMOT:

One detection file per sequence, stored in separate dataset directories, e.g.

    detections/MOT17/test/MOT17-01.txt
    detections/dancetrack/test/dancetrack0003.txt
    detections/sportsmot/test/v_-9kabh1K8UA_c008.txt

SoccerNet-tracking:

One detection file per sequence, following the original SoccerNet naming
convention, e.g.

    detections/SoccerNet_tracking_2022_test_set_dets/
        SNMOT-116__det.txt

Frame directories
^^^^^^^^^^^^^^^^^

For all datasets, the image root contains one directory per sequence, each
containing an ``img1`` subdirectory with the sequence frames, e.g.

    datasets/dancetrack/test/
        dancetrack0003/img1/
        dancetrack0009/img1/
        ...

Detection file formats
----------------------

MOT17, DanceTrack and SportsMOT detections are expected in XYXY format:

    frame,x1,y1,x2,y2,confidence

Example:

    215,1433.7001,480.6000,1479.6001,579.1500,0.02

SoccerNet-tracking detections are expected in MOT format
(as in the original ground truths):

    frame,id,left,top,width,height,confidence,...

Example:

    170,-1,1392,472,37,106,1,-1,-1,-1

The input identity column is ignored because tracker identities are produced by
McByte.

Each sequence is processed independently using a fresh McByte tracker. If a
sequence fails, the error is recorded and benchmark execution continues with
the remaining sequences.
"""

from __future__ import annotations

import gc
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

# ``CMCMethod`` is the only non-stdlib name needed while the module is imported:
# jsonargparse resolves ``benchmark_command``'s annotations when it builds the
# parser, so a type reaching the signature cannot be deferred. It costs nothing
# beyond the ``trackers`` package itself, which importing this module implies.
from trackers.utils.cmc import CMCMethod

if TYPE_CHECKING:
    import numpy as np
    import supervision as sv

    from trackers.core.mcbyte.tracker import McByteTracker

DetectionFileFormat = Literal["xyxy", "mot"]
# Selectable dataset names. Kept in step with the ``DATASETS`` keys below, which
# the type annotation cannot be derived from: jsonargparse turns this alias into
# the ``--dataset`` accept list, so it has to be a literal type.
DatasetName = Literal["mot17", "dancetrack", "sportsmot", "soccernet"]
DEFAULT_OUTPUT_ROOT = Path("outputs/mcbyte_benchmarks")

MOT17_EXISTING = ("01", "03", "06", "07", "08", "12", "14")
MOT17_MISSING = ("02", "04", "05", "09", "10", "11", "13")
MOT17_SUFFIXES = ("FRCNN", "SDP", "DPM")


@dataclass(frozen=True)
class DetectionRecord:
    """One detection parsed from an input file."""

    xyxy: np.ndarray
    confidence: float


@dataclass(frozen=True)
class DatasetPaths:
    """Where one dataset's files live, supplied per run rather than per checkout.

    This is the whole of what ``--datasets`` accepts. Everything else about a
    dataset — how its detection files are parsed, how its sequences are named —
    stays in :data:`DATASETS`, because it is a property of the benchmark rather
    than of the machine the benchmark runs on.

    Attributes:
        detection_root: Directory holding one detection ``.txt`` per sequence.
        image_root: Directory holding one frame directory per sequence.
    """

    detection_root: Path
    image_root: Path


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-specific paths and parsing behavior.

    Each dataset defines where detections and frames are located,
    how detections should be parsed,
    and any dataset-specific conventions.

    The two roots default to ``None`` because no value can be right for every
    machine; :func:`resolve_datasets` fills them from what the run supplied.
    """

    name: str
    detection_format: DetectionFileFormat
    detection_root: Path | None = None
    image_root: Path | None = None
    frame_rate: float = 30.0
    mot17_layout: bool = False
    soccernet_filename: bool = False
    confidence_override: float | None = None


DATASETS: dict[DatasetName, DatasetConfig] = {
    "mot17": DatasetConfig(
        name="mot17",
        detection_format="xyxy",
        mot17_layout=True,
    ),
    "dancetrack": DatasetConfig(
        name="dancetrack",
        detection_format="xyxy",
    ),
    "sportsmot": DatasetConfig(
        name="sportsmot",
        detection_format="xyxy",
    ),
    "soccernet": DatasetConfig(
        name="soccernet",
        detection_format="mot",
        soccernet_filename=True,
        # Preserves the old SoccerNet runner. Set to None to read column 7.
        confidence_override=1.0,
    ),
}


def _unknown_datasets_error(datasets: dict[str, DatasetPaths] | None) -> str:
    """Return the problem with a ``--datasets`` mapping's keys, if any.

    A key naming no known dataset would otherwise be accepted and then quietly
    ignored, which is the worst way for a mistyped root to fail: the run starts,
    and only the "please configure" error much later hints that the paths never
    landed. jsonargparse does not validate mapping keys against a ``Literal``,
    so the check is made here.

    Args:
        datasets: Roots per dataset name, as supplied by the run.

    Returns:
        One error message, or an empty string when every key names a dataset.

    Examples:
        >>> _unknown_datasets_error(None)
        ''
        >>> _unknown_datasets_error({"mot18": None})
        "Unknown --datasets entry 'mot18'. Known datasets: mot17, dancetrack, sportsmot, soccernet."
    """
    unknown = [name for name in datasets or {} if name not in DATASETS]
    if not unknown:
        return ""
    entries = ", ".join(repr(name) for name in unknown)
    noun = "entry" if len(unknown) == 1 else "entries"
    return f"Unknown --datasets {noun} {entries}. Known datasets: {', '.join(DATASETS)}."


def resolve_datasets(datasets: dict[str, DatasetPaths] | None = None) -> dict[DatasetName, DatasetConfig]:
    """Return the dataset table with the run's roots merged over it.

    Only ``detection_root`` and ``image_root`` are taken from ``datasets``;
    every other field keeps the built-in value, so detection parsing stays
    driven by :data:`DATASETS` and cannot be reconfigured from a config file.

    Values are read by attribute rather than unpacked, because the same mapping
    arrives as :class:`DatasetPaths` from a Python caller and as an equivalent
    namespace from a parser that has not instantiated its classes yet.

    Args:
        datasets: Roots per dataset name. Names absent from the mapping, and
            ``None`` for the mapping itself, leave the built-in entry as it is.

    Returns:
        A copy of :data:`DATASETS`, with the supplied roots applied.

    Raises:
        KeyError: If a key names no known dataset. Reach for
            :func:`_unknown_datasets_error` first to report that as CLI usage.

    Examples:
        >>> merged = resolve_datasets({"mot17": DatasetPaths(Path("dets"), Path("frames"))})
        >>> merged["mot17"].detection_root == Path("dets")
        True
        >>> merged["mot17"].mot17_layout
        True
        >>> resolve_datasets()["mot17"].detection_root is None
        True
    """
    resolved = dict(DATASETS)
    for name, paths in (datasets or {}).items():
        if name not in resolved:
            raise KeyError(_unknown_datasets_error({name: paths}))
        resolved[name] = replace(  # type: ignore[index]
            resolved[name],  # type: ignore[index]
            detection_root=Path(paths.detection_root),
            image_root=Path(paths.image_root),
        )
    return resolved


def _runtime_error(device: str, cmc_downscale: int) -> str:
    """Return the first problem with the global runtime arguments, if any.

    Args:
        device: Requested torch device.
        cmc_downscale: Requested camera-motion-compensation downscale factor.

    Returns:
        One error message, or an empty string when the arguments are usable.

    Examples:
        >>> _runtime_error("cpu", 0)
        'cmc_downscale must be positive.'
        >>> _runtime_error("cpu", 6)
        ''
    """
    if cmc_downscale <= 0:
        return "cmc_downscale must be positive."
    if not device.startswith(("cuda", "mps")):
        return ""

    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        return "CUDA was requested, but CUDA PyTorch is unavailable."
    if device.startswith("mps") and not torch.backends.mps.is_available():
        return "MPS was requested, but MPS PyTorch is unavailable."
    return ""


def configure_logging(run_root: Path) -> logging.Logger:
    """Log to both the terminal and the run directory."""
    logger = logging.getLogger("mcbyte_benchmarks")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(run_root / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def sequence_name(detection_file: Path, config: DatasetConfig) -> str:
    """Resolve a sequence name from a detection filename."""
    if config.soccernet_filename:
        return detection_file.name.split("__", maxsplit=1)[0]
    return detection_file.stem


def image_directory(sequence: str, config: DatasetConfig) -> Path:
    """Resolve the sequence frame directory."""
    if config.image_root is None:
        raise ValueError(f"DATASETS['{config.name}'] has no image_root; supply one with --datasets.")
    directory_name = f"{sequence}-FRCNN" if config.mot17_layout else sequence
    return config.image_root / directory_name / "img1"


def read_detection_file(
    detection_file: Path,
    config: DatasetConfig,
) -> dict[int, list[DetectionRecord]]:
    """Read detections grouped by frame.

    XYXY format: frame,x1,y1,x2,y2,confidence
    MOT format:  frame,id,left,top,width,height,confidence,...
    """
    import numpy as np

    grouped: dict[int, list[DetectionRecord]] = defaultdict(list)

    with detection_file.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            values = line.split(",")

            try:
                if config.detection_format == "xyxy":
                    if len(values) < 6:
                        raise ValueError("expected at least 6 columns")
                    frame_number = int(float(values[0]))
                    x1, y1, x2, y2 = map(float, values[1:5])
                    confidence = float(values[5])
                else:
                    if len(values) < 7:
                        raise ValueError("expected at least 7 columns")
                    frame_number = int(float(values[0]))
                    left, top, width, height = map(float, values[2:6])
                    x1, y1 = left, top
                    x2, y2 = left + width, top + height
                    confidence = (
                        config.confidence_override if config.confidence_override is not None else float(values[6])
                    )
            except ValueError as exc:
                raise ValueError(f"Invalid line {line_number} in {detection_file}: {line}") from exc

            if frame_number <= 0:
                raise ValueError(f"Non-positive frame number on line {line_number}.")
            if x2 <= x1 or y2 <= y1:
                continue

            grouped[frame_number].append(
                DetectionRecord(
                    xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                    confidence=float(confidence),
                )
            )

    return dict(grouped)


def build_detections(records: list[DetectionRecord]) -> sv.Detections:
    """Create Supervision detections from parsed records."""
    import numpy as np
    import supervision as sv

    if not records:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.stack([record.xyxy for record in records]).astype(np.float32),
        confidence=np.asarray(
            [record.confidence for record in records],
            dtype=np.float32,
        ),
    )


def find_frame_path(image_dir: Path, frame_number: int) -> Path:
    """Find a frame using common MOT naming schemes."""
    for pattern in (
        "{:06d}.jpg",
        "{:08d}.jpg",
        "{:06d}.png",
        "{:08d}.png",
    ):
        path = image_dir / pattern.format(frame_number)
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find frame {frame_number} in {image_dir}.")


def load_rgb_frame(frame_path: Path) -> np.ndarray:
    """Load one frame and convert OpenCV BGR channels to RGB."""
    import cv2

    frame_bgr = cv2.imread(str(frame_path))
    if frame_bgr is None:
        raise RuntimeError(f"cv2.imread failed for {frame_path}.")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def create_tracker(
    *,
    device: str,
    frame_rate: float,
    enable_isolated_mask_matching: bool,
    enable_cmc: bool,
    cmc_method: CMCMethod,
    cmc_downscale: int,
) -> McByteTracker:
    """Create a fresh full McByte tracker for one sequence."""
    from trackers.core.mcbyte.tracker import McByteMaskConfig, McByteTracker

    return McByteTracker(
        frame_rate=frame_rate,
        enable_cmc=enable_cmc,
        cmc_method=cmc_method,
        cmc_downscale=cmc_downscale,
        enable_mask_manager=True,
        mask_config=McByteMaskConfig(device=device),
        enable_isolated_mask_matching=enable_isolated_mask_matching,
    )


def write_mot_results(
    output: TextIO,
    frame_number: int,
    tracked: sv.Detections,
) -> None:
    """Write valid tracked detections in MOTChallenge format."""
    if tracked.tracker_id is None:
        return

    for xyxy, tracker_id_value in zip(tracked.xyxy, tracked.tracker_id):
        tracker_id = int(tracker_id_value)
        if tracker_id < 0:
            continue
        left, top, right, bottom = map(float, xyxy)
        output.write(
            f"{frame_number},{tracker_id},{left:.2f},{top:.2f},{right - left:.2f},{bottom - top:.2f},-1,-1,-1,-1\n"
        )


def cleanup_tracker(
    tracker: McByteTracker | None,
    logger: logging.Logger,
    sequence: str,
) -> None:
    """Reset state, collect Python objects, and release cached accelerator memory."""
    import torch

    if tracker is not None:
        try:
            tracker.reset()
        except Exception:
            logger.exception("Tracker reset failed after %s.", sequence)
    del tracker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_sequence(
    *,
    sequence: str,
    detection_file: Path,
    image_dir: Path,
    output_file: Path,
    config: DatasetConfig,
    device: str,
    enable_isolated_mask_matching: bool,
    enable_cmc: bool,
    cmc_method: CMCMethod,
    cmc_downscale: int,
    keep_partial_results: bool,
    logger: logging.Logger,
) -> int:
    """Run one benchmark sequence.

    A fresh tracker is created for the sequence.
    Results are first written to a temporary `.partial` file,
    which is replaced by the final MOT result only after successful completion.
    """
    detections_by_frame = read_detection_file(detection_file, config)
    if not detections_by_frame:
        raise ValueError(f"No detections found in {detection_file}.")

    last_frame = max(detections_by_frame)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    partial_file = output_file.with_suffix(".txt.partial")
    partial_file.unlink(missing_ok=True)
    tracker: McByteTracker | None = None

    try:
        tracker = create_tracker(
            device=device,
            frame_rate=config.frame_rate,
            enable_isolated_mask_matching=enable_isolated_mask_matching,
            enable_cmc=enable_cmc,
            cmc_method=cmc_method,
            cmc_downscale=cmc_downscale,
        )

        with partial_file.open("w", encoding="utf-8") as output:
            for frame_number in range(1, last_frame + 1):
                frame_rgb = load_rgb_frame(find_frame_path(image_dir, frame_number))
                detections = build_detections(detections_by_frame.get(frame_number, []))
                tracked = tracker.update(detections=detections, frame=frame_rgb)
                write_mot_results(output, frame_number, tracked)

                if frame_number == 1 or frame_number % 250 == 0:
                    logger.info(
                        "%s | frame %d/%d",
                        sequence,
                        frame_number,
                        last_frame,
                    )

        partial_file.replace(output_file)
        return last_frame
    except Exception:
        if not keep_partial_results:
            partial_file.unlink(missing_ok=True)
        raise
    finally:
        cleanup_tracker(tracker, logger, sequence)


def run_dataset(
    *,
    config: DatasetConfig,
    output_dir: Path,
    device: str,
    enable_isolated_mask_matching: bool,
    enable_cmc: bool,
    cmc_method: CMCMethod,
    cmc_downscale: int,
    skip_existing: bool,
    keep_partial_results: bool,
    logger: logging.Logger,
) -> tuple[int, int, int]:
    """Run every sequence in one dataset.

    Each detection file is matched with its corresponding image directory,
    processed using a fresh McByte tracker, and written as one MOTChallenge
    result file. Missing or failed sequences are recorded while processing
    continues with the remaining sequences.
    """
    if config.detection_root is None or config.image_root is None:
        raise ValueError(
            f"Please configure DATASETS['{config.name}'] detection_root and image_root before running this script, "
            f"or supply them per run with "
            f'--datasets=\'{{"{config.name}": {{"detection_root": ..., "image_root": ...}}}}\'.'
        )
    if not config.detection_root.is_dir():
        raise NotADirectoryError(config.detection_root)
    if not config.image_root.is_dir():
        raise NotADirectoryError(config.image_root)

    detection_files = sorted(config.detection_root.glob("*.txt"))
    if not detection_files:
        raise FileNotFoundError(f"No detections in {config.detection_root}.")

    # Imported after the layout checks so a misconfigured dataset fails without
    # paying for torch, and before the loop so the ``except`` clause below can
    # resolve the name when a sequence actually raises.
    import torch

    completed = skipped = failed = 0
    for index, detection_file in enumerate(detection_files, start=1):
        seq = sequence_name(detection_file, config)
        final_output = output_dir / f"{seq}.txt"
        frames = image_directory(seq, config)

        if skip_existing and final_output.is_file():
            logger.info("[%s %d/%d] Skip %s", config.name, index, len(detection_files), seq)
            skipped += 1
            continue

        logger.info("[%s %d/%d] Process %s", config.name, index, len(detection_files), seq)
        try:
            last_frame = run_sequence(
                sequence=seq,
                detection_file=detection_file,
                image_dir=frames,
                output_file=final_output,
                config=config,
                device=device,
                enable_isolated_mask_matching=enable_isolated_mask_matching,
                enable_cmc=enable_cmc,
                cmc_method=cmc_method,
                cmc_downscale=cmc_downscale,
                keep_partial_results=keep_partial_results,
                logger=logger,
            )
        except torch.cuda.OutOfMemoryError:
            failed += 1
            logger.exception("CUDA OOM while processing %s/%s", config.name, seq)
        except Exception:
            failed += 1
            logger.exception("Failed while processing %s/%s", config.name, seq)
        else:
            completed += 1
            logger.info("Completed %s/%s (%d frames)", config.name, seq, last_frame)

    return completed, skipped, failed


def prepare_mot17_submission(
    raw_dir: Path,
    submission_dir: Path,
    logger: logging.Logger,
) -> None:
    """Create MOT17 submission files.

    The MOT17 evaluation server expects one result file per detector
    (FRCNN, SDP and DPM). Since McByte is detector-agnostic here,
    the same tracking result is duplicated for all three detector names.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)

    for number in MOT17_EXISTING:
        source = raw_dir / f"MOT17-{number}.txt"
        if not source.is_file():
            logger.warning("Missing MOT17 source result: %s", source)
            continue
        content = source.read_bytes()
        for suffix in MOT17_SUFFIXES:
            (submission_dir / f"MOT17-{number}-{suffix}.txt").write_bytes(content)

    for number in MOT17_MISSING:
        for suffix in MOT17_SUFFIXES:
            (submission_dir / f"MOT17-{number}-{suffix}.txt").touch(exist_ok=True)


def benchmark_command(
    dataset: list[DatasetName] | None = None,
    datasets: dict[str, DatasetPaths] | None = None,
    device: str = "auto",
    enable_isolated_mask_matching: bool = False,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    skip_existing: bool = False,
    enable_cmc: bool = True,
    cmc_method: CMCMethod = "sparseOptFlow",
    cmc_downscale: int = 6,
    keep_partial_results: bool = False,
) -> int:
    """Run all selected datasets and summarize sequence failures.

    Args:
        dataset: Datasets to run, as a list: ``--dataset=[mot17,soccernet]``.
            ``None`` runs every dataset in ``DATASETS``.
        datasets: Where each dataset's files live, keyed by the same names
            ``dataset`` selects, each entry holding a ``detection_root`` and an
            ``image_root``. Neither root has a built-in value, so a dataset is
            runnable only once its entry is supplied, as JSON on the command
            line — ``--datasets='{"mot17": {"detection_root": "/data/dets",
            "image_root": "/data/frames"}}'`` — or as the same mapping in a
            ``--config`` file, which is the readable spelling and the one the
            module docstring shows.
        device: Device for SAM + Cutie, e.g. ``cuda``, ``cpu``, or ``mps``. The
            default ``auto`` resolves to CUDA when available, otherwise CPU; MPS
            is never auto-selected (measured ~an order of magnitude slower than
            CPU for this pipeline) and must be requested explicitly.
        enable_isolated_mask_matching: Match masks in isolation. Negate with
            ``--no_enable_isolated_mask_matching``.
        output_root: Directory holding one timestamped run directory per run.
        skip_existing: Skip a sequence whose result file is already present.
        enable_cmc: Compensate for camera motion. Negate with
            ``--no_enable_cmc``.
        cmc_method: Camera-motion compensation method.
        cmc_downscale: Frame downscale factor applied before compensation.
        keep_partial_results: Keep the ``.partial`` file a failed sequence
            leaves behind instead of deleting it.

    Returns:
        Exit code: ``0`` on success, ``1`` on validation error.
    """
    error = _runtime_error(device, cmc_downscale) or _unknown_datasets_error(datasets)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    dataset_table = resolve_datasets(datasets)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    isolation = "isolation" if enable_isolated_mask_matching else "no_isolation"
    run_root = output_root / f"{timestamp}__{isolation}"
    run_root.mkdir(parents=True, exist_ok=False)
    logger = configure_logging(run_root)

    selected = dataset if dataset is not None else list(dataset_table)
    logger.info("Run root: %s", run_root)
    logger.info("Datasets: %s", selected)
    logger.info("Device: %s", device)
    logger.info("Isolation: %s", enable_isolated_mask_matching)

    total_completed = total_skipped = total_failed = 0

    for name in selected:
        config = dataset_table[name]
        raw_dir = run_root / name / "raw"
        try:
            completed, skipped, failed = run_dataset(
                config=config,
                output_dir=raw_dir,
                device=device,
                enable_isolated_mask_matching=enable_isolated_mask_matching,
                enable_cmc=enable_cmc,
                cmc_method=cmc_method,
                cmc_downscale=cmc_downscale,
                skip_existing=skip_existing,
                keep_partial_results=keep_partial_results,
                logger=logger,
            )
        except Exception:
            logger.exception("Dataset setup failed: %s", name)
            total_failed += 1
            continue

        total_completed += completed
        total_skipped += skipped
        total_failed += failed

        if name == "mot17":
            prepare_mot17_submission(
                raw_dir,
                run_root / "mot17" / "submission",
                logger,
            )

    logger.info(
        "Finished: completed=%d skipped=%d failed=%d",
        total_completed,
        total_skipped,
        total_failed,
    )
    return 0
