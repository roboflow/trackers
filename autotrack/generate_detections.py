#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""generate_detections.py — Generate detections for MOT17-val sequences.

Runs detection inference on each frame and saves predictions in MOT format.
Each detector gets its own sequence-level sibling directory, making the detector
visible directly in the filesystem path.

Usage:
    uv run python generate_detections.py \\
        --model yolox-x-crowdhuman \\
        --weights pretrained/yolox_x.pth        # → YOLOX/  (recommended)
    uv run python generate_detections.py --model rfdetr-l   # → RFDETR/
    uv run python generate_detections.py --seq MOT17-04     # single sequence
    uv run python generate_detections.py --model yolov8x-640 --conf 0.3

Prerequisites:
    1. Install optimize group:
           uv sync --group optimize
    2. API key (Roboflow models only — not required for yolox or rfdetr):
           export ROBOFLOW_API_KEY=your_key_here
       Get a free key at https://app.roboflow.com — Account → API Key.
    3. Download frame images (≈ 4 GB):
           trackers download mot17 --split val --asset annotations,detections,frames

Output:
    {data_dir}/{sequence_base}-{TAG}/det/det.txt   (MOT-format; one file per seq)
    {data_dir}/{sequence_base}-{TAG}/gt   →  symlink to ../{sequence_base}-FRCNN/gt
    {data_dir}/{sequence_base}-{TAG}/img1 →  symlink to ../{sequence_base}-FRCNN/img1

    TAG is auto-derived from the model name (rfdetr → RFDETR,
    yolox-x-crowdhuman → YOLOX, yolov8x → YOLO) and can be overridden
    with ``--detector-tag``.

    Each line:  frame_idx,-1,x,y,w,h,confidence,-1,-1,-1
    where (x, y) is the top-left corner and (w, h) is width/height.
    id=-1 because these are raw detections, not tracked identities.

Detector backends:

    rfdetr (recommended — no API key needed):
        Models:   ``rfdetr-l`` (large) — size suffix is required
        Backend:  native ``rfdetr`` package (>= 1.6)
        Weights:  downloaded automatically on first use
        Returns ``sv.Detections`` directly; person class filtered from COCO output.

    yolox (YOLOX-X CrowdHuman / ByteTrack weights — no API key needed):
        Model:    ``yolox-x-crowdhuman``
        Backend:  local YOLOX package (``pip install yolox torch``)
        Weights:  download ``bytetrack_x_mot17.pth.tar`` from the ByteTrack
                  GitHub releases and pass ``--weights /path/to/file``
        This is the exact detector used in the published ByteTrack MOT17 numbers.
        Outputs are person-only (single class).

    roboflow (any Roboflow-hosted model — requires ROBOFLOW_API_KEY):
        Any Roboflow-hosted model ID, e.g. ``yolov8x-1280``.
        Passed directly to ``inference.get_model()``.
        All COCO models output person as class 0; other classes are discarded.

Notes:
    - Each detector writes to its own sequence-level directory (e.g.
      ``MOT17-04-RFDETR/``, ``MOT17-04-YOLOX/``), keeping detector outputs fully
      transparent in the filesystem.  Use ``--detector-tag`` to override the
      auto-derived tag.
    - Overwrites existing det.txt if present (use --skip-existing).
    - Ground truth (gt/) is never read or written; inference is detection-only.
    - Sequences without an img1/ directory are skipped with a warning.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import fire
import supervision as sv
from loguru import logger as _loguru_logger
from rich.console import Console
from rich.progress import track

console = Console()

# Silence per-frame "Infer time" logs from yolox — they break Rich's live rendering
_loguru_logger.disable("yolox")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_ID = "yolov8x-1280"  # YOLOv8-X 1280 px — best recall, free tier
_DEFAULT_CONFIDENCE = 0.1  # low threshold — let the tracker filter aggressively
_DEFAULT_IOU_THRESHOLD = 0.45  # NMS threshold

# Person class ID varies by backend:
# - Roboflow inference / YOLOX: 0-indexed COCO → person = 0
# - rfdetr native (>= 1.6): 1-indexed COCO category IDs → person = 1
_COCO_PERSON_CLASS_ID = 0
_RFDETR_PERSON_CLASS_ID = 1

# Model names that route to the local YOLOX backend instead of Roboflow inference
_YOLOX_BACKEND_MODELS = frozenset({"yolox-x-crowdhuman", "yolox-x"})

# Model names that route to the native rfdetr package (>= 1.6) — no API key needed
_RFDETR_NATIVE_MODELS = frozenset({"rfdetr-l"})


# ---------------------------------------------------------------------------
# Helper: data directory + detector tag
# ---------------------------------------------------------------------------


def _find_data_dir() -> Path:
    """Locate the MOT17-val root directory, same logic as optimize_tracking.py."""
    for candidate in [
        Path("./mot17/val"),
        Path("./data/mot17/val"),
        Path.home() / ".cache/trackers/mot17/val",
    ]:
        if candidate.exists() and any(candidate.glob("*/gt/gt.txt")):
            return candidate
    raise FileNotFoundError(
        "MOT17 val data not found. Run:\n"
        "  trackers download mot17 --split val --asset annotations,detections,frames"
    )


def _derive_detector_tag(model: str) -> str:
    """Derive an uppercase detector tag from a model identifier.

    The tag is used as the directory suffix, e.g. ``MOT17-04-YOLO/``.

    Args:
        model: Model name or Roboflow model ID, e.g. ``"yolov8x-1280"``,
            ``"rf-detr-l"``, ``"yolox-x-crowdhuman"``.

    Returns:
        Short uppercase tag, e.g. ``"YOLO"``, ``"RFDETR"``, ``"YOLOX"``.

    Examples:
        Default YOLOv8x model maps to ``"YOLO"``::

            tag = _derive_detector_tag("yolov8x-1280")
            # "YOLO"
    """
    m = model.lower()
    if m.startswith("yolox"):
        return "YOLOX"
    if m.startswith("rfdetr") or m.startswith("rf-detr"):
        return "RFDETR"
    if m.startswith("yolo"):
        return "YOLO"
    # Generic fallback: first hyphen-separated component, uppercased
    return model.split("-")[0].upper()


# ---------------------------------------------------------------------------
# Roboflow inference backend
# ---------------------------------------------------------------------------


def _process_frame(
    frame_path: Path,
    model: Any,
    confidence: float,
    iou_threshold: float,
) -> list[str]:
    """Run Roboflow inference on one frame; return MOT-format detection lines.

    Args:
        frame_path: Image file whose stem is a zero-padded integer frame index.
        model: Loaded Roboflow inference model with an ``infer`` method.
        confidence: Minimum detection confidence threshold.
        iou_threshold: NMS IoU threshold passed to the model.

    Returns:
        List of strings in MOT format: ``"frame,-1,x,y,w,h,conf,-1,-1,-1"``.

    Examples:
        Output is a list of comma-separated detection strings::

            lines = _process_frame(frame_path, model, 0.1, 0.45)
            # ["1,-1,120.50,200.30,60.00,150.00,0.9200,-1,-1,-1", ...]
    """
    frame_idx = int(frame_path.stem)
    results = model.infer(
        image=str(frame_path),
        confidence=confidence,
        iou_threshold=iou_threshold,
    )
    result = results[0] if isinstance(results, list) else results
    detections = sv.Detections.from_inference(result)
    detections = detections[detections.class_id == _COCO_PERSON_CLASS_ID]  # type: ignore[assignment]
    lines = []
    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
        w = x2 - x1
        h = y2 - y1
        if detections.confidence is None:
            raise RuntimeError(
                f"Model returned no confidence scores for frame {frame_path.stem}."
                " Use a model that provides per-detection confidence."
            )
        conf = float(detections.confidence[i])
        lines.append(
            f"{frame_idx},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"
        )
    return lines


# ---------------------------------------------------------------------------
# YOLOX backend
# ---------------------------------------------------------------------------


def _load_yolox_predictor(model_name: str, weights_path: str, conf: float) -> Any:
    """Load a YOLOX predictor from a local weights file.

    Args:
        model_name: Model name, e.g. ``"yolox-x-crowdhuman"``.
        weights_path: Path to the local ``.pth`` or ``.pth.tar`` weights file.
        conf: Detection confidence threshold applied during NMS postprocessing.

    Returns:
        A ``yolox.utils.Predictor`` instance ready for inference.

    Raises:
        RuntimeError: If the ``yolox`` or ``torch`` packages are not installed.

    Examples:
        Load ByteTrack's YOLOX-X CrowdHuman model::

            predictor = _load_yolox_predictor(
                "yolox-x-crowdhuman",
                "/path/to/bytetrack_x_mot17.pth.tar",
                conf=0.01,
            )
    """
    try:
        import torch
        from yolox.exp import get_exp
        from yolox.tools.demo import Predictor
    except ImportError as exc:
        raise RuntimeError(
            "YOLOX backend requires the 'yolox' and 'torch' packages.\n"
            "Install: pip install yolox torch\n"
            "Download ByteTrack YOLOX-X weights (bytetrack_x_mot17.pth.tar) from\n"
            "the ByteTrack GitHub releases, then pass: --weights /path/to/file"
        ) from exc

    exp = get_exp(exp_file=None, exp_name="yolox_x")
    if "crowdhuman" in model_name:
        # ByteTrack's CrowdHuman-pretrained model is single-class (person only)
        exp.num_classes = 1
        exp.test_size = (800, 1440)  # ByteTrack default test size for MOT17
    exp.test_conf = conf  # apply user threshold at NMS postprocessing time

    model_obj = exp.get_model()
    model_obj.eval()

    # ByteTrack checkpoints store the model under the "model" key
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model", ckpt)
    model_obj.load_state_dict(model_state)

    device = "gpu" if torch.cuda.is_available() else "cpu"
    return Predictor(
        model=model_obj,
        exp=exp,
        trt_file=None,
        decoder=None,
        device=device,
        fp16=False,
    )


def _process_frame_yolox(
    frame_path: Path,
    predictor: Any,
    conf: float,
) -> list[str]:
    """Run YOLOX inference on one frame; return MOT-format detection lines.

    Args:
        frame_path: Image file whose stem is a zero-padded integer frame index.
        predictor: A loaded ``yolox.utils.Predictor`` instance.
        conf: Secondary confidence filter (applied after model NMS).  Set low
            (e.g. 0.01) to rely on the predictor's built-in threshold instead.

    Returns:
        List of strings in MOT format: ``"frame,-1,x,y,w,h,conf,-1,-1,-1"``.

    Examples:
        Output is a list of comma-separated detection strings::

            lines = _process_frame_yolox(frame_path, predictor, conf=0.01)
            # ["1,-1,120.50,200.30,60.00,150.00,0.8500,-1,-1,-1", ...]
    """
    import torch

    frame_idx = int(frame_path.stem)
    # Predictor.inference() accepts a file path string and handles cv2.imread internally
    outputs, img_info = predictor.inference(str(frame_path))
    if outputs[0] is None:
        return []

    output = outputs[0]
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    # img_info["ratio"] = min(test_h/img_h, test_w/img_w); divide to get original coords
    ratio = img_info["ratio"]
    lines = []
    for det in output:
        x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det[:7]
        x1, y1, x2, y2 = x1 / ratio, y1 / ratio, x2 / ratio, y2 / ratio
        score = float(obj_conf) * float(cls_conf)
        if score < conf:
            continue
        # For multi-class (COCO) YOLOX models, keep only person (class 0)
        if predictor.num_classes > 1 and int(cls_id) != 0:
            continue
        w, h = x2 - x1, y2 - y1
        lines.append(
            f"{frame_idx},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1"
        )
    return lines


# ---------------------------------------------------------------------------
# RF-DETR native backend (rfdetr >= 1.6 — no API key needed)
# ---------------------------------------------------------------------------


def _load_rfdetr_model(model_name: str) -> Any:
    """Load a native RF-DETR model via the rfdetr package (>= 1.6).

    Weights are downloaded automatically on first use — no API key required.

    Args:
        model_name: ``"rfdetr-l"`` (large). A size suffix is required;
            bare ``"rfdetr"`` is not accepted.

    Returns:
        An RF-DETR model instance with a ``predict(image, threshold)`` method
        that returns ``sv.Detections`` directly.

    Raises:
        RuntimeError: If the ``rfdetr`` package (>= 1.6) is not installed.
        ValueError: If ``model_name`` is not one of the accepted size variants.

    Examples:
        Load the large RF-DETR model::

            model = _load_rfdetr_model("rfdetr-l")
    """
    try:
        if model_name == "rfdetr-l":
            from rfdetr import RFDETRLarge

            return RFDETRLarge()
        else:
            raise ValueError(
                f"Unknown RF-DETR model {model_name!r}. Only 'rfdetr-l' is supported."
            )
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR backend requires the 'rfdetr' package (>= 1.6).\n"
            "Install: pip install 'rfdetr>=1.6'\n"
            "Weights are downloaded automatically on first use."
        ) from exc


def _process_frame_rfdetr(
    frame_path: Path,
    model: Any,
    confidence: float,
) -> list[str]:
    """Run native RF-DETR inference on one frame; return MOT-format detection lines.

    Args:
        frame_path: Image file whose stem is a zero-padded integer frame index.
        model: A loaded RF-DETR model instance (``RFDETRLarge`` or ``RFDETRBase``).
        confidence: Minimum detection confidence threshold.

    Returns:
        List of strings in MOT format: ``"frame,-1,x,y,w,h,conf,-1,-1,-1"``.

    Examples:
        Output is a list of comma-separated detection strings::

            lines = _process_frame_rfdetr(frame_path, model, confidence=0.1)
            # ["1,-1,120.50,200.30,60.00,150.00,0.9200,-1,-1,-1", ...]
    """
    from PIL import Image

    frame_idx = int(frame_path.stem)
    image = Image.open(frame_path).convert("RGB")
    detections: sv.Detections = model.predict(image, threshold=confidence)
    # rfdetr returns 1-indexed COCO category IDs (person = 1), not 0-indexed
    detections = detections[detections.class_id == _RFDETR_PERSON_CLASS_ID]  # type: ignore[assignment]
    lines = []
    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
        w = x2 - x1
        h = y2 - y1
        if detections.confidence is None:
            raise RuntimeError(
                f"RF-DETR returned no confidence scores for frame {frame_path.stem}."
            )
        conf = float(detections.confidence[i])
        lines.append(
            f"{frame_idx},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"
        )
    return lines


# ---------------------------------------------------------------------------
# Sequence runner
# ---------------------------------------------------------------------------


def _run_on_sequence(
    seq_dir: Path,
    out_dir: Path,
    model: Any,
    backend: str,
    confidence: float,
    iou_threshold: float,
    skip_existing: bool,
) -> int:
    """Run inference on one sequence; return number of frames processed.

    Args:
        seq_dir: Path to the source sequence directory with frames (e.g.
            ``.../MOT17-04-FRCNN``).
        out_dir: Output directory for this detector (e.g. ``.../MOT17-04-YOLO``).
            Detections are written to ``out_dir/det/det.txt``.
        model: Loaded Roboflow model or YOLOX predictor.
        backend: Either ``"roboflow"`` or ``"yolox"``.
        confidence: Minimum detection confidence.
        iou_threshold: NMS IoU threshold (Roboflow backend only).
        skip_existing: If True and ``out_dir/det/det.txt`` already exists, skip.

    Returns:
        Number of frames processed (0 if skipped or no frames found).

    Examples:
        Returns 0 for a sequence without frames::

            n = _run_on_sequence(
                Path("/missing"), Path("/out"), model, "roboflow", 0.1, 0.45, False
            )
            # n == 0
    """
    img_dir = seq_dir / "img1"
    output_file = out_dir / "det" / "det.txt"

    if not img_dir.exists():
        console.print(
            f"  [yellow]skip[/yellow] {seq_dir.name}: img1/ not found — "
            "run: trackers download mot17 --split val --asset frames"
        )
        return 0

    if skip_existing and output_file.exists():
        console.print(
            f"  [yellow]skip[/yellow] {seq_dir.name}: {out_dir.name}/det/det.txt exists"
        )
        return 0

    frames = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
    if not frames:
        console.print(
            f"  [yellow]skip[/yellow] {seq_dir.name}: no .jpg/.png frames in img1/"
        )
        return 0

    lines: list[str] = []
    for frame_path in track(frames, description=f"  {seq_dir.name}", console=console):
        if backend == "yolox":
            lines.extend(_process_frame_yolox(frame_path, model, confidence))
        elif backend == "rfdetr":
            lines.extend(_process_frame_rfdetr(frame_path, model, confidence))
        else:
            lines.extend(_process_frame(frame_path, model, confidence, iou_threshold))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n" if lines else "")
    return len(frames)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    model: str = _DEFAULT_MODEL_ID,
    conf: float = _DEFAULT_CONFIDENCE,
    iou_threshold: float = _DEFAULT_IOU_THRESHOLD,
    weights: str | None = None,
    detector_tag: str | None = None,
    data_dir: str | None = None,
    seq: str | None = None,
    skip_existing: bool = False,
) -> None:
    """Generate detections for MOT17-val sequences.

    Args:
        model: Detector to use.  Any Roboflow-hosted model ID (e.g.
            ``"yolov8x-1280"``, ``"rfdetr-l"``), or ``"yolox-x-crowdhuman"``
            to use the local YOLOX backend with ByteTrack's CrowdHuman weights.
        conf: Minimum detection confidence threshold (default: 0.1).
        iou_threshold: NMS IoU threshold; Roboflow backend only (default: 0.45).
        weights: Path to a local weights file.  Required when
            ``model="yolox-x-crowdhuman"``; unused for Roboflow models.
        detector_tag: Uppercase tag appended to the output directory name, e.g.
            ``"YOLO"`` → ``MOT17-04-YOLO/``.  Auto-derived from the model name
            if unset: ``yolov8x-*`` → YOLO, ``rfdetr-*`` → RFDETR,
            ``yolox-*`` → YOLOX.
        data_dir: MOT17 val directory. Auto-detected if unset.
        seq: Filter to a single sequence prefix, e.g. ``"MOT17-04"``. If unset,
            runs on all sequences with ground-truth annotations and frames.
        skip_existing: Skip sequences where ``{TAG}/det/det.txt`` already exists.

    Examples:
        Run on all sequences with the default YOLOv8x model::

            uv run python generate_detections.py

        RF-DETR large (Roboflow)::

            uv run python generate_detections.py --model rfdetr-l

        YOLOX-X fine-tuned on CrowdHuman (ByteTrack paper detector)::

            uv run python generate_detections.py \\
                --model yolox-x-crowdhuman \\
                --weights /path/to/bytetrack_x_mot17.pth.tar
    """
    is_yolox = model in _YOLOX_BACKEND_MODELS
    is_rfdetr = model in _RFDETR_NATIVE_MODELS

    if not is_yolox and not is_rfdetr and not os.environ.get("ROBOFLOW_API_KEY"):
        raise RuntimeError(
            "ROBOFLOW_API_KEY not set. "
            "Get a free key at https://app.roboflow.com (Account → API Key), "
            "then: export ROBOFLOW_API_KEY=your_key_here"
        )

    if is_yolox and weights is None:
        raise RuntimeError(
            f"--weights is required for model '{model}'.\n"
            "Download bytetrack_x_mot17.pth.tar from the ByteTrack GitHub releases\n"
            "and pass: --weights /path/to/bytetrack_x_mot17.pth.tar"
        )

    _data_dir = Path(data_dir) if data_dir else _find_data_dir()

    # Source sequences: dirs that have actual frames (img1/) and ground truth (gt/)
    source_seqs = sorted(
        d
        for d in _data_dir.iterdir()
        if d.is_dir() and (d / "img1").exists() and (d / "gt" / "gt.txt").exists()
    )
    if not source_seqs:
        raise FileNotFoundError(
            f"No annotated sequences with frames found in {_data_dir}.\n"
            "Download frames with:\n"
            "  trackers download mot17 --split val"
            " --asset annotations,detections,frames"
        )

    if seq is not None:
        source_seqs = [s for s in source_seqs if seq in s.name]
        if not source_seqs:
            raise ValueError(f"No sequences match filter {seq!r}")

    tag = detector_tag or _derive_detector_tag(model)

    console.print(
        f"Loading model [bold]{model}[/bold] ... (detector tag: [bold]{tag}[/bold])"
    )
    if is_yolox:
        loaded_model = _load_yolox_predictor(model, weights, conf)  # type: ignore[arg-type]
        backend = "yolox"
    elif is_rfdetr:
        loaded_model = _load_rfdetr_model(model)
        backend = "rfdetr"
    else:
        from inference import get_model  # lazy import — requires ROBOFLOW_API_KEY

        loaded_model = get_model(model)
        backend = "roboflow"

    total_frames = 0
    for seq_dir in source_seqs:
        # e.g. "MOT17-04-FRCNN" → base "MOT17-04" → out dir "MOT17-04-YOLO"
        seq_base = seq_dir.name.rsplit("-", 1)[0]
        out_dir = _data_dir / f"{seq_base}-{tag}"

        # Create output dir and symlinks so the directory mirrors the FRCNN structure
        out_dir.mkdir(parents=True, exist_ok=True)
        gt_link = out_dir / "gt"
        if not gt_link.exists():
            gt_link.symlink_to(f"../{seq_dir.name}/gt")
        img1_link = out_dir / "img1"
        if not img1_link.exists():
            img1_link.symlink_to(f"../{seq_dir.name}/img1")

        n = _run_on_sequence(
            seq_dir,
            out_dir,
            loaded_model,
            backend,
            conf,
            iou_threshold,
            skip_existing,
        )
        total_frames += n
        if n > 0:
            out = out_dir / "det" / "det.txt"
            n_dets = sum(1 for line in out.read_text().splitlines() if line.strip())
            console.print(
                f"  [green]✓[/green] {n} frames · {n_dets} dets"
                f" → {out.relative_to(_data_dir)}"
            )

    console.print(
        f"\n[bold green]Done.[/bold green] {total_frames} frames across"
        f" {len(source_seqs)} sequence(s) → {tag}/ sibling directories."
    )
    console.print(
        f"Run campaign:\n"
        f"  [dim]cd autotrack &&"
        f" uv run python optimize_tracking.py bytetrack {tag.lower()}[/dim]"
    )


if __name__ == "__main__":
    fire.Fire(main)
