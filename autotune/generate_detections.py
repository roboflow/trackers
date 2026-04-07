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
    uv run python generate_detections.py --model rfdetr/l              # → RFDETR/
    uv run python generate_detections.py --model yolo_world/l         # → YOLOWORLD/
    uv run python generate_detections.py --model yolo_world/l \\
        --api-key YOUR_ROBOFLOW_KEY                                    # explicit key
    uv run python generate_detections.py --seq MOT17-04               # single sequence

Prerequisites:
    1. Install optimize group:
           uv sync --group optimize
    2. Download frame images (≈ 4 GB):
           trackers download mot17 --split val --asset annotations,detections,frames

Output:
    {data_dir}/{sequence_base}-{TAG}/det/det.txt   (MOT-format; one file per seq)
    {data_dir}/{sequence_base}-{TAG}/gt   →  symlink to ../{sequence_base}-FRCNN/gt
    {data_dir}/{sequence_base}-{TAG}/img1 →  symlink to ../{sequence_base}-FRCNN/img1

    TAG is auto-derived from the model name (rfdetr → RFDETR,
    yolo_world/l → YOLOWORLD) and can be overridden with ``--detector-tag``.

    Each line:  frame_idx,-1,x,y,w,h,confidence,-1,-1,-1
    where (x, y) is the top-left corner and (w, h) is width/height.
    id=-1 because these are raw detections, not tracked identities.

Detector backends:

    rfdetr (default — no API key needed):
        Model:    ``rfdetr/l`` (large) — size suffix is required
        Backend:  native ``rfdetr`` package (>= 1.6)
        Weights:  downloaded automatically on first use

    yolo_world (open-vocabulary person detector):
        Model:    ``yolo_world/s``, ``yolo_world/m``, ``yolo_world/l`` (default),
                  ``yolo_world/x``
        Backend:  ``inference-models`` package (>= 0.19.0)
        API key:  set ``ROBOFLOW_API_KEY`` env var, or pass ``--api-key``
                  (needed to download weights on first use)
        Text:     searches for ``"person"`` via CLIP text embeddings
        Outputs are person-only (single text class).

Notes:
    - Each detector writes to its own sequence-level directory (e.g.
      ``MOT17-04-RFDETR/``, ``MOT17-04-YOLOWORLD/``), keeping detector outputs
      fully transparent in the filesystem.  Use ``--detector-tag`` to override.
    - Overwrites existing det.txt if present (use --skip-existing).
    - Ground truth (gt/) is never read or written; inference is detection-only.
    - Sequences without an img1/ directory are skipped with a warning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fire
import supervision as sv
from rich.console import Console
from rich.progress import track

console = Console()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_ID = "rfdetr/l"
_DEFAULT_CONFIDENCE = 0.1  # low threshold — let the tracker filter aggressively

# rfdetr native (>= 1.6) uses 1-indexed COCO category IDs; person = 1
_RFDETR_PERSON_CLASS_ID = 1

_RFDETR_NATIVE_MODELS = frozenset({"rfdetr/n", "rfdetr/s", "rfdetr/m", "rfdetr/l"})
# YOLO-World with text=["person"] outputs person-only (single text class)
_YOLO_WORLD_MODELS = frozenset(
    {"yolo_world/s", "yolo_world/m", "yolo_world/l", "yolo_world/x"}
)


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

    The tag is used as the directory suffix, e.g. ``MOT17-04-RFDETR/``.

    Args:
        model: Model name or Roboflow model ID, e.g. ``"rfdetr/l"``,
            ``"yolo_world/x"``.

    Returns:
        Short uppercase tag, e.g. ``"RFDETR"``, ``"YOLOWORLD"``.

    Examples:
        RF-DETR maps to ``"RFDETR"``::

            tag = _derive_detector_tag("rfdetr/l")
            # "RFDETR"
    """
    m = model.lower()
    if m.startswith("yolo_world") or m.startswith("yolo-world"):
        return "YOLOWORLD"
    if m.startswith("rfdetr") or m.startswith("rf-detr"):
        return "RFDETR"
    if m.startswith("yolo"):
        return "YOLO"
    # Generic fallback: first hyphen-separated component, uppercased
    return model.split("-")[0].upper()


# ---------------------------------------------------------------------------
# YOLO-World backend (inference-models >= 0.19.0)
# ---------------------------------------------------------------------------


def _load_yolo_world_model(model_id: str, api_key: str | None) -> Any:
    """Load a YOLO-World model via the inference-models package.

    Weights are downloaded from Roboflow on first use.  Set
    ``ROBOFLOW_API_KEY`` or pass ``api_key`` explicitly.

    Args:
        model_id: One of ``"yolo_world/s"``, ``"yolo_world/m"``,
            ``"yolo_world/l"`` (default), ``"yolo_world/x"``.
        api_key: Roboflow API key.  Falls back to the ``ROBOFLOW_API_KEY``
            environment variable when ``None``.

    Returns:
        A ``YOLOWorld`` model instance ready for ``.infer()`` calls.

    Raises:
        RuntimeError: If the ``inference-models`` package is not installed.

    Examples:
        Load the large YOLO-World model::

            model = _load_yolo_world_model("yolo_world/l", api_key=None)
    """
    try:
        from inference.models.yolo_world import YOLOWorld
    except ImportError as exc:
        raise RuntimeError(
            "YOLO-World backend requires the 'inference-models' package.\n"
            "Install: uv sync --group optimize"
        ) from exc
    return YOLOWorld(model_id=model_id, api_key=api_key)


def _process_frame_yolo_world(
    frame_path: Path,
    model: Any,
    confidence: float,
) -> list[str]:
    """Run YOLO-World inference on one frame; return MOT-format detection lines.

    The model is queried with text prompt ``["person"]``.  Predictions are
    returned in center-format (x_center, y_center, width, height) and are
    converted to top-left corner format for MOT output.

    Args:
        frame_path: Image file whose stem is a zero-padded integer frame index.
        model: A loaded ``YOLOWorld`` instance.
        confidence: Minimum detection confidence threshold.

    Returns:
        List of strings in MOT format: ``"frame,-1,x,y,w,h,conf,-1,-1,-1"``.

    Examples:
        Output is a list of comma-separated detection strings::

            lines = _process_frame_yolo_world(frame_path, model, confidence=0.1)
            # ["1,-1,120.50,200.30,60.00,150.00,0.9200,-1,-1,-1", ...]
    """
    frame_idx = int(frame_path.stem)
    response = model.infer(
        image=str(frame_path),
        text=["person"],
        confidence=confidence,
    )
    lines = []
    for pred in response.predictions:
        # inference returns center-format; convert to top-left for MOT
        x_tl = pred.x - pred.width / 2
        y_tl = pred.y - pred.height / 2
        lines.append(
            f"{frame_idx},-1,{x_tl:.2f},{y_tl:.2f}"
            f",{pred.width:.2f},{pred.height:.2f},{pred.confidence:.4f},-1,-1,-1"
        )
    return lines


# ---------------------------------------------------------------------------
# RF-DETR native backend (rfdetr >= 1.6 — no API key needed)
# ---------------------------------------------------------------------------


def _load_rfdetr_model(model_name: str) -> Any:
    """Load a native RF-DETR model via the rfdetr package (>= 1.6).

    Weights are downloaded automatically on first use — no API key required.

    Args:
        model_name: ``"rfdetr/l"`` (large). A size suffix is required;
            bare ``"rfdetr"`` is not accepted.

    Returns:
        An RF-DETR model instance with a ``predict(image, threshold)`` method
        that returns ``sv.Detections`` directly.

    Raises:
        RuntimeError: If the ``rfdetr`` package (>= 1.6) is not installed.
        ValueError: If ``model_name`` is not one of the accepted size variants.

    Examples:
        Load the large RF-DETR model::

            model = _load_rfdetr_model("rfdetr/l")
    """
    _RFDETR_CLASS_MAP = {
        "rfdetr/n": "RFDETRNano",
        "rfdetr/s": "RFDETRSmall",
        "rfdetr/m": "RFDETRMedium",
        "rfdetr/l": "RFDETRLarge",
    }
    class_name = _RFDETR_CLASS_MAP.get(model_name)
    if class_name is None:
        raise ValueError(
            f"Unknown RF-DETR model {model_name!r}. "
            f"Supported: {', '.join(sorted(_RFDETR_CLASS_MAP))}"
        )
    try:
        import rfdetr as _rfdetr

        return getattr(_rfdetr, class_name)()
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
    skip_existing: bool,
) -> int:
    """Run inference on one sequence; return number of frames processed.

    Args:
        seq_dir: Path to the source sequence directory with frames (e.g.
            ``.../MOT17-04-FRCNN``).
        out_dir: Output directory for this detector (e.g. ``.../MOT17-04-YOLOWORLD``).
            Detections are written to ``out_dir/det/det.txt``.
        model: Loaded model instance (rfdetr or YOLO-World).
        backend: Either ``"rfdetr"`` or ``"yolo_world"``.
        confidence: Minimum detection confidence.
        skip_existing: If True and ``out_dir/det/det.txt`` already exists, skip.

    Returns:
        Number of frames processed (0 if skipped or no frames found).

    Examples:
        Returns 0 for a sequence without frames::

            n = _run_on_sequence(
                Path("/missing"), Path("/out"), model, "rfdetr", 0.1, False
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
        if backend == "yolo_world":
            lines.extend(_process_frame_yolo_world(frame_path, model, confidence))
        else:
            lines.extend(_process_frame_rfdetr(frame_path, model, confidence))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n" if lines else "")
    return len(frames)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    model: str = _DEFAULT_MODEL_ID,
    conf: float = _DEFAULT_CONFIDENCE,
    api_key: str | None = None,
    detector_tag: str | None = None,
    data_dir: str | None = None,
    seq: str | None = None,
    skip_existing: bool = False,
) -> None:
    """Generate detections for MOT17-val sequences.

    Args:
        model: Detector to use. RF-DETR: ``"rfdetr/n"``, ``"rfdetr/s"``,
            ``"rfdetr/m"``, ``"rfdetr/l"`` (default). YOLO-World:
            ``"yolo_world/s"``, ``"yolo_world/m"``, ``"yolo_world/l"``,
            ``"yolo_world/x"``.
        conf: Minimum detection confidence threshold (default: 0.1).
        api_key: Roboflow API key for YOLO-World weight download.  Falls back
            to the ``ROBOFLOW_API_KEY`` environment variable when unset.
        detector_tag: Uppercase tag appended to the output directory name, e.g.
            ``"RFDETR"`` → ``MOT17-04-RFDETR/``. Auto-derived if unset.
        data_dir: MOT17 val directory. Auto-detected if unset.
        seq: Filter to a single sequence prefix, e.g. ``"MOT17-04"``. If unset,
            runs on all sequences with ground-truth annotations and frames.
        skip_existing: Skip sequences where ``{TAG}/det/det.txt`` already exists.

    Examples:
        RF-DETR large (default)::

            uv run python generate_detections.py --model rfdetr/l

        YOLO-World large (open-vocabulary person detector)::

            uv run python generate_detections.py --model yolo_world/l
    """
    is_yolo_world = model in _YOLO_WORLD_MODELS
    is_rfdetr = model in _RFDETR_NATIVE_MODELS

    if not is_yolo_world and not is_rfdetr:
        rfdetr_opts = ", ".join(sorted(_RFDETR_NATIVE_MODELS))
        yoloworld_opts = ", ".join(sorted(_YOLO_WORLD_MODELS))
        raise ValueError(
            f"Unknown model {model!r}.\n"
            f"RF-DETR options: {rfdetr_opts}\n"
            f"YOLO-World options: {yoloworld_opts}"
        )

    _data_dir = Path(data_dir) if data_dir else _find_data_dir()

    # Source sequences: dirs that have actual frames (img1/) and ground truth (gt/)
    # Only use dirs where img1/ is a real directory (not a symlink) — this
    # excludes RFDETR/YOLOWORLD sibling dirs whose img1/ points back to FRCNN.
    source_seqs = sorted(
        d
        for d in _data_dir.iterdir()
        if d.is_dir()
        and (d / "img1").is_dir()
        and not (d / "img1").is_symlink()
        and (d / "gt" / "gt.txt").exists()
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
    if is_yolo_world:
        loaded_model = _load_yolo_world_model(model, api_key)
        backend = "yolo_world"
    else:
        loaded_model = _load_rfdetr_model(model)
        backend = "rfdetr"

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
        f"  [dim]cd autotune &&"
        f" uv run python optimize_tracking.py bytetrack {tag.lower()}[/dim]"
    )


if __name__ == "__main__":
    fire.Fire(main)
