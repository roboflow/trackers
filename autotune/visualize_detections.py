#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""visualize_detections.py — Overlay detections on MOT17 sequence frames.

Reads ``det/det.txt`` and ``img1/`` from one or more sequence directories and
writes annotated JPEG frames to ``vis/`` inside each sequence directory.

Usage:
    cd autotune
    uv run python visualize_detections.py MOT17-04-YOLOWORLD
    uv run python visualize_detections.py MOT17-04-YOLOWORLD MOT17-02-RFDETR
    uv run python visualize_detections.py --det-source yoloworld
    uv run python visualize_detections.py MOT17-04-YOLOWORLD --conf 0.3

Output:
    {seq_dir}/vis/{frame_stem}.jpg   (one file per frame, skipped if exists)

Notes:
    - Box colour is a red→green gradient keyed to confidence (0 = red, 1 = green).
    - Confidence score is shown as a small label above each box.
    - Bundled detectors (DPM, SDP) fall back to the FRCNN sibling for frames.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import fire
import numpy as np
import supervision as sv
from rich.console import Console
from rich.progress import track

console = Console()

_DEFAULT_DATA_DIR = Path("./mot17/val")


def _conf_color(conf: float) -> tuple[int, int, int]:
    """BGR colour interpolated from red (0) to green (1).

    Args:
        conf: Confidence score in [0, 1].

    Returns:
        BGR tuple suitable for cv2 drawing functions.
    """
    r = int(255 * (1.0 - conf))
    g = int(255 * conf)
    return (0, g, r)


def _load_detections(det_txt: Path, conf_threshold: float) -> dict[int, sv.Detections]:
    """Parse a MOT-format det.txt; return a dict keyed by 1-based frame index.

    Args:
        det_txt: Path to ``det/det.txt``.
        conf_threshold: Minimum confidence — lower detections are dropped.

    Returns:
        Mapping from frame index to ``sv.Detections`` for that frame.
    """
    by_frame: dict[int, list[list[float]]] = {}
    for line in det_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 7:
            continue
        frame_idx = int(parts[0])
        conf = float(parts[6])
        if conf < conf_threshold:
            continue
        x, y, w, h = map(float, parts[2:6])
        by_frame.setdefault(frame_idx, []).append([x, y, x + w, y + h, conf])

    result: dict[int, sv.Detections] = {}
    for frame_idx, rows in by_frame.items():
        arr = np.array(rows, dtype=np.float32)
        result[frame_idx] = sv.Detections(xyxy=arr[:, :4], confidence=arr[:, 4])
    return result


def _annotate_frame(
    image: np.ndarray,
    detections: sv.Detections | None,
) -> np.ndarray:
    """Draw detection boxes and confidence labels on a BGR image.

    Each box is coloured on a red→green gradient keyed to its confidence score.

    Args:
        image: BGR image array (H, W, 3).
        detections: Detections for this frame, or ``None`` if no boxes.

    Returns:
        Annotated BGR image.
    """
    if detections is None or len(detections) == 0:
        return image

    if detections.confidence is None:
        return image
    for (x1, y1, x2, y2), conf in zip(detections.xyxy, detections.confidence):
        color = _conf_color(float(conf))
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{conf:.2f}"
        cv2.putText(
            image,
            label,
            (int(x1), max(int(y1) - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


def _process_sequence(
    seq_dir: Path,
    conf_threshold: float,
) -> int:
    """Visualise one sequence; returns the number of frames written.

    Args:
        seq_dir: Path to the sequence directory (must contain ``det/det.txt``).
            ``img1/`` is used if present; otherwise the FRCNN sibling's frames
            are used automatically (covers bundled DPM/SDP sequences).
        conf_threshold: Minimum confidence to include a detection.

    Returns:
        Number of frames actually written.
    """
    det_txt = seq_dir / "det" / "det.txt"
    vis_dir = seq_dir / "vis"

    img_dir = seq_dir / "img1"
    if not img_dir.exists():
        base = seq_dir.name.rsplit("-", 1)[0]  # "MOT17-02-DPM" → "MOT17-02"
        frcnn_img1 = seq_dir.parent / f"{base}-FRCNN" / "img1"
        if frcnn_img1.exists():
            img_dir = frcnn_img1
        else:
            console.print(f"[yellow]  skip {seq_dir.name} — no img1/[/yellow]")
            return 0
    if not det_txt.exists():
        console.print(f"[yellow]  skip {seq_dir.name} — no det/det.txt[/yellow]")
        return 0

    vis_dir.mkdir(exist_ok=True)
    det_by_frame = _load_detections(det_txt, conf_threshold)

    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not img_files:
        console.print(f"[yellow]  skip {seq_dir.name} — no images in img1/[/yellow]")
        return 0

    def _frame_idx(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0

    written = 0
    for img_path in track(img_files, description=f"  {seq_dir.name}", console=console):
        out_path = vis_dir / f"{img_path.stem}.jpg"
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        image = _annotate_frame(image, det_by_frame.get(_frame_idx(img_path)))
        cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1

    return written


def main(
    *sequences: str,
    det_source: str = "",
    data_dir: str = str(_DEFAULT_DATA_DIR),
    conf: float = 0.1,
) -> None:
    """Visualise detections for one or more MOT17 sequence directories.

    Args:
        *sequences: Sequence directory names or absolute paths (e.g.
            ``MOT17-04-RFDETR``). Relative names resolve under ``data_dir``.
            If omitted, ``--det-source`` selects all matching sequences.
        det_source: Detector tag shorthand (``yoloworld``, ``rfdetr``, ``frcnn``,
            ``sdp``, ``dpm``). Selects all matching sequences when no positional
            sequences are given.
        data_dir: Root directory containing MOT17-val sequences (default: cwd).
        conf: Minimum detection confidence to include (default 0.1).

    Examples:
        Single sequence::

            uv run python visualize_detections.py MOT17-04-YOLOWORLD

        All RF-DETR sequences::

            uv run python visualize_detections.py --det-source rfdetr
    """
    root = Path(data_dir).expanduser()
    seq_dirs: list[Path] = []

    if sequences:
        for s in sequences:
            p = Path(s).expanduser()
            seq_dirs.append(p if p.is_absolute() else root / s)
    elif det_source:
        tag = det_source.upper()
        seq_dirs = sorted(
            d for d in root.iterdir() if d.is_dir() and d.name.endswith(f"-{tag}")
        )
        if not seq_dirs:
            root_abs = root.resolve()
            dirs = [d.name for d in root_abs.iterdir() if d.is_dir()]
            console.print(f"[red]No sequences found for tag -{tag} in {root_abs}[/red]")
            console.print(f"Directories found: {dirs or '(none)'}")
            return
    else:
        console.print("[red]Provide a sequence name or --det-source TAG[/red]")
        raise SystemExit(1)

    total = 0
    for seq_dir in seq_dirs:
        if not seq_dir.exists():
            console.print(f"[red]Not found: {seq_dir}[/red]")
            continue
        console.print(f"[cyan]{seq_dir.name}[/cyan] → {seq_dir / 'vis'}")
        written = _process_sequence(seq_dir, conf)
        total += written
        console.print(f"  [green]✓ {written} frames[/green]")

    console.print(f"\n[bold green]Done — {total} total frames[/bold green]")


if __name__ == "__main__":
    fire.Fire(main)
