#!/usr/bin/env python3
"""Format a MOT prediction directory for Codabench submission.

Steps applied in order:

1. Normalize every line to ``frame,id,left,top,w,h,-1,-1,-1,-1`` and drop rows with id < 0.
2. For MOT17, triplicate ``MOT17-XX`` → ``MOT17-XX-{FRCNN,SDP,DPM}`` and stub the
   sequences not present in the YOLOX detection set.
3. Zip every ``*.txt`` at the archive root (no nested directories).

Usage:

    python mot_format.py --dataset mot17 --pred-dir <preds> --out-zip <bundle.zip>
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from datasets import DATASETS, mot17_server_filenames


def _normalize_line(raw: str) -> str | None:
    parts = raw.strip().split(",")
    if len(parts) < 7:
        return None
    try:
        frame = int(float(parts[0]))
        track_id = int(float(parts[1]))
    except ValueError:
        return None
    if track_id < 0:
        return None
    left, top, w, h = (float(parts[i]) for i in range(2, 6))
    return f"{frame},{track_id},{left:.1f},{top:.1f},{w:.1f},{h:.1f},-1,-1,-1,-1"


def normalize_dir(pred_dir: Path) -> None:
    for path in sorted(pred_dir.glob("*.txt")):
        cleaned = [line for raw in path.read_text().splitlines() if (line := _normalize_line(raw))]
        path.write_text("\n".join(cleaned) + ("\n" if cleaned else ""))


def mot17_triplicate(pred_dir: Path) -> None:
    """MOT17 Codabench server expects FRCNN/SDP/DPM triplets and zero-fills for missing seqs."""
    existing, missing, suffixes = mot17_server_filenames()
    for num in existing:
        src = pred_dir / f"MOT17-{num}.txt"
        if not src.is_file():
            continue
        data = src.read_bytes()
        for suf in suffixes:
            (pred_dir / f"MOT17-{num}-{suf}.txt").write_bytes(data)
        src.unlink()
    for num in missing:
        for suf in suffixes:
            (pred_dir / f"MOT17-{num}-{suf}.txt").touch(exist_ok=True)


def zip_dir(pred_dir: Path, out_zip: Path) -> Path:
    if out_zip.is_file():
        out_zip.unlink()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(pred_dir.glob("*.txt")):
            zf.write(path, arcname=path.name)
    return out_zip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--pred-dir", type=Path, required=True, help="Directory of MOT prediction txt files (modified in place).")
    p.add_argument("--out-zip", type=Path, required=True)
    args = p.parse_args(argv)

    if not args.pred_dir.is_dir():
        print(f"missing pred dir: {args.pred_dir}", file=sys.stderr)
        return 1
    normalize_dir(args.pred_dir)
    if args.dataset == "mot17":
        mot17_triplicate(args.pred_dir)
    zip_path = zip_dir(args.pred_dir, args.out_zip)
    print(f"wrote {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
