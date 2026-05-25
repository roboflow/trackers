#!/usr/bin/env python3
"""Expand MOT17-XX.txt tracker outputs into Codabench/MOTChallenge server layout."""

from __future__ import annotations

import argparse
from pathlib import Path

# Sequences with YOLOX test detections in this benchmark setup.
_EXISTING = ("01", "03", "06", "07", "08", "12", "14")
# Sequences without test detections — server expects empty placeholder files.
_MISSING = ("02", "04", "05", "09", "10", "11", "13")
_SUFFIXES = ("FRCNN", "SDP", "DPM")


def write_mot17_server_format(out_dir: Path) -> int:
    """Triplicate tracked results and add empty files for missing sequences."""
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {out_dir}")

    written = 0
    for num in _EXISTING:
        src = out_dir / f"MOT17-{num}.txt"
        if not src.is_file():
            print(f"  Missing expected source: {src}", flush=True)
            continue
        content = src.read_bytes()
        for suf in _SUFFIXES:
            (out_dir / f"MOT17-{num}-{suf}.txt").write_bytes(content)
            written += 1
        src.unlink()

    for num in _MISSING:
        for suf in _SUFFIXES:
            (out_dir / f"MOT17-{num}-{suf}.txt").touch(exist_ok=True)
            written += 1

    print(f"  MOT17 server format: {written} files in {out_dir}", flush=True)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "out_dir",
        type=Path,
        help="Directory containing MOT17-XX.txt tracker outputs (modified in place).",
    )
    args = p.parse_args(argv)
    try:
        write_mot17_server_format(args.out_dir.resolve())
    except FileNotFoundError as exc:
        print(str(exc), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
