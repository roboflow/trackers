#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trackers.calibration.export import load_calibration_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize calibration coverage and confidence from JSONL output."
    )
    parser.add_argument("calibration", help="Calibration JSONL file")
    return parser.parse_args()


def _longest_gap(valid_flags: list[bool]) -> int:
    longest = 0
    current = 0
    for is_valid in valid_flags:
        if is_valid:
            current = 0
            continue
        current += 1
        longest = max(longest, current)
    return longest


def main(args: argparse.Namespace) -> int:
    path = Path(args.calibration).expanduser().resolve()
    frames = load_calibration_jsonl(path)
    valid_frames = [frame for frame in frames if frame.has_homography]
    valid_flags = [frame.has_homography for frame in frames]
    confidences = [
        frame.confidence for frame in valid_frames if frame.confidence is not None
    ]

    summary = {
        "path": str(path),
        "num_frames": len(frames),
        "num_valid_frames": len(valid_frames),
        "coverage": 0.0 if not frames else len(valid_frames) / len(frames),
        "mean_confidence": (
            None if not confidences else sum(confidences) / len(confidences)
        ),
        "longest_invalid_gap_frames": _longest_gap(valid_flags),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args))
