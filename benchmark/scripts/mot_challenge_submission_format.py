#!/usr/bin/env python3
"""Normalize tracker MOT outputs for MOTChallenge / Codabench submission.

Tracking output should already omit unassigned rows (``tracker_id=-1``) and use
0-based track IDs with ``.1f`` box coordinates and ``conf=-1``. This step
drops any negative IDs as a safety net and rewrites rows to the notebook /
docs submission layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def normalize_mot_submission_line(line: str) -> str | None:
    parts = line.strip().split(",")
    if len(parts) < 7:
        return None
    try:
        frame = int(float(parts[0]))
        track_id = int(float(parts[1]))
    except ValueError:
        return None
    if track_id < 0:
        return None

    left, top, width, height = (float(parts[i]) for i in range(2, 6))
    return (
        f"{frame},{track_id},{left:.1f},{top:.1f},{width:.1f},{height:.1f},"
        f"-1,-1,-1,-1"
    )


def normalize_mot_submission_file(path: Path) -> int:
    lines_out: list[str] = []
    for raw in path.read_text().splitlines():
        normalized = normalize_mot_submission_line(raw)
        if normalized is not None:
            lines_out.append(normalized)
    path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return len(lines_out)


def normalize_mot_submission_dir(out_dir: Path) -> int:
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {out_dir}")
    total_lines = 0
    n_files = 0
    for path in sorted(out_dir.glob("*.txt")):
        total_lines += normalize_mot_submission_file(path)
        n_files += 1
    print(f"  MOT submission format: {n_files} files, {total_lines} lines in {out_dir}")
    return total_lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir", type=Path, help="Directory of per-sequence .txt files (modified in place).")
    args = p.parse_args(argv)
    try:
        normalize_mot_submission_dir(args.out_dir.resolve())
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
