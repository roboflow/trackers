#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Aggregate benchmark score JSONs into markdown tables.

Two layouts (mirroring ``docs/trackers/comparison.md``):

1. **Per tracker** (``--tracker``): rows = datasets, columns = HOTA/IDF1/MOTA.
   Writes ``<output-dir>/<tracker>/tables.md``.

2. **Per dataset** (``--compare-dataset`` + ``--trackers``): rows = trackers.
   Writes ``<output-dir>/comparison/<dataset>/tables.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import (
    COMPARISON_TRACKERS,
    DATASETS,
    LABELS,
    TRACKER_LABELS,
    job_dir,
)

_CONFIGS = ("default", "tuned")
_TARGETS = ("HOTA", "IDF1", "MOTA")


def _from_eval_json(path: Path) -> dict[str, float]:
    """Read TrackEval-style JSON saved by `trackers eval --output ...` and scale to percent."""
    data = json.loads(path.read_text())
    agg = data.get("aggregate", {})
    out: dict[str, float] = {}
    for family in ("HOTA", "Identity", "CLEAR"):
        block = agg.get(family)
        if not isinstance(block, dict):
            continue
        for key in _TARGETS:
            if key in block and key not in out:
                out[key] = float(block[key]) * 100.0
    return out


def _from_codabench_json(path: Path) -> dict[str, float]:
    """Read Codabench summary written by `codabench_submit.py --output ...`. Scores already 0-100."""
    data = json.loads(path.read_text())
    scores = data.get("scores") or {}
    return {k: float(v) for k, v in scores.items() if k in _TARGETS}


def _row_scores(out_dir: Path, tracker: str, dataset: str, config: str) -> dict[str, float] | None:
    base = job_dir(out_dir, tracker, dataset) / config
    eval_json = base / "eval.json"
    cb_json = base / "codabench.json"
    if cb_json.is_file():
        return _from_codabench_json(cb_json)
    if eval_json.is_file():
        return _from_eval_json(eval_json)
    return None


def _format_table(
    rows: list[tuple[str, dict[str, float] | None]],
    *,
    row_header: str,
    row_width: int,
) -> str:
    header = f"|  {row_header}  |   HOTA   |   IDF1   |   MOTA   |"
    sep = "| :-------: | :------: | :------: | :------: |"
    body = []
    for label, scores in rows:
        label_cell = f"{label:^{row_width}}"
        if scores is None:
            body.append(f"| {label_cell} |   —    |   —    |   —    |")
        else:
            hota = f"{scores['HOTA']:6.1f}" if "HOTA" in scores else "  —   "
            idf1 = f"{scores['IDF1']:6.1f}" if "IDF1" in scores else "  —   "
            mota = f"{scores['MOTA']:6.1f}" if "MOTA" in scores else "  —   "
            body.append(f"| {label_cell} | {hota} | {idf1} | {mota} |")
    return "\n".join([header, sep, *body])


def _collect_tracker(out_dir: Path, tracker: str, datasets: list[str]) -> int:
    summary: dict[str, dict[str, dict[str, float] | None]] = {}
    sections: list[str] = []

    for config in _CONFIGS:
        rows = []
        any_present = False
        for dataset in datasets:
            scores = _row_scores(out_dir, tracker, dataset, config)
            rows.append((LABELS.get(dataset, dataset), scores))
            summary.setdefault(dataset, {})[config] = scores
            if scores is not None:
                any_present = True
        if not any_present:
            continue
        title = "Default parameters" if config == "default" else "Tuned parameters"
        sections.append(f"## {title}\n\n{_format_table(rows, row_header='Dataset', row_width=9)}\n")

    out = out_dir / tracker
    out.mkdir(parents=True, exist_ok=True)
    md = f"# {tracker} benchmark\n\n" + ("\n".join(sections) if sections else "_No scores found yet._\n")
    (out / "tables.md").write_text(md)
    (out / "summary.json").write_text(json.dumps({"tracker": tracker, "datasets": summary}, indent=2))

    print(md)
    print(f"saved → {out / 'tables.md'}")
    print(f"saved → {out / 'summary.json'}")
    return 0 if sections else 1


def _collect_comparison(out_dir: Path, dataset: str, trackers: list[str]) -> int:
    if dataset not in DATASETS:
        print(f"unknown dataset: {dataset!r}", file=sys.stderr)
        return 1

    summary: dict[str, dict[str, dict[str, float] | None]] = {}
    sections: list[str] = []
    dataset_label = LABELS.get(dataset, dataset)

    for config in _CONFIGS:
        rows = []
        any_present = False
        for tracker in trackers:
            label = TRACKER_LABELS.get(tracker, tracker)
            scores = _row_scores(out_dir, tracker, dataset, config)
            rows.append((label, scores))
            summary.setdefault(tracker, {})[config] = scores
            if scores is not None:
                any_present = True
        if not any_present:
            continue
        title = "Default parameters" if config == "default" else "Tuned parameters"
        sections.append(f"## {title}\n\n{_format_table(rows, row_header='Tracker', row_width=9)}\n")

    out = out_dir / "comparison" / dataset
    out.mkdir(parents=True, exist_ok=True)
    md = f"# {dataset_label} — tracker comparison\n\n" + (
        "\n".join(sections) if sections else "_No scores found yet._\n"
    )
    (out / "tables.md").write_text(md)
    (out / "summary.json").write_text(
        json.dumps({"dataset": dataset, "trackers": summary}, indent=2),
    )

    print(md)
    print(f"saved → {out / 'tables.md'}")
    print(f"saved → {out / 'summary.json'}")
    return 0 if sections else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tracker", help="Single tracker — rows are datasets (default collect mode).")
    p.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated subset; default=all.")
    p.add_argument(
        "--compare-dataset",
        help="One dataset — rows are trackers (comparison.md layout). Requires --trackers.",
    )
    p.add_argument(
        "--trackers",
        default=",".join(COMPARISON_TRACKERS),
        help="Comma-separated tracker ids for --compare-dataset.",
    )
    args = p.parse_args(argv)

    if args.compare_dataset:
        trackers = [t.strip() for t in args.trackers.split(",") if t.strip()]
        return _collect_comparison(args.output_dir, args.compare_dataset, trackers)

    if not args.tracker:
        p.error("pass --tracker or --compare-dataset")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    return _collect_tracker(args.output_dir, args.tracker, datasets)


if __name__ == "__main__":
    raise SystemExit(main())
