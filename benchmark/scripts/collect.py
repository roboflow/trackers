#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Aggregate per-dataset eval/Codabench score JSONs into a single doc-style markdown table.

Looks under ``<output-dir>/<tracker>/<dataset>/<config>/`` (config ∈ {default, tuned}) for:

  - ``eval.json``    → from `trackers eval --output ...` (SoccerNet local eval)
  - ``codabench.json`` → from `codabench_submit.py --output ...` (MOT17/SportsMOT/DanceTrack)

Writes ``<output-dir>/<tracker>/tables.md`` and ``<output-dir>/<tracker>/summary.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import DATASETS, LABELS, job_dir

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


def _format_table(rows: list[tuple[str, dict[str, float] | None]]) -> str:
    header = "|  Dataset  | HOTA | IDF1 | MOTA |"
    sep = "| :-------: | :--: | :--: | :--: |"
    body = []
    for label, scores in rows:
        if scores is None:
            body.append(f"| {label:^9} |  —   |  —   |  —   |")
        else:
            cells = " | ".join(f"{scores.get(k, float('nan')):4.1f}" if k in scores else "  — " for k in _TARGETS)
            body.append(f"| {label:^9} | {cells} |")
    return "\n".join([header, sep, *body])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracker", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated subset; default=all.")
    args = p.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    summary: dict[str, dict[str, dict[str, float] | None]] = {}
    sections: list[str] = []

    for config in _CONFIGS:
        rows = []
        any_present = False
        for d in datasets:
            scores = _row_scores(args.output_dir, args.tracker, d, config)
            rows.append((LABELS.get(d, d), scores))
            summary.setdefault(d, {})[config] = scores
            if scores is not None:
                any_present = True
        if not any_present:
            continue
        title = "Default parameters" if config == "default" else "Tuned parameters"
        sections.append(f"## {title}\n\n{_format_table(rows)}\n")

    out = args.output_dir / args.tracker
    out.mkdir(parents=True, exist_ok=True)
    md = f"# {args.tracker} benchmark\n\n" + ("\n".join(sections) if sections else "_No scores found yet._\n")
    (out / "tables.md").write_text(md)
    (out / "summary.json").write_text(json.dumps({"tracker": args.tracker, "datasets": summary}, indent=2))

    print(md)
    print(f"saved → {out / 'tables.md'}")
    print(f"saved → {out / 'summary.json'}")
    return 0 if sections else 1


if __name__ == "__main__":
    raise SystemExit(main())
