# AGENTS.md

Guidance for AI coding agents working in this repo. Read by GitHub Copilot, Claude Code, and other agent tooling. This file is the single committed source of that guidance — everything an agent needs is here or linked from here.

Roboflow Trackers is a Python library for multi-object tracking (MOT). It provides clean-room implementations of SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU, and McByte that plug into any detection model via the `supervision` library. End users install with `pip install trackers`; everything below targets work *inside* this repo.

Docs-only guidance (the benchmark-number cross-reference map, one per BENCH-XREF-tagged file) lives in [docs/AGENTS.md](docs/AGENTS.md) instead of here — it doesn't apply to source/test work, so keeping it out of this root file keeps every non-docs session from loading it. Touching `docs/` or `README.md` benchmark numbers? Read that file first.

## Build, Test & Validate (run before opening a PR)

```bash
uv sync --frozen --group dev      # install dev deps (Python >=3.10)
uv run pytest -m "not integration"  # unit tests
pre-commit run --all-files          # lint, format, types, license, spelling
```

Run the full `pre-commit` suite — never a single linter on its own. It wraps ruff, mypy, docformatter, mdformat, and codespell with the exact settings CI uses, so a passing hook run is the only meaningful signal.

Occasionally needed:

```bash
uv run pytest -m integration -v   # integration tests; downloads benchmark data
uv build                          # build package
uv sync --frozen --group docs && uv run mkdocs build --verbose  # build docs
```

## Code Conventions

Configured in `pyproject.toml`, enforced by the hooks in `.pre-commit-config.yaml`. The GitHub Actions workflows run only tests, build, and docs — style and typing are gated by `pre-commit.ci`, which auto-fixes PRs. So a clean local hook run is what keeps a PR quiet:

- **License header** — every `.py` file needs the Apache 2.0 header from `.github/LICENSE_HEADER.txt`. The `insert-license` pre-commit hook adds it; don't hand-write it.
- **Docstrings** — Google style (`Args:`, `Returns:`, `Raises:`, `Example:`). `Example:` blocks are executed as doctests (`--doctest-modules` is in `addopts`), so they must actually run.
- **Ruff** — line length 120, double quotes, max cyclomatic complexity 10, max 5 function args. Active rule sets: `E`, `F`, `I`, `A`, `Q`, `W`, `RUF`, `S`, `UP`.
- **No bare `assert`** outside `tests/` (ruff `S101`; `tests/**` is the only exemption).
- **Typing** — mypy runs over `src`, `tests`, and `demo`. Project is typed (`py.typed` shipped).
- **Public API** — anything exported from `src/trackers/__init__.py` is the stable surface; adding to it is an API change, not an implementation detail.

## File Structure

Src-layout package — code lives under `src/trackers/`, not at repo root:

```
src/trackers/
├── core/          # Tracker implementations
│   ├── base.py    # BaseTracker ABC — defines update(detections, image) interface
│   ├── sort/      # SORT
│   ├── bytetrack/ # ByteTrack
│   ├── botsort/   # BoT-SORT
│   ├── cbiou/     # C-BIoU
│   ├── ocsort/    # OC-SORT
│   ├── mcbyte/    # McByte
│   └── masks/     # Shared mask-pipeline utilities (SAM/Cutie) used by McByte
├── motion/        # MotionEstimator, homography compensation
├── utils/         # Kalman filter, coordinate converters (xcycsr ↔ xyxy)
├── annotators/    # MotionAwareTraceAnnotator
├── io/            # Video/webcam/RTSP frame reader, MOT format I/O
├── datasets/      # Dataset manifests and download helpers
├── eval/          # CLEAR / HOTA / Identity metrics
└── cli/           # `trackers` CLI entry point (jsonargparse-based)
```

## Core API

All trackers share the same interface:

```python
from trackers import SORTTracker, ByteTrackTracker, OCSORTTracker

tracker = ByteTrackTracker()
tracked = tracker.update(detections)  # detections: supervision.Detections
# tracked.tracker_id contains assigned track IDs
```

Detection format: input is `supervision.Detections` with `.xyxy` bounding boxes; output is the same object with `.tracker_id` populated. Internal Kalman state uses `xcycsr` (center-x, center-y, area, aspect ratio).

## Tracker Parameters

Constructor signatures differ per tracker — do not assume a parameter exists everywhere. Read the target tracker's `__init__` before using or documenting a parameter; the docstrings are the source of truth (they are parsed at import time to build the CLI).

Present on **all six** trackers: `lost_track_buffer`, `minimum_consecutive_frames`, `frame_rate`, `state_estimator_class`.

Deliberately **not** universal (common mistakes):

- `minimum_iou_threshold` — SORT / ByteTrack / OC-SORT only. BoT-SORT, C-BIoU, and McByte instead split it into `minimum_iou_threshold_{first,second,unconfirmed}_assoc`.
- `track_activation_threshold` — all trackers **except** OC-SORT.
- `high_conf_det_threshold` — all trackers **except** SORT.

Tracker-specific: OC-SORT adds `direction_consistency_weight` and `delta_t`; C-BIoU adds `buffer_ratio_{first,second}`; BoT-SORT and McByte add the `enable_cmc` / `cmc_method` / `cmc_downscale` motion-compensation trio; McByte adds the `*_mask_*` family.

## Documentation

Docs source lives in-repo under `docs/` (published at https://trackers.roboflow.com). Read the local files — they match the checked-out revision:

- [Quickstart](docs/index.md)
- [Install Trackers](docs/guides/install.md)
- [API Reference](docs/api/trackers.md)
- [Track Objects Guide](docs/guides/track.md)
- [CLI Reference](docs/guides/cli.md)
- [Evaluation Guide](docs/evaluations/evaluate.md)
- [Benchmark Results](docs/evaluations/results.md)

Canonical HOTA benchmark numbers live in [docs/evaluations/results.md](docs/evaluations/results.md). `docs/index.md` and `README.md` are hand-maintained mirrors — don't add a third copy. See [docs/AGENTS.md](docs/AGENTS.md) for the BENCH-XREF sync map.

Project is Apache 2.0 — https://github.com/roboflow/trackers
