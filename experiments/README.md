# autotrack

Autonomous improvement of ByteTrack on MOT17, using the
[autoresearch](https://github.com/karpathy/autoresearch) pattern with
[Roboflow trackers](https://github.com/roboflow/trackers).

Point any coding agent at this folder and let it run.
~500 experiments/hour on CPU, no GPU needed.

## Motivation

Multi-object tracking quality depends on two largely independent axes: the
**algorithm design** (state representation, association logic, track lifecycle) and
**hyperparameter tuning** (confidence thresholds, buffer sizes, Kalman noise scales).
Most published improvements conflate the two — a well-tuned weak algorithm can
outperform a poorly-tuned strong one, making it hard to know what actually matters.

This project separates the axes. An autonomous agent iterates over structural code
changes to the ByteTrack implementation, measuring HOTA after each change at fixed
default parameters. Optuna provides a second-pass validation: after a code change is
accepted, a short tuning run confirms the improvement holds under optimised parameters
and is not a parameter artefact. The iteration log — including all reverted changes —
is the primary research artifact.

**Why ByteTrack?** It is the simplest practically-competitive tracker: pure IoU
association, constant-velocity Kalman filter, no appearance features. Its simplicity
makes it easy to isolate the effect of individual algorithmic changes.

**Why MOT17 with FRCNN detections?** FRCNN public detections are bundled with the
benchmark and require no detector to reproduce. They are weaker than modern detectors
(HOTA ~50 vs ~60 with YOLOX), which creates genuine headroom for algorithmic
improvement. Any agent, on any machine, sees the same inputs.

**Why HOTA?** HOTA (Higher Order Tracking Accuracy, Luiten et al. 2021) decomposes
tracking quality into detection accuracy and association accuracy with equal weight.
MOTA is dominated by false positives/negatives and misses ID-switch quality; IDF1 is
purely association-focused. HOTA is the most informative single scalar for overall
tracker health.

## Approach

The research loop follows the autoresearch pattern: propose one change, measure it,
keep improvements, revert regressions. Each committed iteration is one atomic
hypothesis. The JSONL experiment log captures every attempt — failures are as
informative as successes.

```
Human defines:  research question  ·  metric  ·  hard boundaries
Agent decides:  what to change  ·  what to try next
```

Two tools govern the loop:

| Tool                                | Role                                                                                   |
| ----------------------------------- | -------------------------------------------------------------------------------------- |
| `optimize_tracking.py --n-trials 1` | Campaign metric — evaluates default params, gives a clean code-change signal           |
| `optimize_tracking.py --n-trials N` | Optuna study — warm-starts from `best_config.json`, finds best params for current code |

The agent is free to update `optimize_tracking.py` itself as the tracker architecture
evolves — adding parameters that newly exist, removing ones that were absorbed into
the implementation, tightening search ranges as knowledge accumulates.

## Target analysis

The campaign target of HOTA = 60.0 is set at the theoretical ceiling for IoU-only
trackers with FRCNN public detections. Here is the derivation.

**HOTA formula**: HOTA = √(DetA × AssA) × 100, where DetA measures bounding-box
detection accuracy and AssA measures ID-consistency over time.

**DetA ceiling from FRCNN**: FRCNN public detections on MOT17 have limited recall and
precision — DetA is empirically bounded to ≈ 0.55–0.62 regardless of the tracker.
A perfect tracker cannot recover detections the detector missed.

**AssA potential**: ByteTrack-class IoU-only association without ReID features achieves
AssA ≈ 0.65–0.75 with well-tuned parameters. With algorithmic improvements (better
Kalman dynamics, improved association) this could approach 0.78–0.82.

**Estimated ceilings**:

| Scenario                                | DetA | AssA | HOTA |
| --------------------------------------- | ---- | ---- | ---- |
| Default params, current code (baseline) | 0.57 | 0.44 | 50.4 |
| Optuna only, no code changes            | 0.57 | 0.55 | 56.0 |
| Code improvements + Optuna              | 0.59 | 0.65 | 61.9 |
| Theoretical IoU-only ceiling            | 0.62 | 0.65 | 63.5 |

The DetA/AssA split at baseline is estimated from published ByteTrack HOTA
decompositions on comparable benchmarks. Exact values depend on the evaluation
threshold distribution used by HOTA.

**Published reference points** (IoU-only, no ReID, FRCNN public detections):

- ByteTrack — MOT17 test: HOTA ≈ 47.5; val ≈ 50–52 (detector-limited)
- OC-SORT — MOT17 test: HOTA ≈ 52.4; val ≈ 55–57
- BoT-SORT (no ReID) — MOT17 test: HOTA ≈ 53.1; val ≈ 56–58

Val scores run ~3–5 points above test scores on MOT17 (easier sequences). Reaching
60.0 on val means the evolved ByteTrack is competitive with BoT-SORT at best, or has
genuinely pushed the IoU-only ceiling — either outcome is a publishable result.

## Hard boundaries

See `program.md` for the full contract. The short version:

- Metrics are computed via `trackers.eval` — no substitution
- Data is MOT17-val FRCNN public detections — no oracle, no ground truth at inference
- The tracker must remain ByteTrack — evolve it, do not replace it with a different algorithm

## Setup (first run only)

```bash
uv sync --group optimize          # installs optuna[rdb] + fire
trackers download mot17 --split val --asset annotations,detections
uv run python optimize_tracking.py --fast  # ~3s sanity check, expect HOTA ~50.4
```

## Pre-flight checks

Before starting the campaign loop, all three setup steps must pass:

| Check | Command | Expected result |
| ----- | ------- | --------------- |
| Dependencies | `uv sync --group optimize` | Resolves without error |
| MOT17 data | `trackers download mot17 --split val --asset annotations,detections` | Downloads ~2 GB to `~/.cache/trackers/mot17/val/` |
| Metric sanity | `cd experiments && uv run python optimize_tracking.py --fast` | Prints `__METRICS__: HOTA≈50.4` |

The campaign metric command uses `uv run` — bare `python` will fail with
`ModuleNotFoundError: No module named 'fire'` because `fire` only lives in
the `uv` virtualenv.

## Run the agent

```bash
claude  # or any coding agent
> Read program.md and start the experiment loop.
```

### Run with /optimize campaign

If you use [Borda's Claude Code skill suite](https://github.com/Borda/.ai-home), the
`/optimize` skill drives the loop directly from `program.md`:

```bash
claude
> /optimize campaign experiments/program.md
```

The skill handles the full iteration loop — baseline measurement, agent-driven code
changes, metric verification, auto-rollback on regression, and a final results report.
To run a tuning-only pass (Optuna, no code changes), set `agent_strategy: perf` in
`program.md` before launching. See the skill docs for `--team` and `--codex` flags.

## Files

| File                   | Who edits | Purpose                                    |
| ---------------------- | --------- | ------------------------------------------ |
| `README.md`            | Human     | This file                                  |
| `program.md`           | Human     | Research contract + hard boundaries        |
| `optimize_tracking.py` | Agent     | Optuna runner — agent updates search space |
| `best_config.json`     | Agent     | Best params found so far                   |

## References

- **ByteTrack**: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022
- **SORT**: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016
- **HOTA**: Luiten et al., "HOTA: A Higher Order Metric for Evaluating Multi-object Tracking", IJCV 2021
- **Optuna**: Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework", KDD 2019
- **MOT17**: Milan et al., "MOT16: A Benchmark for Multi-Object Tracking", arXiv 2016
- **autoresearch pattern**: Karpathy, autonomous research loop via coding agents
