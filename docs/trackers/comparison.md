# Tracker Comparison

This page compares Trackers implementations on common MOT benchmarks using `trackers.eval` with TrackEval-compatible metrics. Use these numbers for relative comparisons; absolute scores can shift with different detector quality, data splits, or evaluation settings.

!!! note
    We will provide hosted benchmark datasets through our API in a future release. For now, download datasets locally and evaluate with the same protocol for apples-to-apples comparisons.

## MOT17

Pedestrian tracking with crowded scenes and frequent occlusions. Strongly tests re-identification and identity stability.

| Tracker    | HOTA | IDF1 | MOTA |
|------------|------|------|------|
| SORT       | 58.4 | 69.9 | 67.2 |
| ByteTrack  | 60.1 | 73.2 | 74.1 |

## SportsMOT

Sports broadcast tracking with fast motion, camera pans, and similar-looking targets. Tests association under speed and appearance ambiguity.

| Tracker   | HOTA | IDF1 | MOTA |
|-----------|------|------|------|
| SORT      | 70.9 | 68.9 | 95.7 |
| ByteTrack | 73.0 | 72.5 | 96.4 |

## SoccerNet-tracking

Long sequences with dense interactions and partial occlusions. Tests long-term ID consistency.

| Tracker    | HOTA | IDF1 | MOTA |
|------------|------|------|------|
| SORT       | 81.6 | 76.2 | 95.1 |
| ByteTrack  | 84.0 | 78.1 | 97.8 |

## DanceTrack

Fast-moving, overlapping targets with frequent motion changes. Tests association robustness under rapid motion.

| Tracker    | HOTA | IDF1 | MOTA |
|------------|------|------|------|
| SORT       | N/A  | N/A  | N/A  |
| ByteTrack  | N/A  | N/A  | N/A  |

See also: [Evaluate guide](../learn/evaluate.md)
