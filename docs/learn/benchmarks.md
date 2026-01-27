# Benchmarks

This page presents performance results of the trackers available in the `trackers` Python package on standard multiple object tracking (MOT) benchmarks.

The goal of this page is not only to report scores, but also to help you interpret them for real world projects.

## Datasets

### MOT17

Classic pedestrian MOT benchmark based on MOT16 sequences but with significantly improved, higher-accuracy ground truth annotations. Contains crowded street scenes captured from static and slowly moving cameras, featuring frequent occlusions, moderate density (up to ~30–40 pedestrians), and partial camera motion in some sequences. Strong emphasis on handling short- to medium-term occlusions and re-acquiring identities after crossings or background clutter.

| Tracker    | HOTA     | IDF1     | MOTA     |
|------------|----------|----------|----------|
| SORT       | 58.4     | 69.9     | 67.2     |
| ByteTrack  | **60.1** | **73.2** | **74.1** |

### SportsMOT

Large-scale MOT dataset targeting sports analysis, containing 240 video clips (basketball, football/soccer, volleyball) collected from professional matches (Olympics, NCAA, NBA). Characterized by fast and highly variable player motion, rapid camera panning/zooming, complex backgrounds, and players with very similar appearance (uniforms within team). Only players on the court/field are annotated (excluding referees, coaches, spectators), making it particularly challenging for association under speed and visual similarity.

| Tracker   | HOTA     | IDF1     | MOTA     |
|-----------|----------|----------|----------|
| SORT      | 70.9     | 68.9     | 95.7     |
| ByteTrack | **73.0** | **72.5** | **96.4** |

### SoccerNet

Specialized soccer MOT dataset derived from SoccerNet broadcast videos, containing 200 short 30-second clips + one full 45-minute half-time sequence. Focuses on professional soccer matches with main camera view, including players (both teams), goalkeepers, referees, and the ball; features long sequences with occlusions by players, fast directional changes, and non-linear trajectories. Particularly useful for evaluating long-term association and robustness to crowded penalty areas or midfield clusters.

| Tracker    | HOTA     | IDF1     | MOTA     |
|------------|----------|----------|----------|
| SORT       | 81.6     | 76.2     | 95.1     |
| ByteTrack  | **84.0** | **78.1** | **97.8** |

## Metrics

### HOTA (Higher Order Tracking Accuracy)

HOTA is a balanced tracking metric that evaluates detection quality, localization accuracy, and identity association at the same time. The score is computed over a range of IoU thresholds, from loose to strict, which helps capture both easy and hard matching cases. Detection and association scores are then combined into a single value, giving a more complete view of tracker behavior than metrics focused on one aspect only. The metric is computed after processing the full video sequence and gives limited penalty when the same object is represented by several shorter tracks instead of one continuous track.

Use HOTA when you need an overall assessment of tracking quality and want to compare systems under realistic conditions where both detection errors and association errors matter.

### IDF1 (Identification F1 Score)

IDF1 measures how well a tracker keeps the same ID for the same object over time. It matches predicted IDs to ground truth IDs across the whole video and scores identity precision and recall. The metric reacts strongly to ID switches and lost identities over long periods. Detection coverage has little effect on the score if the remaining IDs stay consistent.

Use IDF1 when stable track IDs matter more than full object coverage, for example trajectory analysis, player tracking, or any logic built on long lived identities.

### MOTA (Multi-Object Tracking Accuracy)

MOTA sums three types of errors. Missed objects, false detections, and ID switches. The score reflects how often a tracker makes basic mistakes frame to frame. Missed detections and false positives dominate the result. The metric does not account for box accuracy and does not reflect long term identity consistency.

Use MOTA when detection coverage represents the main objective and identity stability plays a secondary role.

## Reproducibility

All reported numbers were obtained using the official [TrackEval](https://github.com/JonathonLuiten/TrackEval) library with public MOT Challenge detections for MOT17 and dataset-provided detections for SportsMOT and SoccerNet.

We are currently working on releasing dedicated benchmarking utilities in the `trackers` package that will allow users to easily reproduce these exact numbers, run evaluation on custom trackers using the same protocol, and benchmark their own models on these datasets with standardized splits and settings.

Stay tuned for the upcoming release of the benchmarking module.