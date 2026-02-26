# Tracker Comparison

This page shows head-to-head performance of SORT and ByteTrack on standard MOT benchmarks.

## [MOT17](https://arxiv.org/abs/1603.00831)

Pedestrian tracking with crowded scenes and frequent occlusions. Strongly tests re-identification and identity stability.

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/datasets/MOT17_MOT17-04-DPM-1280x720.mp4" type="video/mp4">
</video>

|  Tracker  |   HOTA   |   IDF1   |   MOTA   |
| :-------: | :------: | :------: | :------: |
|   SORT    |   58.4   |   69.9   |   67.2   |
| ByteTrack |   60.1   |   73.2   |   74.1   |
|  OC-SORT  | **61.9** | **76.1** | **76.7** |

## [SportsMOT](https://arxiv.org/abs/2304.05170)

Sports broadcast tracking with fast motion, camera pans, and similar-looking targets. Tests association under speed and appearance ambiguity.

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/datasets/SportsMOT_v_-6Os86HzwCs_c001-1280x720.mp4" type="video/mp4">
</video>

|  Tracker  |   HOTA   |   IDF1   |   MOTA   |
| :-------: | :------: | :------: | :------: |
|   SORT    |   70.9   |   68.9   |   95.7   |
| ByteTrack | **73.0** | **72.5** | **96.4** |
|  OC-SORT  |   71.5   |   71.2   |   95.2   |

## [SoccerNet-tracking](https://arxiv.org/abs/2204.06918)

Long sequences with dense interactions and partial occlusions. Tests long-term ID consistency.

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/datasets/SoccerNet-tracking_SNMOT-060-1280x720.mp4" type="video/mp4">
</video>

|  Tracker  |   HOTA   |   IDF1   |   MOTA   |
| :-------: | :------: | :------: | :------: |
|   SORT    |   81.6   |   76.2   |   95.1   |
| ByteTrack | **84.0** | **78.1** | **97.8** |
|  OC-SORT  |   78.6   |   72.7   |   94.5   |

## [DanceTrack](https://arxiv.org/abs/2111.14690)

Group dancing tracking with uniform appearance, diverse motions, and extreme articulation. Tests motion-based association without relying on visual discrimination.

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/datasets/DanceTrack_dancetrack0052-1280x720.mp4" type="video/mp4">
</video>

!!! tip "Stay Tuned"

    We're actively benchmarking trackers on DanceTrack. Results will be added here shortly.
