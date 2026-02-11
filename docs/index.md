---
comments: true
---

<div align="center">

<img src="assets/logo-trackers-violet.svg" alt="Trackers Logo" width="200" height="200">

</div>

`trackers` gives you clean, modular re-implementations of leading multi-object tracking algorithms released under the permissive Apache 2.0 license. You combine them with any detection model you already use.

## Install

You can install and use `trackers` in a [**Python>=3.10**](https://www.python.org/) environment. For detailed installation instructions, including installing from source and setting up a local development environment, check out our [install](learn/install.md) page.

!!! example "Installation"

    [![version](https://badge.fury.io/py/trackers.svg)](https://badge.fury.io/py/trackers)
    [![downloads](https://img.shields.io/pypi/dm/trackers)](https://pypistats.org/packages/trackers)
    [![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/trackers/blob/main/LICENSE.md)
    [![python-version](https://img.shields.io/pypi/pyversions/trackers)](https://badge.fury.io/py/trackers)

    === "pip"

        ```bash
        pip install trackers
        ```

    === "uv"

        ```bash
        uv pip install trackers
        ```

## Tutorials

<div class="grid cards" markdown>

- **How to Track Objects with SORT**

    ---

    [![](https://storage.googleapis.com/com-roboflow-marketing/trackers/assets/sort-sample.png)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-sort-tracker.ipynb)

    End-to-end example showing how to run RF-DETR detection with the SORT tracker.

    [:simple-googlecolab: Run Google Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-sort-tracker.ipynb)

- **How to Track Objects with ByteTrack**

    ---

    [![](https://storage.googleapis.com/com-roboflow-marketing/trackers/assets/bytetrack-sample.png)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-bytetrack-tracker.ipynb)

    End-to-end example showing how to run RF-DETR detection with the ByteTrack tracker.

    [:simple-googlecolab: Run Google Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-bytetrack-tracker.ipynb)

-   **How to Track Objects with OC-SORT**

    ---

    [![](url-to-image)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-ocsort-tracker.ipynb)

    End-to-end example showing how to run RF-DETR detection with the OC-SORT tracker.

    [:simple-googlecolab: Run Google Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-ocsort-tracker.ipynb)

</div>

## Tracking Algorithms

`trackers` gives you clean, modular re-implementations of leading multi-object tracking algorithms. The package currently supports [SORT](https://arxiv.org/abs/1602.00763) and [ByteTrack](https://arxiv.org/abs/2110.06864). [OC-SORT](https://arxiv.org/abs/2203.14360), [BoT-SORT](https://arxiv.org/abs/2206.14651), and [McByte](https://arxiv.org/abs/2506.01373) support is coming soon. For comparisons, see the [tracker comparison](trackers/comparison.md) page.

| Algorithm | MOT17 HOTA | MOT17 IDF1 | MOT17 MOTA | SportsMOT HOTA | SoccerNet HOTA |
| :-------: | :--------: | :--------: | :--------: | :------------: | :------------: |
|   SORT    |    58.4    |    69.9    |    67.2    |      70.9      |      81.6      |
| ByteTrack |  **60.1**  |  **73.2**  |  **74.1**  |    **73.0**    |    **84.0**    |
|  OC-SORT  |     —      |     —      |     —      |       —        |       —        |
| BoT-SORT  |     —      |     —      |     —      |       —        |       —        |
|  McByte   |     —      |     —      |     —      |       —        |       —        |

## Integration

With a modular design, `trackers` lets you combine object detectors from different libraries with the tracker of your choice. See the [Track guide](learn/track.md) for CLI usage and more options. These examples use `opencv-python` for decoding and display.

=== "RF-DETR"

    ```python
    import cv2
    from rfdetr import RFDETRNano
    from trackers import ByteTrackTracker

    model = RFDETRNano()
    tracker = ByteTrackTracker()

    cap = cv2.VideoCapture("source.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb)
        detections = tracker.update(detections)
    ```

=== "Inference"

    ```python
    import cv2
    import supervision as sv
    from inference import get_model
    from trackers import ByteTrackTracker

    model = get_model(model_id="rfdetr-nano")
    tracker = ByteTrackTracker()

    cap = cv2.VideoCapture("source.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections)
    ```

=== "Ultralytics"

    ```python
    import cv2
    import supervision as sv
    from ultralytics import YOLO
    from trackers import ByteTrackTracker

    model = YOLO("yolo11n.pt")
    tracker = ByteTrackTracker()

    cap = cv2.VideoCapture("source.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update(detections)
    ```
