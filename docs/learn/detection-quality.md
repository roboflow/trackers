# Detection Quality Matters

Tracking quality starts at the detector. If it misses an object, the tracker never gets a chance. This guide isolates the effect of detector choice by running ByteTrack with three models of increasing accuracy on the MOT17 benchmark.

**What you'll learn:**

- Run the same tracker with different detection models
- Measure how detector choice impacts tracking metrics
- Compare YOLO26 Nano, RF-DETR Nano, and RF-DETR Medium on MOT17

---

## Install

Install `trackers` with the detection extra to enable built-in model support.

```text
pip install trackers[detection]
```

For more options, see the [install guide](install.md).

---

## Detection Models

We pick three models that span a wide accuracy range on COCO, from a lightweight YOLO to a mid-size transformer detector. The gap in COCO accuracy between the weakest and strongest model is nearly 15 AP. The question is how much of that carries over to tracking.

![RF-DETR vs top object detectors on MS COCO](https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/rf_detr_1-2_latency_accuracy_object_detection.png)

<p align="center" style="margin-top: -0.4em;"><small>Comparison of RF-DETR against other top real-time detectors on MS COCO.</small></p>

|     Model      | COCO AP50 | COCO AP50:95 | Latency (ms) |
| :------------: | :-------: | :----------: | :----------: |
|  YOLO26 Nano   |   55.8    |     40.3     |     1.7      |
|  RF-DETR Nano  |   67.6    |     48.4     |     2.3      |
| RF-DETR Medium |   73.6    |     54.7     |     4.4      |

---

## Download Data

Pull the MOT17 validation split. You need frames for detection and annotations for evaluation.

```text
trackers download mot17 \
    --split val \
    --asset frames,annotations \
    --output ./data
```

---

## Run the Experiment

Run ByteTrack with default parameters three times, changing only the detection model each time.

### YOLO26 Nano

=== "Single sequence"

    ```bash
    trackers track \
        --source ./data/mot17/val/MOT17-13-FRCNN/img1 \
        --model yolo26n-640 \
        --tracker bytetrack \
        --classes person \
        --mot-output results/yolo26n/MOT17-13-FRCNN.txt
    ```

=== "All sequences"

    ```bash
    for seq in MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN; do
        trackers track \
            --source ./data/mot17/val/$seq/img1 \
            --model yolo26n-640 \
            --tracker bytetrack \
            --classes person \
            --mot-output results/yolo26n/$seq.txt
    done
    ```

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/yolo26n_MOT17-13-FRCNN.mp4" type="video/mp4">
</video>
<p align="center" style="margin-top: -0.4em;"><small>ByteTrack with YOLO26 Nano on MOT17-13.</small></p>

### RF-DETR Nano

=== "Single sequence"

    ```bash
    trackers track \
        --source ./data/mot17/val/MOT17-13-FRCNN/img1 \
        --model rfdetr-nano \
        --tracker bytetrack \
        --classes person \
        --mot-output results/rfdetr-nano/MOT17-13-FRCNN.txt
    ```

=== "All sequences"

    ```bash
    for seq in MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN; do
        trackers track \
            --source ./data/mot17/val/$seq/img1 \
            --model rfdetr-nano \
            --tracker bytetrack \
            --classes person \
            --mot-output results/rfdetr-nano/$seq.txt
    done
    ```

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/rfdetr_nano_MOT17-13-FRCNN.mp4" type="video/mp4">
</video>
<p align="center" style="margin-top: -0.4em;"><small>ByteTrack with RF-DETR Nano on MOT17-13.</small></p>

### RF-DETR Medium

=== "Single sequence"

    ```bash
    trackers track \
        --source ./data/mot17/val/MOT17-13-FRCNN/img1 \
        --model rfdetr-medium \
        --tracker bytetrack \
        --classes person \
        --mot-output results/rfdetr-medium/MOT17-13-FRCNN.txt
    ```

=== "All sequences"

    ```bash
    for seq in MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN; do
        trackers track \
            --source ./data/mot17/val/$seq/img1 \
            --model rfdetr-medium \
            --tracker bytetrack \
            --classes person \
            --mot-output results/rfdetr-medium/$seq.txt
    done
    ```

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/rfdetr_medium_MOT17-13-FRCNN.mp4" type="video/mp4">
</video>
<p align="center" style="margin-top: -0.4em;"><small>ByteTrack with RF-DETR Medium on MOT17-13.</small></p>

---

## Evaluate

Evaluate each run against ground truth using CLEAR, HOTA, and Identity metrics.

### YOLO26 Nano

```text
trackers eval \
    --gt-dir ./data/mot17/val \
    --tracker-dir results/yolo26n \
    --metrics CLEAR HOTA Identity \
    --columns MOTA HOTA IDF1
```

**Output:**

```
                                MOTA    HOTA    IDF1
----------------------------------------------------
COMBINED                      23.444  32.874  34.411
```

### RF-DETR Nano

```text
trackers eval \
    --gt-dir ./data/mot17/val \
    --tracker-dir results/rfdetr-nano \
    --metrics CLEAR HOTA Identity \
    --columns MOTA HOTA IDF1
```

**Output:**

```
                                MOTA    HOTA    IDF1
----------------------------------------------------
COMBINED                      25.667  35.735  38.182
```

### RF-DETR Medium

```text
trackers eval \
    --gt-dir ./data/mot17/val \
    --tracker-dir results/rfdetr-medium \
    --metrics CLEAR HOTA Identity \
    --columns MOTA HOTA IDF1
```

**Output:**

```
                                MOTA    HOTA    IDF1
----------------------------------------------------
COMBINED                      29.141  38.637  41.950
```

---

## Results

Same tracker, same data, same parameters. The only difference is the detector.

|    Detector    |    MOTA    |    HOTA    |    IDF1    |
| :------------: | :--------: | :--------: | :--------: |
|  YOLO26 Nano   |   23.444   |   32.874   |   34.411   |
|  RF-DETR Nano  |   25.667   |   35.735   |   38.182   |
| RF-DETR Medium | **29.141** | **38.637** | **41.950** |

RF-DETR Medium leads across every metric, showing that a stronger detector directly lifts tracking quality.

---

## Takeaway

Before tweaking tracker hyperparameters, invest in detection quality. The results above show that swapping the detector alone produces larger gains than most tracker-level optimizations.
