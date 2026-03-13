# Detection Quality Matters

TODO: One-liner about how your tracker is only as good as your detections.

**What you'll learn:**

- Run the same tracker with different detection models
- Measure how detector choice impacts tracking metrics
- Compare YOLOv11, YOLO26, RF-DETR Nano, and RF-DETR Medium on MOT17

---

## Install

TODO: One sentence about installing with the detection extra.

```text
pip install trackers[detection]
```

For more options, see the [install guide](install.md).

---

## Download Data

TODO: One sentence about downloading MOT17 val frames and annotations.

```text
trackers download mot17 \
    --split val \
    --asset frames,annotations \
    --output ./data
```

---

## Run the Experiment

TODO: One sentence explaining we run ByteTrack with default parameters four times, swapping only the detector.

### YOLOv11 Nano

```bash
for seq in MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN; do
    trackers track \
        --source ./data/mot17/val/$seq/img1 \
        --model yolov11n-640 \
        --tracker bytetrack \
        --classes person \
        --mot-output results/yolov11n/$seq.txt
done
```

### YOLO26 Nano

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

### RF-DETR Nano

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

### RF-DETR Medium

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

---

## Evaluate

TODO: One sentence about evaluating each detector's results.

### YOLOv11 Nano

```text
trackers eval \
    --gt-dir ./data/mot17/val \
    --tracker-dir results/yolov11n \
    --metrics CLEAR HOTA Identity \
    --columns MOTA HOTA IDF1 IDSW
```

**Output:**

```
Sequence                        MOTA    HOTA    IDF1  IDSW
----------------------------------------------------------
MOT17-02-FRCNN                17.490  27.678  24.520     9
MOT17-04-FRCNN                13.856  29.504  25.647     2
MOT17-05-FRCNN                49.866  43.512  57.008    25
MOT17-09-FRCNN                51.129  41.525  52.672    24
MOT17-10-FRCNN                28.820  33.927  41.535     4
MOT17-11-FRCNN                53.885  47.594  54.685    11
MOT17-13-FRCNN                19.392  28.905  32.856     2
----------------------------------------------------------
COMBINED                      24.081  33.467  35.002    77
```

### YOLO26 Nano

```text
trackers eval \
    --gt-dir ./data/mot17/val \
    --tracker-dir results/yolo26n \
    --metrics CLEAR HOTA Identity \
    --columns MOTA HOTA IDF1 IDSW
```

**Output:**

```
Sequence                        MOTA    HOTA    IDF1  IDSW
----------------------------------------------------------
MOT17-02-FRCNN                 5.010  10.720   9.123    58
MOT17-04-FRCNN                 2.436  19.902  13.552    28
MOT17-05-FRCNN                15.073  22.486  27.471   136
MOT17-09-FRCNN                11.428  18.985  24.202    66
MOT17-10-FRCNN                 6.652  14.671  14.708    46
MOT17-11-FRCNN                11.424  23.055  26.178    89
MOT17-13-FRCNN                 6.401  14.309  16.736    66
----------------------------------------------------------
COMBINED                       5.624  18.308  16.264   489
```

### RF-DETR Nano

```text
trackers eval \
    --gt-dir ./data/mot17/val \
    --tracker-dir results/rfdetr-nano \
    --metrics CLEAR HOTA Identity \
    --columns MOTA HOTA IDF1 IDSW
```

**Output:**

```
Sequence                        MOTA    HOTA    IDF1  IDSW
----------------------------------------------------------
MOT17-02-FRCNN                16.883  29.984  26.664     9
MOT17-04-FRCNN                13.508  29.875  25.546     4
MOT17-05-FRCNN                53.649  47.960  61.448    40
MOT17-09-FRCNN                56.165  47.379  59.742    28
MOT17-10-FRCNN                33.193  39.330  48.518     3
MOT17-11-FRCNN                56.099  48.861  57.435    18
MOT17-13-FRCNN                31.115  36.343  46.808     8
----------------------------------------------------------
COMBINED                      25.671  35.969  38.318   110
```

### RF-DETR Medium

```text
trackers eval \
    --gt-dir ./data/mot17/val \
    --tracker-dir results/rfdetr-medium \
    --metrics CLEAR HOTA Identity \
    --columns MOTA HOTA IDF1 IDSW
```

**Output:**

```
Sequence                        MOTA    HOTA    IDF1  IDSW
----------------------------------------------------------
MOT17-02-FRCNN                20.547  33.568  31.273     8
MOT17-04-FRCNN                15.332  32.508  28.556     1
MOT17-05-FRCNN                54.453  50.108  65.175    33
MOT17-09-FRCNN                63.251  50.954  64.853    19
MOT17-10-FRCNN                39.676  43.046  52.787     6
MOT17-11-FRCNN                57.649  51.554  59.923    13
MOT17-13-FRCNN                42.300  43.160  56.178    20
----------------------------------------------------------
COMBINED                      29.087  39.193  42.451   100
```

---

## Results

TODO: One sentence summarizing the comparison.

|    Detector    |    MOTA    |    HOTA    |    IDF1    | IDSW |
| :------------: | :--------: | :--------: | :--------: | :--: |
|  YOLOv11 Nano  |   24.081   |   33.467   |   35.002   |  77  |
|  YOLO26 Nano   |   5.624    |   18.308   |   16.264   | 489  |
|  RF-DETR Nano  |   25.671   |   35.969   |   38.318   | 110  |
| RF-DETR Medium | **29.087** | **39.193** | **42.451** | 100  |

TODO: One sentence about what the numbers show — same tracker, different scores.

---

## Takeaway

TODO: Two-three sentences about investing in detection quality first, then tuning the tracker. Link to the tracker comparison page.
