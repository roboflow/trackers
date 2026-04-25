# State Estimators

Every tracker in `trackers` uses a Kalman filter to predict where objects will appear in the next frame. The **state estimator** controls how bounding boxes are represented inside that filter. Different representations make different assumptions about object motion, and picking the right one can improve tracking quality without changing anything else.

**What you'll learn:**

- What state estimators are and why they matter
- How `XYXYStateEstimator` and `XCYCSRStateEstimator` represent bounding boxes
- When to use each representation
- How to swap the state estimator in any tracker

---

## Install

Get started by installing the package.

```text
pip install trackers
```

For more options, see the [install guide](install.md).

---

## What Is a State Estimator?

A state estimator wraps a Kalman filter and defines how bounding boxes are encoded into the filter's state vector. The Kalman filter then predicts the next position of each tracked object and corrects that prediction when a new detection arrives.

Two representations are available:

|       Estimator        | State Dimensions | Representation                                        | Aspect Ratio  |
| :--------------------: | :--------------: | :---------------------------------------------------- | :-----------: |
|  `XYXYStateEstimator`  |        8         | Top-left and bottom-right corners + their velocities  |  Can change   |
| `XCYCSRStateEstimator` |        7         | Center point, area, their velocities and aspect ratio | Held constant |

They accept `[x1, y1, x2, y2]` bounding boxes on input and produce `[x1, y1, x2, y2]` bounding boxes on output. The difference is entirely in how the filter models motion internally.

---

## XYXY — Corner-Based

`XYXYStateEstimator` tracks the four corner coordinates independently. Each corner gets its own velocity term, giving the filter 8 state variables:

```
State:   [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
Measure: [x1, y1, x2, y2]
```

The transition matrix $F$ defines how the state evolves from one frame to the next.

State order: $[x_1, y_1, x_2, y_2, v_{x_1}, v_{y_1}, v_{x_2}, v_{y_2}]$

$$
F =
\begin{bmatrix}
1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$

Equivalent update equations:

```text
x1'  = x1  + vx1
y1'  = y1  + vy1
x2'  = x2  + vx2
y2'  = y2  + vy2
vx1' = vx1
vy1' = vy1
vx2' = vx2
vy2' = vy2
```

| Row | Meaning                                                  |
| :-- | :------------------------------------------------------- |
| 1-4 | Each corner coordinate is updated by adding its velocity |
| 5-8 | Velocities persist unchanged from frame to frame         |

Because each corner moves freely, the box width and height can change between frames. This makes XYXY a natural fit when objects change shape — due to camera perspective, non-rigid motion, or inconsistent detections.

**In Trackers, this is the configurable default** for `ByteTrackTracker` and `SORTTracker` via the `state_estimator_class` parameter. Note: previous versions used hand-rolled Kalman filters internally — `XYXYStateEstimator` is the new unified implementation introduced in this refactoring.

---

## XCYCSR — Center-Based

`XCYCSRStateEstimator` tracks the box center, area (scale), and aspect ratio. Only the center and scale get velocity terms; aspect ratio is treated as constant. This gives 7 state variables:

```
State:   [x_center, y_center, scale, aspect_ratio, vx, vy, vs]
Measure: [x_center, y_center, scale, aspect_ratio]
```

The transition matrix $F$ shows the key difference: the aspect ratio is propagated without a velocity term.

State order: $[x_c, y_c, s, r, v_x, v_y, v_s]$

$$
F =
\begin{bmatrix}
1 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$

Equivalent update equations:

```text
x_center'     = x_center + vx
y_center'     = y_center + vy
scale'        = scale + vs
aspect_ratio' = aspect_ratio
vx'           = vx
vy'           = vy
vs'           = vs
```

| Row | Meaning                                                   |
| :-- | :-------------------------------------------------------- |
| 1-3 | Center position and scale follow constant-velocity motion |
| 4   | Aspect ratio is copied forward unchanged                  |
| 5-7 | Velocities persist unchanged from frame to frame          |

The aspect ratio `r = w / h` is carried forward unchanged. This acts as a regularizer — the filter resists sudden shape changes. It works well for rigid objects whose proportions stay consistent, like pedestrians walking or cars on a highway.

**This is the default** for `OCSORTTracker`, matching the original OC-SORT paper.

---

## When to Use Each

| Scenario                                     |      Recommended       | Why                                                        |
| :------------------------------------------- | :--------------------: | :--------------------------------------------------------- |
| Pedestrians, vehicles, rigid objects         | `XCYCSRStateEstimator` | Constant aspect ratio stabilizes predictions               |
| Non-rigid or deformable objects              |  `XYXYStateEstimator`  | Corners move independently to track shape changes          |
| Noisy detections with fluctuating box sizes  | `XCYCSRStateEstimator` | Aspect ratio constraint absorbs size noise                 |
| Strong perspective changes (camera pan/zoom) |  `XYXYStateEstimator`  | Box proportions shift with viewpoint; corners adapt freely |
| Default choice when unsure                   |  `XYXYStateEstimator`  | More general, fewer assumptions                            |

We can also benchmark the trackers using the different State Estimators and we get:

- In **Dancetrack**, with default parameters all trackers perform better with XYXYStateEstimator, but with tuned parameters, SORT tracker with XCYCSRStateEstimator gets +0.8% HOTA.
- In the **Soccernet dataset**, with default parameters, SORT tracker with XYXYStateEstimator has ~5% more HOTA than using XCYC. When tuning parameters with grid search, this difference is reduced to 2%. For the other trackers, we don't find significant advantages of using a different StateEstimators, just having up to 0.2% better HOTA.
- In **SportsMOT**, for OC-SORT and ByteTrack, the StateEstimator doesn't affect the performance, while for SORT XYXYStateEstimator gives a small advantage of ~2% HOTA with default parameters and 0.4% when tuning both.
- In **MOT17**, with default parameters XYXYStateEstimator performs slightly better than XCYCSRStateEstimator with SORT and ByteTrack with up to 0.7% better results, but for OC-SORT XCYCSRStateEstimator gives 0.2% better HOTA. When tuning parameters, XCYCSRStateEstimator performs the best with all the trackers by a small margin, ranging in 0.2-0.4% HOTA.

But lets visualize where these differences are, here is an example where using XCYCSR State Estimator associates an occluded track correctly, while using XYXY changes the ID:

<div style="display: flex; justify-content: center;">
  <video style="width: 50%; height: auto;" controls>
    <source src="https://github.com/user-attachments/assets/219acc15-c6c5-4bf2-93d2-8c1b5523f4f1" type="video/mp4">
  </video>
</div>

---

## Swapping the Estimator

All trackers accept a `state_estimator_class` parameter. Import the class you want and pass it to the constructor.

=== "ByteTrack with XCYCSR"

    ```python
    from trackers import ByteTrackTracker
    from trackers.utils.state_representations import XCYCSRStateEstimator

    tracker = ByteTrackTracker(
        state_estimator_class=XCYCSRStateEstimator,
    )
    ```

=== "OC-SORT with XYXY"

    ```python
    from trackers import OCSORTTracker
    from trackers.utils.state_representations import XYXYStateEstimator

    tracker = OCSORTTracker(
        state_estimator_class=XYXYStateEstimator,
    )
    ```

=== "SORT with XCYCSR"

    ```python
    from trackers import SORTTracker
    from trackers.utils.state_representations import XCYCSRStateEstimator

    tracker = SORTTracker(
        state_estimator_class=XCYCSRStateEstimator,
    )
    ```

Everything else stays the same — detection, association, and visualization work identically regardless of which estimator you choose.

---

## Full Example

Run ByteTrack with both estimators on the same video and compare the results side by side.

```python
import cv2

import supervision as sv
from inference import get_model
from trackers import ByteTrackTracker
from trackers.utils.state_representations import (
    XCYCSRStateEstimator,
    XYXYStateEstimator,
)

model = get_model("rfdetr-nano")

tracker_xyxy = ByteTrackTracker(
    state_estimator_class=XYXYStateEstimator,
)
tracker_xcycsr = ByteTrackTracker(
    state_estimator_class=XCYCSRStateEstimator,
)

cap = cv2.VideoCapture("source.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.infer(frame)[0]
    detections = sv.Detections.from_inference(result)

    tracked_xyxy = tracker_xyxy.update(detections)
    tracked_xcycsr = tracker_xcycsr.update(detections)

    # Compare tracker_id assignments, box smoothness, etc.
    print(f"XYXY IDs:   {tracked_xyxy.tracker_id}")
    print(f"XCYCSR IDs: {tracked_xcycsr.tracker_id}")
```

---

## Takeaway

The state estimator is a single-line change that controls how the Kalman filter models bounding box motion. Use `XCYCSRStateEstimator` when objects keep a consistent shape, and `XYXYStateEstimator` when shape varies or you want fewer assumptions. Try it on your case, the best choice depends on the scene.
