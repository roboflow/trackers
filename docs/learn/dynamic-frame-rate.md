---
title: Dynamic Frame Rate — Variable-Gap Tracking | Trackers
description: Track with irregular frame timing by passing timestamps to tracker.update(). Scale Kalman prediction and lost-track pruning to real wall-clock gaps on SORT, ByteTrack, OC-SORT, BoT-SORT, and C-BIoU.
---

# Dynamic Frame Rate

Most tracking examples assume you call `update()` once per video frame, in order, at a steady FPS. That breaks when **frames are dropped** (you skip frame 2 but still call `update()` on frame 3) or when you **only process some frames**. Processing delay is fine, a frame that arrives late but with the right capture timestamp still has the correct gap to the previous frame you tracked. What matters is the **time between captured frames you actually update on**, not how long inference took.

Pass an optional **`timestamp`** (seconds) on `update()` so the tracker knows how much time actually passed. Prediction and pruning then follow the real gap. Omit `timestamp` and nothing changes: fixed-rate behaviour.

**What you'll learn:**

- How to use dynamic frame rate mode
- When it's worth to use it
- How fixed-rate and dynamic-rate modes differ

---

## Install

Get started by installing the package.

```text
pip install trackers
```

For more options, see the [install guide](install.md).

---

## When to Use It

Turn on timestamps when the **capture-time gap** between two `update()` calls is not always one frame period, we call that **dynamic frame rate**, and some cases where it happens are:

- **Dropped or skipped frames**: you process frame 100 after frame 97; the tracker should treat that as three frame periods of motion, not one
- **Sparse decode**: variable-FPS files or sampling every Nth frame, using the container or camera clock
- **Out-of-order completion**: workers may finish in any order, but you should call `update()` in **non-decreasing capture time** (sort by capture time or frame index before updating). A later frame must not be updated before an earlier one

Stick with the default when you process **every** frame in order at a steady rate. A normal `VideoCapture` loop that reads all frames does not need timestamps.

`SORTTracker`, `ByteTrackTracker`, `OCSORTTracker`, `BoTSORTTracker`, and `CBIoUTracker` all accept `timestamp` on `update()`.

---

## Fixed Rate vs Dynamic Rate

|                     | Fixed rate (default)         | Dynamic rate                                 |
| ------------------- | ---------------------------- | -------------------------------------------- |
| `timestamp`         | `None` (omit)                | Monotonic seconds, e.g. video clock          |
| Kalman `frame_step` | `1.0` per call (frame units) | `elapsed_seconds × frame_rate` (frame units) |

**`lost_track_buffer` is always the same integer** — frames at a **30 FPS reference** (MOT-style tuning), not seconds you type in. The tracker converts it for you:

- **Fixed rate:** `maximum_frames_without_update = int(frame_rate / 30 × lost_track_buffer)` so a 60 FPS pipeline with `lost_track_buffer=30` keeps lost tracks for 60 frames (~1 s if you update every frame).
- **Dynamic rate:** `maximum_time_without_update = lost_track_buffer / 30` seconds between `update()` calls. The same `lost_track_buffer=30` gives a **1.0 s** budget; that does **not** scale with `frame_rate`.

Set **`frame_rate`** on the tracker in both modes. In fixed mode it scales the frame budget above. When you pass `timestamp`, it also turns elapsed seconds into Kalman frame units (`elapsed_seconds × frame_rate`).

If the video is actually steady 25 FPS and you pass timestamps every frame, `frame_step` stays `1.0` — dynamic mode lines up with fixed mode.

---

## What Happens on a Long Gap

On each predict, the filter adjusts how far it extrapolates and how uncertain it is:

- **Position** moves with constant velocity, scaled by the difference in timestamps scaled by the previous frame rate (`frame_step`).
- **Process noise** grows on longer gaps so the box does not stay artificially tight. At `frame_step = 1.0` you get the same as with default usage. On bigger steps the library rescales noise the way a constant-velocity model expects (stronger growth on position than velocity), instead of naively multiplying a one-frame matrix by Δt.

---

## Usage

Feed **`timestamp`** in seconds into `tracker.update()` alongside your detections. Use **capture time** of each frame in the video, or `frame_index / fps`.

=== "Python"

    Read the video clock from OpenCV and pass it to ByteTrack each frame.

    ```python
    import cv2

    import supervision as sv
    from inference import get_model
    from trackers import ByteTrackTracker

    model = get_model("rfdetr-medium")
    tracker = ByteTrackTracker(frame_rate=30.0, lost_track_buffer=30)

    cap = cv2.VideoCapture("source.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections, timestamp=timestamp)
    ```

Call **`tracker.reset()`** when you switch videos so the last timestamp does not leak into the next file.

---

## Things to Know

**You can mix modes per call.** Only steps that pass `timestamp` use seconds for pruning and scaled predict. Calling `update()` without `timestamp` resets the internal timestamp anchor — the next `update()` that *does* pass a `timestamp` will be treated as a fresh start (Kalman frame step `1 / frame_rate`), not a measured gap from the last timestamped call.

**The first timestamped step is special.** The tracker does not use your absolute clock value as the first gap. It assumes `1 / frame_rate` seconds so the first Kalman predict still looks like one frame.

**Timestamps must not go backwards.** If `timestamp` is less than the previous call, `update()` **warns and skips the whole step** (tracks unchanged, output IDs are `-1`). Reorder by capture time and call again. **Duplicate** timestamps (same capture time twice) only skip predict; association still runs on the last state.

**`frame_rate` should match your reference timeline.** If the file is 24 FPS but you set `frame_rate=30`, each step over-predicts motion. Same parameter already mattered for threshold scaling in fixed mode; it matters more when gaps are measured in seconds.

**OC-SORT:** ORU still steps in frame units, not wall-clock gaps.

---

## Takeaway

Timestamps are optional on `update()`. Skip them when you track every frame in order. Use them when frames are **dropped or sparse**, pass **capture time** (not inference finish time), call `update()` in **time order**, set `frame_rate` to the FPS you tuned against, and `reset()` between videos.
