# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Drawing shared by every command that renders frames.

Everything here delegates to ``supervision`` annotators rather than calling
OpenCV directly. The library is already a hard dependency and already speaks
:class:`sv.Detections`, so a bespoke box or mask drawer buys nothing but a
second visual style to keep in sync.

One consequence worth stating: ``trackers track`` and ``trackers inspect`` now
draw from the same palette, so a track ID keeps its colour whether you are
watching a live run or inspecting one stage of the mask pipeline.

Infrastructure module, not a command — see :mod:`trackers.cli` for the layout
rule.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

__all__ = [
    "COLOR_PALETTE",
    "LIFECYCLE_LABELS",
    "LIFECYCLE_MASKED",
    "LIFECYCLE_NEW",
    "LIFECYCLE_PENDING",
    "LIFECYCLE_TRACKED",
    "annotate_lifecycle_boxes",
    "annotate_masks",
    "annotate_tracklet_boxes",
    "draw_status_lines",
]

COLOR_PALETTE = sv.ColorPalette.from_hex(
    [
        "#ffff00",
        "#ff9b00",
        "#ff8080",
        "#ff66b2",
        "#ff66ff",
        "#b266ff",
        "#9999ff",
        "#3399ff",
        "#66ffff",
        "#33ff99",
        "#66ff66",
        "#99ff00",
    ]
)
"""Track-ID palette shared by every command that draws tracklets."""

LIFECYCLE_NEW = 0
LIFECYCLE_PENDING = 1
LIFECYCLE_MASKED = 2
LIFECYCLE_TRACKED = 3
LIFECYCLE_LABELS = ("new", "pending", "masked", "tracked")
"""Mask-lifecycle states, used as ``class_id`` so colour follows state not ID.

Ground-truth replay knows whether a tracklet is newly visible, waiting for mask
creation, already masked, or none of those. Encoding that as a class lets
``supervision`` colour it, instead of a hand-rolled branch per state.
"""

_LIFECYCLE_PALETTE = sv.ColorPalette.from_hex(
    [
        "#3399ff",  # new
        "#ffff00",  # pending
        "#66ff66",  # masked
        "#b266ff",  # tracked
    ]
)

_MASK_ANNOTATOR = sv.MaskAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.TRACK, opacity=0.45)
_BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.TRACK)
_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.TRACK,
    text_color=sv.Color.BLACK,
    text_position=sv.Position.TOP_LEFT,
)
_LIFECYCLE_BOX_ANNOTATOR = sv.BoxAnnotator(color=_LIFECYCLE_PALETTE, color_lookup=sv.ColorLookup.CLASS)
_LIFECYCLE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=_LIFECYCLE_PALETTE,
    color_lookup=sv.ColorLookup.CLASS,
    text_color=sv.Color.BLACK,
    text_position=sv.Position.TOP_LEFT,
)

_STATUS_LINE_HEIGHT = 22
_STATUS_ORIGIN = (10, 20)


def annotate_masks(
    image: np.ndarray,
    masks: np.ndarray,
    tracker_ids: list[int],
) -> np.ndarray:
    """Overlay per-tracklet masks, coloured by track ID.

    Args:
        image: RGB image with shape ``(H, W, 3)``.
        masks: Boolean mask array with shape ``(N, H, W)``.
        tracker_ids: Track ID per mask, in the same order as ``masks``.

    Returns:
        A new RGB image with the masks overlaid.

    Examples:
        >>> image = np.zeros((4, 4, 3), dtype=np.uint8)
        >>> masks = np.ones((1, 4, 4), dtype=bool)
        >>> annotate_masks(image, masks, [1]).shape
        (4, 4, 3)
    """
    if len(masks) == 0:
        return image.copy()

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),
        mask=masks,
        tracker_id=np.asarray(tracker_ids, dtype=int),
    )
    return _MASK_ANNOTATOR.annotate(scene=image.copy(), detections=detections)


def annotate_tracklet_boxes(
    image: np.ndarray,
    xyxy: np.ndarray,
    tracker_ids: list[int],
    labels: list[str] | None = None,
) -> np.ndarray:
    """Draw tracklet boxes and ID labels, coloured by track ID.

    Args:
        image: RGB image with shape ``(H, W, 3)``.
        xyxy: Boxes with shape ``(N, 4)``.
        tracker_ids: Track ID per box.
        labels: Text per box; defaults to the track ID.

    Returns:
        A new RGB image with boxes and labels drawn.

    Examples:
        >>> image = np.zeros((8, 8, 3), dtype=np.uint8)
        >>> boxes = np.array([[1.0, 1.0, 5.0, 5.0]])
        >>> annotate_tracklet_boxes(image, boxes, [7]).shape
        (8, 8, 3)
    """
    if len(xyxy) == 0:
        return image.copy()

    detections = sv.Detections(xyxy=np.asarray(xyxy, dtype=float), tracker_id=np.asarray(tracker_ids, dtype=int))
    scene = _BOX_ANNOTATOR.annotate(scene=image.copy(), detections=detections)
    texts = labels if labels is not None else [str(tracker_id) for tracker_id in tracker_ids]
    return _LABEL_ANNOTATOR.annotate(scene=scene, detections=detections, labels=texts)


def annotate_lifecycle_boxes(
    image: np.ndarray,
    xyxy: np.ndarray,
    tracker_ids: list[int],
    states: list[int],
) -> np.ndarray:
    """Draw tracklet boxes coloured by mask-lifecycle state rather than by ID.

    Args:
        image: RGB image with shape ``(H, W, 3)``.
        xyxy: Boxes with shape ``(N, 4)``.
        tracker_ids: Track ID per box, used for the label text.
        states: One of the ``LIFECYCLE_*`` constants per box.

    Returns:
        A new RGB image with boxes and ``id:state`` labels drawn.

    Examples:
        >>> image = np.zeros((8, 8, 3), dtype=np.uint8)
        >>> boxes = np.array([[1.0, 1.0, 5.0, 5.0]])
        >>> annotate_lifecycle_boxes(image, boxes, [7], [LIFECYCLE_PENDING]).shape
        (8, 8, 3)
    """
    if len(xyxy) == 0:
        return image.copy()

    detections = sv.Detections(
        xyxy=np.asarray(xyxy, dtype=float),
        class_id=np.asarray(states, dtype=int),
        tracker_id=np.asarray(tracker_ids, dtype=int),
    )
    scene = _LIFECYCLE_BOX_ANNOTATOR.annotate(scene=image.copy(), detections=detections)
    texts = [f"{tracker_id} {LIFECYCLE_LABELS[state]}" for tracker_id, state in zip(tracker_ids, states)]
    return _LIFECYCLE_LABEL_ANNOTATOR.annotate(scene=scene, detections=detections, labels=texts)


def draw_status_lines(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a stacked status panel in the top-left corner.

    Args:
        image: RGB image with shape ``(H, W, 3)``.
        lines: Status lines, drawn top to bottom.

    Returns:
        A new RGB image with the status panel drawn.

    Examples:
        >>> image = np.zeros((64, 64, 3), dtype=np.uint8)
        >>> draw_status_lines(image, ["Frame: 1", "Masks: [1, 2]"]).shape
        (64, 64, 3)
    """
    scene = image.copy()
    origin_x, origin_y = _STATUS_ORIGIN
    for index, line in enumerate(lines):
        scene = sv.draw_text(
            scene=scene,
            text=line,
            text_anchor=sv.Point(x=origin_x + _text_anchor_offset(line), y=origin_y + index * _STATUS_LINE_HEIGHT),
            text_color=sv.Color.WHITE,
            background_color=sv.Color.BLACK,
        )
    return scene


def _text_anchor_offset(line: str) -> int:
    """Return half the rendered width of ``line``, since ``draw_text`` centres on its anchor."""
    return int(len(line) * 4.0)
