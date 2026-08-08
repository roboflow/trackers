# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Visual sanity check for :class:`SAMBoxMaskGenerator`.

Intended for local validation and development debugging. No image assets are
bundled and nothing here runs as part of the test suite. The caller supplies an
image and one or more bounding boxes; the command saves an image with SAM masks
and boxes overlaid.

Usage
-----

::

    trackers inspect sam --image_path frame.jpg --box='[[10,20,110,220]]'

Options come from the :func:`sam_command` signature, parsed with jsonargparse
through the shared ``trackers`` parser, so the shared conventions hold:
``--image-path`` and ``--image_path`` are the same option, and ``--config
run.yaml`` supplies the same keys from a file. Boxes are supplied as one list
rather than a repeated option: ``--box='[[10,20,110,220]]'`` for one box,
``--box='[[10,20,110,220],[30,40,130,240]]'`` for two, and
``--box+='[[30,40,130,240]]'`` to append to boxes already given.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from trackers.cli.inspect._common import (
    INSPECT_OUTPUT_ROOT,
    load_rgb_image,
    print_device_info,
    save_rgb_image,
    validate_device,
)
from trackers.core.masks.base import TrackletSnapshot

DEFAULT_OUTPUT_PATH = INSPECT_OUTPUT_ROOT / "sam" / "sam_masks.jpg"


def overlay_masks(
    image: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay binary masks on an RGB image.

    Args:
        image: RGB image with shape ``(H, W, 3)``.
        masks: Boolean mask array with shape ``(N, H, W)``.
        alpha: Mask overlay opacity.

    Returns:
        RGB image with semi-transparent mask overlays.

    Examples:
        >>> image = np.zeros((2, 2, 3), dtype=np.uint8)
        >>> masks = np.ones((1, 2, 2), dtype=bool)
        >>> overlay_masks(image, masks).shape
        (2, 2, 3)
    """
    output = image.copy()
    rng = np.random.default_rng(0)

    for mask in masks:
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        colored_mask = np.zeros_like(output)
        colored_mask[mask] = color

        output = np.where(
            mask[..., None],
            (alpha * colored_mask + (1.0 - alpha) * output).astype(np.uint8),
            output,
        )

    return output


def draw_boxes(
    image: np.ndarray,
    tracklets: list[TrackletSnapshot],
) -> np.ndarray:
    """Draw tracklet bounding boxes and tracker IDs on an RGB image.

    Args:
        image: RGB image with shape ``(H, W, 3)``.
        tracklets: Tracklets whose boxes are drawn.

    Returns:
        RGB image with boxes and IDs drawn.

    Examples:
        >>> image = np.zeros((8, 8, 3), dtype=np.uint8)
        >>> box = np.array([1, 1, 5, 5], dtype=np.float32)
        >>> draw_boxes(image, [TrackletSnapshot(tracker_id=1, xyxy=box)]).shape
        (8, 8, 3)
    """
    output = image.copy()

    for tracklet in tracklets:
        x1, y1, x2, y2 = tracklet.xyxy.astype(int)
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color=(91, 10, 145),
            thickness=2,
        )
        cv2.putText(
            output,
            str(tracklet.tracker_id),
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def validate_and_clip_xyxy_box(
    box: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Validate and clip an ``xyxy`` box to image boundaries.

    Args:
        box: Bounding box in ``(x1, y1, x2, y2)`` format.
        image_shape: Image shape as ``(height, width)``.

    Returns:
        Clipped box as a float32 NumPy array.

    Raises:
        ValueError: If the box has non-positive area before or after clipping.

    Examples:
        >>> validate_and_clip_xyxy_box((1.0, 2.0, 5.0, 6.0), (10, 10))
        array([1., 2., 5., 6.], dtype=float32)
    """
    height, width = image_shape
    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid xyxy box with non-positive size: {box}")

    x1 = np.clip(x1, 0, width)
    x2 = np.clip(x2, 0, width)
    y1 = np.clip(y1, 0, height)
    y2 = np.clip(y2, 0, height)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Box is outside image after clipping: {box}")

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def sam_command(
    image_path: Path,
    box: list[tuple[float, float, float, float]],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    device: str = "cpu",
    model_type: str = "vit_b",
) -> int:
    """Run SAM mask generation, report the execution device, and save a visualization.

    Every option is also accepted with hyphens in place of underscores, so
    ``--image-path`` and ``--image_path`` name the same thing.

    Args:
        image_path: Path to the input image.
        box: Bounding boxes in xyxy format, given as one list rather than a
            repeated option: ``--box='[[x1,y1,x2,y2]]'`` for one box,
            ``--box='[[10,20,110,220],[30,40,130,240]]'`` for two, and
            ``--box+='[[30,40,130,240]]'`` to append to boxes already given.
        output_path: Path to save the visualization.
        device: Device used by SAM, for example ``cpu`` or ``cuda``.
        model_type: SAM model type.

    Returns:
        Exit code: ``0`` on success, ``1`` on validation error.

    Raises:
        RuntimeError: If SAM returns no masks for the given boxes.

    Examples:
        An unreadable image is reported on stderr and exits non-zero, so only
        the return value shows up here.

        >>> sam_command(Path("missing.jpg"), box=[(1.0, 2.0, 3.0, 4.0)])
        1
    """
    try:
        image_rgb = load_rgb_image(image_path)
        tracklets = [
            TrackletSnapshot(
                tracker_id=index,
                xyxy=validate_and_clip_xyxy_box(
                    box=single_box,
                    image_shape=image_rgb.shape[:2],
                ),
            )
            for index, single_box in enumerate(box)
        ]
        resolved_device = validate_device(device, label="SAM")
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    from trackers.core.masks.sam import SAMBoxMaskGenerator

    generator = SAMBoxMaskGenerator(
        model_type=model_type,
        device=resolved_device,
    )
    print_device_info(generator.device, label="SAM")

    mask_output = generator.generate(
        frame=image_rgb,
        tracklets=tracklets,
    )
    if mask_output.masks is None:
        raise RuntimeError("SAM did not return masks.")

    visual_rgb = overlay_masks(image_rgb, mask_output.masks)
    visual_rgb = draw_boxes(visual_rgb, tracklets)
    save_rgb_image(visual_rgb, output_path)

    print(f"Saved visualization to {output_path.resolve()}")
    return 0
