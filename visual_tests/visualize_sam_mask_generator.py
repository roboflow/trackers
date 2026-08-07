# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Visual sanity check for SAMBoxMaskGenerator.

This script is intended for local/manual validation and development debugging
only. It does not bundle any image assets and does not run as part of the test
suite. The user provides an input image and one or more bounding boxes, and the
script saves an image with SAM masks and boxes overlaid.

Usage
-----

::

    python visual_tests/visualize_sam_mask_generator.py \\
        --image_path frame.jpg --box='[[10,20,110,220]]'

Options come from the :func:`visualize_command` signature, parsed with
jsonargparse through the shared ``trackers`` parser, so the shared conventions
hold here too: ``--image-path`` and ``--image_path`` are the same option, and
``--config run.yaml`` supplies the same keys from a file. Boxes are supplied as
one list rather than a repeated option: ``--box='[[10,20,110,220]]'`` for one
box, ``--box='[[10,20,110,220],[30,40,130,240]]'`` for two, and
``--box+='[[30,40,130,240]]'`` to append to boxes already given.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from jsonargparse import CLI

from trackers.cli.__main__ import _CLIParser, _normalise_option
from trackers.core.mcbyte.masks.base import TrackletSnapshot
from trackers.core.mcbyte.masks.sam import SAMBoxMaskGenerator

DEFAULT_OUTPUT_PATH = Path("visual_tests/outputs/sam_mask_generator_output.jpg")


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
    """Draw tracklet bounding boxes and tracker IDs on an RGB image."""
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


def validate_device(device: str) -> str:
    """Validate the requested SAM execution device."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Use --device cpu or install a CUDA-enabled PyTorch build."
        )
    return device


def visualize_command(
    image_path: Path,
    box: list[tuple[float, float, float, float]],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    device: str = "cpu",
    model_type: str = "vit_b",
) -> int:
    """Run SAM mask generation, report the execution device, and save a visualization image.

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
    """
    try:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

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

        resolved_device = validate_device(device)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    generator = SAMBoxMaskGenerator(
        model_type=model_type,
        device=resolved_device,
    )

    print(f"Generator device: {generator.device}")
    if generator.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(generator.device)} (CUDA {torch.version.cuda})")
    else:
        print("GPU: N/A (running on CPU)")

    mask_output = generator.generate(
        frame=image_rgb,
        tracklets=tracklets,
    )

    if mask_output.masks is None:
        raise RuntimeError("SAM did not return masks.")

    visual_rgb = overlay_masks(image_rgb, mask_output.masks)
    visual_rgb = draw_boxes(visual_rgb, tracklets)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    visual_bgr = cv2.cvtColor(visual_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), visual_bgr)

    print(f"Saved visualization to {output_path}")
    return 0


def main() -> int:
    """Parse visualization arguments with jsonargparse and run SAM mask generation.

    Returns:
        Exit code from :func:`visualize_command`.
    """
    args = [_normalise_option(arg) for arg in sys.argv[1:]]
    rc = CLI(
        visualize_command,
        args=args,
        as_positional=False,
        prog="python visual_tests/visualize_sam_mask_generator.py",
        description="Visualize masks generated by SAMBoxMaskGenerator.",
        parser_class=_CLIParser,
    )
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
