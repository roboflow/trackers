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

from trackers.cli._annotate import annotate_masks, annotate_tracklet_boxes
from trackers.cli.inspect._common import (
    INSPECT_OUTPUT_ROOT,
    get_mask_tracklet_ids_in_order,
    load_rgb_image,
    print_device_info,
    save_rgb_image,
    tracklet_boxes,
    validate_and_clip_xyxy_box,
    validate_device,
)
from trackers.core.masks.base import TrackletSnapshot

DEFAULT_OUTPUT_PATH = INSPECT_OUTPUT_ROOT / "sam" / "sam_masks.jpg"


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

    tracker_ids = get_mask_tracklet_ids_in_order(mask_output.tracklet_mask_dict)
    visual_rgb = annotate_masks(image_rgb, mask_output.masks, tracker_ids)
    visual_rgb = annotate_tracklet_boxes(
        visual_rgb,
        tracklet_boxes(tracklets),
        [tracklet.tracker_id for tracklet in tracklets],
    )
    save_rgb_image(visual_rgb, output_path)

    print(f"Saved visualization to {output_path.resolve()}")
    return 0
