from __future__ import annotations

import importlib
from collections.abc import Callable


def _resolve_box_iou_batch() -> Callable:
    for module_name in (
        "supervision.detection.utils.iou_and_nms",
        "supervision.detection.utils",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        try:
            return module.box_iou_batch
        except AttributeError:
            continue

    raise ImportError(
        "Unable to import box_iou_batch from supervision compatibility paths."
    )


box_iou_batch = _resolve_box_iou_batch()
