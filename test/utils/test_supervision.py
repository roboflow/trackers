# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trackers.utils import supervision as supervision_utils


def test_box_iou_batch_uses_current_supervision_implementation() -> None:
    boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
    boxes2 = np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float32)

    iou_matrix = supervision_utils.box_iou_batch(boxes1, boxes2)

    np.testing.assert_allclose(iou_matrix, np.array([[1.0, 0.0]], dtype=np.float32))


def test_resolve_box_iou_batch_prefers_new_supervision_path(
    monkeypatch,
) -> None:
    modern_box_iou_batch = object()
    legacy_box_iou_batch = object()
    import_calls: list[str] = []

    def fake_import_module(module_name: str) -> SimpleNamespace:
        import_calls.append(module_name)
        if module_name == "supervision.detection.utils.iou_and_nms":
            return SimpleNamespace(box_iou_batch=modern_box_iou_batch)
        if module_name == "supervision.detection.utils":
            return SimpleNamespace(box_iou_batch=legacy_box_iou_batch)
        raise AssertionError(f"Unexpected import: {module_name}")

    monkeypatch.setattr(supervision_utils.importlib, "import_module", fake_import_module)

    resolved = supervision_utils._resolve_box_iou_batch()

    assert resolved is modern_box_iou_batch
    assert import_calls == ["supervision.detection.utils.iou_and_nms"]


def test_resolve_box_iou_batch_falls_back_to_legacy_supervision_path(
    monkeypatch,
) -> None:
    legacy_box_iou_batch = object()
    import_calls: list[str] = []

    def fake_import_module(module_name: str) -> SimpleNamespace:
        import_calls.append(module_name)
        if module_name == "supervision.detection.utils.iou_and_nms":
            raise ImportError(module_name)
        if module_name == "supervision.detection.utils":
            return SimpleNamespace(box_iou_batch=legacy_box_iou_batch)
        raise AssertionError(f"Unexpected import: {module_name}")

    monkeypatch.setattr(supervision_utils.importlib, "import_module", fake_import_module)

    resolved = supervision_utils._resolve_box_iou_batch()

    assert resolved is legacy_box_iou_batch
    assert import_calls == [
        "supervision.detection.utils.iou_and_nms",
        "supervision.detection.utils",
    ]
