# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import pytest


@pytest.mark.network
def test_fastreid_mot17_alias_loads_with_finite_normalized_output() -> None:
    pytest.importorskip("torch")
    import numpy as np
    import supervision as sv

    from trackers.core.reid.model import ReIDModel

    model = ReIDModel.from_pretrained("fastreid_mot17_sbs50", device="cpu")
    frame = np.zeros((384, 128, 3), dtype=np.uint8)
    dets = sv.Detections(xyxy=np.array([[0.0, 0.0, 128.0, 384.0]], dtype=np.float32))
    embs = model.extract_features(dets, frame)
    assert embs.shape == (1, 2048)
    assert np.isfinite(embs).all()
    np.testing.assert_allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-4)
    assert model.preprocessing.input_size == (384, 128)
