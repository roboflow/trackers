from __future__ import annotations

import numpy as np

from trackers.calibration.pitch import PitchModel
from trackers.calibration.types import PitchDimensions


def test_pitch_metric_and_normalized_round_trip() -> None:
    pitch = PitchModel(dimensions=PitchDimensions(length_m=105.0, width_m=68.0))
    metric_points = np.array([[0.0, 0.0], [52.5, 34.0], [105.0, 68.0]])

    normalized = pitch.metric_to_normalized(metric_points)
    restored = pitch.normalized_to_metric(normalized)

    np.testing.assert_allclose(normalized[1], np.array([0.5, 0.5]))
    np.testing.assert_allclose(restored, metric_points)
