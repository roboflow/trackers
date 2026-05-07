# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

# Epsilon for floating point comparisons.
# Must match TrackEval exactly for numerical parity.
# References:
#   - trackeval/metrics/clear.py:82,86 (threshold comparisons)
#   - trackeval/metrics/hota.py:59,92 (similarity masking)
#   - trackeval/datasets/_base_dataset.py:274-285 (IoU computation)
EPS = np.finfo("float").eps
