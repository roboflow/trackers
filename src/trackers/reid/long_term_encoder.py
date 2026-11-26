# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Explicit encoder selection for long-term McByte++ joining."""

from trackers.reid.encoder import ReIDEncoder, _load_reid_encoder

SPORTS_REID_MODEL = "mcbyteplusplus_sports"


def _load_long_term_encoder(source: str, *, device: str = "auto", architecture: str | None = None) -> ReIDEncoder:
    """Load a Roboflow encoder, converting the original sports model on request.

    ``mcbyteplusplus_sports`` is a Trackers alias for the verified original McByte++ checkpoint. All other sources use
    the existing Roboflow loader. The conversion helper imports its optional dependencies only when selected.
    """
    if source == SPORTS_REID_MODEL:
        if architecture is not None:
            raise ValueError("mcbyteplusplus_sports fixes architecture to osnet_x1_0; omit architecture.")
        from trackers.reid.sports_model import prepare_sports_reid_model

        source = str(prepare_sports_reid_model())
    return _load_reid_encoder(source, device=device, architecture=architecture)
