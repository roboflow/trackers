# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from trackers.calibration.types import CalibrationFrame, PitchDimensions


class PitchCalibrator(ABC):
    """Common interface for pitch-calibration backends.

    Calibrators are responsible for producing per-frame field geometry that can
    later be used to project tracked players into pitch coordinates.
    """

    def __init__(self, pitch_dimensions: PitchDimensions | None = None) -> None:
        self.pitch_dimensions = pitch_dimensions or PitchDimensions()

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    def is_available(self) -> bool:
        """Return whether the backing implementation is installed."""
        return True

    def availability_hint(self) -> str | None:
        """Return an optional install hint when `is_available` is false."""
        return None

    def describe(self) -> dict[str, object]:
        """Return serializable provider metadata for manifests and logs."""
        return {
            "provider": self.name,
            "pitch_dimensions": self.pitch_dimensions.to_dict(),
            "available": self.is_available(),
            "availability_hint": self.availability_hint(),
        }

    @abstractmethod
    def calibrate_video(
        self,
        source: str | Path,
        output_dir: str | Path,
    ) -> list[CalibrationFrame]:
        """Run calibration on a video or clip and return per-frame results."""
