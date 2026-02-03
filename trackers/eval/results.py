# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Result classes for tracking evaluation metrics.

This module provides dataclasses for storing and manipulating evaluation results
with methods for serialization, display, and persistence.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# TrackEval summary field order for CLEAR metrics
CLEAR_FLOAT_FIELDS = [
    "MOTA",
    "MOTP",
    "MODA",
    "CLR_Re",
    "CLR_Pr",
    "MTR",
    "PTR",
    "MLR",
    "sMOTA",
]
CLEAR_INT_FIELDS = [
    "CLR_TP",
    "CLR_FN",
    "CLR_FP",
    "IDSW",
    "MT",
    "PT",
    "ML",
    "Frag",
]
CLEAR_SUMMARY_FIELDS = CLEAR_FLOAT_FIELDS + CLEAR_INT_FIELDS

# TrackEval summary field order for HOTA metrics
HOTA_FLOAT_FIELDS = [
    "HOTA",
    "DetA",
    "AssA",
    "DetRe",
    "DetPr",
    "AssRe",
    "AssPr",
    "LocA",
]
HOTA_INT_FIELDS = [
    "HOTA_TP",
    "HOTA_FN",
    "HOTA_FP",
]
HOTA_SUMMARY_FIELDS = HOTA_FLOAT_FIELDS + HOTA_INT_FIELDS

# All float fields for formatting
ALL_FLOAT_FIELDS = CLEAR_FLOAT_FIELDS + HOTA_FLOAT_FIELDS


@dataclass
class CLEARMetrics:
    """CLEAR metrics with TrackEval-compatible field names.

    All float metrics are stored as fractions (0-1 range), not percentages.
    Use `to_percentage()` for display formatting.

    Attributes:
        MOTA: Multiple Object Tracking Accuracy.
        MOTP: Multiple Object Tracking Precision.
        MODA: Multiple Object Detection Accuracy.
        CLR_Re: Recall (TP / (TP + FN)).
        CLR_Pr: Precision (TP / (TP + FP)).
        MTR: Mostly Tracked ratio.
        PTR: Partially Tracked ratio.
        MLR: Mostly Lost ratio.
        sMOTA: Summed MOTA.
        CLR_TP: True positives.
        CLR_FN: False negatives.
        CLR_FP: False positives.
        IDSW: ID switches.
        MT: Mostly Tracked count.
        PT: Partially Tracked count.
        ML: Mostly Lost count.
        Frag: Fragmentations.
        MOTP_sum: Raw MOTP sum for aggregation.
        CLR_Frames: Number of frames.
    """

    MOTA: float
    MOTP: float
    MODA: float
    CLR_Re: float
    CLR_Pr: float
    MTR: float
    PTR: float
    MLR: float
    sMOTA: float
    CLR_TP: int
    CLR_FN: int
    CLR_FP: int
    IDSW: int
    MT: int
    PT: int
    ML: int
    Frag: int
    MOTP_sum: float = 0.0
    CLR_Frames: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CLEARMetrics:
        """Create CLEARMetrics from a dictionary.

        Args:
            data: Dictionary with metric values.

        Returns:
            CLEARMetrics instance.
        """
        return cls(
            MOTA=float(data["MOTA"]),
            MOTP=float(data["MOTP"]),
            MODA=float(data["MODA"]),
            CLR_Re=float(data["CLR_Re"]),
            CLR_Pr=float(data["CLR_Pr"]),
            MTR=float(data["MTR"]),
            PTR=float(data["PTR"]),
            MLR=float(data["MLR"]),
            sMOTA=float(data["sMOTA"]),
            CLR_TP=int(data["CLR_TP"]),
            CLR_FN=int(data["CLR_FN"]),
            CLR_FP=int(data["CLR_FP"]),
            IDSW=int(data["IDSW"]),
            MT=int(data["MT"]),
            PT=int(data["PT"]),
            ML=int(data["ML"]),
            Frag=int(data["Frag"]),
            MOTP_sum=float(data.get("MOTP_sum", 0.0)),
            CLR_Frames=int(data.get("CLR_Frames", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary with all metric values.
        """
        return dataclasses.asdict(self)


@dataclass
class HOTAMetrics:
    """HOTA metrics with TrackEval-compatible field names.

    HOTA (Higher Order Tracking Accuracy) evaluates both detection and
    association quality. All float metrics are stored as fractions (0-1 range).

    Attributes:
        HOTA: Higher Order Tracking Accuracy (geometric mean of DetA and AssA).
        DetA: Detection Accuracy.
        AssA: Association Accuracy.
        DetRe: Detection Recall.
        DetPr: Detection Precision.
        AssRe: Association Recall.
        AssPr: Association Precision.
        LocA: Localization Accuracy.
        HOTA_TP: True positives (summed over all alpha thresholds).
        HOTA_FN: False negatives (summed over all alpha thresholds).
        HOTA_FP: False positives (summed over all alpha thresholds).
    """

    HOTA: float
    DetA: float
    AssA: float
    DetRe: float
    DetPr: float
    AssRe: float
    AssPr: float
    LocA: float
    HOTA_TP: int
    HOTA_FN: int
    HOTA_FP: int
    # Per-alpha arrays for aggregation (not serialized to JSON by default)
    _arrays: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HOTAMetrics:
        """Create HOTAMetrics from a dictionary.

        Args:
            data: Dictionary with metric values.

        Returns:
            HOTAMetrics instance.
        """
        # Extract arrays if present (for aggregation)
        arrays = {}
        for key in [
            "HOTA_TP_array",
            "HOTA_FN_array",
            "HOTA_FP_array",
            "AssA_array",
            "AssRe_array",
            "AssPr_array",
            "LocA_array",
        ]:
            if key in data:
                arrays[key] = np.array(data[key])

        return cls(
            HOTA=float(data["HOTA"]),
            DetA=float(data["DetA"]),
            AssA=float(data["AssA"]),
            DetRe=float(data["DetRe"]),
            DetPr=float(data["DetPr"]),
            AssRe=float(data["AssRe"]),
            AssPr=float(data["AssPr"]),
            LocA=float(data["LocA"]),
            HOTA_TP=int(data["HOTA_TP"]),
            HOTA_FN=int(data["HOTA_FN"]),
            HOTA_FP=int(data["HOTA_FP"]),
            _arrays=arrays,
        )

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        """Convert to dictionary.

        Args:
            include_arrays: Whether to include per-alpha arrays. Defaults to False.

        Returns:
            Dictionary with metric values.
        """
        result = {
            "HOTA": self.HOTA,
            "DetA": self.DetA,
            "AssA": self.AssA,
            "DetRe": self.DetRe,
            "DetPr": self.DetPr,
            "AssRe": self.AssRe,
            "AssPr": self.AssPr,
            "LocA": self.LocA,
            "HOTA_TP": self.HOTA_TP,
            "HOTA_FN": self.HOTA_FN,
            "HOTA_FP": self.HOTA_FP,
        }
        if include_arrays and self._arrays:
            for key, arr in self._arrays.items():
                result[key] = arr.tolist() if isinstance(arr, np.ndarray) else arr
        return result


@dataclass
class SequenceResult:
    """Result for a single sequence evaluation.

    Attributes:
        sequence: Name of the sequence.
        CLEAR: CLEAR metrics for this sequence.
        HOTA: HOTA metrics for this sequence (optional).
    """

    sequence: str
    CLEAR: CLEARMetrics
    HOTA: HOTAMetrics | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SequenceResult:
        """Create SequenceResult from a dictionary.

        Args:
            data: Dictionary with sequence name and metrics.

        Returns:
            SequenceResult instance.
        """
        hota = None
        if "HOTA" in data and data["HOTA"] is not None:
            hota = HOTAMetrics.from_dict(data["HOTA"])

        return cls(
            sequence=data["sequence"],
            CLEAR=CLEARMetrics.from_dict(data["CLEAR"]),
            HOTA=hota,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        result: dict[str, Any] = {
            "sequence": self.sequence,
            "CLEAR": self.CLEAR.to_dict(),
        }
        if self.HOTA is not None:
            result["HOTA"] = self.HOTA.to_dict()
        return result

    def json(self, indent: int = 2) -> str:
        """Serialize to JSON string.

        Args:
            indent: JSON indentation level. Defaults to 2.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def table(self, columns: list[str] | None = None) -> str:
        """Format metrics as a table string.

        Args:
            columns: List of metric columns to include. If None, uses default
                columns based on available metrics.

        Returns:
            Formatted table string.
        """
        if columns is None:
            columns = _get_default_columns(self.HOTA is not None)

        return _format_sequence_table(self, columns)


@dataclass
class BenchmarkResult:
    """Result for benchmark (multi-sequence) evaluation.

    Attributes:
        sequences: Dictionary mapping sequence names to their results.
        aggregate: Aggregated metrics across all sequences.
    """

    sequences: dict[str, SequenceResult]
    aggregate: SequenceResult

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Create BenchmarkResult from a dictionary.

        Args:
            data: Dictionary with sequences and aggregate results.

        Returns:
            BenchmarkResult instance.
        """
        sequences = {
            name: SequenceResult.from_dict(seq_data)
            for name, seq_data in data["sequences"].items()
        }
        aggregate = SequenceResult.from_dict(data["aggregate"])
        return cls(sequences=sequences, aggregate=aggregate)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "sequences": {name: seq.to_dict() for name, seq in self.sequences.items()},
            "aggregate": self.aggregate.to_dict(),
        }

    def json(self, indent: int = 2) -> str:
        """Serialize to JSON string.

        Args:
            indent: JSON indentation level. Defaults to 2.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def table(self, columns: list[str] | None = None) -> str:
        """Format metrics as a table string.

        Args:
            columns: List of metric columns to include. If None, uses default
                columns based on available metrics.

        Returns:
            Formatted table string with all sequences and aggregate.
        """
        if columns is None:
            has_hota = self.aggregate.HOTA is not None
            columns = _get_default_columns(has_hota)

        return _format_benchmark_table(self.sequences, self.aggregate, columns)

    def save(self, path: str | Path) -> None:
        """Save results to a JSON file.

        Args:
            path: Path to save the JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.json())

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkResult:
        """Load results from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            BenchmarkResult instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        data = json.loads(path.read_text())
        return cls.from_dict(data)


def _get_default_columns(has_hota: bool) -> list[str]:
    """Get default columns based on available metrics.

    Args:
        has_hota: Whether HOTA metrics are available.

    Returns:
        List of default column names.
    """
    if has_hota:
        # Show key metrics from both HOTA and CLEAR
        return ["HOTA", "DetA", "AssA", "MOTA", "MOTP", "IDSW"]
    else:
        # CLEAR-only defaults
        return ["MOTA", "MOTP", "IDSW", "CLR_FP", "CLR_FN", "MT", "ML"]


def _get_metrics_dict(result: SequenceResult, col: str) -> float | int:
    """Get metric value from a SequenceResult.

    Args:
        result: The sequence result.
        col: Column name.

    Returns:
        The metric value.
    """
    # Check HOTA metrics first
    if result.HOTA is not None:
        hota_dict = result.HOTA.to_dict()
        if col in hota_dict:
            return hota_dict[col]

    # Fall back to CLEAR metrics
    clear_dict = result.CLEAR.to_dict()
    return clear_dict.get(col, 0)


def _format_value(value: float | int, is_float: bool) -> str:
    """Format a metric value for display.

    Float metrics are displayed as percentages with 3 decimal places
    (e.g., 99.353 for MOTA=0.99353), matching TrackEval output.
    Integer metrics are displayed as-is.

    Args:
        value: The metric value.
        is_float: Whether this is a float metric.

    Returns:
        Formatted string.
    """
    if is_float:
        # Display as percentage with 3 decimal places (TrackEval format)
        return f"{value * 100:.3f}"
    return str(value)


def _format_sequence_table(result: SequenceResult, columns: list[str]) -> str:
    """Format single sequence metrics as a table.

    Args:
        result: Sequence result.
        columns: Columns to include.

    Returns:
        Formatted table string.
    """
    # Determine column widths
    col_widths = {}
    for col in columns:
        value = _get_metrics_dict(result, col)
        is_float = col in ALL_FLOAT_FIELDS
        formatted = _format_value(value, is_float)
        col_widths[col] = max(len(col), len(formatted))

    # Build header
    header = "Sequence".ljust(30) + "  ".join(
        col.rjust(col_widths[col]) for col in columns
    )
    separator = "-" * len(header)

    # Build row
    row_values = []
    for col in columns:
        value = _get_metrics_dict(result, col)
        is_float = col in ALL_FLOAT_FIELDS
        formatted = _format_value(value, is_float)
        row_values.append(formatted.rjust(col_widths[col]))
    row = result.sequence.ljust(30) + "  ".join(row_values)

    return f"{header}\n{separator}\n{row}"


def _format_benchmark_table(
    sequences: dict[str, SequenceResult],
    aggregate: SequenceResult,
    columns: list[str],
) -> str:
    """Format benchmark metrics as a table.

    Args:
        sequences: Dictionary of sequence results.
        aggregate: Aggregate result.
        columns: Columns to include.

    Returns:
        Formatted table string.
    """
    # Collect all results for column width calculation
    all_results = [*list(sequences.values()), aggregate]

    col_widths = {}
    for col in columns:
        max_width = len(col)
        for result in all_results:
            value = _get_metrics_dict(result, col)
            is_float = col in ALL_FLOAT_FIELDS
            formatted = _format_value(value, is_float)
            max_width = max(max_width, len(formatted))
        col_widths[col] = max_width

    # Build header
    header = "Sequence".ljust(30) + "  ".join(
        col.rjust(col_widths[col]) for col in columns
    )
    separator = "-" * len(header)

    # Build rows
    lines = [header, separator]
    for seq_name in sorted(sequences.keys()):
        seq_result = sequences[seq_name]
        row_values = []
        for col in columns:
            value = _get_metrics_dict(seq_result, col)
            is_float = col in ALL_FLOAT_FIELDS
            formatted = _format_value(value, is_float)
            row_values.append(formatted.rjust(col_widths[col]))
        lines.append(seq_name.ljust(30) + "  ".join(row_values))

    # Add aggregate row
    lines.append(separator)
    agg_values = []
    for col in columns:
        value = _get_metrics_dict(aggregate, col)
        is_float = col in ALL_FLOAT_FIELDS
        formatted = _format_value(value, is_float)
        agg_values.append(formatted.rjust(col_widths[col]))
    lines.append("COMBINED".ljust(30) + "  ".join(agg_values))

    return "\n".join(lines)
