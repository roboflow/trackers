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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
class SequenceResult:
    """Result for a single sequence evaluation.

    Attributes:
        sequence: Name of the sequence.
        CLEAR: CLEAR metrics for this sequence.
    """

    sequence: str
    CLEAR: CLEARMetrics

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SequenceResult:
        """Create SequenceResult from a dictionary.

        Args:
            data: Dictionary with sequence name and metrics.

        Returns:
            SequenceResult instance.
        """
        return cls(
            sequence=data["sequence"],
            CLEAR=CLEARMetrics.from_dict(data["CLEAR"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "sequence": self.sequence,
            "CLEAR": self.CLEAR.to_dict(),
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
            columns: List of metric columns to include. If None, uses all
                summary fields in TrackEval order.

        Returns:
            Formatted table string.
        """
        if columns is None:
            columns = CLEAR_SUMMARY_FIELDS

        return _format_sequence_table(self.sequence, self.CLEAR, columns)


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
            columns: List of metric columns to include. If None, uses all
                summary fields in TrackEval order.

        Returns:
            Formatted table string with all sequences and aggregate.
        """
        if columns is None:
            columns = CLEAR_SUMMARY_FIELDS

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


def _format_sequence_table(
    sequence: str, metrics: CLEARMetrics, columns: list[str]
) -> str:
    """Format single sequence metrics as a table.

    Args:
        sequence: Sequence name.
        metrics: CLEAR metrics.
        columns: Columns to include.

    Returns:
        Formatted table string.
    """
    metrics_dict = metrics.to_dict()

    # Determine column widths
    col_widths = {}
    for col in columns:
        value = metrics_dict.get(col, 0)
        is_float = col in CLEAR_FLOAT_FIELDS
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
        value = metrics_dict.get(col, 0)
        is_float = col in CLEAR_FLOAT_FIELDS
        formatted = _format_value(value, is_float)
        row_values.append(formatted.rjust(col_widths[col]))
    row = sequence.ljust(30) + "  ".join(row_values)

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
    # Collect all values to determine column widths
    all_metrics = [seq.CLEAR.to_dict() for seq in sequences.values()]
    all_metrics.append(aggregate.CLEAR.to_dict())

    col_widths = {}
    for col in columns:
        max_width = len(col)
        for metrics_dict in all_metrics:
            value = metrics_dict.get(col, 0)
            is_float = col in CLEAR_FLOAT_FIELDS
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
        metrics_dict = seq_result.CLEAR.to_dict()
        row_values = []
        for col in columns:
            value = metrics_dict.get(col, 0)
            is_float = col in CLEAR_FLOAT_FIELDS
            formatted = _format_value(value, is_float)
            row_values.append(formatted.rjust(col_widths[col]))
        lines.append(seq_name.ljust(30) + "  ".join(row_values))

    # Add aggregate row
    lines.append(separator)
    agg_dict = aggregate.CLEAR.to_dict()
    agg_values = []
    for col in columns:
        value = agg_dict.get(col, 0)
        is_float = col in CLEAR_FLOAT_FIELDS
        formatted = _format_value(value, is_float)
        agg_values.append(formatted.rjust(col_widths[col]))
    lines.append("COMBINED".ljust(30) + "  ".join(agg_values))

    return "\n".join(lines)
