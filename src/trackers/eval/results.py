# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Result classes for tracking evaluation metrics.

This module provides dataclasses for storing and manipulating evaluation results with methods for serialization,
display, and persistence.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from trackers.eval.errors import AggregationIncompatibleError

AggregationMode = Literal["tp_weighted", "scene_mean"]

# Official AI City 2024 scene-mean fields. All other HOTA summary fields are
# undefined under an unweighted scene mean and serialize as JSON null.
_SCENE_MEAN_DEFINED_HOTA_FIELDS = frozenset({"HOTA", "DetA", "AssA", "LocA"})


def _reject_nonfinite_json_constant(value: str) -> float:
    """Reject NaN/Infinity so ``json.loads(..., parse_constant=...)`` stays strict."""
    raise ValueError(f"Non-finite JSON constant is not allowed: {value!r}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


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
    "OWTA",
]
HOTA_INT_FIELDS = [
    "HOTA_TP",
    "HOTA_FN",
    "HOTA_FP",
]
HOTA_SUMMARY_FIELDS = HOTA_FLOAT_FIELDS + HOTA_INT_FIELDS

# TrackEval summary field order for Identity metrics
IDENTITY_FLOAT_FIELDS = [
    "IDF1",
    "IDR",
    "IDP",
]
IDENTITY_INT_FIELDS = [
    "IDTP",
    "IDFN",
    "IDFP",
]
IDENTITY_SUMMARY_FIELDS = IDENTITY_FLOAT_FIELDS + IDENTITY_INT_FIELDS

# All float fields for formatting
ALL_FLOAT_FIELDS = CLEAR_FLOAT_FIELDS + HOTA_FLOAT_FIELDS + IDENTITY_FLOAT_FIELDS


@dataclass
class CLEARMetrics:
    """CLEAR metrics with TrackEval-compatible field names.

    Float metrics are stored as fractions (0-1 range), not percentages. The values follow the original CLEAR
    MOT definitions.

    Attributes:
        MOTA: Multiple Object Tracking Accuracy. Penalizes false negatives,
            false positives, and ID switches: `(TP - FP - IDSW) / (TP + FN)`.
            Can be negative when errors exceed matches.
        MOTP: Multiple Object Tracking Precision. Mean IoU of matched pairs.
            Measures localization quality only.
        MODA: Multiple Object Detection Accuracy. Like MOTA but ignores ID
            switches: `(TP - FP) / (TP + FN)`.
        CLR_Re: CLEAR recall. Fraction of GT detections matched:
            `TP / (TP + FN)`.
        CLR_Pr: CLEAR precision. Fraction of tracker detections correct:
            `TP / (TP + FP)`.
        MTR: Mostly tracked ratio. Fraction of GT tracks tracked for >80% of
            their lifespan.
        PTR: Partially tracked ratio. Fraction of GT tracks tracked for 20-80%.
        MLR: Mostly lost ratio. Fraction of GT tracks tracked for <20%.
        sMOTA: Summed MOTA. Replaces TP count with IoU sum:
            `(MOTP_sum - FP - IDSW) / (TP + FN)`.
        CLR_TP: True positives. Number of correct matches.
        CLR_FN: False negatives. Number of missed GT detections.
        CLR_FP: False positives. Number of spurious tracker detections.
        IDSW: ID switches. Times a GT track changes its matched tracker ID.
        MT: Mostly tracked count. Number of GT tracks tracked >80%.
        PT: Partially tracked count. Number of GT tracks tracked 20-80%.
        ML: Mostly lost count. Number of GT tracks tracked <20%.
        Frag: Fragmentations. Times a tracked GT becomes untracked then tracked
            again.
        MOTP_sum: Raw IoU sum for aggregation across sequences.
        CLR_Frames: Number of frames evaluated.
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
        """Create `CLEARMetrics` from a dictionary.

        Args:
            data: Dictionary with metric values.

        Returns:
            `CLEARMetrics` instance.
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
        """Convert to dictionary representation.

        Returns:
            Dictionary with all metric values.
        """
        return dataclasses.asdict(self)


@dataclass
class HOTAMetrics:
    """HOTA metrics with TrackEval-compatible field names.

    HOTA evaluates both detection quality and association quality. Float metrics are stored as fractions
    (0-1 range).

    Attributes:
        HOTA: Higher Order Tracking Accuracy. Geometric mean of DetA and
            AssA, averaged over 19 IoU thresholds (0.05 to 0.95).
        DetA: Detection accuracy: `TP / (TP + FN + FP)`.
        AssA: Association accuracy for matched detections over time.
        DetRe: Detection recall: `TP / (TP + FN)`.
        DetPr: Detection precision: `TP / (TP + FP)`.
        AssRe: Association recall. For each GT ID, measures how consistently
            it maps to a single tracker ID across time.
        AssPr: Association precision. For each tracker ID, measures how
            consistently it maps to a single GT ID across time.
        LocA: Localization accuracy. Mean IoU for matched pairs.
        OWTA: Open World Tracking Accuracy. `sqrt(DetRe * AssA)`, useful when
            precision is less meaningful.
        HOTA_TP: True positive count summed over all 19 thresholds.
        HOTA_FN: False negative count summed over all 19 thresholds.
        HOTA_FP: False positive count summed over all 19 thresholds.
    """

    HOTA: float
    DetA: float
    AssA: float
    DetRe: float | None
    DetPr: float | None
    AssRe: float | None
    AssPr: float | None
    LocA: float
    OWTA: float | None
    HOTA_TP: int | None
    HOTA_FN: int | None
    HOTA_FP: int | None
    # Per-alpha arrays for aggregation (not serialized to JSON by default)
    _arrays: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HOTAMetrics:
        """Create `HOTAMetrics` from a dictionary.

        Args:
            data: Dictionary with metric values. Undefined scene-mean fields may
                be JSON ``null`` / ``None``.

        Returns:
            `HOTAMetrics` instance.
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
            DetRe=_optional_float(data.get("DetRe")),
            DetPr=_optional_float(data.get("DetPr")),
            AssRe=_optional_float(data.get("AssRe")),
            AssPr=_optional_float(data.get("AssPr")),
            LocA=float(data["LocA"]),
            OWTA=_optional_float(data.get("OWTA")),
            HOTA_TP=_optional_int(data.get("HOTA_TP")),
            HOTA_FN=_optional_int(data.get("HOTA_FN")),
            HOTA_FP=_optional_int(data.get("HOTA_FP")),
            _arrays=arrays,
        )

    def to_dict(self, include_arrays: bool = False, arrays_as_list: bool = True) -> dict[str, Any]:
        """Convert to dictionary representation.

        Args:
            include_arrays: Whether to include per-alpha arrays. Defaults to `False`.
            arrays_as_list: Whether to convert arrays to lists for JSON serialization.
                Defaults to `True`. Set to `False` to keep numpy arrays.

        Returns:
            Dictionary with all metric values. Undefined optional fields are
            ``None`` (JSON ``null``); NaN is never emitted.
        """

        def _finite_or_none(value: float | None) -> float | None:
            if value is None:
                return None
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
            return value

        result = {
            "HOTA": float(self.HOTA),
            "DetA": float(self.DetA),
            "AssA": float(self.AssA),
            "DetRe": _finite_or_none(self.DetRe),
            "DetPr": _finite_or_none(self.DetPr),
            "AssRe": _finite_or_none(self.AssRe),
            "AssPr": _finite_or_none(self.AssPr),
            "LocA": float(self.LocA),
            "OWTA": _finite_or_none(self.OWTA),
            "HOTA_TP": self.HOTA_TP,
            "HOTA_FN": self.HOTA_FN,
            "HOTA_FP": self.HOTA_FP,
        }
        if include_arrays and self._arrays:
            for key, arr in self._arrays.items():
                if arrays_as_list:
                    result[key] = arr.tolist() if isinstance(arr, np.ndarray) else arr
                else:
                    result[key] = arr
        return result


@dataclass
class IdentityMetrics:
    """Identity metrics with TrackEval-compatible field names.

    Identity metrics measure global ID consistency using an optimal one-to-one assignment between GT
    and tracker IDs across the full sequence.

    Attributes:
        IDF1: ID F1 score. Harmonic mean of IDR and IDP, the primary
            identity metric.
        IDR: ID recall. `IDTP / (IDTP + IDFN)`, fraction of GT detections
            with correct global ID assignment.
        IDP: ID precision. `IDTP / (IDTP + IDFP)`, fraction of tracker
            detections with correct global ID assignment.
        IDTP: ID true positives. Detections matched with globally consistent
            IDs.
        IDFN: ID false negatives. GT detections not matched or matched to the
            wrong global ID.
        IDFP: ID false positives. Tracker detections not matched or matched
            to the wrong global ID.
    """

    IDF1: float
    IDR: float
    IDP: float
    IDTP: int
    IDFN: int
    IDFP: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentityMetrics:
        """Create `IdentityMetrics` from a dictionary.

        Args:
            data: Dictionary with metric values.

        Returns:
            `IdentityMetrics` instance.
        """
        return cls(
            IDF1=float(data["IDF1"]),
            IDR=float(data["IDR"]),
            IDP=float(data["IDP"]),
            IDTP=int(data["IDTP"]),
            IDFN=int(data["IDFN"]),
            IDFP=int(data["IDFP"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all metric values.
        """
        return dataclasses.asdict(self)


@dataclass
class SequenceResult:
    """Result for a single sequence evaluation.

    Attributes:
        sequence: Name of the sequence.
        CLEAR: CLEAR metrics for this sequence, or `None` if not requested.
        HOTA: HOTA metrics for this sequence, or `None` if not requested.
        Identity: Identity metrics for this sequence, or `None` if not
            requested.
    """

    sequence: str
    CLEAR: CLEARMetrics | None = None
    HOTA: HOTAMetrics | None = None
    Identity: IdentityMetrics | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SequenceResult:
        """Create `SequenceResult` from a dictionary.

        Args:
            data: Dictionary with sequence name and metrics.

        Returns:
            `SequenceResult` instance.
        """
        clear = None
        if "CLEAR" in data and data["CLEAR"] is not None:
            clear = CLEARMetrics.from_dict(data["CLEAR"])

        hota = None
        if "HOTA" in data and data["HOTA"] is not None:
            hota = HOTAMetrics.from_dict(data["HOTA"])

        identity = None
        if "Identity" in data and data["Identity"] is not None:
            identity = IdentityMetrics.from_dict(data["Identity"])

        return cls(
            sequence=data["sequence"],
            CLEAR=clear,
            HOTA=hota,
            Identity=identity,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all metric values.
        """
        result: dict[str, Any] = {
            "sequence": self.sequence,
        }
        if self.CLEAR is not None:
            result["CLEAR"] = self.CLEAR.to_dict()
        if self.HOTA is not None:
            result["HOTA"] = self.HOTA.to_dict()
        if self.Identity is not None:
            result["Identity"] = self.Identity.to_dict()
        return result

    def json(self, indent: int = 2) -> str:
        """Serialize to JSON string.

        Args:
            indent: Indentation level for formatting. Defaults to `2`.

        Returns:
            JSON string representation. Uses ``allow_nan=False`` so non-finite
            floats cannot sneak into the payload.
        """
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def table(self, columns: list[str] | None = None) -> str:
        """Format as a table string.

        Args:
            columns: Metric columns to include. If `None`, includes all available
                metrics.

        Returns:
            Formatted table string.
        """
        if columns is None:
            columns = _get_available_columns(
                has_clear=self.CLEAR is not None,
                has_hota=self.HOTA is not None,
                has_identity=self.Identity is not None,
            )

        return _format_sequence_table(self, columns)


@dataclass
class BenchmarkCoverage:
    """Serialized identity and completeness of a benchmark run.

    Examples:
        >>> coverage = BenchmarkCoverage(
        ...     benchmark="AI City Challenge 2024",
        ...     split="test",
        ...     protocol="MTMC_Tracking_2024",
        ...     file_format="aicity-2024",
        ...     canonical_scene_camera_map_sha256="abc",
        ...     scene_camera_map_sha256="abc",
        ...     expected_scenes=["scene_061"],
        ...     evaluated_scenes=["scene_061"],
        ...     missing_scenes=[],
        ...     complete=True,
        ... )
        >>> coverage.complete
        True
    """

    benchmark: str
    split: str
    protocol: str
    file_format: str
    canonical_scene_camera_map_sha256: str
    scene_camera_map_sha256: str
    expected_scenes: list[str]
    evaluated_scenes: list[str]
    missing_scenes: list[str]
    complete: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkCoverage:
        """Create benchmark coverage from serialized data."""
        string_fields = (
            "benchmark",
            "split",
            "protocol",
            "file_format",
            "canonical_scene_camera_map_sha256",
        )
        if any(not isinstance(data.get(field), str) for field in string_fields):
            raise ValueError("Benchmark coverage identity fields must be strings.")
        for scene_field in ("expected_scenes", "evaluated_scenes", "missing_scenes"):
            value = data.get(scene_field)
            if not isinstance(value, list) or not all(isinstance(scene, str) for scene in value):
                raise ValueError(f"Benchmark coverage {scene_field} must be a list of strings.")
            if len(value) != len(set(value)):
                raise ValueError(f"Benchmark coverage {scene_field} must not contain duplicates.")
        if type(data.get("complete")) is not bool:
            raise ValueError("Benchmark coverage complete must be a boolean.")
        canonical_hash = data["canonical_scene_camera_map_sha256"]
        scene_hash = data.get("scene_camera_map_sha256", canonical_hash)
        if not isinstance(scene_hash, str):
            raise ValueError("Benchmark coverage scene_camera_map_sha256 must be a string.")

        expected = list(data["expected_scenes"])
        evaluated = list(data["evaluated_scenes"])
        missing = list(data["missing_scenes"])
        expected_set = set(expected)
        if missing != [scene for scene in expected if scene not in set(evaluated)]:
            raise ValueError("Benchmark coverage missing_scenes is inconsistent with evaluated_scenes.")
        complete = data["complete"]
        hashes_match = scene_hash == canonical_hash
        if complete != (not missing and set(evaluated) == expected_set and hashes_match):
            raise ValueError("Benchmark coverage complete is inconsistent with scene coverage.")
        return cls(
            benchmark=data["benchmark"],
            split=data["split"],
            protocol=data["protocol"],
            file_format=data["file_format"],
            canonical_scene_camera_map_sha256=canonical_hash,
            scene_camera_map_sha256=scene_hash,
            expected_scenes=expected,
            evaluated_scenes=evaluated,
            missing_scenes=missing,
            complete=complete,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert benchmark coverage to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)


@dataclass
class BenchmarkResult:
    """Result for multi-sequence evaluation.

    Attributes:
        sequences: Dictionary mapping sequence names to their results.
        aggregate: Combined metrics across all sequences.
        aggregation: How ``aggregate`` was produced. ``\"tp_weighted\"`` is the
            MOT TrackEval path (`aggregate_hota_metrics`). ``\"scene_mean\"`` is
            the AI City 2024 unweighted mean of per-scene HOTA/DetA/AssA/LocA.
            Defaults to ``\"tp_weighted\"`` so existing serialized JSON loads
            unchanged.
        coverage: Benchmark and scene coverage identity. Legacy payloads omit
            this field and load with ``None``.
    """

    sequences: dict[str, SequenceResult]
    aggregate: SequenceResult
    aggregation: AggregationMode = "tp_weighted"
    coverage: BenchmarkCoverage | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Create `BenchmarkResult` from a dictionary.

        Args:
            data: Dictionary with sequences and aggregate results.

        Returns:
            `BenchmarkResult` instance.
        """
        sequences = {name: SequenceResult.from_dict(seq_data) for name, seq_data in data["sequences"].items()}
        aggregate = SequenceResult.from_dict(data["aggregate"])
        aggregation = data.get("aggregation", "tp_weighted")
        if aggregation not in ("tp_weighted", "scene_mean"):
            raise ValueError(f"Unknown aggregation mode: {aggregation!r}")
        coverage_data = data.get("coverage")
        coverage = BenchmarkCoverage.from_dict(coverage_data) if coverage_data is not None else None
        if coverage is not None and set(sequences) != set(coverage.evaluated_scenes):
            raise ValueError("Benchmark coverage evaluated_scenes must match serialized sequence results.")
        return cls(sequences=sequences, aggregate=aggregate, aggregation=aggregation, coverage=coverage)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all metric values.
        """
        result: dict[str, Any] = {
            "sequences": {name: seq.to_dict() for name, seq in self.sequences.items()},
            "aggregate": self.aggregate.to_dict(),
            "aggregation": self.aggregation,
        }
        if self.coverage is not None:
            result["coverage"] = self.coverage.to_dict()
        return result

    def json(self, indent: int = 2) -> str:
        """Serialize to JSON string.

        Args:
            indent: Indentation level for formatting. Defaults to `2`.

        Returns:
            JSON string representation. Uses ``allow_nan=False`` so non-finite
            floats cannot sneak into the payload.
        """
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def tp_weighted_hota(self) -> HOTAMetrics:
        """Recompute a TP-weighted HOTA aggregate from per-sequence results.

        This is the public re-aggregation entry point. Scene-mean multicamera
        results cannot be re-aggregated this way and raise a typed error rather
        than a bare ``KeyError`` from missing per-alpha arrays.

        Returns:
            TP-weighted `HOTAMetrics` across ``self.sequences``.

        Raises:
            AggregationIncompatibleError: If ``aggregation == \"scene_mean\"`` or
                any sequence lacks the per-alpha arrays required by
                `aggregate_hota_metrics`.
        """
        if self.aggregation == "scene_mean":
            raise AggregationIncompatibleError(
                "Cannot TP-weight a scene_mean BenchmarkResult. "
                "Per-scene HOTA/DetA/AssA/LocA were already averaged unweighted; "
                "re-aggregation via aggregate_hota_metrics is undefined."
            )
        from trackers.eval.hota import aggregate_hota_metrics

        sequence_metrics = [
            sequence.HOTA.to_dict(include_arrays=True, arrays_as_list=False)
            for sequence in self.sequences.values()
            if sequence.HOTA is not None
        ]
        if not sequence_metrics:
            raise ValueError("No HOTA results available to aggregate.")
        return HOTAMetrics.from_dict(aggregate_hota_metrics(sequence_metrics))

    def table(self, columns: list[str] | None = None) -> str:
        """Format as a table string.

        Args:
            columns: Metric columns to include. If `None`, includes all available
                metrics. For ``aggregation=\"scene_mean\"``, defaults to the four
                official fields only.

        Returns:
            Formatted table string.
        """
        if columns is None:
            if self.aggregation == "scene_mean":
                columns = ["HOTA", "DetA", "AssA", "LocA"]
            else:
                columns = _get_available_columns(
                    has_clear=self.aggregate.CLEAR is not None,
                    has_hota=self.aggregate.HOTA is not None,
                    has_identity=self.aggregate.Identity is not None,
                )

        return _format_benchmark_table(self.sequences, self.aggregate, columns)

    def save(self, path: str | Path) -> None:
        """Save to a JSON file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.json())

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkResult:
        """Load from a JSON file.

        Args:
            path: Source file path.

        Returns:
            `BenchmarkResult` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        data = json.loads(path.read_text(), parse_constant=_reject_nonfinite_json_constant)
        _reject_nonfinite_numeric_strings(data)
        return cls.from_dict(data)


def _reject_nonfinite_numeric_strings(value: Any, *, field_name: str | None = None) -> None:
    """Reject string spellings that numeric result fields would coerce to non-finite floats."""
    _reject_nonfinite_result_scalar(value, field_name=field_name)
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite_numeric_strings(child, field_name=str(key))
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite_numeric_strings(child, field_name=field_name)


def _reject_nonfinite_result_scalar(value: Any, *, field_name: str | None) -> None:
    """Reject one decoded scalar when its field is numeric."""
    numeric_fields = set(CLEAR_SUMMARY_FIELDS + HOTA_SUMMARY_FIELDS + IDENTITY_SUMMARY_FIELDS)
    if field_name is None or (field_name not in numeric_fields and not field_name.endswith("_array")):
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value is not allowed for {field_name}: {value!r}")
    if not isinstance(value, str):
        return
    try:
        number = float(value)
    except ValueError:
        return
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric string is not allowed for {field_name}: {value!r}")


def _get_available_columns(has_clear: bool = False, has_hota: bool = False, has_identity: bool = False) -> list[str]:
    """Get column names for the metrics that were computed.

    Returns all summary fields for each metric type that is available.

    Args:
        has_clear: Whether CLEAR metrics are available.
        has_hota: Whether HOTA metrics are available.
        has_identity: Whether Identity metrics are available.

    Returns:
        List of column names for available metrics.
    """
    columns: list[str] = []
    if has_clear:
        columns.extend(CLEAR_SUMMARY_FIELDS)
    if has_hota:
        columns.extend(HOTA_SUMMARY_FIELDS)
    if has_identity:
        columns.extend(IDENTITY_SUMMARY_FIELDS)
    return columns


def _get_metrics_dict(result: SequenceResult, col: str) -> float | int | None:
    """Get metric value from a SequenceResult.

    Args:
        result: The sequence result.
        col: Column name.

    Returns:
        The metric value, or ``None`` when the field is undefined.
    """
    # Check CLEAR metrics
    if result.CLEAR is not None:
        clear_dict = result.CLEAR.to_dict()
        if col in clear_dict:
            return clear_dict[col]

    # Check HOTA metrics
    if result.HOTA is not None:
        hota_dict = result.HOTA.to_dict()
        if col in hota_dict:
            return hota_dict[col]

    # Check Identity metrics
    if result.Identity is not None:
        identity_dict = result.Identity.to_dict()
        if col in identity_dict:
            return identity_dict[col]

    return 0


def _format_value(value: float | int | None, is_float: bool) -> str:
    """Format a metric value for display.

    Float metrics are displayed as percentages with 3 decimal places
    (e.g., 99.353 for MOTA=0.99353), matching TrackEval output.
    Integer metrics are displayed as-is. Undefined (``None``) fields render
    as ``-``.

    Args:
        value: The metric value.
        is_float: Whether this is a float metric.

    Returns:
        Formatted string.
    """
    if value is None:
        return "-"
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
    header = "Sequence".ljust(30) + "  ".join(col.rjust(col_widths[col]) for col in columns)
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
    header = "Sequence".ljust(30) + "  ".join(col.rjust(col_widths[col]) for col in columns)
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
    lines.append(aggregate.sequence.ljust(30) + "  ".join(agg_values))

    return "\n".join(lines)
