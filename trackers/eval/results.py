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

    Float metrics are stored as fractions (0-1 range), not percentages. The
    values follow the original CLEAR MOT definitions.

    Attributes:
        `MOTA`: Multiple Object Tracking Accuracy. Penalizes false negatives,
            false positives, and ID switches: `(TP - FP - IDSW) / (TP + FN)`.
            Can be negative when errors exceed matches.
        `MOTP`: Multiple Object Tracking Precision. Mean IoU of matched pairs.
            Measures localization quality only.
        `MODA`: Multiple Object Detection Accuracy. Like `MOTA` but ignores ID
            switches: `(TP - FP) / (TP + FN)`.
        `CLR_Re`: CLEAR recall. Fraction of GT detections matched:
            `TP / (TP + FN)`.
        `CLR_Pr`: CLEAR precision. Fraction of tracker detections correct:
            `TP / (TP + FP)`.
        `MTR`: Mostly tracked ratio. Fraction of GT tracks tracked for >80% of
            their lifespan.
        `PTR`: Partially tracked ratio. Fraction of GT tracks tracked for 20-80%.
        `MLR`: Mostly lost ratio. Fraction of GT tracks tracked for <20%.
        `sMOTA`: Summed MOTA. Replaces TP count with IoU sum:
            `(MOTP_sum - FP - IDSW) / (TP + FN)`.
        `CLR_TP`: True positives. Number of correct matches.
        `CLR_FN`: False negatives. Number of missed GT detections.
        `CLR_FP`: False positives. Number of spurious tracker detections.
        `IDSW`: ID switches. Times a GT track changes its matched tracker ID.
        `MT`: Mostly tracked count. Number of GT tracks tracked >80%.
        `PT`: Partially tracked count. Number of GT tracks tracked 20-80%.
        `ML`: Mostly lost count. Number of GT tracks tracked <20%.
        `Frag`: Fragmentations. Times a tracked GT becomes untracked then tracked
            again.
        `MOTP_sum`: Raw IoU sum for aggregation across sequences.
        `CLR_Frames`: Number of frames evaluated.
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

        Examples:
            ```pycon
            >>> from trackers.eval import CLEARMetrics
            >>>
            >>> data = {
            ...     "MOTA": 0.756, "MOTP": 0.813, "MODA": 0.789,
            ...     "CLR_Re": 0.824, "CLR_Pr": 0.915, "MTR": 0.62,
            ...     "PTR": 0.27, "MLR": 0.11, "sMOTA": 0.741,
            ...     "CLR_TP": 123, "CLR_FN": 26, "CLR_FP": 11,
            ...     "IDSW": 4, "MT": 52, "PT": 23, "ML": 9, "Frag": 12,
            ... }
            >>> metrics = CLEARMetrics.from_dict(data)
            >>> print(metrics.MOTA)
            # 0.756
            ```
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

        Examples:
            ```pycon
            >>> from trackers.eval import CLEARMetrics
            >>>
            >>> metrics = CLEARMetrics(
            ...     MOTA=0.756, MOTP=0.813, MODA=0.789,
            ...     CLR_Re=0.824, CLR_Pr=0.915, MTR=0.62,
            ...     PTR=0.27, MLR=0.11, sMOTA=0.741,
            ...     CLR_TP=123, CLR_FN=26, CLR_FP=11,
            ...     IDSW=4, MT=52, PT=23, ML=9, Frag=12,
            ... )
            >>> print(metrics.to_dict()["MOTA"])
            # 0.756
            ```
        """
        return dataclasses.asdict(self)


@dataclass
class HOTAMetrics:
    """HOTA metrics with TrackEval-compatible field names.

    HOTA evaluates both detection quality and association quality. Float
    metrics are stored as fractions (0-1 range).

    Attributes:
        `HOTA`: Higher Order Tracking Accuracy. Geometric mean of `DetA` and
            `AssA`, averaged over 19 IoU thresholds (0.05 to 0.95).
        `DetA`: Detection accuracy: `TP / (TP + FN + FP)`.
        `AssA`: Association accuracy for matched detections over time.
        `DetRe`: Detection recall: `TP / (TP + FN)`.
        `DetPr`: Detection precision: `TP / (TP + FP)`.
        `AssRe`: Association recall. For each GT ID, measures how consistently
            it maps to a single tracker ID across time.
        `AssPr`: Association precision. For each tracker ID, measures how
            consistently it maps to a single GT ID across time.
        `LocA`: Localization accuracy. Mean IoU for matched pairs.
        `OWTA`: Open World Tracking Accuracy. `sqrt(DetRe * AssA)`, useful when
            precision is less meaningful.
        `HOTA_TP`: True positive count summed over all 19 thresholds.
        `HOTA_FN`: False negative count summed over all 19 thresholds.
        `HOTA_FP`: False positive count summed over all 19 thresholds.
    """

    HOTA: float
    DetA: float
    AssA: float
    DetRe: float
    DetPr: float
    AssRe: float
    AssPr: float
    LocA: float
    OWTA: float
    HOTA_TP: int
    HOTA_FN: int
    HOTA_FP: int
    # Per-alpha arrays for aggregation (not serialized to JSON by default)
    _arrays: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HOTAMetrics:
        """Create `HOTAMetrics` from a dictionary.

        Args:
            data: Dictionary with metric values.

        Returns:
            `HOTAMetrics` instance.

        Examples:
            ```pycon
            >>> from trackers.eval import HOTAMetrics
            >>>
            >>> data = {
            ...     "HOTA": 0.623, "DetA": 0.712, "AssA": 0.548,
            ...     "DetRe": 0.731, "DetPr": 0.695, "AssRe": 0.534,
            ...     "AssPr": 0.562, "LocA": 0.779, "OWTA": 0.627,
            ...     "HOTA_TP": 8900, "HOTA_FN": 2100, "HOTA_FP": 1700,
            ... }
            >>> metrics = HOTAMetrics.from_dict(data)
            >>> print(metrics.HOTA)
            # 0.623
            ```
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
            OWTA=float(data["OWTA"]),
            HOTA_TP=int(data["HOTA_TP"]),
            HOTA_FN=int(data["HOTA_FN"]),
            HOTA_FP=int(data["HOTA_FP"]),
            _arrays=arrays,
        )

    def to_dict(
        self, include_arrays: bool = False, arrays_as_list: bool = True
    ) -> dict[str, Any]:
        """Convert to dictionary.

        Args:
            include_arrays: Whether to include per-alpha arrays. Defaults to `False`.
            arrays_as_list: Whether to convert arrays to lists (for JSON).
                Defaults to `True`. Set to `False` to keep numpy arrays.

        Returns:
            Dictionary with metric values.

        Examples:
            ```pycon
            >>> from trackers.eval import HOTAMetrics
            >>>
            >>> metrics = HOTAMetrics(
            ...     HOTA=0.623, DetA=0.712, AssA=0.548,
            ...     DetRe=0.731, DetPr=0.695, AssRe=0.534,
            ...     AssPr=0.562, LocA=0.779, OWTA=0.627,
            ...     HOTA_TP=8900, HOTA_FN=2100, HOTA_FP=1700,
            ... )
            >>> print(metrics.to_dict()["HOTA"])
            # 0.623
            ```
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
            "OWTA": self.OWTA,
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

    Identity metrics measure global ID consistency using an optimal one-to-one
    assignment between GT and tracker IDs across the full sequence.

    Attributes:
        `IDF1`: ID F1 score. Harmonic mean of `IDR` and `IDP`, the primary
            identity metric.
        `IDR`: ID recall. `IDTP / (IDTP + IDFN)`, fraction of GT detections
            with correct global ID assignment.
        `IDP`: ID precision. `IDTP / (IDTP + IDFP)`, fraction of tracker
            detections with correct global ID assignment.
        `IDTP`: ID true positives. Detections matched with globally consistent
            IDs.
        `IDFN`: ID false negatives. GT detections not matched or matched to the
            wrong global ID.
        `IDFP`: ID false positives. Tracker detections not matched or matched
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

        Examples:
            ```pycon
            >>> from trackers.eval import IdentityMetrics
            >>>
            >>> data = {
            ...     "IDF1": 0.721, "IDR": 0.704, "IDP": 0.739,
            ...     "IDTP": 11000, "IDFN": 3900, "IDFP": 3800,
            ... }
            >>> metrics = IdentityMetrics.from_dict(data)
            >>> print(metrics.IDF1)
            # 0.721
            ```
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
        """Convert to dictionary.

        Returns:
            Dictionary with all metric values.

        Examples:
            ```pycon
            >>> from trackers.eval import IdentityMetrics
            >>>
            >>> metrics = IdentityMetrics(
            ...     IDF1=0.721, IDR=0.704, IDP=0.739,
            ...     IDTP=11000, IDFN=3900, IDFP=3800,
            ... )
            >>> print(metrics.to_dict()["IDF1"])
            # 0.721
            ```
        """
        return dataclasses.asdict(self)


@dataclass
class SequenceResult:
    """Result for a single sequence evaluation.

    Attributes:
        `sequence`: Name of the sequence.
        `CLEAR`: `CLEARMetrics` for this sequence, or `None` if not requested.
        `HOTA`: `HOTAMetrics` for this sequence, or `None` if not requested.
        `Identity`: `IdentityMetrics` for this sequence, or `None` if not
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

        Examples:
            ```pycon
            >>> from trackers.eval import SequenceResult
            >>>
            >>> data = {
            ...     "sequence": "MOT17-02",
            ...     "CLEAR": {
            ...         "MOTA": 0.756,
            ...         "MOTP": 0.813,
            ...         "MODA": 0.789,
            ...         "CLR_Re": 0.824,
            ...         "CLR_Pr": 0.915,
            ...         "MTR": 0.62,
            ...         "PTR": 0.27,
            ...         "MLR": 0.11,
            ...         "sMOTA": 0.741,
            ...         "CLR_TP": 123,
            ...         "CLR_FN": 26,
            ...         "CLR_FP": 11,
            ...         "IDSW": 4,
            ...         "MT": 52,
            ...         "PT": 23,
            ...         "ML": 9,
            ...         "Frag": 12,
            ...     },
            ... }
            >>>
            >>> result = SequenceResult.from_dict(data)
            >>> print(result.sequence)
            # MOT17-02
            ```
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
        """Convert to dictionary.

        Returns:
            Dictionary representation.

        Examples:
            ```pycon
            >>> from trackers.eval import SequenceResult
            >>>
            >>> result = SequenceResult(sequence="MOT17-02")
            >>> data = result.to_dict()
            >>> print(data)
            # {'sequence': 'MOT17-02'}
            >>> print(data["sequence"])
            # MOT17-02
            ```
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
            indent: JSON indentation level. Defaults to 2.

        Returns:
            JSON string representation.

        Examples:
            ```pycon
            >>> from trackers.eval import SequenceResult
            >>>
            >>> result = SequenceResult(sequence="MOT17-02")
            >>> print(result.json())
            # {
            #   "sequence": "MOT17-02"
            # }
            ```
        """
        return json.dumps(self.to_dict(), indent=indent)

    def table(self, columns: list[str] | None = None) -> str:
        """Format metrics as a table string.

        Args:
            columns: List of metric columns to include. If None, uses default
                columns based on available metrics.

        Returns:
            Formatted table string.

        Examples:
            ```pycon
            >>> from trackers.eval import (
            ...     SequenceResult, CLEARMetrics, HOTAMetrics, IdentityMetrics
            ... )
            >>>
            >>> result = SequenceResult(
            ...     sequence="MOT17-02",
            ...     CLEAR=CLEARMetrics(
            ...         MOTA=0.756, MOTP=0.813, MODA=0.789,
            ...         CLR_Re=0.824, CLR_Pr=0.915, MTR=0.62,
            ...         PTR=0.27, MLR=0.11, sMOTA=0.741,
            ...         CLR_TP=123, CLR_FN=26, CLR_FP=11,
            ...         IDSW=42, MT=52, PT=23, ML=9, Frag=12,
            ...     ),
            ...     HOTA=HOTAMetrics(
            ...         HOTA=0.623, DetA=0.712, AssA=0.548,
            ...         DetRe=0.731, DetPr=0.695, AssRe=0.534,
            ...         AssPr=0.562, LocA=0.779, OWTA=0.627,
            ...         HOTA_TP=8900, HOTA_FN=2100, HOTA_FP=1700,
            ...     ),
            ...     Identity=IdentityMetrics(
            ...         IDF1=0.721, IDR=0.704, IDP=0.739,
            ...         IDTP=11000, IDFN=3900, IDFP=3800,
            ...     ),
            ... )
            >>>
            >>> print(result.table(columns=["MOTA", "HOTA", "IDF1", "IDSW"]))
            # Sequence                        MOTA    HOTA    IDF1  IDSW
            # ----------------------------------------------------------
            # MOT17-02                      75.600  62.300  72.100    42
            ```
        """
        if columns is None:
            columns = _get_available_columns(
                has_clear=self.CLEAR is not None,
                has_hota=self.HOTA is not None,
                has_identity=self.Identity is not None,
            )

        return _format_sequence_table(self, columns)


@dataclass
class BenchmarkResult:
    """Result for multi-sequence evaluation.

    Attributes:
        `sequences`: Dictionary mapping sequence names to `SequenceResult`.
        `aggregate`: Combined metrics across all sequences. Has
            `sequence="COMBINED"`.
    """

    sequences: dict[str, SequenceResult]
    aggregate: SequenceResult

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Create `BenchmarkResult` from a dictionary.

        Args:
            data: Dictionary with sequences and aggregate results.

        Returns:
            `BenchmarkResult` instance.

        Examples:
            ```pycon
            >>> from trackers.eval import BenchmarkResult
            >>>
            >>> data = {
            ...     "sequences": {},
            ...     "aggregate": {"sequence": "COMBINED"},
            ... }
            >>> result = BenchmarkResult.from_dict(data)
            >>> print(result.aggregate.sequence)
            # COMBINED
            ```
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

        Examples:
            ```pycon
            >>> from trackers.eval import BenchmarkResult, SequenceResult
            >>>
            >>> bench = BenchmarkResult(
            ...     sequences={"seq1": SequenceResult(sequence="seq1")},
            ...     aggregate=SequenceResult(sequence="COMBINED"),
            ... )
            >>> data = bench.to_dict()
            >>> print("aggregate" in data)
            # True
            ```
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

        Examples:
            ```pycon
            >>> from trackers.eval import BenchmarkResult, SequenceResult
            >>>
            >>> bench = BenchmarkResult(
            ...     sequences={},
            ...     aggregate=SequenceResult(sequence="COMBINED"),
            ... )
            >>> print(bench.json())
            # {
            #   "sequences": {},
            #   "aggregate": {
            #     "sequence": "COMBINED"
            #   }
            # }
            ```
        """
        return json.dumps(self.to_dict(), indent=indent)

    def table(self, columns: list[str] | None = None) -> str:
        """Format metrics as a table string.

        Args:
            columns: List of metric columns to include. If None, uses default
                columns based on available metrics.

        Returns:
            Formatted table string with all sequences and aggregate.

        Examples:
            ```pycon
            >>> from trackers.eval import (
            ...     BenchmarkResult, SequenceResult, CLEARMetrics
            ... )
            >>>
            >>> seq1 = SequenceResult(
            ...     sequence="MOT17-02",
            ...     CLEAR=CLEARMetrics(
            ...         MOTA=0.748, MOTP=0.810, MODA=0.780,
            ...         CLR_Re=0.820, CLR_Pr=0.910, MTR=0.60,
            ...         PTR=0.28, MLR=0.12, sMOTA=0.735,
            ...         CLR_TP=120, CLR_FN=28, CLR_FP=12,
            ...         IDSW=37, MT=50, PT=24, ML=10, Frag=11,
            ...     ),
            ... )
            >>> seq2 = SequenceResult(
            ...     sequence="MOT17-04",
            ...     CLEAR=CLEARMetrics(
            ...         MOTA=0.761, MOTP=0.815, MODA=0.795,
            ...         CLR_Re=0.828, CLR_Pr=0.920, MTR=0.64,
            ...         PTR=0.26, MLR=0.10, sMOTA=0.748,
            ...         CLR_TP=126, CLR_FN=24, CLR_FP=10,
            ...         IDSW=45, MT=54, PT=22, ML=8, Frag=13,
            ...     ),
            ... )
            >>> aggregate = SequenceResult(
            ...     sequence="COMBINED",
            ...     CLEAR=CLEARMetrics(
            ...         MOTA=0.755, MOTP=0.813, MODA=0.788,
            ...         CLR_Re=0.824, CLR_Pr=0.915, MTR=0.62,
            ...         PTR=0.27, MLR=0.11, sMOTA=0.742,
            ...         CLR_TP=246, CLR_FN=52, CLR_FP=22,
            ...         IDSW=82, MT=104, PT=46, ML=18, Frag=24,
            ...     ),
            ... )
            >>> bench = BenchmarkResult(
            ...     sequences={"MOT17-02": seq1, "MOT17-04": seq2},
            ...     aggregate=aggregate,
            ... )
            >>>
            >>> print(bench.table(columns=["MOTA", "IDSW"]))
            # Sequence                        MOTA  IDSW
            # ------------------------------------------
            # MOT17-02                      74.800    37
            # MOT17-04                      76.100    45
            # ------------------------------------------
            # COMBINED                      75.500    82
            ```
        """
        if columns is None:
            columns = _get_available_columns(
                has_clear=self.aggregate.CLEAR is not None,
                has_hota=self.aggregate.HOTA is not None,
                has_identity=self.aggregate.Identity is not None,
            )

        return _format_benchmark_table(self.sequences, self.aggregate, columns)

    def save(self, path: str | Path) -> None:
        """Save results to a JSON file.

        Args:
            path: Path to save the JSON file.

        Examples:
            ```pycon
            >>> import tempfile
            >>> from trackers.eval import BenchmarkResult, SequenceResult
            >>>
            >>> bench = BenchmarkResult(
            ...     sequences={"seq1": SequenceResult(sequence="seq1")},
            ...     aggregate=SequenceResult(sequence="COMBINED"),
            ... )
            >>>
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     path = f"{tmpdir}/results.json"
            ...     bench.save(path)
            ...     print("Saved successfully")
            # Saved successfully
            ```
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
            `BenchmarkResult` instance.

        Raises:
            FileNotFoundError: If the file does not exist.

        Examples:
            ```pycon
            >>> import tempfile
            >>> from trackers.eval import BenchmarkResult, SequenceResult
            >>>
            >>> bench = BenchmarkResult(
            ...     sequences={"seq1": SequenceResult(sequence="seq1")},
            ...     aggregate=SequenceResult(sequence="COMBINED"),
            ... )
            >>>
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     path = f"{tmpdir}/results.json"
            ...     bench.save(path)
            ...     loaded = BenchmarkResult.load(path)
            ...     print(loaded.aggregate.sequence)
            # COMBINED
            ```
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        data = json.loads(path.read_text())
        return cls.from_dict(data)


def _get_available_columns(
    has_clear: bool = False, has_hota: bool = False, has_identity: bool = False
) -> list[str]:
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


def _get_metrics_dict(result: SequenceResult, col: str) -> float | int:
    """Get metric value from a SequenceResult.

    Args:
        result: The sequence result.
        col: Column name.

    Returns:
        The metric value.
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
