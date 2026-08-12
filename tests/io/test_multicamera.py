# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for AI City 2024 multi-camera file preparation semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import trackers.io.multicamera as multicamera
from trackers.io.multicamera import (
    _COL_XWORLD,
    _COL_YWORLD,
    _euclidean_similarity,
    _prepare_multicamera_sequence,
    _truncate_multicamera_rows,
    load_multicamera_file,
    load_scene_camera_map,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "multicamera"


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


class TestLoadMulticameraFile:
    """Parser contract for AI City 2024 9-column text files."""

    def test_column_index_to_field_mapping(self, tmp_path: Path) -> None:
        """Distinguishable columns map to the documented field indices."""
        # Values chosen so an xworld/yworld swap changes similarity.
        # frame_id=0 becomes frame 1 after the protocol offset.
        path = _write(
            tmp_path / "cols.txt",
            "1 2 0 10 20 30 40 100.0 200.0\n",
        )
        data = load_multicamera_file(path)
        assert data.shape == (1, 9)
        assert data[0, 0] == 1  # camera_id
        assert data[0, 1] == 2  # obj_id
        assert data[0, 2] == 0  # frame_id
        assert data[0, 3] == 10  # x
        assert data[0, 4] == 20  # y
        assert data[0, 5] == 30  # w
        assert data[0, 6] == 40  # h
        assert data[0, _COL_XWORLD] == 100.0
        assert data[0, _COL_YWORLD] == 200.0

        gt = data
        pred = data.copy()
        pred[0, _COL_XWORLD] = 101.0  # differ in x only
        pred[0, _COL_YWORLD] = 200.0
        seq = _prepare_multicamera_sequence(gt, pred)
        assert seq.similarity_scores[0][0, 0] == pytest.approx(0.5)

        swapped_pred = pred.copy()
        swapped_pred[0, _COL_XWORLD], swapped_pred[0, _COL_YWORLD] = (
            swapped_pred[0, _COL_YWORLD],
            swapped_pred[0, _COL_XWORLD],
        )
        swapped_seq = _prepare_multicamera_sequence(gt, swapped_pred)
        assert swapped_seq.similarity_scores[0][0, 0] != seq.similarity_scores[0][0, 0]

    def test_retained_row_budget_rejects_limit_plus_one_before_concatenation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(multicamera, "_MAX_DETECTIONS_PER_SEQUENCE", 1)
        path = _write(
            tmp_path / "too-many.txt",
            "1 1 0 0 0 1 1 0 0\n1 2 0 0 0 1 1 1 0\n",
        )

        with pytest.raises(ValueError, match="MAX_DETECTIONS_PER_SEQUENCE"):
            load_multicamera_file(path)

    def test_negative_identifiers_raise(self, tmp_path: Path) -> None:
        """Negative camera_id / obj_id / frame_id raise for the whole file."""
        for bad in (
            "-1 1 0 0 0 1 1 0 0\n",
            "1 -1 0 0 0 1 1 0 0\n",
            "1 1 -1 0 0 1 1 0 0\n",
        ):
            path = _write(tmp_path / "neg.txt", bad)
            with pytest.raises(ValueError, match="Invalid negative values"):
                load_multicamera_file(path)

    @pytest.mark.parametrize(
        ("column", "token"),
        [
            pytest.param(0, "1.0", id="camera-fractional-lexeme"),
            pytest.param(1, "1e0", id="object-scientific-lexeme"),
            pytest.param(2, "1E0", id="frame-scientific-lexeme"),
        ],
    )
    def test_identifier_lexemes_require_decimal_digits(
        self,
        tmp_path: Path,
        column: int,
        token: str,
    ) -> None:
        """Integral-valued float syntax is rejected before numeric conversion."""
        fields = ["1", "1", "0", "0", "0", "1", "1", "0", "0"]
        fields[column] = token
        path = _write(tmp_path / "lexical-id.txt", " ".join(fields) + "\n")

        with pytest.raises(ValueError, match=r"decimal|identifier|integer"):
            load_multicamera_file(path)

    def test_max_float_safe_identifier_is_preserved_exactly(self, tmp_path: Path) -> None:
        """The largest exactly representable integer ID remains unchanged."""
        max_safe_integer = 2**53 - 1
        path = _write(tmp_path / "max-safe.txt", f"1 {max_safe_integer} 0 0 0 1 1 0 0\n")

        data = load_multicamera_file(path)

        assert int(data[0, 1]) == max_safe_integer

    @pytest.mark.parametrize(
        "identifier",
        [
            pytest.param(2**53, id="first-float-unsafe-integer"),
            pytest.param(2**63 - 1, id="int64-maximum"),
        ],
    )
    def test_float_unsafe_identifier_rejected_before_storage(
        self,
        tmp_path: Path,
        identifier: int,
    ) -> None:
        """IDs that float64 cannot preserve exactly are rejected before storage."""
        path = _write(tmp_path / "unsafe-id.txt", f"1 {identifier} 0 0 0 1 1 0 0\n")

        with pytest.raises(ValueError, match=r"safe|2\*\*53|exact"):
            load_multicamera_file(path)

    @pytest.mark.parametrize(
        "camera_ids",
        [
            pytest.param([True], id="boolean"),
            pytest.param([1.5], id="fractional"),
            pytest.param(["1e0"], id="scientific-string"),
        ],
    )
    def test_camera_filter_rejects_ambiguous_identifiers(
        self,
        tmp_path: Path,
        camera_ids: list[object],
    ) -> None:
        """Camera filters reject coercions that can silently select another camera."""
        path = _write(tmp_path / "camera.txt", "1 1 0 0 0 1 1 0 0\n")

        with pytest.raises((TypeError, ValueError)):
            load_multicamera_file(path, camera_ids=camera_ids)  # type: ignore[arg-type]

    def test_negative_world_coordinates_accepted(self, tmp_path: Path) -> None:
        """World coordinates may be negative."""
        path = _write(tmp_path / "neg_world.txt", "1 1 0 0 0 1 1 -20.5 -3.25\n")
        data = load_multicamera_file(path)
        assert data[0, _COL_XWORLD] == pytest.approx(-20.5)
        assert data[0, _COL_YWORLD] == pytest.approx(-3.25)

    def test_unsupported_file_format(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "ok.txt", "1 1 0 0 0 1 1 0 0\n")
        with pytest.raises(ValueError, match=r"Unsupported file_format"):
            load_multicamera_file(path, file_format="aicity-2025")

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            ("1 1 0 0 0 1 1 0\n", "Expected 9 columns"),
            ("1 1 0 0 0 1 1 0 0 1\n", "Expected 9 columns"),
            ("1 1 0 0 0 1 1 abc 0\n", "Non-numeric"),
            ("1 1 0 0 0 1 1 nan 0\n", "Non-finite"),
            ("1 1 0 0 0 1 1 inf 0\n", "Non-finite"),
            ("camera_id obj_id frame_id x y w h xworld yworld\n", "Header or comment"),
            ("# comment\n1 1 0 0 0 1 1 0 0\n", "Header or comment"),
            ("1 1 0 0 0 1 1 0 0\n\n1 1 1 0 0 1 1 0 0\n", "Blank line"),
            ("1 1 1.5 0 0 1 1 0 0\n", "Non-integer frame_id"),
        ],
    )
    def test_malformed_input(self, tmp_path: Path, content: str, match: str) -> None:
        path = _write(tmp_path / "bad.txt", content)
        with pytest.raises(ValueError, match=match):
            load_multicamera_file(path)

    def test_empty_and_whitespace_only_raise(self, tmp_path: Path) -> None:
        for content in ("", "   \n  \n"):
            path = _write(tmp_path / "empty.txt", content)
            with pytest.raises(ValueError, match="empty"):
                load_multicamera_file(path)

    def test_empty_after_camera_filter_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "cam.txt", "1 1 0 0 0 1 1 0 0\n")
        with pytest.raises(ValueError, match="empty after camera filtering"):
            load_multicamera_file(path, camera_ids=[99])


class TestPrepareSemantics:
    """Numbered AI City 2024 preparation semantics."""

    def test_camera_filter_then_dedup_composition(self, tmp_path: Path) -> None:
        """Excluded-camera first duplicate must not win after filtering."""
        gt_path = _write(
            tmp_path / "comp.txt",
            "99 10 0 0 0 1 1 9.0 9.0\n1 10 0 0 0 1 1 0.0 0.0\n",
        )
        pred_path = _write(tmp_path / "counterpart.txt", "1 10 0 0 0 1 1 0.0 0.0\n")
        seq = _prepare_multicamera_sequence(
            load_multicamera_file(gt_path),
            load_multicamera_file(pred_path),
            camera_ids=[1],
        )
        assert seq.num_gt_dets == 1
        assert seq.similarity_scores[0][0, 0] == pytest.approx(1.0)

    def test_half_to_even_rounding_reaches_similarity(self, tmp_path: Path) -> None:
        """Rounding at an exact ``x.xxx5`` tie affects the similarity input."""
        # 0.1235 -> 0.124 (half-to-even toward even digit 4? 0.124; np.round)
        # GT at 0.0; pred at 0.1235 rounds to 0.124 → sim = 1 - 0.124/2
        gt = _write(tmp_path / "gt.txt", "1 1 0 0 0 1 1 0.0 0.0\n")
        pred = _write(tmp_path / "pred.txt", "1 1 0 0 0 1 1 0.1235 0.0\n")
        seq = _prepare_multicamera_sequence(
            load_multicamera_file(gt),
            load_multicamera_file(pred),
        )
        expected = max(0.0, 1.0 - 0.124 / 2.0)
        assert seq.similarity_scores[0][0, 0] == pytest.approx(expected)

        # Confirm banker's rounding itself.
        assert float(np.round(0.1235, 3)) == pytest.approx(0.124)
        assert float(np.round(1.2345, 3)) == pytest.approx(1.234)

    def test_dedup_keep_first_selects_world_point(self, tmp_path: Path) -> None:
        gt_path = _write(
            tmp_path / "dedup.txt",
            "1 10 0 0 0 1 1 0.0 0.0\n2 10 0 0 0 1 1 5.0 0.0\n",
        )
        pred_path = _write(tmp_path / "fixed.txt", "1 10 0 0 0 1 1 0.0 0.0\n")
        seq = _prepare_multicamera_sequence(
            load_multicamera_file(gt_path),
            load_multicamera_file(pred_path),
        )
        assert seq.num_gt_dets == 1
        assert seq.similarity_scores[0][0, 0] == pytest.approx(1.0)

    def test_frame_zero_becomes_frame_one(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "f0.txt", "1 1 0 0 0 1 1 0 0\n")
        data = load_multicamera_file(path)
        seq = _prepare_multicamera_sequence(data, data)
        assert seq.num_frames == 1
        assert len(seq.gt_ids[0]) == 1  # emitted under frame index 0 == file frame 1

    def test_interior_gap_frames_are_empty_entries(self, tmp_path: Path) -> None:
        """Rows at frames 1 and 5 (0-based 0 and 4) yield five entries, three empty."""
        gt = _write(
            tmp_path / "gap_gt.txt",
            "1 1 0 0 0 1 1 0 0\n1 1 4 0 0 1 1 0 0\n",
        )
        pred = _write(
            tmp_path / "gap_pred.txt",
            "1 1 0 0 0 1 1 0 0\n1 1 4 0 0 1 1 0 0\n",
        )
        seq = _prepare_multicamera_sequence(
            load_multicamera_file(gt),
            load_multicamera_file(pred),
        )
        assert seq.num_frames == 5
        assert len(seq.gt_ids) == 5
        nonempty = [i for i, ids in enumerate(seq.gt_ids) if len(ids) > 0]
        assert nonempty == [0, 4]
        assert seq.similarity_scores[1].shape == (0, 0)
        assert seq.similarity_scores[2].shape == (0, 0)
        assert seq.similarity_scores[3].shape == (0, 0)

    def test_sequence_length_driven_by_predictions(self, tmp_path: Path) -> None:
        gt = _write(tmp_path / "gt.txt", "1 1 0 0 0 1 1 0 0\n")
        pred = _write(
            tmp_path / "pred.txt",
            "1 1 0 0 0 1 1 0 0\n1 2 3 0 0 1 1 1 0\n",
        )
        seq = _prepare_multicamera_sequence(
            load_multicamera_file(gt),
            load_multicamera_file(pred),
        )
        assert seq.num_frames == 4
        assert seq.similarity_scores[3].shape == (0, 1)  # (N=0, M=1)

    def test_one_sided_similarity_shapes(self, tmp_path: Path) -> None:
        gt = _write(tmp_path / "gt.txt", "1 1 0 0 0 1 1 0 0\n")
        pred = _write(tmp_path / "pred.txt", "1 1 1 0 0 1 1 0 0\n")
        seq = _prepare_multicamera_sequence(
            load_multicamera_file(gt),
            load_multicamera_file(pred),
        )
        assert seq.similarity_scores[0].shape == (1, 0)
        assert seq.similarity_scores[1].shape == (0, 1)

    def test_row_conservation_no_preprocessing(self) -> None:
        """num_gt_dets equals rows surviving camera filter + dedup only."""
        gt = load_multicamera_file(FIXTURE_DIR / "scene_a_gt.txt", camera_ids=[1, 2])
        pred = load_multicamera_file(FIXTURE_DIR / "scene_a_pred.txt", camera_ids=[1, 2])
        seq = _prepare_multicamera_sequence(gt, pred)
        # scene_a_gt after cam filter: 7 rows minus cam99 already filtered at load
        # -> 6 rows; dedup collapses two (frame0,id10) rows -> 5
        assert seq.num_gt_dets == 5
        assert seq.num_tracker_dets == 5

    def test_unmapped_ids_both_directions(self, tmp_path: Path) -> None:
        gt = _write(tmp_path / "gt.txt", "1 1 0 0 0 1 1 0 0\n")
        pred = _write(tmp_path / "pred.txt", "1 99 0 0 0 1 1 0 0\n")
        seq = _prepare_multicamera_sequence(
            load_multicamera_file(gt),
            load_multicamera_file(pred),
        )
        assert seq.num_gt_ids == 1
        assert seq.num_tracker_ids == 1
        assert seq.similarity_scores[0].shape == (1, 1)

    def test_single_frame_scene(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "one.txt", "1 1 0 0 0 1 1 0 0\n")
        data = load_multicamera_file(path)
        seq = _prepare_multicamera_sequence(data, data)
        assert seq.num_frames == 1
        assert seq.similarity_scores[0][0, 0] == pytest.approx(1.0)

    def test_sparse_sequence_accepts_official_length_bound(self, tmp_path: Path) -> None:
        """A minimal sparse sequence may reach exactly 23,994 frames."""
        path = _write(tmp_path / "sparse.txt", "1 1 23993 0 0 1 1 0 0\n")
        data = load_multicamera_file(path)

        sequence = _prepare_multicamera_sequence(data, data)

        assert sequence.num_frames == 23_994

    def test_sparse_sequence_rejects_official_length_plus_one(self, tmp_path: Path) -> None:
        """A minimal sparse sequence is rejected at 23,995 frames."""
        path = _write(tmp_path / "sparse.txt", "1 1 23994 0 0 1 1 0 0\n")
        data = load_multicamera_file(path)

        with pytest.raises(ValueError, match=r"MAX_SEQUENCE_LENGTH|23.?994"):
            _prepare_multicamera_sequence(data, data)

    @pytest.mark.parametrize(
        "limit_name",
        [
            pytest.param("_MAX_FRAME_PAIR_COUNT", id="frame-pairs"),
            pytest.param("_MAX_IDENTITY_PAIR_COUNT", id="identity-pairs"),
        ],
    )
    def test_pair_allocation_guard_accepts_exact_limit(self, limit_name: str) -> None:
        """Allocation guard accepts an integer product exactly at its limit."""
        limit = getattr(multicamera, limit_name)
        validate = getattr(multicamera, "_validate_pair_allocation")

        validate(limit, 1, limit)

    @pytest.mark.parametrize(
        "limit_name",
        [
            pytest.param("_MAX_FRAME_PAIR_COUNT", id="frame-pairs"),
            pytest.param("_MAX_IDENTITY_PAIR_COUNT", id="identity-pairs"),
        ],
    )
    def test_pair_allocation_guard_rejects_limit_plus_one(self, limit_name: str) -> None:
        """Allocation guard rejects one pair beyond the limit without allocating."""
        limit = getattr(multicamera, limit_name)
        validate = getattr(multicamera, "_validate_pair_allocation")

        with pytest.raises(ValueError, match=limit_name.lstrip("_")):
            validate(limit + 1, 1, limit)


class TestEuclideanSimilarity:
    def test_distance_thresholds(self) -> None:
        p0 = np.array([[0.0, 0.0]])
        assert _euclidean_similarity(p0, np.array([[0.0, 0.0]]), 2.0)[0, 0] == 1.0
        assert _euclidean_similarity(p0, np.array([[1.0, 0.0]]), 2.0)[0, 0] == pytest.approx(0.5)
        assert _euclidean_similarity(p0, np.array([[2.0, 0.0]]), 2.0)[0, 0] == 0.0
        assert _euclidean_similarity(p0, np.array([[3.0, 0.0]]), 2.0)[0, 0] == 0.0

    def test_non_default_zero_distance(self) -> None:
        p0 = np.array([[0.0, 0.0]])
        assert _euclidean_similarity(p0, np.array([[1.0, 0.0]]), 1.0)[0, 0] == 0.0

    @pytest.mark.parametrize(
        "zero_distance",
        [
            pytest.param(0.0, id="zero"),
            pytest.param(-1.0, id="negative"),
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="positive-infinity"),
            pytest.param(float("-inf"), id="negative-infinity"),
        ],
    )
    def test_invalid_zero_distance_rejected(self, zero_distance: float) -> None:
        """Non-finite and non-positive distance scales are rejected."""
        with pytest.raises(ValueError, match="zero_distance"):
            _euclidean_similarity(np.zeros((1, 2)), np.zeros((1, 2)), zero_distance)


class TestSceneCameraMap:
    def test_load_scene_camera_map(self) -> None:
        mapping = load_scene_camera_map(FIXTURE_DIR / "scene_camera_map.json")
        assert mapping["scene_a"] == [1, 2]
        assert mapping["scene_b"] == [3, 4]

    def test_boolean_camera_id_rejected(self, tmp_path: Path) -> None:
        """JSON booleans are not valid camera integers."""
        path = _write(tmp_path / "bool-map.json", '[{"scene_name": "scene_a", "camera_ids": [true]}]\n')

        with pytest.raises(ValueError, match="camera_ids"):
            load_scene_camera_map(path)


def test_multicamera_public_surface_is_narrow() -> None:
    """Supported loaders are public while verification helpers stay private."""
    assert multicamera.load_multicamera_file is load_multicamera_file
    assert multicamera.load_scene_camera_map is load_scene_camera_map
    assert not hasattr(multicamera, "truncate_multicamera_rows")


class TestTruncateAndStreamingBounds:
    def test_truncate_keeps_unsorted_later_in_window_rows(self, tmp_path: Path) -> None:
        """Default scan must not early-exit: AI City files interleave cameras/frames."""
        path = _write(
            tmp_path / "unsorted.txt",
            "1 1 0 0 0 1 1 0 0\n1 1 5 0 0 1 1 0 0\n1 1 1 0 0 1 1 0 0\n",
        )
        rows = _truncate_multicamera_rows(path, max_frame=2)
        assert [int(row[2]) for row in rows] == [0, 1]

    def test_truncate_sorted_early_exit_skips_trailing_malformed(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "sorted.txt",
            "1 1 0 0 0 1 1 0 0\n1 1 1 0 0 1 1 0 0\n1 1 5 0 0 1 1 0 0\nNOT A VALID ROW\n",
        )
        rows = _truncate_multicamera_rows(path, max_frame=2, assume_sorted_frames=True)
        assert len(rows) == 2
        assert [int(row[2]) for row in rows] == [0, 1]

    def test_loader_source_avoids_whole_file_text_apis(self) -> None:
        """Regression guard: do not reintroduce read_text / readlines parsing."""
        source = (Path(__file__).resolve().parents[2] / "src" / "trackers" / "io" / "multicamera.py").read_text(
            encoding="utf-8"
        )
        assert "def _stream_parse_aicity_2024" in source
        assert "read_text(" not in source
        assert ".readlines(" not in source
        assert "np.loadtxt" not in source
        assert "np.genfromtxt" not in source


@pytest.mark.integration
def test_load_scene_061_peak_rss_bounded() -> None:
    """Pinned GT-vs-GT preparation has bounded memory above package imports."""
    import subprocess
    import sys

    from tests.conftest import MULTICAMERA_HF_REVISION, hf_fixture_file

    gt_path = hf_fixture_file(
        "MTMC_Tracking_2024/test/scene_061/ground_truth.txt",
        revision=MULTICAMERA_HF_REVISION,
    )
    file_mb = gt_path.stat().st_size / (1024 * 1024)
    assert 30.0 < file_mb < 45.0, file_mb

    script = "\n".join(
        [
            "import resource",
            "import sys",
            "from pathlib import Path",
            "",
            f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2] / 'src')!r})",
            "from trackers.eval import evaluate_multicamera_scene",
            "",
            "baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
            f"path = Path({str(gt_path)!r})",
            "result = evaluate_multicamera_scene(",
            '    scene="scene_061",',
            "    gt_path=path,",
            "    tracker_path=path,",
            "    camera_ids=list(range(535, 545)),",
            ")",
            "peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
            "# macOS reports bytes; Linux reports kilobytes.",
            'divisor = 1024 * 1024 if sys.platform == "darwin" else 1024',
            "baseline_mb = baseline_rss / divisor",
            "peak_mb = peak_rss / divisor",
            "print(",
            '    f"hota={result.HOTA.HOTA if result.HOTA else None} "',
            '    f"baseline_mb={baseline_mb:.3f} peak_mb={peak_mb:.3f} "',
            '    f"peak_delta_mb={max(0.0, peak_mb - baseline_mb):.3f}"',
            ")",
        ]
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    # Last stdout line carries the measurement (imports may warn on stderr).
    measurement = completed.stdout.strip().splitlines()[-1]
    peak_delta_mb = float(measurement.split("peak_delta_mb=")[1].split()[0])
    assert peak_delta_mb < 8.0 * file_mb, measurement
