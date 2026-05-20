# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trackers import ByteTrackTracker
from trackers.eval.results import (
    BenchmarkResult,
    CLEARMetrics,
    HOTAMetrics,
    IdentityMetrics,
    SequenceResult,
)
from trackers.tune.tuner import Tuner, _create_optuna_study, _default_trial_params, _extract_metric

optuna = pytest.importorskip("optuna")


_MOT_LINE = "1,-1,10,20,100,80,0.9,1\n"


def _make_benchmark_result(
    mota: float = 0.75,
    hota: float | None = None,
    idf1: float | None = None,
) -> BenchmarkResult:
    clear = CLEARMetrics(
        MOTA=mota,
        MOTP=0.8,
        MODA=0.76,
        CLR_Re=0.8,
        CLR_Pr=0.9,
        MTR=0.7,
        PTR=0.2,
        MLR=0.1,
        sMOTA=0.72,
        CLR_TP=100,
        CLR_FN=20,
        CLR_FP=10,
        IDSW=5,
        MT=7,
        PT=2,
        ML=1,
        Frag=3,
    )
    hota_metrics = (
        HOTAMetrics(
            HOTA=hota,
            DetA=0.7,
            AssA=0.65,
            DetRe=0.72,
            DetPr=0.85,
            AssRe=0.68,
            AssPr=0.9,
            LocA=0.78,
            OWTA=0.69,
            HOTA_TP=1000,
            HOTA_FN=300,
            HOTA_FP=200,
        )
        if hota is not None
        else None
    )
    identity_metrics = (
        IdentityMetrics(IDF1=idf1, IDR=0.7, IDP=0.8, IDTP=90, IDFN=15, IDFP=10) if idf1 is not None else None
    )
    return BenchmarkResult(
        sequences={},
        aggregate=SequenceResult(
            sequence="COMBINED",
            CLEAR=clear,
            HOTA=hota_metrics,
            Identity=identity_metrics,
        ),
    )


def _setup_dirs(tmp_path: Path) -> tuple[Path, Path]:
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "seq1.txt").write_text(_MOT_LINE)

    det_dir = tmp_path / "det"
    det_dir.mkdir()
    (det_dir / "seq1.txt").write_text(_MOT_LINE)

    return gt_dir, det_dir


@pytest.mark.parametrize(
    "metric,make_kwargs,expected",
    [
        ("MOTA", {"mota": 0.75}, 0.75),
        ("HOTA", {"hota": 0.62}, 0.62),
        ("IDF1", {"idf1": 0.71}, 0.71),
    ],
)
def test_extract_metric(metric: str, make_kwargs: dict, expected: float) -> None:
    result = _make_benchmark_result(**make_kwargs)
    assert _extract_metric(result, metric) == pytest.approx(expected)


@pytest.mark.parametrize("metric", ["NONEXISTENT", "HOTA"])
def test_extract_metric_raises(metric: str) -> None:
    # Default result has no HOTA/Identity families — both should raise
    result = _make_benchmark_result()
    with pytest.raises(ValueError, match=r"not found in BenchmarkResult\.aggregate"):
        _extract_metric(result, metric)


class TestDefaultTrialParams:
    def test_bytetrack_defaults_match_init(self) -> None:
        assert ByteTrackTracker.search_space is not None
        defaults = _default_trial_params(ByteTrackTracker, ByteTrackTracker.search_space)
        assert defaults["lost_track_buffer"] == 30
        assert defaults["track_activation_threshold"] == 0.7
        assert defaults["minimum_iou_threshold"] == 0.1
        assert defaults["high_conf_det_threshold"] == 0.6
        assert defaults["minimum_consecutive_frames"] == 2


class TestTunerFixedParams:
    def test_fixed_params_merged_into_best_result(self, tmp_path: Path) -> None:
        gt_dir, det_dir = _setup_dirs(tmp_path)

        with (
            patch(
                "trackers.tune.tuner.evaluate_mot_sequences",
                return_value=_make_benchmark_result(),
            ),
            patch("trackers.tune.tuner._run_tracker_on_detections"),
        ):
            tuner = Tuner(
                "bytetrack",
                gt_dir,
                det_dir,
                n_trials=1,
                enqueue_defaults=False,
                fixed_params={"frame_rate": 25.0},
            )
            best = tuner.run()

        assert best["frame_rate"] == 25.0

    def test_raises_when_enable_cmc_without_images(self, tmp_path: Path) -> None:
        gt_dir, det_dir = _setup_dirs(tmp_path)
        with pytest.raises(ValueError, match=r"images_dir"):
            Tuner(
                "botsort",
                gt_dir,
                det_dir,
                fixed_params={"enable_cmc": True},
            )

    def test_raises_on_unknown_fixed_param(self, tmp_path: Path) -> None:
        gt_dir, det_dir = _setup_dirs(tmp_path)
        with pytest.raises(ValueError, match=r"Unknown tracker parameter"):
            Tuner("bytetrack", gt_dir, det_dir, fixed_params={"not_a_param": 1})

    def test_fixed_params_excluded_from_tunable_search_space(self, tmp_path: Path) -> None:
        gt_dir, det_dir = _setup_dirs(tmp_path)
        tuner = Tuner(
            "bytetrack",
            gt_dir,
            det_dir,
            fixed_params={"lost_track_buffer": 42},
        )
        assert "lost_track_buffer" in tuner._search_space
        assert "lost_track_buffer" not in tuner._tunable_search_space
        assert tuner._fixed_params["lost_track_buffer"] == 42


class TestTunerInit:
    def test_raises_for_unknown_tracker(self, tmp_path: Path) -> None:
        gt_dir, det_dir = _setup_dirs(tmp_path)
        with pytest.raises(ValueError, match=r"not registered"):
            Tuner("nonexistent_tracker", gt_dir, det_dir)

    def test_raises_for_tracker_without_search_space(self, tmp_path: Path) -> None:
        from trackers.core.base import BaseTracker

        class _NoSearchSpaceTracker(BaseTracker):
            tracker_id = "_test_no_ss"
            search_space = None

            def update(self, detections):  # type: ignore[override]
                return detections

            def reset(self) -> None:
                pass

        gt_dir, det_dir = _setup_dirs(tmp_path)
        try:
            with pytest.raises(ValueError, match=r"does not define a search_space"):
                Tuner("_test_no_ss", gt_dir, det_dir)
        finally:
            BaseTracker._registry.pop("_test_no_ss", None)

    def test_raises_when_no_sequences_found(self, tmp_path: Path) -> None:
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        det_dir = tmp_path / "det"
        det_dir.mkdir()
        with pytest.raises(ValueError, match=r"No sequences found"):
            Tuner("bytetrack", gt_dir, det_dir)

    def test_valid_init_stores_attributes(self, tmp_path: Path) -> None:
        gt_dir, det_dir = _setup_dirs(tmp_path)
        tuner = Tuner("bytetrack", gt_dir, det_dir, n_trials=10)
        assert tuner._tracker_id == "bytetrack"
        assert tuner._objective_metric == "MOTA"
        assert tuner._metrics == ["CLEAR"]
        assert tuner._n_trials == 10
        assert tuner._sequences == ["seq1"]

    def test_seqmap_filters_sequences(self, tmp_path: Path) -> None:
        gt_dir, det_dir = _setup_dirs(tmp_path)
        (det_dir / "seq2.txt").write_text(_MOT_LINE)
        seqmap = tmp_path / "seqmap.txt"
        seqmap.write_text("seq1\n")
        tuner = Tuner("bytetrack", gt_dir, det_dir, seqmap=seqmap)
        assert tuner._sequences == ["seq1"]

    @pytest.mark.parametrize(
        "objective,initial_metrics,expected_metrics",
        [
            ("HOTA", ["CLEAR"], ["CLEAR", "HOTA"]),
            ("IDF1", ["CLEAR"], ["CLEAR", "Identity"]),
            ("MOTA", ["CLEAR"], ["CLEAR"]),
            ("HOTA", ["CLEAR", "HOTA"], ["CLEAR", "HOTA"]),
        ],
    )
    def test_auto_adds_required_metric_family(
        self,
        tmp_path: Path,
        objective: str,
        initial_metrics: list[str],
        expected_metrics: list[str],
    ) -> None:
        """Tuner auto-adds the metric family required by the objective."""
        gt_dir, det_dir = _setup_dirs(tmp_path)
        tuner = Tuner("bytetrack", gt_dir, det_dir, metrics=initial_metrics, objective=objective)
        assert tuner._metrics == expected_metrics

    def test_objective_normalized_to_uppercase(self, tmp_path: Path) -> None:
        """Objective string is normalized to uppercase regardless of input case."""
        gt_dir, det_dir = _setup_dirs(tmp_path)
        tuner = Tuner("bytetrack", gt_dir, det_dir, objective="mota")
        assert tuner._objective_metric == "MOTA"


class TestTunerSeed:
    def test_create_optuna_study_uses_seeded_sampler(self) -> None:
        with patch.object(optuna, "create_study", wraps=optuna.create_study) as mock_create:
            study = _create_optuna_study(optuna, "bytetrack", 42)
        sampler = mock_create.call_args.kwargs["sampler"]
        assert isinstance(sampler, optuna.samplers.TPESampler)
        assert isinstance(study.sampler, optuna.samplers.TPESampler)

    def test_create_optuna_study_without_seed_uses_default_sampler(self) -> None:
        with patch.object(optuna, "create_study", wraps=optuna.create_study) as mock_create:
            study = _create_optuna_study(optuna, "bytetrack", None)
        assert "sampler" not in mock_create.call_args.kwargs
        assert study.sampler is not None

    def test_tuner_passes_seed_to_study(self, tmp_path: Path) -> None:
        gt_dir, det_dir = _setup_dirs(tmp_path)
        mock_study = MagicMock()
        with (
            patch("trackers.tune.tuner._create_optuna_study", return_value=mock_study) as mock_create,
            patch(
                "trackers.tune.tuner.evaluate_mot_sequences",
                return_value=_make_benchmark_result(),
            ),
            patch("trackers.tune.tuner._run_tracker_on_detections"),
        ):
            tuner = Tuner("bytetrack", gt_dir, det_dir, n_trials=1, seed=7, enqueue_defaults=False)
            tuner.run()
        mock_create.assert_called_once_with(tuner._optuna, "bytetrack", 7)

    def test_same_seed_reproduces_sampled_trials(self, tmp_path: Path) -> None:
        gt_dir, det_dir = _setup_dirs(tmp_path)
        trial_params_runs: list[list[dict]] = []

        for _ in range(2):
            with (
                patch(
                    "trackers.tune.tuner.evaluate_mot_sequences",
                    return_value=_make_benchmark_result(),
                ),
                patch("trackers.tune.tuner._run_tracker_on_detections"),
            ):
                tuner = Tuner(
                    "bytetrack",
                    gt_dir,
                    det_dir,
                    n_trials=4,
                    seed=42,
                    enqueue_defaults=False,
                )
                tuner.run()
            assert tuner.study is not None
            trial_params_runs.append([dict(t.params) for t in tuner.study.trials])

        assert trial_params_runs[0] == trial_params_runs[1]


class TestTunerRun:
    def test_run_enqueues_defaults_as_first_trial(self, tmp_path: Path) -> None:
        assert ByteTrackTracker.search_space is not None
        expected_defaults = _default_trial_params(ByteTrackTracker, ByteTrackTracker.search_space)
        gt_dir, det_dir = _setup_dirs(tmp_path)

        with (
            patch(
                "trackers.tune.tuner.evaluate_mot_sequences",
                return_value=_make_benchmark_result(),
            ),
            patch("trackers.tune.tuner._run_tracker_on_detections"),
        ):
            tuner = Tuner("bytetrack", gt_dir, det_dir, n_trials=2)
            tuner.run()

        assert tuner.study is not None
        assert dict(tuner.study.trials[0].params) == expected_defaults
        assert len(tuner.study.trials) == 2

    def test_run_baseline_enqueue_excludes_fixed_params(self, tmp_path: Path) -> None:
        """Baseline enqueue only includes tunable search_space keys, not fixed_params."""
        gt_dir, det_dir = _setup_dirs(tmp_path)
        enqueue_mock = patch.object(optuna.Study, "enqueue_trial")

        with (
            patch(
                "trackers.tune.tuner.evaluate_mot_sequences",
                return_value=_make_benchmark_result(),
            ),
            patch("trackers.tune.tuner._run_tracker_on_detections"),
            enqueue_mock as mock_enqueue,
        ):
            tuner = Tuner(
                "bytetrack",
                gt_dir,
                det_dir,
                n_trials=1,
                fixed_params={"frame_rate": 20.0, "lost_track_buffer": 42},
            )
            tuner.run()

        enqueued = mock_enqueue.call_args[0][0]
        assert "frame_rate" not in enqueued  # not in search_space
        assert "lost_track_buffer" not in enqueued  # fixed → excluded from sampling
        assert enqueued["track_activation_threshold"] == 0.7

    def test_run_without_enqueue_defaults_skips_default_trial(self, tmp_path: Path) -> None:
        gt_dir, det_dir = _setup_dirs(tmp_path)

        with (
            patch(
                "trackers.tune.tuner.evaluate_mot_sequences",
                return_value=_make_benchmark_result(),
            ),
            patch("trackers.tune.tuner._run_tracker_on_detections"),
        ):
            tuner = Tuner("bytetrack", gt_dir, det_dir, n_trials=1, enqueue_defaults=False)
            assert tuner._default_trial_params is None
            tuner.run()

        assert tuner.study is not None
        assert len(tuner.study.trials) == 1

    def test_run_returns_dict_with_search_space_keys(self, tmp_path: Path) -> None:
        from trackers import ByteTrackTracker

        assert ByteTrackTracker.search_space is not None
        expected_keys = set(ByteTrackTracker.search_space.keys())
        gt_dir, det_dir = _setup_dirs(tmp_path)

        with (
            patch(
                "trackers.tune.tuner.evaluate_mot_sequences",
                return_value=_make_benchmark_result(),
            ),
            patch("trackers.tune.tuner._run_tracker_on_detections"),
        ):
            tuner = Tuner("bytetrack", gt_dir, det_dir, n_trials=2)
            best = tuner.run()

        assert isinstance(best, dict)
        assert set(best.keys()) == expected_keys

    def test_run_calls_tracker_reset_per_sequence(self, tmp_path: Path) -> None:
        """reset() must be called once per sequence per trial."""
        from trackers import SORTTracker

        reset_calls: list[int] = []
        gt_dir, det_dir = _setup_dirs(tmp_path)
        (det_dir / "seq2.txt").write_text(_MOT_LINE)  # two sequences → two resets
        (gt_dir / "seq2.txt").write_text(_MOT_LINE)

        original_reset = SORTTracker.reset

        def _counting_reset(self_tracker: SORTTracker) -> None:
            reset_calls.append(1)
            original_reset(self_tracker)

        with (
            patch(
                "trackers.tune.tuner.evaluate_mot_sequences",
                return_value=_make_benchmark_result(),
            ),
            patch("trackers.tune.tuner._run_tracker_on_detections"),
            patch.object(SORTTracker, "reset", _counting_reset),
        ):
            tuner = Tuner("sort", gt_dir, det_dir, n_trials=1)
            tuner.run()

        assert len(reset_calls) == 2  # 1 trial * 2 sequences


class TestLoadMotSequenceFrame:
    def test_raises_when_frame_image_missing(self, tmp_path: Path) -> None:
        from trackers.tune.tuner import _load_mot_sequence_frame

        images_dir = tmp_path / "images"
        (images_dir / "seq_a" / "img1").mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match=r"seq_a.*frame 1"):
            _load_mot_sequence_frame(images_dir, "seq_a", 1)

    def test_loads_frame_with_supported_extension(self, tmp_path: Path) -> None:
        import cv2
        import numpy as np

        from trackers.tune.tuner import _load_mot_sequence_frame

        images_dir = tmp_path / "images"
        frame_dir = images_dir / "seq_a" / "img1"
        frame_dir.mkdir(parents=True)
        image_path = frame_dir / "000001.jpg"
        cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))

        frame = _load_mot_sequence_frame(images_dir, "seq_a", 1)

        assert frame.shape == (8, 8, 3)

    def test_loads_eight_digit_dancetrack_frame(self, tmp_path: Path) -> None:
        import cv2
        import numpy as np

        from trackers.tune.tuner import _load_mot_sequence_frame

        images_dir = tmp_path / "images"
        frame_dir = images_dir / "dancetrack0001" / "img1"
        frame_dir.mkdir(parents=True)
        image_path = frame_dir / "00000001.jpg"
        cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))

        frame = _load_mot_sequence_frame(images_dir, "dancetrack0001", 1)

        assert frame.shape == (8, 8, 3)


class TestRunTrackerOnDetections:
    """End-to-end tests for the _run_tracker_on_detections helper."""

    def test_creates_valid_mot_output_file(self, tmp_path: Path) -> None:
        """Output file exists and each line is valid 10-column MOT format."""
        from trackers import ByteTrackTracker
        from trackers.tune.tuner import _run_tracker_on_detections

        # Two detections in frame 1, one in frame 3 — frame 2 is intentionally
        # absent so the code path feeding sv.Detections.empty() is exercised.
        det_content = (
            "1,-1,10,20,100,80,0.90,-1,-1,-1\n1,-1,200,150,80,60,0.85,-1,-1,-1\n3,-1,15,25,100,80,0.88,-1,-1,-1\n"
        )
        det_path = tmp_path / "seq.txt"
        pred_path = tmp_path / "pred.txt"
        det_path.write_text(det_content)

        tracker = ByteTrackTracker()
        _run_tracker_on_detections(tracker, det_path, pred_path)

        assert pred_path.exists(), "prediction file must be created"
        lines = [ln for ln in pred_path.read_text().splitlines() if ln.strip()]
        assert len(lines) > 0, "prediction file must contain at least one tracked box"
        for line in lines:
            fields = line.split(",")
            assert len(fields) >= 7, f"invalid MOT line: {line!r}"
            frame_idx = int(fields[0])
            assert 1 <= frame_idx <= 3, f"unexpected frame index: {frame_idx}"

    def test_empty_frames_do_not_crash(self, tmp_path: Path) -> None:
        """Frames with no detections are fed as sv.Detections.empty() without error."""
        from trackers import SORTTracker
        from trackers.tune.tuner import _run_tracker_on_detections

        # Only frame 1 has a detection; frames 2-5 are absent → empty detections
        det_path = tmp_path / "sparse.txt"
        pred_path = tmp_path / "pred.txt"
        det_path.write_text("5,-1,50,50,60,60,0.95,-1,-1,-1\n")

        tracker = SORTTracker()
        _run_tracker_on_detections(tracker, det_path, pred_path)

        assert pred_path.exists()

    def test_raises_when_images_dir_set_but_frame_missing(self, tmp_path: Path) -> None:
        from trackers import SORTTracker
        from trackers.tune.tuner import _run_tracker_on_detections

        det_path = tmp_path / "sparse.txt"
        pred_path = tmp_path / "pred.txt"
        det_path.write_text("1,-1,10,20,100,80,0.90,-1,-1,-1\n")
        images_dir = tmp_path / "images"
        (images_dir / "seq" / "img1").mkdir(parents=True)

        tracker = SORTTracker()
        with pytest.raises(FileNotFoundError, match=r"seq.*frame 1"):
            _run_tracker_on_detections(
                tracker,
                det_path,
                pred_path,
                images_dir=images_dir,
                seq_name="seq",
            )
