# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the ``trackers benchmark mcbyte`` benchmark subcommand."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import get_args

import cv2
import numpy as np
import pytest
import supervision as sv

from trackers.cli.__main__ import _CLIParser, _normalise_option
from trackers.cli._detections import (
    DetectionRecord,
    build_detections,
    read_detection_file,
)
from trackers.cli.benchmark.mcbyte import (
    DATASETS,
    MOT17_EXISTING,
    MOT17_MISSING,
    MOT17_SUFFIXES,
    DatasetConfig,
    DatasetName,
    DatasetPaths,
    _read_sequence_frame_rate,
    _runtime_error,
    _unknown_datasets_error,
    benchmark_command,
    image_directory,
    prepare_mot17_submission,
    resolve_datasets,
    run_dataset,
    run_sequence,
    sequence_name,
)


@pytest.fixture
def benchmark_parser() -> _CLIParser:
    """Parser exposing the benchmark options, matching the script's own wiring."""
    parser = _CLIParser(exit_on_error=False)
    parser.add_function_arguments(benchmark_command)
    return parser


@pytest.fixture
def config_parser() -> _CLIParser:
    """Benchmark parser that also accepts ``--config``, as the real CLI does."""
    parser = _CLIParser(exit_on_error=False)
    parser.add_argument("--config", action="config")
    parser.add_function_arguments(benchmark_command)
    return parser


class TestBenchmarkOptions:
    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            pytest.param([], 6, id="default"),
            pytest.param(["--cmc_downscale", "2"], 2, id="underscore_override"),
            pytest.param(["--cmc-downscale", "2"], 2, id="hyphen_override"),
            pytest.param(["--cmc_downscale=2"], 2, id="inline_value"),
        ],
    )
    def test_cmc_downscale(self, benchmark_parser: _CLIParser, arguments: list[str], expected: int) -> None:
        """The benchmark follows McByte's default and retains explicit overrides."""
        namespace = benchmark_parser.parse_args([_normalise_option(arg) for arg in arguments])

        assert namespace.cmc_downscale == expected

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            pytest.param([], True, id="enabled_by_default"),
            pytest.param(["--no_enable_cmc"], False, id="negated"),
            pytest.param(["--no-enable-cmc"], False, id="negated_hyphenated"),
            pytest.param(["--enable_cmc=false"], False, id="explicit_false"),
        ],
    )
    def test_enable_cmc(self, benchmark_parser: _CLIParser, arguments: list[str], expected: bool) -> None:
        """Camera-motion compensation is on unless a negative spelling turns it off."""
        namespace = benchmark_parser.parse_args([_normalise_option(arg) for arg in arguments])

        assert namespace.enable_cmc is expected

    def test_dataset_accepts_a_list(self, benchmark_parser: _CLIParser) -> None:
        """Datasets are selected as one list rather than a repeated option."""
        namespace = benchmark_parser.parse_args(["--dataset=[mot17,soccernet]"])

        assert namespace.dataset == ["mot17", "soccernet"]

    def test_unknown_dataset_is_rejected(self, benchmark_parser: _CLIParser) -> None:
        """An unregistered dataset name fails while parsing."""
        with pytest.raises(Exception, match="dataset"):
            benchmark_parser.parse_args(["--dataset=[nonexistent]"])

    def test_selectable_names_match_the_dataset_table(self) -> None:
        """No dataset can be configured without also being selectable."""
        assert set(get_args(DatasetName)) == set(DATASETS)


class TestRuntimeError:
    @pytest.mark.parametrize(
        ("device", "cmc_downscale", "expected"),
        [
            pytest.param("cpu", 6, "", id="usable"),
            pytest.param("cpu", 0, "cmc_downscale must be positive.", id="zero_downscale"),
            pytest.param("cpu", -1, "cmc_downscale must be positive.", id="negative_downscale"),
        ],
    )
    def test_reports_the_first_problem(self, device: str, cmc_downscale: int, expected: str) -> None:
        """Runtime arguments are validated before any run directory is created."""
        assert _runtime_error(device, cmc_downscale) == expected

    def test_invalid_arguments_exit_non_zero(self, capsys: pytest.CaptureFixture) -> None:
        """The command reports the problem on stderr instead of raising."""
        code = benchmark_command(device="cpu", cmc_downscale=0)

        assert code == 1
        assert "cmc_downscale must be positive." in capsys.readouterr().err

    def test_a_failed_dataset_exits_non_zero(self, tmp_path: Path) -> None:
        """A run that reaches the datasets but fails one of them still reports failure."""
        code = benchmark_command(dataset=["mot17"], device="cpu", output_root=tmp_path / "runs")

        assert code == 1


class TestDatasetRoots:
    def test_inline_json_supplies_both_roots(self, benchmark_parser: _CLIParser) -> None:
        """A dataset's roots are configurable on the command line as one JSON mapping."""
        namespace = benchmark_parser.parse_args(
            ['--dataset_roots={"mot17": {"detection_root": "/data/dets", "image_root": "/data/imgs"}}']
        )
        instantiated = benchmark_parser.instantiate_classes(namespace)

        assert instantiated.dataset_roots == {"mot17": DatasetPaths(Path("/data/dets"), Path("/data/imgs"))}

    def test_config_file_supplies_both_roots(self, config_parser: _CLIParser, tmp_path: Path) -> None:
        """The same mapping reaches the command from a ``--config`` file."""
        config = tmp_path / "run.yaml"
        config.write_text(
            "dataset: [mot17]\ndataset_roots:\n  mot17:\n    detection_root: /data/dets\n    image_root: /data/imgs\n"
        )

        namespace = config_parser.parse_args([f"--config={config}"])
        instantiated = config_parser.instantiate_classes(namespace)

        assert instantiated.dataset == ["mot17"]
        assert instantiated.dataset_roots == {"mot17": DatasetPaths(Path("/data/dets"), Path("/data/imgs"))}

    def test_both_roots_are_required_together(self, benchmark_parser: _CLIParser) -> None:
        """Half an entry is rejected while parsing rather than failing mid-run."""
        with pytest.raises(Exception, match=r"detection_root|image_root"):
            benchmark_parser.parse_args(['--dataset_roots={"mot17": {"detection_root": "/data/dets"}}'])

    def test_defaults_leave_both_roots_unset(self) -> None:
        """No root has a built-in value, so an unconfigured run cannot half-work."""
        assert all(config.detection_root is None and config.image_root is None for config in DATASETS.values())

    def test_merge_keeps_dataset_specific_behaviour(self) -> None:
        """Only the roots are user-supplied; parsing behaviour stays table-driven."""
        merged = resolve_datasets({"soccernet": DatasetPaths(Path("dets"), Path("imgs"))})

        assert merged["soccernet"].detection_root == Path("dets")
        assert merged["soccernet"].image_root == Path("imgs")
        assert merged["soccernet"].detection_format == "mot"
        assert merged["soccernet"].soccernet_filename is True
        assert merged["soccernet"].confidence_override == 1.0

    def test_merge_leaves_unnamed_datasets_alone(self) -> None:
        """Configuring one dataset does not disturb the others."""
        merged = resolve_datasets({"mot17": DatasetPaths(Path("dets"), Path("imgs"))})

        assert merged["dancetrack"].detection_root is None

    @pytest.mark.parametrize(
        ("dataset_roots", "expected"),
        [
            pytest.param(None, "", id="unsupplied"),
            pytest.param({}, "", id="empty"),
            pytest.param({"mot17": None}, "", id="known"),
            pytest.param({"mot18": None}, "Unknown --dataset_roots entry 'mot18'.", id="typo"),
        ],
    )
    def test_unknown_entries_are_reported(self, dataset_roots: dict | None, expected: str) -> None:
        """A key naming no dataset is caught rather than silently ignored."""
        assert _unknown_datasets_error(dataset_roots).startswith(expected)

    def test_unknown_entry_exits_non_zero(self, capsys: pytest.CaptureFixture) -> None:
        """A mistyped dataset name stops the run before a run directory is created."""
        code = benchmark_command(device="cpu", dataset_roots={"mot18": DatasetPaths(Path("a"), Path("b"))})

        assert code == 1
        assert "Unknown --dataset_roots entry 'mot18'." in capsys.readouterr().err

    def test_direct_call_with_unknown_key_raises_keyerror(self) -> None:
        """Bypassing the CLI-level check and calling ``resolve_datasets`` directly still fails loudly."""
        with pytest.raises(KeyError, match="mot18"):
            resolve_datasets({"mot18": DatasetPaths(Path("a"), Path("b"))})


class TestUnconfiguredDataset:
    def test_run_dataset_asks_for_the_missing_roots(self) -> None:
        """A dataset neither the source nor the run supplied roots for says so."""
        with pytest.raises(ValueError, match=r"Please configure DATASETS\['mot17'\]"):
            run_dataset(
                config=DATASETS["mot17"],
                output_dir=Path("unused"),
                device="cpu",
                enable_isolated_mask_matching=False,
                enable_cmc=True,
                cmc_method="sparseOptFlow",
                cmc_downscale=6,
                skip_existing=False,
                keep_partial_results=False,
                logger=logging.getLogger("test_mcbyte"),
            )

    def test_supplied_roots_reach_the_filesystem(self, tmp_path: Path) -> None:
        """Roots from the run are what the dataset is actually looked for under."""
        detection_root = tmp_path / "dets"
        image_root = tmp_path / "frames"
        detection_root.mkdir()
        image_root.mkdir()
        resolved = resolve_datasets({"mot17": DatasetPaths(detection_root, image_root)})

        with pytest.raises(FileNotFoundError, match=re.escape(str(detection_root))):
            run_dataset(
                config=resolved["mot17"],
                output_dir=tmp_path / "out",
                device="cpu",
                enable_isolated_mask_matching=False,
                enable_cmc=True,
                cmc_method="sparseOptFlow",
                cmc_downscale=6,
                skip_existing=False,
                keep_partial_results=False,
                logger=logging.getLogger("test_mcbyte"),
            )

    def test_the_error_names_the_command_line_route(self) -> None:
        """The message stays actionable now that editing the source is not the only fix."""
        with pytest.raises(ValueError, match="--dataset_roots"):
            run_dataset(
                config=DATASETS["mot17"],
                output_dir=Path("unused"),
                device="cpu",
                enable_isolated_mask_matching=False,
                enable_cmc=True,
                cmc_method="sparseOptFlow",
                cmc_downscale=6,
                skip_existing=False,
                keep_partial_results=False,
                logger=logging.getLogger("test_mcbyte"),
            )


class TestReadDetectionFile:
    def test_parses_xyxy_format(self, tmp_path: Path) -> None:
        """The XYXY branch reads frame, box corners and confidence directly."""
        detection_file = tmp_path / "seq.txt"
        detection_file.write_text("1,10,20,30,40,0.9\n")
        grouped = read_detection_file(detection_file, "xyxy")

        record = grouped[1][0]
        np.testing.assert_allclose(record.xyxy, [10.0, 20.0, 30.0, 40.0])
        assert record.confidence == pytest.approx(0.9)

    def test_parses_mot_format(self, tmp_path: Path) -> None:
        """The MOT branch converts left/top/width/height into XYXY corners."""
        detection_file = tmp_path / "seq.txt"
        detection_file.write_text("1,-1,10,20,15,25,0.5,-1,-1,-1\n")
        grouped = read_detection_file(detection_file, "mot")

        record = grouped[1][0]
        np.testing.assert_allclose(record.xyxy, [10.0, 20.0, 25.0, 45.0])
        assert record.confidence == pytest.approx(0.5)

    def test_confidence_override_replaces_the_parsed_column(self, tmp_path: Path) -> None:
        """SoccerNet-style datasets can pin confidence instead of trusting column 7."""
        detection_file = tmp_path / "seq.txt"
        detection_file.write_text("1,-1,10,20,15,25,0.5,-1,-1,-1\n")
        grouped = read_detection_file(detection_file, "mot", confidence_override=1.0)

        assert grouped[1][0].confidence == 1.0

    def test_degenerate_boxes_are_dropped_silently(self, tmp_path: Path) -> None:
        """A box with zero or negative width/height is filtered, not raised."""
        detection_file = tmp_path / "seq.txt"
        detection_file.write_text("1,10,20,10,40,0.9\n1,10,20,30,40,0.8\n")
        grouped = read_detection_file(detection_file, "xyxy")

        assert len(grouped[1]) == 1
        assert grouped[1][0].confidence == pytest.approx(0.8)

    def test_non_positive_frame_number_raises(self, tmp_path: Path) -> None:
        """A zero or negative frame number is rejected rather than silently grouped."""
        detection_file = tmp_path / "seq.txt"
        detection_file.write_text("0,10,20,30,40,0.9\n")
        with pytest.raises(ValueError, match="Non-positive frame number"):
            read_detection_file(detection_file, "xyxy")

    def test_malformed_line_error_names_the_line_number(self, tmp_path: Path) -> None:
        """A parse failure points back at the offending line, not just the file."""
        detection_file = tmp_path / "seq.txt"
        detection_file.write_text("1,10,20,30,40,0.9\n1,not,a,number,20,0.5\n")
        with pytest.raises(ValueError, match="line 2"):
            read_detection_file(detection_file, "xyxy")


class TestBuildDetections:
    def test_empty_records_produce_empty_detections(self) -> None:
        """No parsed records still produce a valid, empty Detections object."""
        detections = build_detections([])

        assert len(detections) == 0

    def test_populated_records_produce_matching_arrays(self) -> None:
        """Parsed boxes and confidences reach the Detections object unchanged."""
        records = [
            DetectionRecord(xyxy=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), confidence=0.9),
            DetectionRecord(xyxy=np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32), confidence=0.5),
        ]

        detections = build_detections(records)

        assert detections.xyxy.shape == (2, 4)
        np.testing.assert_allclose(detections.xyxy, [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        np.testing.assert_allclose(detections.confidence, [0.9, 0.5])


class TestSequenceName:
    def test_soccernet_filename_splits_on_the_double_underscore(self) -> None:
        """SoccerNet sequence names are recovered before the ``__det`` suffix."""
        config = DatasetConfig(name="soccernet", detection_format="mot", soccernet_filename=True)

        assert sequence_name(Path("SNMOT-116__det.txt"), config) == "SNMOT-116"

    def test_default_uses_the_file_stem(self) -> None:
        """Non-SoccerNet datasets use the detection filename stem as-is."""
        config = DatasetConfig(name="mot17", detection_format="xyxy")

        assert sequence_name(Path("MOT17-01.txt"), config) == "MOT17-01"


class TestImageDirectory:
    def test_mot17_layout_appends_the_frcnn_suffix(self, tmp_path: Path) -> None:
        """MOT17 frame directories carry the historical ``-FRCNN`` suffix."""
        config = DatasetConfig(name="mot17", detection_format="xyxy", mot17_layout=True, image_root=tmp_path)

        assert image_directory("MOT17-01", config) == tmp_path / "MOT17-01-FRCNN" / "img1"

    def test_default_uses_the_sequence_name_unchanged(self, tmp_path: Path) -> None:
        """Other datasets' frame directories match the sequence name exactly."""
        config = DatasetConfig(name="dancetrack", detection_format="xyxy", image_root=tmp_path)

        assert image_directory("dancetrack0003", config) == tmp_path / "dancetrack0003" / "img1"

    def test_raises_without_a_configured_image_root(self) -> None:
        """A missing ``image_root`` fails with the ``--dataset_roots`` hint, not a bad path."""
        config = DatasetConfig(name="mot17", detection_format="xyxy")

        with pytest.raises(ValueError, match="--dataset_roots"):
            image_directory("MOT17-01", config)


class TestReadSequenceFrameRate:
    """``seqinfo.ini``, not the dataset-wide default, should decide a sequence's frame rate.

    Regression coverage for H-FRAME-RATE-NEVER-PROPAGATED: ``DatasetConfig.frame_rate`` is a single 30.0 fallback
    for every sequence in a dataset, but real sequences run at a different rate (verified locally: every cached
    SportsMOT-val sequence is 25 fps, every cached DanceTrack-val sequence is 20 fps).
    """

    def test_missing_seqinfo_falls_back_to_default(self, tmp_path: Path) -> None:
        """No ``seqinfo.ini`` next to the sequence: the dataset default is used verbatim."""
        image_dir = tmp_path / "seq" / "img1"
        image_dir.mkdir(parents=True)

        assert _read_sequence_frame_rate(image_dir, default=30.0) == 30.0

    def test_reads_the_real_frame_rate_from_seqinfo(self, tmp_path: Path) -> None:
        """A real ``seqinfo.ini`` (SportsMOT-val's own format) overrides the default."""
        seq_dir = tmp_path / "seq"
        image_dir = seq_dir / "img1"
        image_dir.mkdir(parents=True)
        (seq_dir / "seqinfo.ini").write_text(
            "[Sequence]\nname=v_0kUtTtmLaJA_c006\nimDir=img1\nframeRate=25\nseqLength=346\n"
        )

        assert _read_sequence_frame_rate(image_dir, default=30.0) == 25.0

    def test_missing_frame_rate_key_falls_back_to_default(self, tmp_path: Path) -> None:
        """A ``seqinfo.ini`` present but without ``frameRate`` still falls back."""
        seq_dir = tmp_path / "seq"
        image_dir = seq_dir / "img1"
        image_dir.mkdir(parents=True)
        (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=no-frame-rate\n")

        assert _read_sequence_frame_rate(image_dir, default=30.0) == 30.0

    def test_malformed_ini_falls_back_to_default(self, tmp_path: Path) -> None:
        """Unparsable INI content does not raise; it falls back like a missing file."""
        seq_dir = tmp_path / "seq"
        image_dir = seq_dir / "img1"
        image_dir.mkdir(parents=True)
        (seq_dir / "seqinfo.ini").write_text("not an ini file at all\n===\n")

        assert _read_sequence_frame_rate(image_dir, default=30.0) == 30.0

    @pytest.mark.parametrize(
        "frame_rate_value",
        ["0", "-25", "nan", "inf"],
    )
    def test_non_positive_or_non_finite_frame_rate_falls_back(self, tmp_path: Path, frame_rate_value: str) -> None:
        """A degenerate ``frameRate`` (zero, negative, NaN, inf) is rejected like a missing one."""
        seq_dir = tmp_path / "seq"
        image_dir = seq_dir / "img1"
        image_dir.mkdir(parents=True)
        (seq_dir / "seqinfo.ini").write_text(f"[Sequence]\nframeRate={frame_rate_value}\n")

        assert _read_sequence_frame_rate(image_dir, default=30.0) == 30.0

    def test_missing_seqinfo_does_not_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """No ``seqinfo.ini`` at all is unremarkable: no warning, unlike a present-but-bad file."""
        image_dir = tmp_path / "seq" / "img1"
        image_dir.mkdir(parents=True)

        with caplog.at_level(logging.WARNING, logger="test_mcbyte_seqinfo"):
            _read_sequence_frame_rate(
                image_dir, default=30.0, logger=logging.getLogger("test_mcbyte_seqinfo"), sequence="seq"
            )

        assert caplog.records == []

    def test_malformed_ini_warns_with_logger(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A ``seqinfo.ini`` present but unparsable warns, distinguishing it from a missing file."""
        seq_dir = tmp_path / "seq"
        image_dir = seq_dir / "img1"
        image_dir.mkdir(parents=True)
        (seq_dir / "seqinfo.ini").write_text("not an ini file at all\n===\n")

        with caplog.at_level(logging.WARNING, logger="test_mcbyte_seqinfo"):
            result = _read_sequence_frame_rate(
                image_dir, default=30.0, logger=logging.getLogger("test_mcbyte_seqinfo"), sequence="seq"
            )

        assert result == 30.0
        assert len(caplog.records) == 1
        assert "seq" in caplog.records[0].message
        assert "seqinfo.ini" in caplog.records[0].message

    def test_degenerate_frame_rate_warns_with_logger(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A ``seqinfo.ini`` with a non-finite/non-positive ``frameRate`` warns too."""
        seq_dir = tmp_path / "seq"
        image_dir = seq_dir / "img1"
        image_dir.mkdir(parents=True)
        (seq_dir / "seqinfo.ini").write_text("[Sequence]\nframeRate=nan\n")

        with caplog.at_level(logging.WARNING, logger="test_mcbyte_seqinfo"):
            result = _read_sequence_frame_rate(
                image_dir, default=30.0, logger=logging.getLogger("test_mcbyte_seqinfo"), sequence="seq"
            )

        assert result == 30.0
        assert len(caplog.records) == 1
        assert "nan" in caplog.records[0].message


class TestRunSequenceUsesRealFrameRate:
    """``run_sequence`` must build the tracker with the sequence's real frame rate, not always ``config.frame_rate``.

    ``create_tracker`` is monkeypatched to a stub that only records its ``frame_rate`` kwarg and returns a minimal fake
    tracker — the real ``McByteTracker`` needs SAM/Cutie, far too heavy for this wiring check. One real 1x1 frame and a
    one-line detection file are the only filesystem fixtures, matching the sequence layout ``run_sequence`` actually
    reads (``image_dir`` for frames, ``image_dir.parent / "seqinfo.ini"``).
    """

    class _FakeTracker:
        def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
            return sv.Detections.empty()

        def reset(self) -> None:
            pass

    def _make_sequence(self, tmp_path: Path) -> tuple[Path, Path]:
        """Write one detection file and one real frame; return (detection_file, image_dir)."""
        image_dir = tmp_path / "seq" / "img1"
        image_dir.mkdir(parents=True)
        cv2.imwrite(str(image_dir / "000001.jpg"), np.zeros((4, 4, 3), dtype=np.uint8))
        detection_file = tmp_path / "seq.txt"
        detection_file.write_text("1,10,20,30,40,0.9\n")
        return detection_file, image_dir

    def test_seqinfo_frame_rate_reaches_create_tracker(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a ``seqinfo.ini`` present, its ``frameRate`` — not the dataset default — reaches the tracker."""
        detection_file, image_dir = self._make_sequence(tmp_path)
        (image_dir.parent / "seqinfo.ini").write_text("[Sequence]\nframeRate=25\n")

        captured: dict[str, float] = {}

        def fake_create_tracker(
            *, frame_rate: float, **_kwargs: object
        ) -> TestRunSequenceUsesRealFrameRate._FakeTracker:
            captured["frame_rate"] = frame_rate
            return TestRunSequenceUsesRealFrameRate._FakeTracker()

        monkeypatch.setattr("trackers.cli.benchmark.mcbyte.create_tracker", fake_create_tracker)

        config = DatasetConfig(name="sportsmot", detection_format="xyxy", frame_rate=30.0)
        run_sequence(
            sequence="seq",
            detection_file=detection_file,
            image_dir=image_dir,
            output_file=tmp_path / "out" / "seq.txt",
            config=config,
            device="cpu",
            enable_isolated_mask_matching=False,
            enable_cmc=False,
            cmc_method="sparseOptFlow",
            cmc_downscale=6,
            keep_partial_results=False,
            logger=logging.getLogger("test_mcbyte"),
        )

        assert captured["frame_rate"] == 25.0

    def test_without_seqinfo_falls_back_to_dataset_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ``seqinfo.ini``: ``config.frame_rate`` reaches the tracker exactly as before this fix.

        Without the fix — ``create_tracker(frame_rate=config.frame_rate, ...)`` unconditionally — this test would
        still pass, but :meth:`test_seqinfo_frame_rate_reaches_create_tracker` above would fail (30.0 instead of
        25.0), which is the actual regression guard.
        """
        detection_file, image_dir = self._make_sequence(tmp_path)

        captured: dict[str, float] = {}

        def fake_create_tracker(
            *, frame_rate: float, **_kwargs: object
        ) -> TestRunSequenceUsesRealFrameRate._FakeTracker:
            captured["frame_rate"] = frame_rate
            return TestRunSequenceUsesRealFrameRate._FakeTracker()

        monkeypatch.setattr("trackers.cli.benchmark.mcbyte.create_tracker", fake_create_tracker)

        config = DatasetConfig(name="sportsmot", detection_format="xyxy", frame_rate=30.0)
        run_sequence(
            sequence="seq",
            detection_file=detection_file,
            image_dir=image_dir,
            output_file=tmp_path / "out" / "seq.txt",
            config=config,
            device="cpu",
            enable_isolated_mask_matching=False,
            enable_cmc=False,
            cmc_method="sparseOptFlow",
            cmc_downscale=6,
            keep_partial_results=False,
            logger=logging.getLogger("test_mcbyte"),
        )

        assert captured["frame_rate"] == 30.0


class TestPrepareMot17Submission:
    def test_duplicates_source_content_across_required_suffixes(self, tmp_path: Path) -> None:
        """One tracked result is copied under each detector name McByte does not distinguish."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "MOT17-01.txt").write_bytes(b"1,2,3,4,5,6,-1,-1,-1,-1\n")
        submission_dir = tmp_path / "submission"

        prepare_mot17_submission(raw_dir, submission_dir, logging.getLogger("test_mcbyte"))

        for suffix in MOT17_SUFFIXES:
            assert (submission_dir / f"MOT17-01-{suffix}.txt").read_bytes() == b"1,2,3,4,5,6,-1,-1,-1,-1\n"

    def test_creates_empty_placeholders_for_missing_sequences(self, tmp_path: Path) -> None:
        """Sequences absent from the raw results still get a submittable empty file."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        submission_dir = tmp_path / "submission"

        prepare_mot17_submission(raw_dir, submission_dir, logging.getLogger("test_mcbyte"))

        for number in MOT17_MISSING:
            for suffix in MOT17_SUFFIXES:
                path = submission_dir / f"MOT17-{number}-{suffix}.txt"
                assert path.is_file()
                assert path.stat().st_size == 0

    def test_produces_the_full_sequence_by_suffix_file_count(self, tmp_path: Path) -> None:
        """Every existing and missing sequence ends up covered, three files each."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for number in MOT17_EXISTING:
            (raw_dir / f"MOT17-{number}.txt").write_bytes(b"data\n")
        submission_dir = tmp_path / "submission"

        prepare_mot17_submission(raw_dir, submission_dir, logging.getLogger("test_mcbyte"))

        expected_count = (len(MOT17_EXISTING) + len(MOT17_MISSING)) * len(MOT17_SUFFIXES)
        assert len(list(submission_dir.glob("*.txt"))) == expected_count
