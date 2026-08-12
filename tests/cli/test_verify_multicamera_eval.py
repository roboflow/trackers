# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for scripts/verify_multicamera_eval.py helpers."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_multicamera_eval.py"
OFFICIAL_SCENES = tuple(f"scene_{index:03d}" for index in range(61, 91))
CANONICAL_EVALUATOR_TREE_SHA256 = "5a715f92f089a640da3a325d9648e4437cd3dedf8d9edcf22b63d86594e4676c"
TIER3_INPUT_SHA256 = {
    "ground_truth_test_full.txt": "76fc83dae03807622ef62246ba7ebdf43f8109f5a99a2447e681fd8c94955c14",
    "pred.txt": "a51d3f9ff529cfcc1ed7c7e5dbe65307f05ce1e634b369cfedb4d23f5c83fcc3",
    "scene_name_2_cam_id_full.json": "f1f1c873d40a50e075d85a364554d902968b2c6717f16ebd5e63d43300f50bac",
}


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("verify_multicamera_eval", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


class _RecordingHandle:
    """Proxy an output handle while exposing deterministic close state."""

    def __init__(self, handle: TextIO, *, fail_write: bool = False) -> None:
        self._handle = handle
        self._fail_write = fail_write

    @property
    def closed(self) -> bool:
        """Return whether the wrapped handle was closed."""
        return self._handle.closed

    def write(self, text: str) -> int:
        """Write text or inject the configured failure."""
        if self._fail_write:
            raise OSError("injected write failure")
        return self._handle.write(text)

    def close(self) -> None:
        """Close the wrapped handle."""
        self._handle.close()


def test_split_preserves_order_and_shared_cameras(tmp_path: Path) -> None:
    """Streaming splitter keeps file order and fans out shared cameras."""
    module = _load_verify_module()
    source = _write(
        tmp_path / "mono.txt",
        "1 10 0 0 0 1 1 0 0\n3 20 1 0 0 1 1 1 1\n2 11 2 0 0 1 1 2 2\n99 1 3 0 0 1 1 3 3\n",
    )
    mapping = {"scene_a": [1, 2], "scene_b": [2, 3]}
    written = module.split_multicamera_file_by_scene(source, mapping, tmp_path / "out")
    assert set(written) == {"scene_a", "scene_b"}
    assert written["scene_a"].read_text().splitlines() == [
        "1 10 0 0 0 1 1 0 0",
        "2 11 2 0 0 1 1 2 2",
    ]
    assert written["scene_b"].read_text().splitlines() == [
        "3 20 1 0 0 1 1 1 1",
        "2 11 2 0 0 1 1 2 2",
    ]


def test_split_malformed_late_row_closes_handles_and_stops_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late parse error closes prior outputs and emits no subsequent scene."""
    module = _load_verify_module()
    source = _write(
        tmp_path / "mono.txt",
        "1 10 0 0 0 1 1 0 0\nmalformed late row\n2 20 1 0 0 1 1 1 1\n",
    )
    output_dir = tmp_path / "out"
    original_open = Path.open
    opened: list[_RecordingHandle] = []

    def recording_open(path: Path, *args: Any, **kwargs: Any) -> TextIO:
        handle = original_open(path, *args, **kwargs)
        if path.parent == output_dir:
            recorded = _RecordingHandle(handle)
            opened.append(recorded)
            return recorded  # type: ignore[return-value]
        return handle

    monkeypatch.setattr(Path, "open", recording_open)

    with pytest.raises(ValueError, match="Expected 9 columns"):
        module.split_multicamera_file_by_scene(
            source,
            {"scene_a": [1], "scene_b": [2]},
            output_dir,
        )

    assert opened and all(handle.closed for handle in opened)
    assert not (output_dir / "scene_b.txt").exists()


def test_split_open_failure_closes_prior_handles_and_stops_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An output-open failure closes earlier files and prevents later scenes."""
    module = _load_verify_module()
    source = _write(
        tmp_path / "mono.txt",
        "1 10 0 0 0 1 1 0 0\n2 20 1 0 0 1 1 1 1\n3 30 2 0 0 1 1 2 2\n",
    )
    output_dir = tmp_path / "out"
    original_open = Path.open
    opened: list[_RecordingHandle] = []

    def failing_open(path: Path, *args: Any, **kwargs: Any) -> TextIO:
        if path == output_dir / "scene_b.txt":
            raise OSError("injected open failure")
        handle = original_open(path, *args, **kwargs)
        if path.parent == output_dir:
            recorded = _RecordingHandle(handle)
            opened.append(recorded)
            return recorded  # type: ignore[return-value]
        return handle

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(OSError, match="injected open failure"):
        module.split_multicamera_file_by_scene(
            source,
            {"scene_a": [1], "scene_b": [2], "scene_c": [3]},
            output_dir,
        )

    assert opened and all(handle.closed for handle in opened)
    assert not (output_dir / "scene_c.txt").exists()


def test_split_write_failure_closes_handle_and_stops_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An output-write failure closes its file and prevents later scenes."""
    module = _load_verify_module()
    source = _write(
        tmp_path / "mono.txt",
        "1 10 0 0 0 1 1 0 0\n2 20 1 0 0 1 1 1 1\n",
    )
    output_dir = tmp_path / "out"
    original_open = Path.open
    opened: list[_RecordingHandle] = []

    def failing_write_open(path: Path, *args: Any, **kwargs: Any) -> TextIO:
        handle = original_open(path, *args, **kwargs)
        if path.parent == output_dir:
            recorded = _RecordingHandle(handle, fail_write=True)
            opened.append(recorded)
            return recorded  # type: ignore[return-value]
        return handle

    monkeypatch.setattr(Path, "open", failing_write_open)

    with pytest.raises(OSError, match="injected write failure"):
        module.split_multicamera_file_by_scene(
            source,
            {"scene_a": [1], "scene_b": [2]},
            output_dir,
        )

    assert opened and all(handle.closed for handle in opened)
    assert not (output_dir / "scene_b.txt").exists()


@pytest.mark.parametrize(
    "scene_name_factory",
    [
        pytest.param(lambda root: "../escaped", id="traversal"),
        pytest.param(lambda root: str(root / "absolute-escaped"), id="absolute"),
    ],
)
def test_split_rejects_scene_names_outside_destination(
    tmp_path: Path,
    scene_name_factory,
) -> None:
    """Traversal and absolute scene names cannot escape the output directory."""
    module = _load_verify_module()
    source = _write(tmp_path / "mono.txt", "1 10 0 0 0 1 1 0 0\n")
    output_dir = tmp_path / "out"
    scene_name = scene_name_factory(tmp_path)
    escaped_path = output_dir / f"{scene_name}.txt"

    with pytest.raises(ValueError, match=r"scene name|path|absolute|traversal"):
        module.split_multicamera_file_by_scene(source, {scene_name: [1]}, output_dir)

    assert not escaped_path.exists()


def test_tier1_goldens_persist_nvidia_oracle_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Golden regeneration persists NVIDIA's result object after parity."""
    module = _load_verify_module()
    fields = {"HOTA": 0.1, "DetA": 0.1, "AssA": 0.1, "LocA": 0.1}
    oracle_fields = {"HOTA": 0.2, "DetA": 0.3, "AssA": 0.4, "LocA": 0.5}
    ours_hota = SimpleNamespace(**fields)
    ours = SimpleNamespace(
        sequences={"scene_a": SimpleNamespace(HOTA=ours_hota)},
        aggregate=SimpleNamespace(HOTA=ours_hota),
    )
    monkeypatch.setattr(module, "FIXTURE_DIR", tmp_path)
    monkeypatch.setattr(
        module,
        "run_nvidia_oracle",
        lambda *_args, **_kwargs: {
            "scene_a": oracle_fields,
            "FINAL": oracle_fields,
        },
    )
    monkeypatch.setattr(module, "_compare_headline", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("trackers.eval.evaluate_multicamera_scenes", lambda **_kwargs: ours)

    module._run_tier1(tmp_path / "eval", write_goldens=True)
    persisted = json.loads((tmp_path / "expected_results.json").read_text())

    assert persisted["scenes"]["scene_a"] == oracle_fields
    assert persisted["SCENE_MEAN"] == oracle_fields


def test_tier2_goldens_persist_nvidia_oracle_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier-2 regeneration persists NVIDIA's result object after parity."""
    module = _load_verify_module()
    revision = module.HF_REVISION
    recipe = {
        "source": "scene_061.txt",
        "revision": revision,
        "max_frame": 1,
        "drop_every_k": 7,
        "id_swap_frame_start": 10,
        "id_swap_frame_end": 20,
        "dedup_dup_rows": 0,
        "camera_ids": [1],
    }
    (tmp_path / "tier2_expected.json").write_text(json.dumps({"recipe": recipe, "scene_061": {}}))
    ours_fields = {"HOTA": 0.1, "DetA": 0.1, "AssA": 0.1, "LocA": 0.1}
    oracle_fields = {"HOTA": 0.2, "DetA": 0.3, "AssA": 0.4, "LocA": 0.5}
    monkeypatch.setattr(module, "FIXTURE_DIR", tmp_path)
    monkeypatch.setattr(module, "_hf_download", lambda _filename: tmp_path / "source.txt")
    monkeypatch.setattr(module, "_compare_headline", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "run_nvidia_oracle",
        lambda *_args, **_kwargs: {
            "scene_061": oracle_fields,
            "FINAL": oracle_fields,
        },
    )
    monkeypatch.setattr(
        "trackers.io.multicamera._truncate_multicamera_rows",
        lambda *_args, **_kwargs: [["1", "1", "0", "0", "0", "1", "1", "0", "0"]],
    )
    monkeypatch.setattr(
        "trackers.eval.evaluate_multicamera_scene",
        lambda *_args, **_kwargs: SimpleNamespace(HOTA=SimpleNamespace(**ours_fields)),
    )

    module._run_tier2(tmp_path / "eval", write_goldens=True)
    persisted = json.loads((tmp_path / "tier2_expected.json").read_text())

    assert persisted["scene_061"] == oracle_fields


def test_provenance_contains_auditable_tier3_receipt() -> None:
    """Durable provenance authenticates exact values from both evaluators."""
    module = _load_verify_module()
    provenance = json.loads((REPO_ROOT / "tests" / "data" / "multicamera" / "provenance.json").read_text())
    receipt = provenance["tier3"]
    artifact_path = REPO_ROOT / "tests" / "data" / "multicamera" / receipt["comparison_artifact"]

    assert receipt["command"] == (
        "python scripts/verify_multicamera_eval.py --tier1 --tier2 --tier3 "
        "--tier3-sample-dir TIER3_SAMPLE_DIR "
        "--num-cores 1 --write-goldens"
    )
    assert receipt["revision"] == "1eebcf0f74a510994fe4c886f4fa77fbc6724ea8"
    assert receipt["runtime_seconds"] == 716.133
    assert receipt["scene_count"] == 30
    assert receipt["scene_range"] == ["scene_061", "scene_090"]
    assert receipt["comparison_tolerance"] == {"rel": 1e-4, "abs": 1e-4}
    assert receipt["headline_percent"] == {
        "HOTA": 49.2825,
        "DetA": 49.1998,
        "AssA": 49.3655,
        "LocA": 77.0547,
    }
    assert receipt["comparison_artifact"] == "tier3_comparison.jsonl"
    assert module._sha256_file(artifact_path) == receipt["comparison_artifact_sha256"]
    records = [json.loads(line) for line in artifact_path.read_text().splitlines()]
    assert [record["scene"] for record in records] == list(OFFICIAL_SCENES)
    assert len({record["scene"] for record in records}) == receipt["scene_count"]
    assert all(set(record) == {"scene", "ours", "nvidia"} for record in records)
    assert all(
        set(record[system]) == set(module.HEADLINE_FIELDS) for record in records for system in ("ours", "nvidia")
    )
    assert all(
        isinstance(record[system][field], (int, float)) and math.isfinite(record[system][field])
        for record in records
        for system in ("ours", "nvidia")
        for field in module.HEADLINE_FIELDS
    )
    tolerance = receipt["comparison_tolerance"]
    assert all(
        module._approx_equal(
            record["ours"][field],
            record["nvidia"][field],
            rel=tolerance["rel"],
            abs_=tolerance["abs"],
        )
        for record in records
        for field in module.HEADLINE_FIELDS
    )
    assert set(receipt["environment"]) >= {"python", "numpy", "scipy", "pandas"}
    module._validate_tier3_receipt(receipt)


def test_provenance_records_actual_evaluator_and_input_hashes() -> None:
    """Provenance authenticates executed evaluator bytes and Tier-3 inputs."""
    module = _load_verify_module()
    provenance = json.loads((REPO_ROOT / "tests" / "data" / "multicamera" / "provenance.json").read_text())
    receipt = provenance["tier3"]

    assert provenance["evaluator"]["tree_sha256"] == CANONICAL_EVALUATOR_TREE_SHA256
    assert "path" not in provenance["evaluator"]
    assert "tier3" in provenance["tiers_validated"]
    assert receipt["input_sha256"] == TIER3_INPUT_SHA256
    assert module._comparison_receipt_sha256(receipt) == receipt["comparison_receipt_sha256"]


def test_tier3_artifact_writing_is_deterministic(tmp_path: Path) -> None:
    """Artifact writing sorts scenes and emits canonical compact JSONL."""
    module = _load_verify_module()
    metrics = {"HOTA": 0.1, "DetA": 0.2, "AssA": 0.3, "LocA": 0.4}
    comparisons = {
        "scene_062": {"ours": metrics, "nvidia": metrics},
        "scene_061": {"ours": metrics, "nvidia": metrics},
    }
    artifact = tmp_path / "comparison.jsonl"

    module._write_tier3_comparison_artifact(artifact, comparisons)
    lines = artifact.read_text().splitlines()

    assert [json.loads(line)["scene"] for line in lines] == ["scene_061", "scene_062"]
    assert lines == [json.dumps(json.loads(line), sort_keys=True, separators=(",", ":")) for line in lines]


def test_tier3_receipt_rejects_artifact_hash_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Receipt validation authenticates artifact bytes before parsing them."""
    module = _load_verify_module()
    monkeypatch.setattr(module, "FIXTURE_DIR", tmp_path)
    artifact = tmp_path / "tier3.jsonl"
    artifact.write_text("{}\n")
    receipt = {
        "comparison_artifact": artifact.name,
        "comparison_artifact_sha256": "0" * 64,
    }
    receipt["comparison_receipt_sha256"] = module._comparison_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="hash"):
        module._validate_tier3_receipt(receipt)


@pytest.mark.parametrize(
    ("matches_canonical", "expected_revision", "expected_verified"),
    [
        pytest.param(True, "1eebcf0f74a510994fe4c886f4fa77fbc6724ea8", True, id="matching"),
        pytest.param(False, None, False, id="mismatched"),
    ],
)
def test_evaluator_override_revision_requires_canonical_bytes(
    tmp_path: Path,
    matches_canonical: bool,
    expected_revision: str | None,
    expected_verified: bool,
) -> None:
    """Only an override matching canonical evaluator bytes may claim the pin."""
    module = _load_verify_module()
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "main.py").write_text("canonical\n" if matches_canonical else "modified\n")
    canonical_hash = module._sha256_evaluator_tree(eval_dir) if matches_canonical else "0" * 64

    identity = module._evaluator_identity(eval_dir, canonical_tree_sha256=canonical_hash)

    assert identity["revision"] == expected_revision
    assert identity["verified"] is expected_verified


def test_unverified_evaluator_cannot_write_goldens() -> None:
    module = _load_verify_module()

    with pytest.raises(ValueError, match=r"verified|golden|evaluator"):
        module._require_verified_evaluator_for_goldens(
            {"verified": False, "tree_sha256": "0" * 64},
            write_goldens=True,
        )


def test_provenance_rewrite_preserves_validated_tier3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_verify_module()
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    metrics = {"HOTA": 0.1, "DetA": 0.1, "AssA": 0.1, "LocA": 0.1}
    comparisons = {"scene_061": {"ours": metrics, "nvidia": metrics}}
    artifact = fixture_dir / "tier3.jsonl"
    module._write_tier3_comparison_artifact(artifact, comparisons)
    receipt = {
        "comparison_artifact": artifact.name,
        "comparison_artifact_sha256": module._sha256_file(artifact),
    }
    receipt["comparison_receipt_sha256"] = module._comparison_receipt_sha256(receipt)
    (fixture_dir / "provenance.json").write_text(
        json.dumps({"tiers_validated": ["tier1", "tier2", "tier3"], "tier3": receipt})
    )
    monkeypatch.setattr(module, "FIXTURE_DIR", fixture_dir)
    monkeypatch.setattr(
        module, "_dependency_versions", lambda: dict.fromkeys(("python", "numpy", "scipy", "pandas"), "1")
    )
    monkeypatch.setattr(module, "_evaluator_identity", lambda _path: {"verified": True})

    module._write_provenance(
        eval_dir=tmp_path,
        tiers_run=["tier1", "tier2"],
        tier3_receipt=None,
        write_goldens=True,
    )
    rewritten = json.loads((fixture_dir / "provenance.json").read_text())

    assert rewritten["tiers_validated"] == ["tier1", "tier2", "tier3"]
    assert rewritten["tier3"] == receipt


def test_provenance_rewrite_rejects_corrupted_tier3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_verify_module()
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "provenance.json").write_text(
        json.dumps(
            {
                "tiers_validated": ["tier3"],
                "tier3": {
                    "scene_comparison": {},
                    "comparison_receipt_sha256": "0" * 64,
                },
            }
        )
    )
    monkeypatch.setattr(module, "FIXTURE_DIR", fixture_dir)

    with pytest.raises(ValueError, match=r"digest|provenance"):
        module._write_provenance(
            eval_dir=tmp_path,
            tiers_run=["tier1"],
            tier3_receipt=None,
            write_goldens=True,
        )


def test_tier3_builds_final_layout_without_copy_step() -> None:
    """Tier-3 splitting writes its final evaluator layout without copy2 duplication."""
    source = SCRIPT_PATH.read_text()
    tier3_source = source.split("def _run_tier3", maxsplit=1)[1].split("def main", maxsplit=1)[0]

    assert "shutil.copy2" not in tier3_source
