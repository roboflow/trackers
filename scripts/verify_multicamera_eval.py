#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Regenerate multicamera evaluation goldens and run NVIDIA oracle parity.

Executable tiers:

* ``--tier1`` — hermetic fixtures vs NVIDIA ``main.py`` + committed goldens
* ``--tier2`` — truncated ``scene_061`` recipe vs NVIDIA + committed goldens
* ``--tier3`` — stream-split full bundled sample files, compare per-scene and
  headline HOTA/DetA/AssA/LocA (downloads ~2.8 GB if not cached)

Downloads the pinned NVIDIA evaluator from Hugging Face revision
``1eebcf0f74a510994fe4c886f4fa77fbc6724ea8`` (no GCS mirror). Records exact
evaluator revision and numpy/scipy versions in ``provenance.json`` when
``--write-goldens`` is set.

Examples::

    python scripts/verify_multicamera_eval.py --tier1 --tier2
    python scripts/verify_multicamera_eval.py --tier1 --tier2 --write-goldens
    python scripts/verify_multicamera_eval.py --tier3
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import shlex
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "data" / "multicamera"
HF_REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
HF_REVISION = "1eebcf0f74a510994fe4c886f4fa77fbc6724ea8"
CANONICAL_EVALUATOR_TREE_SHA256 = "5a715f92f089a640da3a325d9648e4437cd3dedf8d9edcf22b63d86594e4676c"
HEADLINE_FIELDS = ("HOTA", "DetA", "AssA", "LocA")
_SCENE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# Published AI City 2024 README headline numbers (percentages).
PUBLISHED_HEADLINE_PCT = {
    "HOTA": 49.2825,
    "DetA": 49.1998,
    "AssA": 49.3655,
    "LocA": 77.0546,
}

# Evaluator source files only — never the multi-GB sample payloads by default.
_EVALUATOR_FILES = (
    "MTMC_Tracking_2024/eval/main.py",
    "MTMC_Tracking_2024/eval/README.md",
    "MTMC_Tracking_2024/eval/3rdParty_Licenses.md",
    "MTMC_Tracking_2024/eval/utils/__init__.py",
    "MTMC_Tracking_2024/eval/utils/io_utils.py",
    "MTMC_Tracking_2024/eval/trackeval/__init__.py",
    "MTMC_Tracking_2024/eval/trackeval/_timing.py",
    "MTMC_Tracking_2024/eval/trackeval/eval.py",
    "MTMC_Tracking_2024/eval/trackeval/utils.py",
    "MTMC_Tracking_2024/eval/trackeval/plotting.py",
    "MTMC_Tracking_2024/eval/trackeval/datasets/__init__.py",
    "MTMC_Tracking_2024/eval/trackeval/datasets/_base_dataset.py",
    "MTMC_Tracking_2024/eval/trackeval/datasets/mot_challenge_2d_box.py",
    "MTMC_Tracking_2024/eval/trackeval/datasets/mot_challenge_3d_location.py",
    "MTMC_Tracking_2024/eval/trackeval/datasets/test_mot.py",
    "MTMC_Tracking_2024/eval/trackeval/metrics/__init__.py",
    "MTMC_Tracking_2024/eval/trackeval/metrics/_base_metric.py",
    "MTMC_Tracking_2024/eval/trackeval/metrics/clear.py",
    "MTMC_Tracking_2024/eval/trackeval/metrics/count.py",
    "MTMC_Tracking_2024/eval/trackeval/metrics/hota.py",
    "MTMC_Tracking_2024/eval/trackeval/metrics/identity.py",
    "MTMC_Tracking_2024/eval/sample_file/scene_name_2_cam_id_full.json",
)

_TIER3_SAMPLE_FILES = (
    "MTMC_Tracking_2024/eval/sample_file/ground_truth_test_full.txt",
    "MTMC_Tracking_2024/eval/sample_file/pred.txt",
    "MTMC_Tracking_2024/eval/sample_file/scene_name_2_cam_id_full.json",
)


def _ensure_repo_src_on_path() -> None:
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _build_scene_routes(
    scene_camera_map: Mapping[str, Sequence[int]],
    output_dir: Path,
    *,
    ground_truth_layout: bool,
) -> tuple[dict[int, list[str]], dict[str, Path]]:
    """Validate scene paths and map each camera to destination scenes."""
    resolved_output = output_dir.resolve()
    camera_to_scenes: dict[int, list[str]] = {}
    destinations: dict[str, Path] = {}
    for scene_name, camera_ids in scene_camera_map.items():
        if not _SCENE_NAME_PATTERN.fullmatch(scene_name):
            raise ValueError(f"Invalid scene name {scene_name!r}; absolute and traversal paths are forbidden.")
        relative_path = Path(scene_name) / "ground_truth.txt" if ground_truth_layout else Path(f"{scene_name}.txt")
        destination = (output_dir / relative_path).resolve()
        if not destination.is_relative_to(resolved_output):
            raise ValueError(f"Scene output path escapes destination: {scene_name!r}")
        destinations[scene_name] = destination
        for camera_id in camera_ids:
            camera_to_scenes.setdefault(int(camera_id), []).append(scene_name)
    return camera_to_scenes, destinations


def _parse_split_row(
    raw_line: str,
    *,
    path: Path,
    line_number: int,
    parse_row: Callable[..., tuple[float, ...]] | None,
) -> tuple[int, str]:
    """Validate one splitter row and return its camera and normalized text."""
    line_body = raw_line.rstrip("\r\n")
    line = line_body.strip()
    if not line:
        raise ValueError(f"Blank line at {path}:{line_number} is not allowed in AI City 2024 files.")
    if line.startswith("#") or line.lower().startswith("camera"):
        raise ValueError(f"Header or comment line at {path}:{line_number} is not allowed: {line_body!r}")
    tokens = line.split()
    if parse_row is not None:
        return int(parse_row(tokens, path=path, line_number=line_number)[0]), line_body
    if len(tokens) != 9 or not tokens[0].isdecimal() or int(tokens[0]) > 2**53 - 1:
        raise ValueError(f"Malformed camera identifier or column count at {path}:{line_number}.")
    return int(tokens[0]), line_body


def split_multicamera_file_by_scene(
    path: str | Path,
    scene_camera_map: Mapping[str, Sequence[int]],
    output_dir: str | Path,
    *,
    ground_truth_layout: bool = False,
    validate_rows: bool = True,
) -> dict[str, Path]:
    """Stream a monolithic AI City file into per-scene text files.

    Verification-only helper (not part of the public ``trackers`` API). Routes
    each row to the scene that owns its ``camera_id``, preserving file order
    within each scene. Cameras shared by multiple scenes are written to every
    matching scene file. Unknown cameras are dropped silently.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Multi-camera file not found: {path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_repo_src_on_path()
    from trackers.io.multicamera import _parse_row_tokens

    camera_to_scenes, destinations = _build_scene_routes(
        scene_camera_map,
        output_dir,
        ground_truth_layout=ground_truth_layout,
    )

    handles: dict[str, TextIO] = {}
    written: dict[str, Path] = {}
    saw_row = False
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            for line_number, raw_line in enumerate(source, start=1):
                camera_id, line_body = _parse_split_row(
                    raw_line,
                    path=path,
                    line_number=line_number,
                    parse_row=_parse_row_tokens if validate_rows else None,
                )
                saw_row = True
                for scene_name in camera_to_scenes.get(camera_id, ()):
                    if scene_name not in handles:
                        out_path = destinations[scene_name]
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        handles[scene_name] = out_path.open("w", encoding="utf-8")
                        written[scene_name] = out_path
                    handles[scene_name].write(line_body + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    if not saw_row:
        raise ValueError(f"Multi-camera file is empty: {path}")
    return written


def _dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for distribution in ("numpy", "scipy", "pandas"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _sha256_file(path: Path) -> str:
    """Hash a file without loading it wholly into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_evaluator_tree(eval_dir: Path) -> str:
    """Hash evaluator relative paths and bytes deterministically."""
    digest = hashlib.sha256()
    files = sorted(path for path in eval_dir.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    for path in files:
        relative = path.relative_to(eval_dir).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _evaluator_identity(
    eval_dir: Path,
    *,
    canonical_tree_sha256: str = CANONICAL_EVALUATOR_TREE_SHA256,
) -> dict[str, Any]:
    """Identify actual evaluator bytes and verify whether they match the pin."""
    tree_sha256 = _sha256_evaluator_tree(eval_dir)
    verified = tree_sha256 == canonical_tree_sha256
    return {
        "path": str(eval_dir.resolve()),
        "tree_sha256": tree_sha256,
        "canonical_tree_sha256": canonical_tree_sha256,
        "revision": HF_REVISION if verified else None,
        "verified": verified,
    }


def _require_verified_evaluator_for_goldens(identity: Mapping[str, Any], *, write_goldens: bool) -> None:
    """Prevent modified evaluator bytes from producing NVIDIA-labeled fixtures."""
    if write_goldens and identity.get("verified") is not True:
        raise ValueError(
            "Refusing to write NVIDIA goldens with an unverified evaluator "
            f"(tree_sha256={identity.get('tree_sha256')!r})."
        )


def _comparison_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Digest the retained Tier-3 comparison evidence deterministically."""
    if "scene_comparison" not in receipt and "validation_artifact_sha256" not in receipt:
        raise ValueError("Tier-3 receipt requires per-scene comparisons or a hashed independent validation artifact.")
    evidence = {
        key: receipt[key]
        for key in (
            "scene_parity",
            "scene_comparison",
            "headline_percent",
            "input_sha256",
            "validation_artifact",
            "validation_artifact_sha256",
        )
        if key in receipt
    }
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_tier3_receipt(receipt: Mapping[str, Any]) -> None:
    """Authenticate retained Tier-3 evidence before carrying it forward."""
    scene_parity = receipt.get("scene_parity")
    if not isinstance(scene_parity, Mapping) or not scene_parity:
        raise ValueError("Retained Tier-3 receipt lacks per-scene parity evidence.")
    scene_comparison = receipt.get("scene_comparison")
    artifact_path = receipt.get("validation_artifact")
    artifact_sha256 = receipt.get("validation_artifact_sha256")
    if scene_comparison is not None:
        if not isinstance(scene_comparison, Mapping) or set(scene_comparison) != set(scene_parity):
            raise ValueError("Retained Tier-3 scene comparisons do not match scene parity coverage.")
    elif isinstance(artifact_path, str) and isinstance(artifact_sha256, str):
        if _sha256_file(REPO_ROOT / artifact_path) != artifact_sha256:
            raise ValueError("Retained Tier-3 validation artifact hash does not match provenance.")
    else:
        raise ValueError("Retained Tier-3 receipt has neither scene comparisons nor a hashed validation artifact.")
    if receipt.get("comparison_receipt_sha256") != _comparison_receipt_sha256(receipt):
        raise ValueError("Retained Tier-3 comparison receipt digest does not match provenance.")


def _hf_download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
            revision=HF_REVISION,
        )
    )


def ensure_nvidia_eval_dir(cache_dir: Path | None = None) -> Path:
    """Download pinned NVIDIA evaluator sources into a local cache directory."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "trackers-nvidia-eval" / HF_REVISION
    eval_dir = cache_dir / "MTMC_Tracking_2024" / "eval"
    marker = cache_dir / ".revision"
    if eval_dir.joinpath("main.py").exists() and marker.exists() and marker.read_text().strip() == HF_REVISION:
        return eval_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    for filename in _EVALUATOR_FILES:
        local = _hf_download(filename)
        destination = cache_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.resolve() != local.resolve():
            shutil.copy2(local, destination)
    marker.write_text(HF_REVISION + "\n", encoding="utf-8")
    return eval_dir


def _load_nvidia_module(eval_dir: Path) -> Any:
    main_path = eval_dir / "main.py"
    if not main_path.exists():
        raise SystemExit(f"NVIDIA evaluator missing: {main_path}")
    # NVIDIA imports ``trackeval`` and ``utils`` relative to the eval directory.
    eval_str = str(eval_dir)
    if eval_str not in sys.path:
        sys.path.insert(0, eval_str)
    spec = importlib.util.spec_from_file_location("nvidia_mtmc_main", main_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to import NVIDIA evaluator from {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_nvidia_oracle(
    eval_dir: Path,
    *,
    prediction_file: Path,
    ground_truth_file: Path,
    scene_map_file: Path,
    num_cores: int = 1,
) -> dict[str, dict[str, float]]:
    """Run NVIDIA ``computes_mot_metrics`` and return per-scene + FINAL fractions."""
    module = _load_nvidia_module(eval_dir)
    with tempfile.TemporaryDirectory(prefix="nvidia-mtmc-") as tmp:
        sequence_result = module.computes_mot_metrics(
            str(prediction_file),
            str(ground_truth_file),
            tmp,
            num_cores,
            str(scene_map_file),
        )
    data = sequence_result[0]["MotChallenge3DLocation"]["data"]
    import numpy as np

    scores: dict[str, dict[str, float]] = {}
    headline_accum: dict[str, list[float]] = {field: [] for field in HEADLINE_FIELDS}
    for scene_name, payload in data.items():
        if scene_name == "COMBINED_SEQ":
            continue
        hota = payload["pedestrian"]["HOTA"]
        scene_scores = {
            "HOTA": float(np.mean(hota["HOTA"])),
            "DetA": float(np.mean(hota["DetA"])),
            "AssA": float(np.mean(hota["AssA"])),
            "LocA": float(np.mean(hota["LocA"])),
        }
        scores[scene_name] = scene_scores
        for field in HEADLINE_FIELDS:
            headline_accum[field].append(scene_scores[field])
    scores["FINAL"] = {field: float(np.mean(headline_accum[field])) for field in HEADLINE_FIELDS}
    return scores


def _approx_equal(got: float, expected: float, *, rel: float = 1e-4, abs_: float = 1e-4) -> bool:
    return abs(got - expected) <= max(abs_, rel * abs(expected))


def _compare_headline(
    label: str,
    got: dict[str, float],
    expected: dict[str, float],
    *,
    rel: float = 1e-4,
    abs_: float = 1e-4,
) -> None:
    for field in HEADLINE_FIELDS:
        if field not in got or field not in expected:
            raise SystemExit(f"{label}: missing field {field}")
        if not _approx_equal(got[field], expected[field], rel=rel, abs_=abs_):
            raise SystemExit(f"{label}.{field}: got {got[field]!r}, expected {expected[field]!r}")


def _write_provenance(
    *,
    eval_dir: Path,
    tiers_run: list[str],
    tier3_receipt: dict[str, Any] | None,
    write_goldens: bool,
) -> None:
    versions = _dependency_versions()
    previous: dict[str, Any] = {}
    provenance_path = FIXTURE_DIR / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
    if tier3_receipt is not None:
        _validate_tier3_receipt(tier3_receipt)
    elif "tier3" in previous and "tier3" in previous.get("tiers_validated", []):
        _validate_tier3_receipt(previous["tier3"])
    validated_tiers = list(tiers_run)
    if tier3_receipt is not None or ("tier3" in previous.get("tiers_validated", []) and "tier3" in previous):
        if "tier3" not in validated_tiers:
            validated_tiers.append("tier3")
    payload = {
        "oracle": "NVIDIA MTMC_Tracking_2024/eval/main.py from nvidia/PhysicalAI-SmartSpaces",
        "dataset_revision": HF_REVISION,
        "evaluator": _evaluator_identity(eval_dir),
        "download": (
            f"hf_hub_download(repo_id={HF_REPO_ID!r}, repo_type='dataset', revision={HF_REVISION!r}) - no GCS mirror"
        ),
        "command": ("python scripts/verify_multicamera_eval.py --tier1 --tier2 [--tier3] [--write-goldens]"),
        "numpy": versions["numpy"],
        "scipy": versions["scipy"],
        "pandas": versions["pandas"],
        "python": versions["python"],
        "tiers_validated": validated_tiers,
        "notes": "Goldens are fractions in [0, 1]. NVIDIA prints percentages.",
    }
    if tier3_receipt is not None:
        payload["tier3"] = tier3_receipt
    elif "tier3" in previous:
        payload["tier3"] = previous["tier3"]
    if write_goldens:
        provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("Wrote", provenance_path)
    else:
        print("Provenance (not written; pass --write-goldens to persist):")
        print(json.dumps(payload, indent=2))


def _run_tier1(eval_dir: Path, *, write_goldens: bool) -> dict[str, dict[str, float]]:
    _ensure_repo_src_on_path()
    from trackers.eval import evaluate_multicamera_scenes

    ours = evaluate_multicamera_scenes(
        gt_dir=FIXTURE_DIR / "gt",
        tracker_dir=FIXTURE_DIR / "pred",
        scene_camera_map=FIXTURE_DIR / "scene_camera_map.json",
        allow_partial=True,
    )
    nvidia = run_nvidia_oracle(
        eval_dir,
        prediction_file=FIXTURE_DIR / "combined_pred.txt",
        ground_truth_file=FIXTURE_DIR / "combined_gt.txt",
        scene_map_file=FIXTURE_DIR / "scene_camera_map.json",
    )

    ours_scenes: dict[str, dict[str, float]] = {}
    for scene_name, seq in ours.sequences.items():
        if seq.HOTA is None:
            raise SystemExit(f"tier1: missing HOTA for {scene_name}")
        ours_scenes[scene_name] = {field: float(getattr(seq.HOTA, field)) for field in HEADLINE_FIELDS}
        _compare_headline(f"tier1 ours vs NVIDIA {scene_name}", ours_scenes[scene_name], nvidia[scene_name])

    if ours.aggregate.HOTA is None:
        raise SystemExit("tier1: missing SCENE_MEAN HOTA")
    ours_mean = {field: float(getattr(ours.aggregate.HOTA, field)) for field in HEADLINE_FIELDS}
    _compare_headline("tier1 ours vs NVIDIA FINAL", ours_mean, nvidia["FINAL"])

    expected_path = FIXTURE_DIR / "expected_results.json"
    if write_goldens:
        payload = {
            "source": "NVIDIA MTMC_Tracking_2024/eval/main.py",
            "revision": HF_REVISION,
            "tolerance": {"rel": 1e-4, "abs": 1e-4},
            "scenes": {name: {field: nvidia[name][field] for field in HEADLINE_FIELDS} for name in sorted(ours_scenes)},
            "SCENE_MEAN": {field: nvidia["FINAL"][field] for field in HEADLINE_FIELDS},
        }
        expected_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("Wrote", expected_path)
    else:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        for scene_name, values in expected["scenes"].items():
            _compare_headline(f"tier1 vs golden {scene_name}", ours_scenes[scene_name], values)
        _compare_headline("tier1 vs golden SCENE_MEAN", ours_mean, expected["SCENE_MEAN"])

    print("tier1 OK (ours == NVIDIA" + (" / goldens rewritten)" if write_goldens else " == goldens)"))
    return {"SCENE_MEAN": ours_mean, **ours_scenes}


def _build_tier2_prediction_rows(
    gt_rows: list[list[str]],
    *,
    drop_every_k: int,
    id_swap_frame_start: int,
    id_swap_frame_end: int,
    dedup_dup_rows: int,
) -> list[list[str]]:
    pred_rows: list[list[str]] = []
    for index, parts in enumerate(gt_rows):
        if index % drop_every_k == 0:
            continue
        frame = int(parts[2])
        obj = int(parts[1])
        if id_swap_frame_start <= frame < id_swap_frame_end:
            obj = obj + 1000
        new_parts = list(parts)
        new_parts[1] = str(obj)
        pred_rows.append(new_parts)
    for parts in pred_rows[:dedup_dup_rows]:
        duplicate = list(parts)
        duplicate[7] = str(float(duplicate[7]) + 10.0)
        pred_rows.append(duplicate)
    return pred_rows


def _run_tier2(eval_dir: Path, *, write_goldens: bool) -> dict[str, dict[str, float]]:
    _ensure_repo_src_on_path()
    from trackers.eval import evaluate_multicamera_scene
    from trackers.io.multicamera import _truncate_multicamera_rows

    expected_path = FIXTURE_DIR / "tier2_expected.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    recipe = expected["recipe"]
    if "revision" not in recipe:
        raise SystemExit("tier2_expected.json recipe must pin an explicit HF revision")
    if recipe["revision"] != HF_REVISION:
        raise SystemExit(f"tier2 recipe revision {recipe['revision']!r} != script pin {HF_REVISION!r}")

    gt_source = _hf_download(recipe["source"])
    gt_rows = _truncate_multicamera_rows(gt_source, max_frame=recipe["max_frame"])
    pred_rows = _build_tier2_prediction_rows(
        gt_rows,
        drop_every_k=recipe["drop_every_k"],
        id_swap_frame_start=recipe["id_swap_frame_start"],
        id_swap_frame_end=recipe["id_swap_frame_end"],
        dedup_dup_rows=recipe["dedup_dup_rows"],
    )

    with tempfile.TemporaryDirectory(prefix="tier2-mtmc-") as tmp_name:
        tmp = Path(tmp_name)
        gt_path = tmp / "ground_truth.txt"
        pred_path = tmp / "pred.txt"
        scene_map_path = tmp / "scene_map.json"
        gt_path.write_text("\n".join(" ".join(row) for row in gt_rows) + "\n", encoding="utf-8")
        pred_path.write_text("\n".join(" ".join(row) for row in pred_rows) + "\n", encoding="utf-8")
        scene_map_path.write_text(
            json.dumps([{"scene_name": "scene_061", "camera_ids": recipe["camera_ids"]}], indent=2) + "\n",
            encoding="utf-8",
        )

        ours = evaluate_multicamera_scene(
            scene="scene_061",
            gt_path=gt_path,
            tracker_path=pred_path,
            camera_ids=tuple(recipe["camera_ids"]),
        )
        if ours.HOTA is None:
            raise SystemExit("tier2: missing HOTA")
        ours_scores = {field: float(getattr(ours.HOTA, field)) for field in HEADLINE_FIELDS}

        nvidia = run_nvidia_oracle(
            eval_dir,
            prediction_file=pred_path,
            ground_truth_file=gt_path,
            scene_map_file=scene_map_path,
        )
        _compare_headline("tier2 ours vs NVIDIA scene_061", ours_scores, nvidia["scene_061"])
        if write_goldens:
            payload = {
                "recipe": recipe,
                "scene_061": {field: nvidia["scene_061"][field] for field in HEADLINE_FIELDS},
            }
            expected_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print("Wrote", expected_path)
        else:
            _compare_headline("tier2 vs golden scene_061", ours_scores, expected["scene_061"])

    print("tier2 OK (ours == NVIDIA" + (" / goldens rewritten)" if write_goldens else " == goldens)"))
    return {"scene_061": ours_scores}


def _resolve_tier3_sample_dir(sample_dir: Path | None) -> Path:
    if sample_dir is not None:
        return sample_dir
    # Prefer already-downloaded Hugging Face cache / local mirrors.
    candidates = [
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--nvidia--PhysicalAI-SmartSpaces"
        / "snapshots"
        / HF_REVISION
        / "MTMC_Tracking_2024"
        / "eval"
        / "sample_file",
        Path("/tmp/aic24eval_full/MTMC_Tracking_2024/eval/sample_file"),  # noqa: S108
    ]
    for candidate in candidates:
        if (candidate / "pred.txt").exists() and (candidate / "ground_truth_test_full.txt").exists():
            return candidate

    print("Downloading tier-3 sample files from Hugging Face (≈2.8 GB)...")
    paths = {name: _hf_download(name) for name in _TIER3_SAMPLE_FILES}
    return paths["MTMC_Tracking_2024/eval/sample_file/pred.txt"].parent


def _run_tier3(eval_dir: Path, *, sample_dir: Path | None, num_cores: int) -> dict[str, Any]:
    _ensure_repo_src_on_path()
    from trackers.eval import evaluate_multicamera_scenes
    from trackers.io.multicamera import load_scene_camera_map

    sample = _resolve_tier3_sample_dir(sample_dir)
    pred = sample / "pred.txt"
    gt = sample / "ground_truth_test_full.txt"
    scene_map_path = sample / "scene_name_2_cam_id_full.json"
    for path in (pred, gt, scene_map_path):
        if not path.exists():
            raise SystemExit(f"tier3 missing sample file: {path}")

    camera_map = load_scene_camera_map(scene_map_path)
    with tempfile.TemporaryDirectory(prefix="tier3-mtmc-") as tmp_name:
        tmp = Path(tmp_name)
        gt_dir = tmp / "gt"
        tracker_dir = tmp / "pred"
        split_multicamera_file_by_scene(
            gt,
            camera_map,
            gt_dir,
            ground_truth_layout=True,
            validate_rows=False,
        )
        split_multicamera_file_by_scene(pred, camera_map, tracker_dir, validate_rows=False)

        ours = evaluate_multicamera_scenes(
            gt_dir=gt_dir,
            tracker_dir=tracker_dir,
            scene_camera_map=camera_map,
        )
        nvidia = run_nvidia_oracle(
            eval_dir,
            prediction_file=pred,
            ground_truth_file=gt,
            scene_map_file=scene_map_path,
            num_cores=num_cores,
        )

        scene_comparison: dict[str, dict[str, dict[str, float]]] = {}
        for scene_name, seq in ours.sequences.items():
            if seq.HOTA is None:
                raise SystemExit(f"tier3: missing HOTA for {scene_name}")
            ours_scores = {field: float(getattr(seq.HOTA, field)) for field in HEADLINE_FIELDS}
            _compare_headline(f"tier3 ours vs NVIDIA {scene_name}", ours_scores, nvidia[scene_name])
            scene_comparison[scene_name] = {
                "ours": ours_scores,
                "nvidia": nvidia[scene_name],
            }

        if ours.aggregate.HOTA is None:
            raise SystemExit("tier3: missing SCENE_MEAN")
        ours_mean = {field: float(getattr(ours.aggregate.HOTA, field)) for field in HEADLINE_FIELDS}
        _compare_headline("tier3 ours vs NVIDIA FINAL", ours_mean, nvidia["FINAL"])
        published = {field: value / 100.0 for field, value in PUBLISHED_HEADLINE_PCT.items()}
        _compare_headline("tier3 NVIDIA vs published README", nvidia["FINAL"], published, rel=1e-3, abs_=1e-4)
        _compare_headline("tier3 ours vs published README", ours_mean, published, rel=1e-3, abs_=1e-4)

    print("tier3 OK — SCENE_MEAN %: " + ", ".join(f"{field}={ours_mean[field] * 100:.4f}" for field in HEADLINE_FIELDS))
    return {
        "SCENE_MEAN": ours_mean,
        "nvidia_FINAL": nvidia["FINAL"],
        "scene_parity": {scene: "passed" for scene in camera_map},
        "scene_comparison": scene_comparison,
        "input_sha256": {
            "ground_truth_test_full.txt": _sha256_file(gt),
            "pred.txt": _sha256_file(pred),
            "scene_name_2_cam_id_full.json": _sha256_file(scene_map_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier1", action="store_true", help="Hermetic fixtures vs NVIDIA + goldens.")
    parser.add_argument("--tier2", action="store_true", help="Truncated scene_061 vs NVIDIA + goldens.")
    parser.add_argument(
        "--tier3",
        action="store_true",
        help="Full ~2.8 GB sample files: stream-split, compare ours/NVIDIA/README.",
    )
    parser.add_argument(
        "--write-goldens",
        action="store_true",
        help="Rewrite expected_results.json / tier2_expected.json / provenance.json.",
    )
    parser.add_argument(
        "--nvidia-eval-dir",
        type=Path,
        default=None,
        help="Existing MTMC_Tracking_2024/eval directory (skips HF evaluator download).",
    )
    parser.add_argument(
        "--tier3-sample-dir",
        type=Path,
        default=None,
        help="Directory containing pred.txt / ground_truth_test_full.txt / scene map.",
    )
    parser.add_argument("--num-cores", type=int, default=1, help="NVIDIA TrackEval worker count.")
    args = parser.parse_args()

    if not (args.tier1 or args.tier2 or args.tier3):
        parser.error("Specify at least one of --tier1 / --tier2 / --tier3")

    eval_dir = Path(args.nvidia_eval_dir) if args.nvidia_eval_dir else ensure_nvidia_eval_dir()
    evaluator_identity = _evaluator_identity(eval_dir)
    _require_verified_evaluator_for_goldens(evaluator_identity, write_goldens=args.write_goldens)
    print(f"NVIDIA evaluator: {eval_dir}")
    print(f"Evaluator revision: {evaluator_identity['revision'] or 'unverified override'}")
    print("Dependency versions:", json.dumps(_dependency_versions(), sort_keys=True))

    tiers_run: list[str] = []
    tier3_receipt: dict[str, Any] | None = None
    if args.tier1:
        _run_tier1(eval_dir, write_goldens=args.write_goldens)
        tiers_run.append("tier1")
    if args.tier2:
        _run_tier2(eval_dir, write_goldens=args.write_goldens)
        tiers_run.append("tier2")
    if args.tier3:
        started = time.perf_counter()
        tier3_result = _run_tier3(eval_dir, sample_dir=args.tier3_sample_dir, num_cores=args.num_cores)
        runtime_seconds = time.perf_counter() - started
        tier3_receipt = {
            "command": shlex.join(["python", *sys.argv]),
            "revision": evaluator_identity["revision"],
            "runtime_seconds": round(runtime_seconds, 3),
            "scene_parity": tier3_result["scene_parity"],
            "scene_comparison": tier3_result["scene_comparison"],
            "headline_percent": {
                field: round(float(tier3_result["SCENE_MEAN"][field]) * 100, 4) for field in HEADLINE_FIELDS
            },
            "environment": _dependency_versions(),
            "input_sha256": tier3_result["input_sha256"],
        }
        tier3_receipt["comparison_receipt_sha256"] = _comparison_receipt_sha256(tier3_receipt)
        tiers_run.append("tier3")
    else:
        print("tier3: not run")

    _write_provenance(
        eval_dir=eval_dir,
        tiers_run=tiers_run,
        tier3_receipt=tier3_receipt,
        write_goldens=args.write_goldens,
    )


if __name__ == "__main__":
    main()
