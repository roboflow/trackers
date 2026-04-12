#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import tomllib

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trackers.calibration.export import (
    write_calibration_jsonl,
    write_calibration_quality_csv,
    write_homography_jsonl,
    write_manifest,
)
from trackers.calibration.providers.pnlcalib import PnLCalibProvider
from trackers.calibration.types import PitchDimensions


def _repo_root() -> Path:
    return _REPO_ROOT


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _slugify(value: str) -> str:
    safe_chars = []
    for char in value:
        if char.isalnum():
            safe_chars.append(char)
        elif char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    slug = "".join(safe_chars).strip("._")
    return slug or "video"


def _load_pitch_dimensions(path: Path) -> PitchDimensions:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    pitch = data.get("pitch", data)
    return PitchDimensions(
        length_m=float(pitch.get("length_m", 105.0)),
        width_m=float(pitch.get("width_m", 68.0)),
    )


def _load_toml(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _default_output_dir(
    source: Path, provider: str, config_path: Path, root: Path
) -> Path:
    return (
        root
        / "runs"
        / _slugify(source.stem)
        / "calibration"
        / f"{provider}__{_slugify(config_path.stem)}"
    )


def _build_provider(
    provider_name: str,
    *,
    pitch_dimensions: PitchDimensions,
    config_path: Path,
    upstream_root: Path,
    config_data: dict[str, object],
) -> PnLCalibProvider:
    if provider_name != "pnlcalib":
        raise ValueError(f"Unsupported calibration provider: {provider_name}")
    return PnLCalibProvider(
        pitch_dimensions=pitch_dimensions,
        config_path=config_path,
        upstream_root=upstream_root,
        config_data=config_data,
    )


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Prepare and run a pitch-calibration backend on a video or clip."
    )
    parser.add_argument("source", help="Video or clip to calibrate")
    parser.add_argument(
        "--provider",
        default="pnlcalib",
        choices=["pnlcalib"],
        help="Calibration backend to use",
    )
    parser.add_argument(
        "--config",
        default="configs/calibration/pnlcalib/default.toml",
        help="Provider config relative to the repo root",
    )
    parser.add_argument(
        "--pitch-config",
        default="configs/pitch/canonical_105x68.toml",
        help="Pitch model config relative to the repo root",
    )
    parser.add_argument(
        "--upstream-root",
        default="third_party/pnlcalib",
        help="Location of the vendored or checked-out upstream provider",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to runs/<clip>/calibration/<provider>__<config>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan without invoking the provider",
    )
    parser.add_argument(
        "--print-manifest",
        action="store_true",
        help="Echo the manifest JSON that will be written to disk",
    )
    parser.set_defaults(repo_root=root)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root

    source_path = _resolve_path(args.source, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    pitch_config_path = _resolve_path(args.pitch_config, repo_root)
    upstream_root = _resolve_path(args.upstream_root, repo_root)

    if not source_path.exists():
        raise FileNotFoundError(f"Video not found: {source_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Calibration config not found: {config_path}")
    if not pitch_config_path.exists():
        raise FileNotFoundError(f"Pitch config not found: {pitch_config_path}")

    config_data = _load_toml(config_path)
    pitch_dimensions = _load_pitch_dimensions(pitch_config_path)
    provider = _build_provider(
        args.provider,
        pitch_dimensions=pitch_dimensions,
        config_path=config_path,
        upstream_root=upstream_root,
        config_data=config_data,
    )

    output_dir = (
        _resolve_path(args.output_dir, repo_root)
        if args.output_dir is not None
        else _default_output_dir(source_path, args.provider, config_path, repo_root)
    )

    manifest = {
        "source": str(source_path),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "pitch_config_path": str(pitch_config_path),
        "provider": provider.describe(),
    }

    if args.print_manifest or args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / "manifest.json", manifest)

    if not provider.is_available():
        hint = provider.availability_hint()
        if hint is not None:
            print(hint, file=sys.stderr)
        return 2

    try:
        frames = provider.calibrate_video(source_path, output_dir)
    except NotImplementedError as error:
        print(str(error), file=sys.stderr)
        return 3

    output_config = config_data.get("output", {})
    if not isinstance(output_config, dict):
        output_config = {}

    if bool(output_config.get("write_camera_jsonl", True)):
        write_calibration_jsonl(output_dir / "camera.jsonl", frames)
    if bool(output_config.get("write_homography_jsonl", True)):
        write_homography_jsonl(output_dir / "homography.jsonl", frames)
    if bool(output_config.get("write_quality_csv", True)):
        write_calibration_quality_csv(output_dir / "quality.csv", frames)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
