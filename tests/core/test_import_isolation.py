# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ReID-only heavy deps. PIL is intentionally excluded: supervision (base dep) imports it.
_REID_HEAVY_MODULES = ("torch", "torchvision", "timm", "huggingface_hub", "safetensors", "gdown")


def _block_modules_stmt() -> str:
    return "; ".join(f"sys.modules[{name!r}] = None" for name in _REID_HEAVY_MODULES)


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=os.environ.copy(),
    )


def test_base_package_import_without_reid_extra() -> None:
    """Fresh process: base `import trackers` works with ReID heavy modules blocked."""
    code = f"import sys; {_block_modules_stmt()}; import trackers; assert trackers.BoTSORTTracker; print('ok')"
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr or result.stdout


def test_reid_package_import_without_heavy_modules() -> None:
    """Fresh process: `import trackers.core.reid` stays numpy-only."""
    code = (
        "import sys; "
        f"{_block_modules_stmt()}; "
        "import importlib; "
        "reid = importlib.import_module('trackers.core.reid'); "
        "assert reid.FeatureBank; "
        "assert reid.compute_reid_metrics; "
        "assert sys.modules.get('torch') is None; "
        "print('ok')"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr or result.stdout


def test_botsort_spawn_track_without_heavy_modules() -> None:
    """Fresh process: BoTSORT update + track spawn without loading ReID model code."""
    code = (
        "import sys\n"
        f"{_block_modules_stmt()}\n"
        "before = set(sys.modules)\n"
        "import numpy as np\n"
        "import supervision as sv\n"
        "from trackers.core.botsort.tracker import BoTSORTTracker\n"
        "tracker = BoTSORTTracker(enable_cmc=False)\n"
        "frame = np.zeros((64, 64, 3), dtype=np.uint8)\n"
        "d = sv.Detections("
        "xyxy=np.array([[1.0, 1.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]], dtype=np.float32), "
        "confidence=np.array([0.9, 0.9], dtype=np.float32))\n"
        "out = tracker.update(d, frame=frame)\n"
        "assert len(out) == 2\n"
        "assert (out.tracker_id >= 0).all()\n"
        "assert len(tracker.tracks) == 2\n"
        "imported = set(sys.modules) - before\n"
        "assert 'trackers.core.reid.model' not in imported\n"
        "print('ok')\n"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr or result.stdout


def test_reid_model_lazy_access_gives_install_hint() -> None:
    """Fresh process: accessing ReIDModel without deps shows trackers[reid] hint."""
    code = (
        "import sys\n"
        f"{_block_modules_stmt()}\n"
        "import trackers\n"
        "try:\n"
        "    _ = trackers.ReIDModel\n"
        "except ImportError as exc:\n"
        "    assert 'trackers[reid]' in str(exc)\n"
        "    print('ok')\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')\n"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr or result.stdout
