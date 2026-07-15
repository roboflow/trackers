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
