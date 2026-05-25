#!/usr/bin/env python3
"""Print ``trackers track`` flags from a params JSON file or library defaults."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trackers.core.base import BaseTracker  # noqa: E402


def _is_class_param(name: str, param) -> bool:
    if name == "state_estimator_class":
        return True
    default = param.default_value
    return isinstance(default, type)


def tracker_flags(tracker_id: str, params: dict | None = None) -> str:
    """Build CLI flags for one tracker.

    When *params* is empty, emit explicit ``--tracker.*`` flags from that
    tracker's registry defaults. The CLI registers shared parameter names once
    for all trackers (first registration wins), so omitting flags lets SORT and
    others inherit BoT-SORT/ByteTrack defaults by mistake.
    """
    info = BaseTracker._lookup_tracker(tracker_id)
    if info is None:
        raise ValueError(f"unknown tracker: {tracker_id}")

    if not params:
        params = {name: param.default_value for name, param in info.parameters.items()}

    parts: list[str] = []
    for name, value in params.items():
        if name not in info.parameters:
            continue
        param = info.parameters[name]
        if _is_class_param(name, param):
            continue
        if param.param_type is bool:
            if value != param.default_value:
                parts.append(f"--tracker.{name}")
        else:
            parts.extend([f"--tracker.{name}", str(value)])
    return " ".join(parts)


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        print(
            "usage: tracker_flags.py TRACKER [PARAMS.json|-]\n"
            "  Omit PARAMS or pass '-' to use library default hyperparameters.",
            file=sys.stderr,
        )
        return 1

    tracker_id = sys.argv[1]
    params_path = sys.argv[2] if len(sys.argv) == 3 else "-"
    if params_path in {"-", "defaults", ""}:
        params: dict = {}
    else:
        params = json.loads(Path(params_path).read_text())

    try:
        print(tracker_flags(tracker_id, params))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
