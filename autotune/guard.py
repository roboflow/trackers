# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Regression guard: ensures no tracker's HOTA drops >0.5% from the stored best."""

import json
import re
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from optimize_tracking import main as optimize_tracking_main


def _run_search(tracker_name: str) -> str:
    """Run tracker optimization and return captured stdout."""
    stdout_buffer = StringIO()
    with redirect_stdout(stdout_buffer):
        optimize_tracking_main(tracker=tracker_name, det_source="sdp", n_trials=500)
    return stdout_buffer.getvalue()


best = json.loads((Path(__file__).parent / "best_config.json").read_text())
failed = []

for t in ["bytetrack", "sort", "ocsort"]:
    out = _run_search(t)
    m = re.search(r"HOTA=([0-9.]+)", out)
    if not m:
        failed.append(f"{t}: no HOTA output")
        continue
    hota, base = float(m.group(1)), best[t]["sdp"]["hota"]
    drop = (base - hota) / base * 100
    print(f"{t}: HOTA={hota:.3f} vs best={base:.3f} ({drop:+.2f}%)")
    if drop > 0.5:
        failed.append(f"{t}: regressed {drop:.2f}%")

if failed:
    print("GUARD FAILED:", "; ".join(failed))
    sys.exit(1)
