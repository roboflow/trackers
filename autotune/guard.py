"""Regression guard: ensures no tracker's HOTA drops >0.5% from the stored best."""

import json
import re
import subprocess
import sys

best = json.load(open("best_config.json"))
failed = []

for t in ["bytetrack", "sort", "ocsort"]:
    out = subprocess.run(
        ["uv", "run", "python", "optimize_tracking.py", t, "sdp", "--n-trials", "500"],
        capture_output=True,
        text=True,
    ).stdout
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
