#!/usr/bin/env python3
"""Walk the expected ``data/`` layout and print what's present vs missing per dataset.

Use this before running ``make benchmark`` to verify the manual data setup. The
README documents where each asset is downloaded from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DATASETS, EVAL_SPLIT, SUBMIT_SPLIT, TUNE_SPLIT, split_paths


def _check(label: str, path: Path | None, *, required: bool) -> bool:
    if path is None:
        print(f"    {label:<10} (n/a)")
        return True
    ok = path.is_dir() or path.is_file()
    marker = "ok" if ok else ("MISS" if required else "skip")
    print(f"    {label:<10} {marker:<5} {path}")
    return ok or not required


def check_dataset(data_root: Path, dataset: str) -> bool:
    splits = sorted({TUNE_SPLIT[dataset], EVAL_SPLIT[dataset], *([SUBMIT_SPLIT[dataset]] if dataset in SUBMIT_SPLIT else [])})
    print(f"\n[{dataset}]")
    ok = True
    for split in splits:
        try:
            paths = split_paths(data_root, dataset, split)
        except ValueError:
            continue
        print(f"  {split}:")
        ok &= _check("dets", paths.det_dir, required=True)
        ok &= _check("gt", paths.gt_dir, required=split != SUBMIT_SPLIT.get(dataset))
        ok &= _check("images", paths.images_dir, required=False)
        ok &= _check("seqmap", paths.seqmap, required=False)
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = p.parse_args(argv)

    print(f"checking data_root = {args.data_root}")
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    all_ok = True
    for dataset in datasets:
        all_ok &= check_dataset(args.data_root, dataset)
    print("\n" + ("All required assets found." if all_ok else "Some required assets missing — see README for download instructions."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
