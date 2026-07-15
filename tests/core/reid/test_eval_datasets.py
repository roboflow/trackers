# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trackers.core.reid.eval.datasets import (
    MARKET1501_GALLERY_JUNK_PIDS,
    ReIDSplit,
    _parse_market_filename,
    _parse_msmt17_camid,
    load_market1501,
    load_msmt17,
)


class TestMarket1501Loader:
    def test_parse_market_filename(self) -> None:
        assert _parse_market_filename("0001_c1s1_001051_00.jpg") == (1, 0)
        assert _parse_market_filename("0000_c1s1_000151_01.jpg") == (0, 0)
        assert _parse_market_filename("-1_c1s1_000151_01.jpg") == (-1, 0)

    def test_load_market1501_from_temp_tree(self, tmp_path: Path) -> None:
        query_dir = tmp_path / "query"
        gallery_dir = tmp_path / "bounding_box_test"
        query_dir.mkdir()
        gallery_dir.mkdir()
        (query_dir / "0001_c1s1_001051_00.jpg").write_bytes(b"jpeg")
        (gallery_dir / "0000_c1s1_000151_01.jpg").write_bytes(b"jpeg")
        (gallery_dir / "0002_c2s1_000851_01.jpg").write_bytes(b"jpeg")

        query, gallery = load_market1501(tmp_path)
        assert len(query) == 1
        assert query.pids.tolist() == [1]
        assert gallery.pids.tolist() == [0, 2]
        assert gallery.gallery_junk_pids == MARKET1501_GALLERY_JUNK_PIDS


class TestMSMT17Loader:
    def test_parse_msmt17_camid(self) -> None:
        assert _parse_msmt17_camid("0001_019_07_0303morning_0020_1.jpg") == 6

    def test_load_msmt17_from_temp_lists(self, tmp_path: Path) -> None:
        test_root = tmp_path / "test"
        test_root.mkdir()
        rel = "0001/0001_019_07_0303morning_0020_1.jpg"
        image_path = test_root / rel
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(b"jpeg")

        (tmp_path / "list_query.txt").write_text(f"{rel} 42\n")
        (tmp_path / "list_gallery.txt").write_text(f"{rel} 42 6\n")

        query, gallery = load_msmt17(tmp_path)
        assert isinstance(query, ReIDSplit)
        assert query.pids.tolist() == [42]
        assert query.camids.tolist() == [6]
        assert gallery.pids.tolist() == [42]
        assert gallery.camids.tolist() == [6]
        assert gallery.gallery_junk_pids == frozenset({-1})
