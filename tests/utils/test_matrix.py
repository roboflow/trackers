# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for ``trackers.utils.matrix``."""

from __future__ import annotations

import numpy as np
import pytest

from trackers.utils.matrix import _diagonal_matrix


class TestDiagonalMatrix:
    """Regression coverage for ``_diagonal_matrix`` (PR #578 review, findings T1-T3).

    The Q/R/P assertions in the BoT-SORT and McByte tracklet tests build their expected value by calling the production
    method under test, so a stride, offset, or ordering error in ``_diagonal_matrix`` would appear identically on both
    sides of the comparison and cancel. These tests build the expected matrix independently of the tracklet's noise-
    construction methods.
    """

    @pytest.mark.parametrize(
        "size",
        [
            pytest.param(1, id="stride-boundary"),
            pytest.param(4, id="measurement-noise-size"),
            pytest.param(7, id="xcycsr-process-noise-size"),
            pytest.param(8, id="default-process-noise-size"),
        ],
    )
    def test_matches_np_diag(self, size: int) -> None:
        """Output equals ``np.diag`` on the same 1-D input, at every runtime size.

        ``np.diag`` is NumPy's own trusted diagonal constructor, independent of ``_diagonal_matrix``'s hand-rolled
        stride write, so a divergence here is a real defect rather than an error shared with the assertion.
        """
        values = [float(i + 1) for i in range(size)]

        result = _diagonal_matrix(values)

        np.testing.assert_array_equal(result, np.diag(values))
        assert result.dtype == np.float64

    def test_off_diagonal_entries_are_zero(self) -> None:
        """Off-diagonal positions stay zero at the default 8x8 process-noise size.

        The stride trick ``matrix.flat[::size + 1]`` is correctness-critical at
        this width; a wrong stride would leave nonzero off-diagonal noise that
        ``test_matches_np_diag`` and ``test_diagonal_values_preserve_input_order``
        would not catch, since both check only the diagonal's own values.
        """
        values = [float(i + 1) for i in range(8)]

        result = _diagonal_matrix(values)

        assert np.count_nonzero(result - np.diag(np.diag(result))) == 0

    def test_diagonal_values_preserve_input_order(self) -> None:
        """Diagonal entries land in the same order as the input list, unreversed.

        Expected values are written independently of both ``_diagonal_matrix`` and ``np.diag`` via direct fancy
        indexing, so a reversed or shuffled stride write would be caught rather than masked by a shared construction
        path.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
        expected = np.zeros((7, 7), dtype=np.float64)
        expected[np.arange(7), np.arange(7)] = values

        result = _diagonal_matrix(values)

        np.testing.assert_array_equal(result, expected)
