# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for transition matrix / ``Q`` builders in ``motion_models.py``."""

from __future__ import annotations

import numpy as np
import pytest

from trackers.utils.kalman_filter import KalmanFilter
from trackers.utils.motion_models import (
    KalmanMotionModel,
    ScalableProcessNoise,
    constant_velocity_transition_matrix,
)

POS_1D = np.array([0], dtype=np.int64)
VEL_1D = np.array([1], dtype=np.int64)
POS_4D = np.array([0, 1, 2, 3], dtype=np.int64)
VEL_4D = np.array([4, 5, 6, 7], dtype=np.int64)
POS_XCYCSR = np.array([0, 1, 2], dtype=np.int64)
VEL_XCYCSR = np.array([4, 5, 6], dtype=np.int64)


def test_transition_mtx_scales_velocity_coupling_with_frame_step() -> None:
    """Velocity coupling in the transition matrix scales linearly with frame_step."""
    mtx1 = constant_velocity_transition_matrix(8, POS_4D, VEL_4D, 1.0)
    mtx2 = constant_velocity_transition_matrix(8, POS_4D, VEL_4D, 2.0)
    mtx_half = constant_velocity_transition_matrix(8, POS_4D, VEL_4D, 0.5)

    for v in VEL_4D:
        for j in range(8):
            assert mtx1[v, j] == mtx2[v, j] == mtx_half[v, j]

    for p, v in zip(POS_4D, VEL_4D):
        assert mtx1[p, v] == pytest.approx(1.0)
        assert mtx2[p, v] == pytest.approx(2.0)
        assert mtx_half[p, v] == pytest.approx(0.5)


def test_dwna_gap_noise_scales_with_frame_step_polynomial() -> None:
    gap = ScalableProcessNoise(
        dim_x=8,
        pos_idx=POS_4D,
        vel_idx=VEL_4D,
        baseline_Q=np.eye(8, dtype=np.float64) * 0.01,
        sigma_a2=np.ones(4, dtype=np.float64) * 0.01,
        extra_q_diagonal=np.ones(8, dtype=np.float64) * 0.01,
    )
    # Compare two gap steps (doubling) so both go through DWNA, not the baseline.
    Q1 = gap.build_Q(2.0)
    Q2 = gap.build_Q(4.0)

    for p, v in zip(POS_4D, VEL_4D):
        assert Q2[v, v] == pytest.approx(Q1[v, v] * 4.0)
        assert Q2[p, p] == pytest.approx(Q1[p, p] * 16.0)
        assert Q2[p, v] == pytest.approx(Q1[p, v] * 8.0)
        assert Q2[v, p] == pytest.approx(Q1[v, p] * 8.0)


def test_dwna_gap_noise_preserves_non_kinematic_diagonal() -> None:
    extra = np.ones(7, dtype=np.float64) * 0.01
    extra[3] = 7.5
    gap = ScalableProcessNoise(
        dim_x=7,
        pos_idx=POS_XCYCSR,
        vel_idx=VEL_XCYCSR,
        baseline_Q=np.diag(extra),
        sigma_a2=np.ones(3, dtype=np.float64) * 0.01,
        extra_q_diagonal=extra,
    )
    # Non-unit steps exercise the DWNA path; the non-kinematic diagonal is fixed.
    for frame_step in (0.5, 2.0):
        Q_built = gap.build_Q(frame_step)
        assert Q_built[3, 3] == pytest.approx(7.5), f"Q[3,3] not preserved at frame_step={frame_step}"


def test_build_Q_at_frame_step_1_returns_baseline() -> None:
    """``build_Q(1.0)`` returns the configured baseline Q unchanged (backward-compat gate)."""
    baseline = np.eye(8, dtype=np.float64) * 0.01
    gap = ScalableProcessNoise(
        dim_x=8,
        pos_idx=POS_4D,
        vel_idx=VEL_4D,
        baseline_Q=baseline.copy(),
        sigma_a2=np.ones(4, dtype=np.float64) * 0.01,
        extra_q_diagonal=np.ones(8, dtype=np.float64) * 0.01,
    )
    q1 = gap.build_Q(1.0)
    np.testing.assert_array_equal(q1, baseline)


def test_sync_preserves_configured_q_at_unit_frame_step() -> None:
    """At frame_step=1.0, kf.process_noise must equal the configured Q exactly (backward-compat gate)."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    custom_Q = np.diag([0.01, 0.02])
    kf.process_noise = custom_Q.copy()
    model = KalmanMotionModel.from_filter(kf, POS_1D, VEL_1D)
    model.calibrate_from_process_noise(custom_Q)

    model.apply(kf, 1.0)

    # frame_step=1.0 returns the hand-tuned Q unchanged — no DWNA rescaling.
    np.testing.assert_array_equal(kf.process_noise, custom_Q)


def test_sync_gap_scales_q_relative_to_unit_frame_step() -> None:
    """Gap step must scale velocity variance as ``dt^2`` relative to ``frame_step = 1``."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    custom_Q = np.diag([0.01, 0.02])
    kf.process_noise = custom_Q.copy()
    model = KalmanMotionModel.from_filter(kf, POS_1D, VEL_1D)
    model.calibrate_from_process_noise(custom_Q)

    model.apply(kf, 1.0)
    q_unit = kf.process_noise.copy()
    model.apply(kf, 3.0)

    assert kf.process_noise[1, 1] == pytest.approx(custom_Q[1, 1] * 9.0)
    model.apply(kf, 1.0)
    # Returning to frame_step=1 restores the configured baseline Q exactly.
    np.testing.assert_allclose(kf.process_noise, q_unit, rtol=1e-12)
