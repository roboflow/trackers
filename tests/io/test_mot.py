# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from trackers.io.mot import _MOTFrameData, _prepare_mot_sequence


def _frame(
    ids: list[int],
    boxes: list[list[float]],
    confidences: list[float],
    classes: list[int],
) -> _MOTFrameData:
    """Build a single-frame `_MOTFrameData` with xywh boxes."""
    return _MOTFrameData(
        ids=np.array(ids, dtype=np.intp),
        boxes=np.array(boxes, dtype=np.float64),
        confidences=np.array(confidences, dtype=np.float64),
        classes=np.array(classes, dtype=np.intp),
    )


class TestMotDistractorPreprocessing:
    """GT preprocessing must follow TrackEval's class-based distractor handling.

    TrackEval (`mot_challenge_2d_box.py`) scores only the pedestrian class (1)
    and drops tracker detections that best-match a distractor-class region
    `{2, 7, 8, 12}`. These cases never appear in the single-class SportsMOT /
    DanceTrack integration fixtures, so they are covered here directly.
    """

    def test_distractor_class_excluded_and_matching_tracker_removed(self) -> None:
        """A distractor-class GT (conf==1) must not be scored as GT, and a
        tracker detection overlapping it must be removed rather than counted FP.
        A separate ignored pedestrian (conf==0) must also be dropped from GT.
        """
        ground_truth = {
            1: _frame(
                ids=[1, 2, 3],
                boxes=[[0, 0, 10, 10], [100, 100, 10, 10], [200, 200, 10, 10]],
                confidences=[1.0, 1.0, 0.0],
                classes=[1, 8, 1],  # pedestrian, distractor, ignored pedestrian
            )
        }
        tracker = {
            1: _frame(
                ids=[10, 20, 30],
                boxes=[[0, 0, 10, 10], [100, 100, 10, 10], [300, 300, 10, 10]],
                confidences=[1.0, 1.0, 1.0],
                classes=[1, 1, 1],
            )
        }

        sequence = _prepare_mot_sequence(ground_truth, tracker)

        # Only the genuine pedestrian (id 1) is scored as ground truth; the
        # distractor (id 2) and the conf==0 pedestrian (id 3) are excluded.
        assert sequence.num_gt_dets == 1
        assert sequence.num_gt_ids == 1
        assert set(sequence.gt_id_mapping) == {1}

        # Tracker det 20 overlaps the distractor and is dropped from the scored
        # detections; the true positive (10) and the genuine false positive (30)
        # remain. (num_tracker_ids is unaffected: it is built before suppression,
        # and a never-matched id is metric-neutral.)
        assert sequence.num_tracker_dets == 2
        assert len(sequence.tracker_ids[0]) == 2

        # Verify the *correct* detection was suppressed: id20 matched the distractor
        # region and was removed; id10 (TP) and id30 (genuine FP) must survive.
        suppressed_mapped = sequence.tracker_id_mapping[20]
        surviving = set(sequence.tracker_ids[0].tolist())
        assert suppressed_mapped not in surviving
        assert sequence.tracker_id_mapping[10] in surviving
        assert sequence.tracker_id_mapping[30] in surviving

    @pytest.mark.parametrize(
        "distractor_class",
        [
            pytest.param(2, id="person_on_vehicle"),
            pytest.param(7, id="static_person"),
            pytest.param(8, id="distractor"),
            pytest.param(12, id="reflection"),
        ],
    )
    def test_all_distractor_classes_excluded(self, distractor_class: int) -> None:
        """Every class in _DISTRACTOR_CLASSES must be excluded from GT and suppress
        an overlapping tracker detection.
        """
        ground_truth = {1: _frame([1], [[0, 0, 10, 10]], [1.0], [distractor_class])}
        tracker = {1: _frame([10], [[0, 0, 10, 10]], [1.0], [1])}

        sequence = _prepare_mot_sequence(ground_truth, tracker)

        assert sequence.num_gt_dets == 0
        assert sequence.num_tracker_dets == 0

    def test_ignored_non_distractor_gt_does_not_suppress_tracker(self) -> None:
        """GT (conf=0, non-distractor class) is neither scored GT nor a distractor.

        A tracker detection overlapping it must be kept, not suppressed.  Before
        this PR the old `~valid_mask` would have included such rows in the
        distractor mask; the new class-based mask must NOT.
        """
        ground_truth = {
            1: _frame([1], [[0, 0, 10, 10]], [0.0], [5])  # conf=0, class=5 (vehicle)
        }
        tracker = {1: _frame([10], [[0, 0, 10, 10]], [1.0], [1])}

        sequence = _prepare_mot_sequence(ground_truth, tracker)

        assert sequence.num_gt_dets == 0  # conf=0 → not scored GT
        assert sequence.num_tracker_dets == 1  # class 5 is not a distractor → kept

    def test_all_distractor_frame_yields_zero_gt(self) -> None:
        """Frame where every GT row is a distractor class yields zero scored GT."""
        ground_truth = {1: _frame([1, 2], [[0, 0, 10, 10], [50, 50, 10, 10]], [1.0, 1.0], [8, 2])}
        tracker = {1: _frame([10, 20], [[0, 0, 10, 10], [50, 50, 10, 10]], [1.0, 1.0], [1, 1])}

        sequence = _prepare_mot_sequence(ground_truth, tracker)

        assert sequence.num_gt_dets == 0
        assert sequence.num_tracker_dets == 0  # both dets matched to distractors

    def test_empty_gt_frame_no_error(self) -> None:
        """Missing GT frame must produce no error and zero GT dets for that frame."""
        ground_truth: dict[int, _MOTFrameData] = {}
        tracker = {1: _frame([10], [[0, 0, 10, 10]], [1.0], [1])}

        sequence = _prepare_mot_sequence(ground_truth, tracker, num_frames=1)

        assert sequence.num_gt_dets == 0
        assert sequence.num_tracker_dets == 1
        assert sequence.num_frames == 1

    def test_multi_frame_sequence_accumulates_correctly(self) -> None:
        """Multi-frame sequences must accumulate GT/tracker dets across all frames."""
        ground_truth = {
            1: _frame([1], [[0, 0, 10, 10]], [1.0], [1]),
            # frame 2: one scored pedestrian + one ignored pedestrian (conf=0)
            2: _frame([1, 2], [[0, 0, 10, 10], [50, 50, 10, 10]], [1.0, 0.0], [1, 1]),
        }
        tracker = {
            1: _frame([10], [[0, 0, 10, 10]], [1.0], [1]),
            2: _frame([10, 20], [[0, 0, 10, 10], [50, 50, 10, 10]], [1.0, 1.0], [1, 1]),
        }

        sequence = _prepare_mot_sequence(ground_truth, tracker)

        assert sequence.num_frames == 2
        assert sequence.num_gt_dets == 2  # 1 from frame 1, 1 from frame 2 (conf=0 excluded)
        assert sequence.num_tracker_dets == 3  # 1 + 2
        assert len(sequence.gt_ids) == 2
        assert len(sequence.tracker_ids) == 2

    def test_single_class_sequence_unaffected(self) -> None:
        """SportsMOT / DanceTrack-style data (all pedestrian, conf==1) must be
        passed through unchanged, so existing parity is preserved.
        """
        ground_truth = {
            1: _frame([1, 2], [[0, 0, 10, 10], [50, 50, 10, 10]], [1.0, 1.0], [1, 1]),
        }
        tracker = {
            1: _frame([10, 20], [[0, 0, 10, 10], [50, 50, 10, 10]], [1.0, 1.0], [1, 1]),
        }

        sequence = _prepare_mot_sequence(ground_truth, tracker)

        assert sequence.num_gt_dets == 2
        assert sequence.num_tracker_dets == 2
        assert sequence.num_gt_ids == 2
        assert sequence.num_tracker_ids == 2
