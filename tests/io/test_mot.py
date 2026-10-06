# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from trackers.eval import MOTClassConfig
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


# one class-6 (non_mot_vehicle) GT box and one pedestrian tracker detection on the same region
_CLASS_6_GROUND_TRUTH = {1: _frame([1], [[0, 0, 10, 10]], [1.0], [6])}
_OVERLAPPING_TRACKER = {1: _frame([10], [[0, 0, 10, 10]], [1.0], [1])}


class TestMotDistractorPreprocessing:
    """GT preprocessing must follow TrackEval's class-based distractor handling.

    TrackEval (`mot_challenge_2d_box.py`) scores only the pedestrian class (1) and drops tracker detections that best-
    match a distractor-class region `{2, 7, 8, 12}`. These cases never appear in the single-class SportsMOT / DanceTrack
    integration fixtures, so they are covered here directly.
    """

    def test_distractor_class_excluded_and_matching_tracker_removed(self) -> None:
        """A distractor-class GT (conf==1) must not be scored as GT, and a tracker detection overlapping it must be
        removed rather than counted FP.

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
        ("class_config", "distractor_class"),
        [
            pytest.param("mot17", 2, id="mot17-person_on_vehicle"),
            pytest.param("mot17", 7, id="mot17-static_person"),
            pytest.param("mot17", 8, id="mot17-distractor"),
            pytest.param("mot17", 12, id="mot17-reflection"),
            pytest.param("mot20", 2, id="mot20-person_on_vehicle"),
            pytest.param("mot20", 6, id="mot20-non_mot_vehicle"),
            pytest.param("mot20", 7, id="mot20-static_person"),
            pytest.param("mot20", 8, id="mot20-distractor"),
            pytest.param("mot20", 12, id="mot20-reflection"),
        ],
    )
    def test_all_distractor_classes_excluded(self, class_config: str, distractor_class: int) -> None:
        """Every distractor class of a preset is excluded from GT and suppresses an overlapping tracker detection.

        MOT20 adds class 6 to the MOT17 distractor set, so each preset is exercised over its own full class list; a
        preset that silently dropped or gained a class fails here instead of passing on the default alone.
        """
        ground_truth = {1: _frame([1], [[0, 0, 10, 10]], [1.0], [distractor_class])}
        tracker = {1: _frame([10], [[0, 0, 10, 10]], [1.0], [1])}

        sequence = _prepare_mot_sequence(ground_truth, tracker, class_config=class_config)  # type: ignore[arg-type]

        assert sequence.num_gt_dets == 0
        assert sequence.num_tracker_dets == 0

    def test_mot20_distractor_status_follows_class_not_confidence(self) -> None:
        """A class-6 GT row marked ignored (conf=0) still suppresses an overlapping tracker detection under MOT20.

        The distractor mask is class-based, so the confidence flag decides only whether a row is scored, never whether
        it is a distractor.
        """
        ground_truth = {1: _frame([1], [[0, 0, 10, 10]], [0.0], [6])}
        tracker = {1: _frame([10], [[0, 0, 10, 10]], [1.0], [1])}

        sequence = _prepare_mot_sequence(ground_truth, tracker, class_config="mot20")

        assert sequence.num_gt_dets == 0
        assert sequence.num_tracker_dets == 0

    @pytest.mark.parametrize(
        ("class_config", "expected_tracker_dets"),
        [
            pytest.param("mot17", 1, id="mot17-keeps-tracker-detection"),
            pytest.param("mot20", 0, id="mot20-suppresses-tracker-detection"),
        ],
    )
    def test_class_6_is_a_distractor_only_under_mot20(self, class_config: str, expected_tracker_dets: int) -> None:
        """A tracker detection over a class-6 GT row is kept under MOT17 and suppressed under MOT20."""
        sequence = _prepare_mot_sequence(_CLASS_6_GROUND_TRUTH, _OVERLAPPING_TRACKER, class_config=class_config)  # type: ignore[arg-type]

        assert sequence.num_gt_dets == 0
        assert sequence.num_tracker_dets == expected_tracker_dets

    def test_custom_class_config_overrides_presets(self) -> None:
        """A caller-supplied configuration controls class-6 handling directly."""
        class_config = MOTClassConfig(distractor_classes=(), scored_classes=(6,))

        sequence = _prepare_mot_sequence(_CLASS_6_GROUND_TRUTH, _OVERLAPPING_TRACKER, class_config=class_config)

        assert sequence.num_gt_dets == 1
        assert sequence.num_tracker_dets == 1

    def test_custom_scored_class_builds_id_mapping_from_that_class(self) -> None:
        """Scoring class 6 instead of pedestrians maps only the class-6 GT track to a 0-indexed ID."""
        ground_truth = {1: _frame([1, 2], [[0, 0, 10, 10], [50, 50, 10, 10]], [1.0, 1.0], [6, 1])}
        tracker = {1: _frame([10], [[0, 0, 10, 10]], [1.0], [1])}
        class_config = MOTClassConfig(scored_classes=(6,))

        sequence = _prepare_mot_sequence(ground_truth, tracker, class_config=class_config)

        assert sequence.gt_id_mapping == {1: 0}
        assert sequence.num_gt_ids == 1
        assert [ids.tolist() for ids in sequence.gt_ids] == [[0]]

    def test_default_config_leaves_class_6_track_out_of_id_mapping(self) -> None:
        """Under the default config a class-6 GT track gets no ID and the remaining IDs stay contiguous."""
        ground_truth = {
            1: _frame([1, 2, 3], [[0, 0, 10, 10], [50, 50, 10, 10], [100, 100, 10, 10]], [1.0, 1.0, 1.0], [1, 6, 1])
        }
        tracker = {1: _frame([10], [[0, 0, 10, 10]], [1.0], [1])}

        sequence = _prepare_mot_sequence(ground_truth, tracker)

        assert sequence.gt_id_mapping == {1: 0, 3: 1}
        assert sequence.num_gt_ids == 2
        assert [ids.tolist() for ids in sequence.gt_ids] == [[0, 1]]

    def test_multiple_scored_classes_count_every_matching_row(self) -> None:
        """Scoring classes 1 and 6 together counts both rows and maps both GT tracks."""
        ground_truth = {1: _frame([1, 2], [[0, 0, 10, 10], [50, 50, 10, 10]], [1.0, 1.0], [1, 6])}
        tracker = {1: _frame([10, 20], [[0, 0, 10, 10], [50, 50, 10, 10]], [1.0, 1.0], [1, 1])}
        class_config = MOTClassConfig(scored_classes=(1, 6))

        sequence = _prepare_mot_sequence(ground_truth, tracker, class_config=class_config)

        assert sequence.num_gt_dets == 2
        assert sequence.num_gt_ids == 2
        assert sequence.gt_id_mapping == {1: 0, 2: 1}

    @pytest.mark.parametrize(
        "class_config",
        [
            pytest.param("mot17", id="mot17"),
            pytest.param("mot20", id="mot20"),
            pytest.param(MOTClassConfig(scored_classes=(6,)), id="custom-scored-class-6"),
        ],
    )
    def test_tracker_id_mapping_does_not_depend_on_class_config(self, class_config: str | MOTClassConfig) -> None:
        """Tracker IDs are mapped before distractor suppression, so the class config never changes the mapping."""
        ground_truth = {1: _frame([1, 2], [[0, 0, 10, 10], [50, 50, 10, 10]], [1.0, 1.0], [1, 6])}
        tracker = {1: _frame([10, 20], [[0, 0, 10, 10], [50, 50, 10, 10]], [1.0, 1.0], [1, 1])}

        sequence = _prepare_mot_sequence(ground_truth, tracker, class_config=class_config)  # type: ignore[arg-type]

        assert sequence.tracker_id_mapping == {10: 0, 20: 1}
        assert sequence.num_tracker_ids == 2

    def test_ignored_non_distractor_gt_does_not_suppress_tracker(self) -> None:
        """GT (conf=0, non-distractor class) is neither scored GT nor a distractor.

        A tracker detection overlapping it must be kept, not suppressed.  Before this PR the old `~valid_mask` would
        have included such rows in the distractor mask; the new class-based mask must NOT.
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
        """SportsMOT / DanceTrack-style data (all pedestrian, conf==1) must be passed through unchanged, so existing
        parity is preserved."""
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
