# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from trackers.eval.mot_classes import (
    MOT_CLASS_PRESETS,
    MOTClassConfig,
    resolve_mot_class_config,
)
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
        "distractor_class",
        [
            pytest.param(2, id="person_on_vehicle"),
            pytest.param(7, id="static_person"),
            pytest.param(8, id="distractor"),
            pytest.param(12, id="reflection"),
        ],
    )
    def test_all_distractor_classes_excluded(self, distractor_class: int) -> None:
        """Every class in _DISTRACTOR_CLASSES must be excluded from GT and suppress an overlapping tracker detection."""
        ground_truth = {1: _frame([1], [[0, 0, 10, 10]], [1.0], [distractor_class])}
        tracker = {1: _frame([10], [[0, 0, 10, 10]], [1.0], [1])}

        sequence = _prepare_mot_sequence(ground_truth, tracker)

        assert sequence.num_gt_dets == 0
        assert sequence.num_tracker_dets == 0

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

    def test_mot20_treats_non_mot_vehicle_as_distractor(self) -> None:
        """Class 6 (non_mot_vehicle) is a distractor under MOT20 and suppresses overlapping detections."""
        ground_truth = {1: _frame([1], [[0, 0, 10, 10]], [1.0], [6])}
        tracker = {1: _frame([10], [[0, 0, 10, 10]], [1.0], [1])}

        # Under default / mot17: class 6 is not a distractor, so tracker det is NOT suppressed
        seq_mot17 = _prepare_mot_sequence(ground_truth, tracker, class_config="mot17")
        assert seq_mot17.num_gt_dets == 0  # not scored GT
        assert seq_mot17.num_tracker_dets == 1  # kept, not suppressed

        # Under mot20: class 6 is a distractor and suppresses the tracker detection
        seq_mot20 = _prepare_mot_sequence(ground_truth, tracker, class_config="mot20")
        assert seq_mot20.num_gt_dets == 0
        assert seq_mot20.num_tracker_dets == 0  # suppressed

    def test_caller_supplied_class_config_overrides_presets(self) -> None:
        """A user-supplied MOTClassConfig overrides presets for scored and distractor classes."""
        custom_config = MOTClassConfig(scored_classes=(3,), distractor_classes=(4,))

        ground_truth = {
            1: _frame(
                ids=[1, 2, 3],
                boxes=[[0, 0, 10, 10], [50, 50, 10, 10], [100, 100, 10, 10]],
                confidences=[1.0, 1.0, 1.0],
                classes=[3, 4, 1],  # scored, distractor, former pedestrian (now neither)
            )
        }
        tracker = {
            1: _frame(
                ids=[10, 20, 30],
                boxes=[[0, 0, 10, 10], [50, 50, 10, 10], [100, 100, 10, 10]],
                confidences=[1.0, 1.0, 1.0],
                classes=[1, 1, 1],
            )
        }

        seq = _prepare_mot_sequence(ground_truth, tracker, class_config=custom_config)

        # Only class 3 is scored
        assert seq.num_gt_dets == 1
        assert set(seq.gt_id_mapping) == {1}
        # Tracker detection 20 overlaps distractor class 4 and is suppressed;
        # det 10 (TP on class 3) and det 30 (FP on non-distractor class 1) survive.
        assert seq.num_tracker_dets == 2
        suppressed_mapped = seq.tracker_id_mapping[20]
        surviving = set(seq.tracker_ids[0].tolist())
        assert suppressed_mapped not in surviving
        assert seq.tracker_id_mapping[10] in surviving
        assert seq.tracker_id_mapping[30] in surviving


class TestMOTClassConfigResolution:
    """Validation and resolution of MOTClassConfig and presets."""

    def test_resolve_presets(self) -> None:
        assert resolve_mot_class_config("mot17") == MOT_CLASS_PRESETS["mot17"]
        assert resolve_mot_class_config("mot20") == MOT_CLASS_PRESETS["mot20"]
        assert resolve_mot_class_config(None) == MOT_CLASS_PRESETS["mot17"]

    def test_resolve_instance(self) -> None:
        cfg = MOTClassConfig(scored_classes=(1, 2), distractor_classes=(3,))
        assert resolve_mot_class_config(cfg) is cfg

    def test_resolve_dict(self) -> None:
        cfg = resolve_mot_class_config({"scored_classes": [1], "distractor_classes": [2, 6]})
        assert cfg.scored_classes == (1,)
        assert cfg.distractor_classes == (2, 6)

    def test_resolve_unknown_preset_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown MOT class preset: 'mot99'"):
            resolve_mot_class_config("mot99")  # type: ignore[arg-type]

    def test_resolve_invalid_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Expected MOTClassPreset or MOTClassConfig"):
            resolve_mot_class_config(12345)  # type: ignore[arg-type]
