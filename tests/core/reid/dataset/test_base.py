from contextlib import ExitStack as DoesNotRaise

import pytest

from trackers.core.reid.dataset.base import TripletsDataset


@pytest.mark.parametrize(
    "tracker_id_to_images, exception",
    [
        (
                {"0111": []},
                pytest.raises(ValueError),
        ),  # Single tracker with no images - should raise ValueError
        (
                {"0111": ["0111_00000000.jpg"]},
                pytest.raises(ValueError),
        ),  # Single tracker with one image - should raise ValueError
        (
                {"0111": ["0111_00000000.jpg", "0111_00000001.jpg"]},
                pytest.raises(ValueError),
        ),  # Single tracker with multiple images - should raise ValueError
        (
                {
                    "0111": ["0111_00000000.jpg", "0111_00000001.jpg"],
                    "0112": ["0112_00000000.jpg"],
                },
                pytest.raises(ValueError),
        ),  # Two trackers but one has only one image - should raise ValueError
        (
                {
                    "0111": ["0111_00000000.jpg", "0111_00000001.jpg"],
                    "0112": ["0112_00000000.jpg", "0112_00000001.jpg"],
                },
                DoesNotRaise(),
        ),  # Two trackers with multiple images - should not raise
        (
                {
                    "0111": ["0111_00000000.jpg", "0111_00000001.jpg"],
                    "0112": ["0112_00000000.jpg", "0112_00000001.jpg"],
                    "0113": ["0113_00000000.jpg"],
                },
                DoesNotRaise(),
        ),  # Three trackers, one with fewer images - should validate dataset length
    ],
)
def test_triplet_dataset_initialization(
        tracker_id_to_images, exception
):
    with exception:
        _ = TripletsDataset(tracker_id_to_images)


@pytest.mark.parametrize(
    "tracker_id_to_images, split_ratio, expected_train_size, expected_val_size, exception",
    [
        (
                {
                    "0111": ["0111_00000000.jpg", "0111_00000001.jpg"],
                    "0112": ["0112_00000000.jpg", "0112_00000001.jpg"],
                },
                0.5,
                1,
                1,
                pytest.raises(ValueError),
        ),  # Split results in only 1 tracker in test set - should raise ValueError
        (
                {
                    "0111": ["0111_00000000.jpg", "0111_00000001.jpg"],
                    "0112": ["0112_00000000.jpg", "0112_00000001.jpg"],
                    "0113": ["0113_00000000.jpg", "0113_00000001.jpg"],
                    "0114": ["0114_00000000.jpg", "0114_00000001.jpg"],
                    "0115": ["0115_00000000.jpg", "0115_00000001.jpg"],
                },
                0.2,
                1,
                4,
                pytest.raises(ValueError),
        ),  # Split results in only 1 tracker in test set - should raise ValueError
        (
                {
                    "0111": ["0111_00000000.jpg", "0111_00000001.jpg"],
                    "0112": ["0112_00000000.jpg", "0112_00000001.jpg"],
                    "0113": ["0113_00000000.jpg", "0113_00000001.jpg"],
                    "0114": ["0114_00000000.jpg", "0114_00000001.jpg"],
                    "0115": ["0115_00000000.jpg", "0115_00000001.jpg"],
                },
                0.8,
                4,
                1,
                pytest.raises(ValueError),
        ),  # Split results in only 1 tracker in val set - should raise ValueError
        (
                {
                    "0111": ["0111_00000000.jpg", "0111_00000001.jpg"],
                    "0112": ["0112_00000000.jpg", "0112_00000001.jpg"],
                    "0113": ["0113_00000000.jpg", "0113_00000001.jpg"],
                    "0114": ["0114_00000000.jpg", "0114_00000001.jpg"],
                    "0115": ["0115_00000000.jpg", "0115_00000001.jpg"],
                },
                0.6,
                3,
                2,
                DoesNotRaise(),
        ),  # Valid split with multiple trackers in both sets
        (
                {
                    "0111": ["0111_00000000.jpg", "0111_00000001.jpg"],
                    "0112": ["0112_00000000.jpg", "0112_00000001.jpg"],
                    "0113": ["0113_00000000.jpg", "0113_00000001.jpg"],
                    "0114": ["0114_00000000.jpg", "0114_00000001.jpg"],
                },
                0.5,
                2,
                2,
                DoesNotRaise(),
        ),  # 50% train, 50% validation - valid
    ],
)
def test_triplet_dataset_split(
        tracker_id_to_images, split_ratio, expected_train_size, expected_val_size,
        exception
):
    with exception:
        dataset = TripletsDataset(tracker_id_to_images)
        train_dataset, val_dataset = dataset.split(split_ratio=split_ratio,
                                                   random_state=42)

        assert len(
            train_dataset) == expected_train_size, f"Expected train dataset size {expected_train_size}, got {len(train_dataset)}"
        assert len(
            val_dataset) == expected_val_size, f"Expected validation dataset size {expected_val_size}, got {len(val_dataset)}"
