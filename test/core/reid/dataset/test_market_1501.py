from unittest.mock import patch

import pytest

from trackers.core.reid.dataset.market_1501 import parse_market1501_triplet_mapping


@pytest.mark.parametrize(
    "mock_glob_output, expected_result",
    [
        (
            # Empty dataset
            [],
            {},
        ),
        (
            # Single image for one person
            ["0111_00000000.jpg"],
            {"0111": ["0111_00000000.jpg"]},
        ),
        (
            # Multiple images for one person
            ["0111_00000000.jpg", "0111_00000001.jpg"],
            {"0111": ["0111_00000000.jpg", "0111_00000001.jpg"]},
        ),
        (
            # Multiple people with multiple images
            [
                "0111_00000000.jpg",
                "0111_00000001.jpg",
                "0112_00000000.jpg",
                "0112_00000001.jpg",
            ],
            {
                "0111": ["0111_00000000.jpg", "0111_00000001.jpg"],
                "0112": ["0112_00000000.jpg", "0112_00000001.jpg"],
            },
        ),
        (
            # Multiple people with varying number of images
            ["0111_00000000.jpg", "0111_00000001.jpg", "0112_00000000.jpg"],
            {
                "0111": ["0111_00000000.jpg", "0111_00000001.jpg"],
                "0112": ["0112_00000000.jpg"],
            },
        ),
    ],
)
def test_parse_market1501_dataset(mock_glob_output, expected_result):
    with patch("glob.glob", return_value=mock_glob_output):
        result = parse_market1501_triplet_mapping("dummy_path")
        assert result == expected_result
