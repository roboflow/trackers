# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for tracker auto-registration and parameter extraction."""

from typing import Any, Optional, Union

import pytest

from trackers.core.base import (
    BaseTracker,
    TrackerInfo,
    _extract_params_from_init,
    _normalize_type,
    _parse_docstring_arguments,
)


class TestParseDocstringArguments:
    """Tests for _parse_docstring_arguments function."""

    @pytest.mark.parametrize(
        ("docstring", "expected"),
        [
            # Empty docstring
            ("", {}),
            # No Args section
            (
                """
                Some description without Args section.

                Returns:
                    Something.
                """,
                {},
            ),
            # Simple param_name: description format
            (
                """
                Args:
                    param1: Description of param1.
                """,
                {"param1": "Description of param1."},
            ),
            # Multi-line description
            (
                """
                Args:
                    param1: Description that spans
                        multiple lines here.
                """,
                {"param1": "Description that spans multiple lines here."},
            ),
            # Multiple parameters
            (
                """
                Args:
                    param1: First parameter.
                    param2: Second parameter.
                    param3: Third parameter.
                """,
                {
                    "param1": "First parameter.",
                    "param2": "Second parameter.",
                    "param3": "Third parameter.",
                },
            ),
            # Args section ends at Returns
            (
                """
                Args:
                    param1: Description.

                Returns:
                    Something.
                """,
                {"param1": "Description."},
            ),
            # Real-world style with type in description
            (
                """
                Args:
                    lost_track_buffer: `int` specifying number of frames
                        to buffer when track is lost.
                """,
                {
                    "lost_track_buffer": "`int` specifying number of frames "
                    "to buffer when track is lost."
                },
            ),
            # param (type): format
            (
                """
                Args:
                    param1 (int): First parameter.
                    param2 (str): Second parameter.
                """,
                {"param1": "First parameter.", "param2": "Second parameter."},
            ),
            # param (type, optional): format
            (
                """
                Args:
                    param1 (int, optional): Optional integer.
                """,
                {"param1": "Optional integer."},
            ),
            # Dotted parameter name
            (
                """
                Args:
                    config.threshold: The threshold value for detection.
                    model.weights: Path to model weights file.
                """,
                {
                    "config.threshold": "The threshold value for detection.",
                    "model.weights": "Path to model weights file.",
                },
            ),
            # Description containing colon
            (
                """
                Args:
                    format_str: Use format: key=value for configuration.
                    path: File path, e.g.: /home/user/file.txt
                """,
                {
                    "format_str": "Use format: key=value for configuration.",
                    "path": "File path, e.g.: /home/user/file.txt",
                },
            ),
        ],
    )
    def test_parse_docstring(self, docstring: str, expected: dict[str, str]) -> None:
        result = _parse_docstring_arguments(docstring)
        assert result == expected


class TestNormalizeType:
    """Tests for _normalize_type function."""

    @pytest.mark.parametrize(
        ("annotation", "default", "expected"),
        [
            # Simple types
            (int, None, int),
            (str, None, str),
            (float, None, float),
            # Optional types
            (Optional[int], None, int),
            (Optional[str], None, str),
            # Union with None
            (Union[int, None], None, int),
            (Union[str, None], None, str),
            # List and dict
            (list[int], None, list),
            (dict[str, int], None, dict),
            # Nested optional list
            (Optional[list[int]], None, list),
            # Tuple types
            (tuple[int, ...], None, tuple),
            (tuple[int, str], None, tuple),
            # Any with default value
            (Any, 42, int),
            (Any, "hello", str),
            (Any, None, Any),
        ],
    )
    def test_normalize_type(
        self, annotation: Any, default: Any, expected: type
    ) -> None:
        assert _normalize_type(annotation, default) == expected


class TestExtractParamsFromInit:
    """Tests for _extract_params_from_init function."""

    def test_extract_params_with_defaults(self) -> None:
        class TestClass:
            def __init__(
                self,
                param1: int = 10,
                param2: float = 0.5,
                param3: str = "default",
            ) -> None:
                """
                Args:
                    param1: Integer parameter.
                    param2: Float parameter.
                    param3: String parameter.
                """
                pass

        params = _extract_params_from_init(TestClass)

        assert params["param1"].param_type is int
        assert params["param1"].default_value == 10
        assert "Integer parameter" in params["param1"].description

        assert params["param2"].param_type is float
        assert params["param2"].default_value == 0.5

        assert params["param3"].param_type is str
        assert params["param3"].default_value == "default"

    def test_extract_params_without_docstring(self) -> None:
        class TestClass:
            def __init__(self, param1: int = 10) -> None:
                pass

        params = _extract_params_from_init(TestClass)

        assert params["param1"].param_type is int
        assert params["param1"].default_value == 10
        assert params["param1"].description == ""

    def test_extract_params_infers_type_from_default(self) -> None:
        class TestClass:
            def __init__(self, param1=42) -> None:  # No type hint
                pass

        params = _extract_params_from_init(TestClass)

        assert params["param1"].param_type is int  # Inferred from default
        assert params["param1"].default_value == 42

    def test_excludes_self_parameter(self) -> None:
        class TestClass:
            def __init__(self, param1: int = 10) -> None:
                pass

        params = _extract_params_from_init(TestClass)

        assert "self" not in params
        assert len(params) == 1


class TestTrackerAutoRegistration:
    """Tests for BaseTracker auto-registration."""

    @pytest.mark.parametrize("tracker_id", ["bytetrack", "sort"])
    def test_tracker_is_registered(self, tracker_id: str) -> None:
        from trackers import ByteTrackTracker, SORTTracker  # noqa: F401

        assert tracker_id in BaseTracker._registered_trackers()

    @pytest.mark.parametrize("tracker_id", ["bytetrack", "sort"])
    def test_lookup_tracker(self, tracker_id: str) -> None:
        from trackers import ByteTrackTracker, SORTTracker  # noqa: F401

        info = BaseTracker._lookup_tracker(tracker_id)

        assert info is not None
        assert isinstance(info, TrackerInfo)
        assert "lost_track_buffer" in info.parameters

    def test_lookup_tracker_unknown_returns_none(self) -> None:
        info = BaseTracker._lookup_tracker("nonexistent")
        assert info is None

    def test_registered_trackers_returns_sorted_list(self) -> None:
        from trackers import ByteTrackTracker, SORTTracker  # noqa: F401

        registered = BaseTracker._registered_trackers()

        assert isinstance(registered, list)
        assert registered == sorted(registered)

    @pytest.mark.parametrize("tracker_id", ["bytetrack", "sort"])
    def test_tracker_params_have_descriptions(self, tracker_id: str) -> None:
        info = BaseTracker._lookup_tracker(tracker_id)

        assert info is not None
        has_descriptions = any(p.description for p in info.parameters.values())
        assert has_descriptions


class TestTrackerInstantiation:
    """Tests for instantiating trackers via registry."""

    @pytest.mark.parametrize("tracker_id", ["bytetrack", "sort"])
    def test_instantiate_with_defaults(self, tracker_id: str) -> None:
        info = BaseTracker._lookup_tracker(tracker_id)
        assert info is not None
        tracker = info.tracker_class()

        assert tracker is not None
        assert hasattr(tracker, "update")
        assert hasattr(tracker, "reset")

    def test_instantiate_bytetrack_with_custom_params(self) -> None:
        info = BaseTracker._lookup_tracker("bytetrack")
        assert info is not None
        tracker = info.tracker_class(lost_track_buffer=60, frame_rate=60.0)  # type: ignore[call-arg]

        # Internal calculation: maximum_frames_without_update = 60/30 * 60 = 120
        assert tracker.maximum_frames_without_update == 120  # type: ignore[attr-defined]

    def test_instantiate_with_registry_params(self) -> None:
        """Test creating tracker with params dict like CLI would do."""
        info = BaseTracker._lookup_tracker("sort")
        assert info is not None

        kwargs = {name: param.default_value for name, param in info.parameters.items()}
        kwargs["lost_track_buffer"] = 50

        tracker = info.tracker_class(**kwargs)
        assert tracker is not None
