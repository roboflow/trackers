# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for MOT class configuration validation, presets, and preset resolution."""

from __future__ import annotations

import re
from typing import Any, get_args

import numpy as np
import pytest

from trackers.eval.mot_classes import MOT_CLASS_PRESETS, MOTClassConfig, MOTClassPreset, resolve_mot_class_config

_MOT17_CONFIG = MOTClassConfig(scored_classes=(1,), distractor_classes=(2, 7, 8, 12))
_MOT20_CONFIG = MOTClassConfig(scored_classes=(1,), distractor_classes=(2, 6, 7, 8, 12))


class TestMOTClassConfig:
    """`MOTClassConfig` normalizes valid input and rejects contradictory configurations."""

    def test_defaults_match_trackeval_mot17(self) -> None:
        """The default configuration scores pedestrians and uses TrackEval's MOT17 distractor classes."""
        config = MOTClassConfig()

        assert config.scored_classes == (1,)
        assert config.distractor_classes == (2, 7, 8, 12)

    def test_normalizes_lists_and_numpy_ints_to_int_tuples(self) -> None:
        """List input and NumPy integers are stored as tuples of plain Python ints."""
        # the field annotation stays `tuple[int, ...]`; coercion of other iterables is a runtime convenience
        config = MOTClassConfig(
            scored_classes=[np.int64(1), 3],  # type: ignore[arg-type]
            distractor_classes=np.array([2, 8]),  # type: ignore[arg-type]
        )

        assert config.scored_classes == (1, 3)
        assert config.distractor_classes == (2, 8)
        assert all(type(class_id) is int for class_id in config.scored_classes + config.distractor_classes)

    def test_list_input_stays_hashable_and_equal_to_tuple_input(self) -> None:
        """A config built from lists equals and hashes like one built from tuples, so it works as a dict key."""
        from_lists = MOTClassConfig(scored_classes=[1], distractor_classes=[2, 7, 8, 12])  # type: ignore[arg-type]

        assert from_lists == MOTClassConfig()
        assert hash(from_lists) == hash(MOTClassConfig())

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"distractor_classes": ()}, id="empty-distractors"),
            pytest.param({"scored_classes": (1, 6), "distractor_classes": (2,)}, id="multiple-scored"),
        ],
    )
    def test_accepts_valid_variants(self, kwargs: dict[str, Any]) -> None:
        """Empty distractor sets and several scored classes are valid configurations."""
        config = MOTClassConfig(**kwargs)

        assert set(config.scored_classes).isdisjoint(config.distractor_classes)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            pytest.param({"scored_classes": ()}, "at least one class ID", id="empty-scored"),
            pytest.param({"scored_classes": (1,), "distractor_classes": (1, 2)}, r"\[1\]", id="overlap"),
            pytest.param({"scored_classes": (1, 7)}, r"\[7\]", id="overlap-with-default-distractor"),
            pytest.param({"scored_classes": "1"}, "iterable of integer class IDs", id="string-scored"),
            pytest.param({"scored_classes": None}, "iterable of integer class IDs", id="none-scored"),
            pytest.param({"scored_classes": b"1"}, "iterable of integer class IDs", id="bytes-scored"),
            pytest.param({"scored_classes": (1.0,)}, "only integer class IDs", id="float-entry"),
            pytest.param({"scored_classes": (True,)}, "only integer class IDs", id="bool-entry"),
            pytest.param({"distractor_classes": ("2",)}, "only integer class IDs", id="string-entry"),
            pytest.param({"distractor_classes": 8}, "iterable of integer class IDs", id="scalar-distractor"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict[str, Any], message: str) -> None:
        """Contradictory or non-integer class sets raise instead of silently producing wrong metrics."""
        with pytest.raises(ValueError, match=message):
            MOTClassConfig(**kwargs)


class TestMOTClassPresets:
    """The preset registry is complete, read-only, and holds the documented class sets."""

    def test_registry_names_match_preset_literal(self) -> None:
        """Every name in the `MOTClassPreset` literal has a registry entry and vice versa."""
        assert set(MOT_CLASS_PRESETS) == set(get_args(MOTClassPreset))

    @pytest.mark.parametrize(
        ("preset", "expected"),
        [
            pytest.param("mot17", _MOT17_CONFIG, id="mot17"),
            pytest.param("mot20", _MOT20_CONFIG, id="mot20"),
        ],
    )
    def test_preset_contents(self, preset: MOTClassPreset, expected: MOTClassConfig) -> None:
        """Each preset holds exactly the class sets TrackEval uses for that benchmark."""
        assert MOT_CLASS_PRESETS[preset] == expected

    def test_registry_rejects_mutation(self) -> None:
        """Mutating the registry raises, so a caller cannot change the default preset for the whole process."""
        with pytest.raises(TypeError):
            MOT_CLASS_PRESETS["mot17"] = MOTClassConfig(distractor_classes=())  # type: ignore[index]


class TestResolveMOTClassConfig:
    """`resolve_mot_class_config` maps preset names to configs and passes configs through."""

    @pytest.mark.parametrize(
        ("name", "expected_preset"),
        [
            pytest.param("mot17", "mot17", id="mot17"),
            pytest.param("mot20", "mot20", id="mot20"),
            pytest.param("MOT20", "mot20", id="uppercase"),
            pytest.param("Mot17", "mot17", id="mixed-case"),
            pytest.param("  mot20 ", "mot20", id="surrounding-whitespace"),
        ],
    )
    def test_preset_name_resolves_to_registry_entry(self, name: str, expected_preset: MOTClassPreset) -> None:
        """Preset names resolve case-insensitively and ignore surrounding whitespace."""
        resolved = resolve_mot_class_config(name)  # type: ignore[arg-type]

        assert resolved == MOT_CLASS_PRESETS[expected_preset]

    def test_config_instance_passes_through_unchanged(self) -> None:
        """An explicit config is returned as the same object, not copied or re-resolved."""
        config = MOTClassConfig(scored_classes=(1, 6), distractor_classes=())

        assert resolve_mot_class_config(config) is config

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            pytest.param("mot21", "'mot21'", id="unknown-name"),
            pytest.param("", "''", id="empty-string"),
            pytest.param("mot 17", "'mot 17'", id="inner-whitespace"),
            pytest.param(None, "NoneType", id="none"),
            pytest.param(17, "int", id="int"),
            pytest.param(["mot17"], "list", id="list"),
            pytest.param({}, "dict", id="dict"),
            pytest.param(b"mot17", "bytes", id="bytes"),
        ],
    )
    def test_rejects_unsupported_input_with_value_error(self, value: Any, message: str) -> None:
        """Anything that is neither a preset name nor a config raises `ValueError` naming the offending value."""
        with pytest.raises(ValueError, match=re.escape(message)):
            resolve_mot_class_config(value)

    def test_unknown_preset_error_lists_supported_presets(self) -> None:
        """The error for an unknown preset name lists every supported preset."""
        with pytest.raises(ValueError, match=r"mot17.*mot20"):
            resolve_mot_class_config("mot21")  # type: ignore[arg-type]
