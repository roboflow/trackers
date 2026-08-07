# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Transitional CLI spellings, kept parsing until 2.10.0.

Everything in this module exists to let a command written against the previous
release keep working, with a ``FutureWarning`` naming its replacement. It is
deliberately one file so the removal is a deletion rather than an archaeology
exercise: at 2.10.0 this module goes, along with the ``_translate_legacy_args``
call in :func:`trackers.cli.__main__.main`.

TODO(v2.10): two passes in :func:`_translate_legacy_args` are NOT deprecations
and must survive the deletion — the ``_normalise_option`` sweep that makes
hyphens and underscores interchangeable, and ``_expand_tracker_shorthand``.
Both live in :mod:`trackers.cli._parser`; move the calls there rather than
dropping them with this file.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import fields

from trackers.cli._parser import (
    _SUBCOMMANDS,
    _expand_tracker_shorthand,
    _normalise_option,
    _raise_for_detection_source_conflict,
    _target_for_option,
)
from trackers.cli.track import TrackerOptions, _abbreviated_tracker_parameters
from trackers.core.base import BaseTracker

# Release that introduced this batch of CLI deprecations, and the one that drops
# them. Both are pinned literals rather than being derived from the installed
# version: a computed deadline moves with every release, so a user upgrading
# 2.7 -> 2.8 -> 2.9 is told 2.10, then 2.11, then 2.12, and the removal the
# warning promises never actually arrives. A later batch of deprecations gets
# its own pair rather than editing these.
_LEGACY_DEPRECATED_IN = "2.7.0"
_LEGACY_REMOVED_IN = "2.10.0"

_DEVELOP_TRACK_ARGUMENTS = {
    "--model": "--detection.model",
    "--detections": "--detection.mot_file",
    "--model.confidence": "--detection.confidence",
    "--model.device": "--detection.device",
    "--model.api_key": "--detection.api_key",
    "--classes": "--filters.classes",
    "--track_ids": "--filters.track_ids",
    "-o": "--output.video",
    "--output": "--output.video",
    "--mot-output": "--output.mot_results",
    "--overwrite": "--output.overwrite",
    "--show-boxes": "--show.boxes",
    "--show-masks": "--show.masks",
    "--show-labels": "--show.labels",
    "--show-ids": "--show.ids",
    "--show-confidence": "--show.confidence",
    "--show-trajectories": "--show.trajectories",
    "--no-boxes": "--show.no_boxes",
    "--no-ids": "--show.no_ids",
}


def _develop_tracker_parameter_arguments() -> dict[str, str]:
    """Return develop's renamed ``--tracker.<param>`` spellings.

    Every entry is derived from ``TrackerOptions``, so a renamed tracker
    parameter cannot leave a stale mapping behind. Two renames are covered:
    unabbreviated names, superseded by the ``min_`` / ``max_`` spellings, and
    ``iou``, which the parser exposes as ``iou_variant``.

    Parameters whose spelling never changed are absent, so they parse as-is
    without a warning.

    Returns:
        Mapping of develop option string to its current replacement.

    Examples:
        >>> _develop_tracker_parameter_arguments()["--tracker.minimum_iou_threshold"]
        '--tracker.min_iou_threshold'
    """
    replacements = {
        f"--tracker.{long_name}": f"--tracker.{short_name}"
        for long_name, short_name in _abbreviated_tracker_parameters().items()
    }
    replacements["--tracker.iou"] = "--tracker.iou_variant"
    return replacements


def _develop_store_false_tracker_flags() -> frozenset[str]:
    """Return tracker options develop exposed as bare "turn this off" flags.

    The bare positive spelling becomes available again once this goes.

    A registry parameter that is a boolean defaulting to ``True`` was a
    ``store_false`` flag on develop, so the bare spelling has to keep meaning
    ``false``. The current parser takes an explicit value for every tracker
    boolean, because the option fields are ``bool | None`` rather than ``bool``.

    Booleans defaulting to ``False`` are excluded: develop's ``store_true`` flag
    and an explicit ``true`` already agree, so no rewrite is owed.

    Returns:
        Option strings whose bare form must be rewritten to an explicit ``false``.
    """
    option_fields = {field.name for field in fields(TrackerOptions)}
    flags = set()
    for tracker_id in BaseTracker._registered_trackers():
        tracker_info = BaseTracker._lookup_tracker(tracker_id)
        if tracker_info is None:
            continue
        for parameter_name, parameter in tracker_info.parameters.items():
            if parameter.param_type is bool and parameter.default_value and parameter_name in option_fields:
                flags.add(f"--tracker.{parameter_name}")
    return frozenset(flags)


_DEVELOP_STORE_FALSE_TRACKER_FLAGS = _develop_store_false_tracker_flags()


def _normalised_options(arguments: dict[str, str]) -> dict[str, str]:
    """Return one deprecation table re-keyed by canonical option spelling.

    Only the keys are touched, and only their separators; what each deprecated
    option maps to is untouched. Lookups normalise before they reach these
    tables, so a key left in the hyphenated spelling a user is migrating from
    would never match. Normalising them makes both spellings of every deprecated
    option resolve, which is the point: someone porting a develop command should
    not also have to guess which separator that particular option wanted.

    Args:
        arguments: Deprecated option spellings mapped to their replacements.

    Returns:
        The same mapping with each key normalised.

    Examples:
        >>> _normalised_options({"--mot-output": "--output.mot_results"})
        {'--mot_output': '--output.mot_results'}
    """
    return {_normalise_option(option): replacement for option, replacement in arguments.items()}


# Deprecated spelling to current spelling, per subcommand. Keys are normalised
# on the way in so a lookup matches whichever separator the user reached for.
_LEGACY_ARGUMENTS = {
    subcommand: _normalised_options(arguments)
    for subcommand, arguments in {
        "track": {**_DEVELOP_TRACK_ARGUMENTS, **_develop_tracker_parameter_arguments()},
        "eval": {
            "-o": "--output",
        },
        "tune": {
            "-o": "--output",
        },
        "download": {
            "--list": "--list_available",
            "--dataset": "--name",
            "-o": "--output",
        },
        # ``mcbyte`` shipped no earlier spelling, so it has nothing to translate.
        # The entry is still required: ``_translate_legacy_args`` subscripts this
        # table for whichever subcommand it finds, and a missing key raises
        # KeyError before any option is looked at.
        "mcbyte": {},
    }.items()
}
_LEGACY_LIST_ARGUMENTS = {
    "eval": frozenset({"--metrics", "--columns"}),
    "tune": frozenset({"--metrics"}),
}
# Track filters that used to take one comma-separated string and now take a
# list, matching the list-valued options of eval and tune.
_COMMA_LIST_TRACK_ARGUMENTS = frozenset({"--filters.classes", "--filters.track_ids"})
# Download options that consume a following token. The scan for a legacy
# positional runs before deprecated names are rewritten, so the superseded
# ``--dataset`` has to be listed alongside its replacement.
_DOWNLOAD_VALUE_OPTIONS = frozenset(
    _normalise_option(option)
    for option in ("--name", "--dataset", "--split", "--asset", "-o", "--output", "--cache-dir")
)


def _warn_legacy_cli(message: str) -> None:
    """Warn about one legacy CLI form with its scheduled removal release."""
    warnings.warn(
        f"{message} It will be removed in {_LEGACY_REMOVED_IN}.",
        FutureWarning,
        stacklevel=3,
    )


def _translate_legacy_args(args: list[str]) -> list[str]:
    """Translate deprecated CLI spellings to their current argument paths.

    The translator runs before jsonargparse sees argv, allowing legacy scalar
    arguments to target fields in the track command's nested option dataclasses.
    """
    subcommand_index = next((index for index, arg in enumerate(args) if arg in _SUBCOMMANDS), None)
    if subcommand_index is None:
        return args

    subcommand = args[subcommand_index]
    command_args = args[subcommand_index + 1 :]
    if subcommand == "track":
        command_args = _expand_tracker_shorthand(command_args)
        command_args = _translate_develop_boolean_flags(command_args)
    command_args = _translate_legacy_list_args(subcommand, command_args)
    legacy_arguments = _LEGACY_ARGUMENTS[subcommand]
    provided_targets = _provided_targets(command_args, legacy_arguments)
    translated = args[: subcommand_index + 1]

    if subcommand == "download":
        command_args = _translate_download_positional(command_args, provided_targets)

    for index, arg in enumerate(command_args):
        if arg == "--":
            translated.extend(command_args[index:])
            break
        option, separator, value = arg.partition("=")
        # Look the deprecated spelling up in canonical form, so that whichever
        # separator the user reached for resolves to the same entry. The warning
        # still quotes what they typed.
        replacement = legacy_arguments.get(_normalise_option(option))
        if replacement is None:
            translated.append(_normalise_option(arg))
            continue

        target = _target_for_option(replacement)
        _raise_for_canonical_conflict(target, option, replacement, provided_targets)
        _warn_legacy_cli(f"{option} is deprecated; use {replacement} instead.")
        translated.append(f"{replacement}{separator}{value}" if separator else replacement)

    if subcommand == "track":
        translated[subcommand_index + 1 :] = _translate_comma_separated_lists(translated[subcommand_index + 1 :])
        _raise_for_detection_source_conflict(translated[subcommand_index + 1 :])
    return translated


def _translate_develop_boolean_flags(args: list[str]) -> list[str]:
    """Rewrite develop's bare tracker boolean flags to an explicit ``false``.

    Only the bare spelling is rewritten. ``--tracker.enable_cmc=false`` and
    ``--tracker.enable_cmc false`` are the current syntax for the same option,
    so they pass through untouched and without a warning; rewriting them would
    append a second value and produce an unparsable argument.

    Args:
        args: Track arguments, without the subcommand token.

    Returns:
        Arguments where every bare develop boolean flag carries an explicit value.

    Examples:
        >>> _translate_develop_boolean_flags(["--tracker.min_iou_threshold", "0.3"])
        ['--tracker.min_iou_threshold', '0.3']
    """
    translated: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            translated.extend(args[index:])
            break

        option, separator, _ = arg.partition("=")
        next_value = args[index + 1] if index + 1 < len(args) else ""
        carries_value = bool(separator) or (next_value != "" and not next_value.startswith("-"))
        if _normalise_option(option) not in _DEVELOP_STORE_FALSE_TRACKER_FLAGS or carries_value:
            translated.append(arg)
            index += 1
            continue

        _warn_legacy_cli(f"{option} is deprecated; use {option}=false instead.")
        translated.append(f"{option}=false")
        index += 1
    return translated


def _translate_comma_separated_lists(args: list[str]) -> list[str]:
    """Translate comma-separated track filter strings to JSON-list values.

    Runs after the legacy option names have been translated, so the develop-era
    ``--classes person,car`` spelling is covered by the same pass that handles
    ``--filters.classes person,car``.

    Args:
        args: Track arguments with canonical option names.

    Returns:
        Arguments where every comma-separated filter value is a JSON list.
    """
    translated: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            translated.extend(args[index:])
            break

        option, separator, inline_value = arg.partition("=")
        if option not in _COMMA_LIST_TRACK_ARGUMENTS:
            translated.append(arg)
            index += 1
            continue

        next_value = args[index + 1] if index + 1 < len(args) else ""
        if not separator and (not next_value or next_value.startswith("-")):
            translated.append(arg)
            index += 1
            continue

        value = inline_value if separator else next_value
        width = 1 if separator else 2
        if value.startswith("["):
            translated.extend(args[index : index + width])
            index += width
            continue

        _warn_legacy_cli(f"comma-separated {option} values are deprecated; use a list instead.")
        translated.extend([option, json.dumps([token.strip() for token in value.split(",") if token.strip()])])
        index += width
    return translated


def _translate_legacy_list_args(subcommand: str, args: list[str]) -> list[str]:
    """Translate argparse's space-separated list syntax to JSON-list values."""
    list_arguments = _LEGACY_LIST_ARGUMENTS.get(subcommand, frozenset())
    if not list_arguments:
        return args

    translated: list[str] = []
    index = 0
    while index < len(args):
        option, separator, first_value = args[index].partition("=")
        if option not in list_arguments:
            translated.append(args[index])
            index += 1
            continue
        if first_value.startswith("["):
            translated.append(args[index])
            index += 1
            continue
        if not separator and index + 1 < len(args) and args[index + 1].startswith("["):
            translated.extend(args[index : index + 2])
            index += 2
            continue

        values = [first_value] if separator else []
        index += 1
        while index < len(args) and not args[index].startswith("-"):
            values.append(args[index])
            index += 1
        if not values:
            translated.append(option)
            continue

        _warn_legacy_cli(f"space-separated {option} values are deprecated; use a JSON list instead.")
        translated.extend([option, json.dumps(values)])
    return translated


def _translate_download_positional(args: list[str], provided_targets: set[str]) -> list[str]:
    """Translate a legacy download dataset positional without touching option values.

    ``--dataset`` drops out of ``_DOWNLOAD_VALUE_OPTIONS`` at the same time.
    """
    translated = list(args)
    expects_value = False
    for index, arg in enumerate(translated):
        if arg == "--":
            break
        if expects_value:
            expects_value = False
            continue

        option = _normalise_option(arg.partition("=")[0])
        if option in _DOWNLOAD_VALUE_OPTIONS:
            expects_value = "=" not in arg
            continue
        if arg.startswith("-"):
            continue

        _raise_for_canonical_conflict("name", "positional dataset", "--name", provided_targets)
        _warn_legacy_cli("The positional dataset argument is deprecated; use --name instead.")
        translated[index : index + 1] = ["--name", arg]
        break
    return translated


def _provided_targets(args: list[str], legacy_arguments: dict[str, str]) -> set[str]:
    """Return current logical targets explicitly present in one command invocation."""
    targets: set[str] = set()
    for arg in args:
        if arg == "--":
            break
        if _normalise_option(arg.partition("=")[0]) in legacy_arguments:
            continue
        target = _target_for_option(_normalise_option(arg))
        if target:
            targets.add(target)
    return targets


def _raise_for_canonical_conflict(
    target: str,
    legacy_option: str,
    replacement: str,
    canonical_targets: set[str],
) -> None:
    """Reject a deprecated option when its current spelling is also present.

    ``_provided_targets`` exists only to feed this, and goes with it.
    """
    if target and target in canonical_targets:
        raise ValueError(f"{legacy_option} cannot be combined with {replacement}. Use only the current spelling.")
