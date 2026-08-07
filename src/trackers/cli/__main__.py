#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Command-line entry point for the trackers package."""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import fields
from importlib.metadata import version

from jsonargparse import CLI, ActionYesNo, ArgumentParser

from trackers.cli.download import download_command
from trackers.cli.eval import eval_command
from trackers.cli.track import (
    DetectionOptions,
    FilterOptions,
    OutputOptions,
    ShowOptions,
    TrackerOptions,
    _abbreviated_tracker_parameters,
    track_command,
)
from trackers.cli.tune import tune_command
from trackers.core.base import BaseTracker

_SUBCOMMANDS = frozenset({"track", "eval", "tune", "download"})
# Track option dataclasses paired with the nested CLI key each is registered
# under. Argument registration and boolean-syntax derivation read this single
# table, so a dotted CLI path cannot drift from the dataclass that defines it.
_TRACK_OPTION_GROUPS: tuple[tuple[type, str], ...] = (
    (DetectionOptions, "detection"),
    (FilterOptions, "filters"),
    (TrackerOptions, "tracker"),
    (OutputOptions, "output"),
    (ShowOptions, "show"),
)
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


def _normalise_option(arg: str) -> str:
    """Return one argument with its option name in canonical underscore spelling.

    Every ``-`` after the leading ``--`` becomes ``_``, including a leading
    ``no-``, so ``--no-enqueue-defaults`` reaches the parser as
    ``--no_enqueue_defaults``. Only the name is touched: an ``=`` value, a
    following token, and anything that is not a long option are returned
    unchanged, so ``--detection.model=rfdetr-base`` keeps its hyphen.

    Args:
        arg: One command-line argument.

    Returns:
        The argument with a normalised option name.

    Examples:
        >>> _normalise_option("--detection.mot-file")
        '--detection.mot_file'
        >>> _normalise_option("rfdetr-base")
        'rfdetr-base'
    """
    option, separator, value = arg.partition("=")
    if not option.startswith("--"):
        return arg
    normalized = f"--{option.removeprefix('--').replace('-', '_')}"
    return f"{normalized}{separator}{value}" if separator else normalized


def _normalised_options(arguments: dict[str, str]) -> dict[str, str]:
    """Return one deprecation table keyed by canonical option spelling.

    Lookups normalise before they hit these tables, so a table written with the
    hyphenated spelling a user is migrating from would never match. Normalising
    the keys makes both spellings of every deprecated option resolve, which is
    the point: someone porting a develop command should not have to also guess
    which separator that particular option wanted.

    Args:
        arguments: Deprecated option spellings mapped to their replacements.

    Returns:
        The same mapping with each key normalised.

    Examples:
        >>> _normalised_options({"--mot-output": "--output.mot_results"})
        {'--mot_output': '--output.mot_results'}
    """
    return {_normalise_option(option): replacement for option, replacement in arguments.items()}


_LEGACY_ARGUMENTS = {
    "track": _normalised_options({**_DEVELOP_TRACK_ARGUMENTS, **_develop_tracker_parameter_arguments()}),
    "eval": {
        "-o": "--output",
    },
    "tune": {
        "-o": "--output",
    },
    "download": _normalised_options(
        {
            "--list": "--list_available",
            "-o": "--output",
        }
    ),
}
_LEGACY_LIST_ARGUMENTS = {
    "eval": frozenset({"--metrics", "--columns"}),
    "tune": frozenset({"--metrics"}),
}
# Track filters that used to take one comma-separated string and now take a
# list, matching the list-valued options of eval and tune.
_COMMA_LIST_TRACK_ARGUMENTS = frozenset({"--filters.classes", "--filters.track_ids"})
_DOWNLOAD_VALUE_OPTIONS = frozenset(
    _normalise_option(option) for option in ("--dataset", "--split", "--asset", "-o", "--output", "--cache-dir")
)


def _legacy_removal_version() -> str:
    """Return the release three minor versions after the installed package."""
    major, minor, *_ = version("trackers").split(".")
    return f"{major}.{int(minor) + 3}.0"


def _warn_legacy_cli(message: str) -> None:
    """Warn about one legacy CLI form with its scheduled removal release."""
    warnings.warn(
        f"{message} It will be removed in {_legacy_removal_version()}.",
        FutureWarning,
        stacklevel=3,
    )


# Algorithm selector inside the tracker option group. ``--tracker <id>`` is the
# shorthand spelling, expanded to this path before jsonargparse sees argv.
_TRACKER_NAME_OPTION = "--tracker.name"
_TRACKER_SHORTHAND_OPTION = "--tracker"
# Prefix marking the negative half of a boolean option pair.
_NEGATION_PREFIX = "no_"


class _GroupedYesNo(ActionYesNo):
    """Boolean option pair whose negative half stays inside the option group.

    ``ActionYesNo`` builds its negative option by substituting the prefix at the
    *start* of the option string, so ``--show.ids`` becomes ``--no_show.ids``.
    That negates the group rather than the field, which reads as nonsense once a
    group is named after anything but a verb: ``--no_detection.fast`` claims
    there is no detection, when the intent is a detector that is not fast. This
    subclass moves the prefix onto the leaf instead, giving ``--show.no_ids``
    and ``--detection.no_fast``.

    ``nargs="?"`` keeps the explicit-value spelling working, so ``--show.ids``,
    ``--show.no_ids``, ``--show.ids false`` and ``--show.ids=false`` are all
    accepted and a ``--config`` file still holds one plain boolean key per
    field. Two options writing one destination means the last one on the command
    line wins, matching how every other repeated option behaves.
    """

    def __init__(self, yes_prefix: str = "", no_prefix: str = _NEGATION_PREFIX, **kwargs):
        super().__init__(yes_prefix=yes_prefix, no_prefix=no_prefix, **kwargs)
        if kwargs and self._no_prefix is not None:
            path = self.option_strings[-1][2 + len(self._no_prefix) :]
            group, dot, leaf = path.rpartition(".")
            self.option_strings[-1] = f"--{group}{dot}{self._no_prefix}{leaf}"

    def __call__(self, *args, **kwargs):
        """Set the destination from whichever half of the pair was used."""
        if len(args) == 0:
            # argparse re-instantiates the action through this branch, and the
            # base class hardcodes its own type there.
            kwargs["_yes_prefix"] = self._yes_prefix
            kwargs["_no_prefix"] = self._no_prefix
            return _GroupedYesNo(**kwargs)
        value = args[2] if isinstance(args[2], bool) else True
        # Strip the dashes before splitting: an ungrouped option has no dot, so
        # ``rpartition`` would hand back ``--no_list_available`` and the prefix
        # test would silently miss, leaving the negative half setting ``True``.
        leaf = args[3].removeprefix("--").rpartition(".")[2]
        setattr(args[1], self.dest, not value if leaf.startswith(self._no_prefix) else value)
        return None


class _CLIParser(ArgumentParser):
    """Expose track dataclasses while preserving intentional boolean syntax."""

    def add_argument(self, *args, **kwargs):  # type: ignore[override]
        if kwargs.get("type") is bool:
            kwargs.pop("type")
            kwargs["nargs"] = "?"
            kwargs["action"] = _GroupedYesNo(yes_prefix="", no_prefix=_NEGATION_PREFIX)
        if _TRACKER_NAME_OPTION in args:
            # The registry is the accept list, so an unknown tracker is rejected
            # while parsing rather than after a detection model has been loaded.
            # ``type`` is deliberately dropped: combined with ``choices`` it makes
            # jsonargparse print a lambda repr in --help, and registry ids are strings.
            kwargs.pop("type", None)
            kwargs["choices"] = BaseTracker._registered_trackers()
        return super().add_argument(*args, **kwargs)

    def add_function_arguments(self, function, *args, **kwargs):  # type: ignore[override]
        if function is track_command:
            return _add_track_arguments(self)
        return super().add_function_arguments(function, *args, **kwargs)


def _add_track_arguments(parser: ArgumentParser) -> list[str]:
    """Register track arguments while preserving their nested dataclass paths."""
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video file, webcam index, RTSP URL, or image directory.",
    )
    added_args = ["source"]
    for option_class, nested_key in _TRACK_OPTION_GROUPS:
        added_args.extend(parser.add_class_arguments(option_class, nested_key))
    # Registered as a plain boolean rather than a bare ``store_true`` so it gains
    # the same ``--no_display`` half every other boolean option has.
    parser.add_argument("--display", type=bool, default=False, help="Show a live preview window.")
    added_args.append("display")
    return added_args


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


def _expand_tracker_shorthand(args: list[str]) -> list[str]:
    """Expand ``--tracker <id>`` to the ``--tracker.name`` path it selects.

    ``--tracker`` is a supported shorthand rather than a deprecated spelling, so
    no warning is emitted. The expansion is needed because ``--tracker`` is also
    the prefix of the tracker option group: jsonargparse reads a bare
    ``--tracker`` value as JSON for the whole group, which a plain registry ID
    is not. A value that does start with ``{`` is left alone so the group can
    still be supplied as a JSON object.

    Args:
        args: Track arguments, without the subcommand token.

    Returns:
        Arguments where every ``--tracker`` shorthand targets ``--tracker.name``.

    Examples:
        >>> _expand_tracker_shorthand(["--tracker", "sort", "--display"])
        ['--tracker.name', 'sort', '--display']
    """
    expanded: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            expanded.extend(args[index:])
            break

        option, separator, value = arg.partition("=")
        if _normalise_option(option) != _TRACKER_SHORTHAND_OPTION:
            expanded.append(arg)
            index += 1
            continue

        if separator:
            expanded.append(arg if value.startswith("{") else f"{_TRACKER_NAME_OPTION}={value}")
            index += 1
            continue

        next_value = args[index + 1] if index + 1 < len(args) else ""
        if not next_value or next_value.startswith("-") or next_value.startswith("{"):
            expanded.append(arg)
            index += 1
            continue

        expanded.extend([_TRACKER_NAME_OPTION, next_value])
        index += 2
    return expanded


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
    """Translate a legacy download dataset positional without touching option values."""
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

        _raise_for_canonical_conflict("dataset", "positional dataset", "--dataset", provided_targets)
        _warn_legacy_cli("The positional dataset argument is deprecated; use --dataset instead.")
        translated[index : index + 1] = ["--dataset", arg]
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


def _target_for_option(option: str) -> str:
    """Return a normalized logical target for conflict detection.

    Both halves of a boolean pair collapse onto the same target, so mixing a
    deprecated spelling with the current one is still caught whichever polarity
    each carries.
    """
    option_name = option.partition("=")[0]
    if not option_name.startswith("--"):
        return ""
    target = option_name.removeprefix("--").removeprefix("no-").replace("-", "_")
    group, dot, leaf = target.rpartition(".")
    return f"{group}{dot}{leaf.removeprefix(_NEGATION_PREFIX)}"


def _raise_for_detection_source_conflict(args: list[str]) -> None:
    """Preserve develop's mutually exclusive detector-source CLI contract."""
    targets: set[str] = set()
    for arg in args:
        if arg == "--":
            break
        target = _target_for_option(arg)
        if target:
            targets.add(target)
    if {"detection.model", "detection.mot_file"}.issubset(targets):
        raise ValueError("--detection.model cannot be combined with --detection.mot_file.")


def _raise_for_canonical_conflict(
    target: str,
    legacy_option: str,
    replacement: str,
    canonical_targets: set[str],
) -> None:
    """Reject a deprecated option when its current spelling is also present."""
    if target and target in canonical_targets:
        raise ValueError(f"{legacy_option} cannot be combined with {replacement}. Use only the current spelling.")


def main() -> int:
    """Dispatch to track / eval / tune / download via jsonargparse CLI."""
    warnings.warn(
        "The trackers CLI is in beta. APIs may change in future releases.",
        UserWarning,
        stacklevel=2,
    )
    try:
        args = _translate_legacy_args(sys.argv[1:])
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    if args == ["--version"]:
        print(f"trackers {version('trackers')}")
        return 0
    rc = CLI(
        {"track": track_command, "eval": eval_command, "tune": tune_command, "download": download_command},
        args=args,
        as_positional=False,
        prog="trackers",
        description="Command-line tools for multi-object tracking.",
        parser_class=_CLIParser,
    )
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
