# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Parser conventions shared by every trackers command.

Everything here is permanent: the option-name spelling rule, the boolean option pair, and the nested option groups of
``track``. The transitional rewrites that accept the previous release's spellings live in :mod:`trackers.cli._legacy`.

This module deliberately imports nothing from :mod:`trackers.cli.__main__`. The entry point imports every subcommand, so
a subcommand reaching back for the shared parser through ``__main__`` would import a half-initialised module. Import
from here instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jsonargparse import ActionYesNo, ArgumentParser

from trackers.cli.track import (
    DetectionOptions,
    FilterOptions,
    OutputOptions,
    ReIDOptions,
    ShowOptions,
    TrackerOptions,
    track_command,
)
from trackers.core.base import BaseTracker

_SUBCOMMANDS = frozenset({"track", "eval", "tune", "download", "benchmark", "inspect"})
# Track option dataclasses paired with the nested CLI key each is registered
# under. Argument registration and boolean-syntax derivation read this single
# table, so a dotted CLI path cannot drift from the dataclass that defines it.
_TRACK_OPTION_GROUPS: tuple[tuple[type, str], ...] = (
    (DetectionOptions, "detection"),
    (FilterOptions, "filters"),
    (TrackerOptions, "tracker"),
    (ReIDOptions, "reid"),
    (OutputOptions, "output"),
    (ShowOptions, "show"),
)
# Algorithm selector inside the tracker option group. ``--tracker <id>`` is the
# shorthand spelling, expanded to this path before jsonargparse sees argv.
_TRACKER_NAME_OPTION = "--tracker.name"
_TRACKER_SHORTHAND_OPTION = "--tracker"
# Prefix marking the negative half of a boolean option pair.
_NEGATION_PREFIX = "no_"
# Commands whose arguments are registered by hand rather than derived from the
# signature, keyed by the command callable. Each command module registers its
# own entry, so adding a command never edits this module and no command module
# is imported here to find out whether it has one.
_ARGUMENT_ADDERS: dict[Callable[..., Any], Callable[[ArgumentParser], list[str]]] = {}


def register_argument_adder(
    function: Callable[..., Any],
    adder: Callable[[ArgumentParser], list[str]],
) -> None:
    """Register a hand-written argument registration for one command.

    A command whose options cannot be derived from its signature calls this at
    import time. ``add_function_arguments`` then finds it by lookup, which is
    what keeps the parser from importing command modules to identify them.

    Args:
        function: The command callable the registration belongs to.
        adder: Callable registering the arguments and returning their names.

    Examples:
        >>> def _adder(parser):
        ...     return []
        >>> def _command(): ...
        >>> register_argument_adder(_command, _adder)
        >>> _ARGUMENT_ADDERS[_command] is _adder
        True
    """
    _ARGUMENT_ADDERS[function] = adder


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


def _target_for_option(option: str) -> str:
    """Return a normalized logical target for conflict detection.

    Both halves of a boolean pair collapse onto the same target, so mixing a
    deprecated spelling with the current one is still caught whichever polarity
    each carries.

    Args:
        option: One command-line argument.

    Returns:
        The logical destination the option writes, or an empty string when the
        argument is not a long option.

    Examples:
        >>> _target_for_option("--show.no_ids")
        'show.ids'
        >>> _target_for_option("results.json")
        ''
    """
    option_name = option.partition("=")[0]
    if not option_name.startswith("--"):
        return ""
    target = option_name.removeprefix("--").removeprefix("no-").replace("-", "_")
    group, dot, leaf = target.rpartition(".")
    return f"{group}{dot}{leaf.removeprefix(_NEGATION_PREFIX)}"


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
        adder = _ARGUMENT_ADDERS.get(function)
        if adder is not None:
            return adder(self)
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
        if option_class is ReIDOptions:
            added_args.extend(parser.add_class_arguments(option_class, nested_key, skip={"enable"}))
            # jsonargparse handles ``bool | None`` as a value-taking type hint.
            # Registering the field as ``bool`` preserves the paired bare flags
            # while its explicit default retains the omitted state.
            parser.add_argument(
                "--reid.enable",
                type=bool,
                default=None,
                help="Explicitly enable or disable appearance association.",
            )
            added_args.append("reid.enable")
            continue
        added_args.extend(parser.add_class_arguments(option_class, nested_key))
    # Registered as a plain boolean rather than a bare ``store_true`` so it gains
    # the same ``--no_display`` half every other boolean option has.
    parser.add_argument("--display", type=bool, default=False, help="Show a live preview window.")
    added_args.append("display")
    return added_args


register_argument_adder(track_command, _add_track_arguments)


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

        if index + 1 >= len(args) or args[index + 1].startswith("-") or args[index + 1].startswith("{"):
            expanded.append(arg)
            index += 1
            continue

        expanded.extend([_TRACKER_NAME_OPTION, args[index + 1]])
        index += 2
    return expanded


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
