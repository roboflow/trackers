#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import sys
import warnings
from argparse import Action, ArgumentParser

import defopt as _defopt

# Top-level argument groups that should be rewritten from ``--prefix-name``
# to dotted form ``--prefix.name`` on the generated argparse parser. Groups
# are derived from the leading underscore-separated token of each parameter
# in ``track()`` (e.g. ``detection_confidence`` → group ``detection``).
_GROUPS = frozenset({"detection", "filters", "out", "show"})

_orig_create_parser = _defopt._create_parser


def _dotted_create_parser(funcs: object, opts: object) -> ArgumentParser:
    """Wrap ``defopt._create_parser`` and rewrite group flags to dotted form.

    Args:
        funcs: Mapping of subcommand name to callable (passed through to
            ``defopt._create_parser``).
        opts: ``defopt`` options instance (passed through unchanged).

    Returns:
        The argparse ``ArgumentParser`` returned by ``defopt._create_parser``,
        with every option string of the form ``--<group>-<rest>`` (for any
        group in ``_GROUPS``) rewritten to ``--<group>.<rest>``.
    """
    parser = _orig_create_parser(funcs, opts)
    _rewrite_dotted(parser)
    return parser


def _rewrite_dotted(parser: ArgumentParser) -> None:
    """Rewrite ``--prefix-name`` options to ``--prefix.name`` in-place.

    Handles both positive flags (``--show-ids`` → ``--show.ids``) and boolean
    negation flags (``--no-show-ids`` → ``--show.no-ids``). Also updates
    ``action.negative_option_strings`` so ``_BooleanOptionalAction`` True/False
    detection keeps working after the rename.

    Recurses into subparsers so all subcommands are rewritten.

    Args:
        parser: An argparse parser (root or subparser) to rewrite.
    """
    new_map: dict[str, Action] = {}
    for opt, action in list(parser._option_string_actions.items()):
        if not opt.startswith("--"):
            continue
        bare = opt[2:]
        new_opt: str | None = None
        if bare.startswith("no-"):
            # Negation: --no-<group>-<rest> → --<group>.no-<rest>
            after_no = bare[3:]
            prefix, sep, rest = after_no.partition("-")
            if sep and prefix in _GROUPS and rest:
                new_opt = f"--{prefix}.no-{rest}"
        else:
            # Positive: --<group>-<rest> → --<group>.<rest>
            prefix, sep, rest = bare.partition("-")
            if sep and prefix in _GROUPS and rest:
                new_opt = f"--{prefix}.{rest}"
        if new_opt is not None:
            action.option_strings = [new_opt if s == opt else s for s in action.option_strings]
            if hasattr(action, "negative_option_strings"):
                action.negative_option_strings = [new_opt if s == opt else s for s in action.negative_option_strings]
            new_map[new_opt] = action
            del parser._option_string_actions[opt]
    parser._option_string_actions.update(new_map)
    for action in parser._actions:
        if hasattr(action, "_name_parser_map"):
            for sub in action._name_parser_map.values():  # type: ignore[attr-defined]
                _rewrite_dotted(sub)


_defopt._create_parser = _dotted_create_parser


def main() -> int:
    """Main entry point for the trackers CLI."""
    warnings.warn(
        "The trackers CLI is in beta. APIs may change in future releases.",
        UserWarning,
        stacklevel=2,
    )

    from importlib.metadata import version

    from trackers.cli.download import download
    from trackers.cli.eval import eval_cmd
    from trackers.cli.track import track
    from trackers.cli.tune import tune

    result = _defopt.run(
        {"track": track, "eval": eval_cmd, "tune": tune, "download": download},
        argv=sys.argv[1:],
        cli_options="all",
        version=version("trackers"),
        short={"out-output": "o"},
    )
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
