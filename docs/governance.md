---
title: Governance — Roboflow Trackers
description: Who reviews and merges changes to Roboflow Trackers, the branching model, release process, and versioning policy.
---

# Governance

## Maintainers & Code Review

Roboflow Trackers is maintained by the [Roboflow](https://roboflow.com) team. Pull requests are reviewed according to the repository's [`CODEOWNERS`](https://github.com/roboflow/trackers/blob/develop/.github/CODEOWNERS) file, which currently designates [@SkalskiP](https://github.com/SkalskiP) as the default reviewer for all changes.

A full list of contributors is available on the [GitHub contributors page](https://github.com/roboflow/trackers/graphs/contributors).

## Branching & Releases

The project uses a structured branching model — see [Contributing → Branching Strategy](contributing.md#branching-strategy) for the `develop` / `release/stable` / `release/X.Y.Z` branch roles, and [Contributing → Releasing](contributing.md#releasing) for how feature and bugfix releases are cut and tagged.

## Versioning

Roboflow Trackers follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — see the [Changelog](changelog.md) for the full release history.

## What's not yet documented here

The project doesn't yet have a written policy for release cadence or a formal multi-maintainer decision-making process beyond the `CODEOWNERS`-based review above. This page will be extended once that's defined — flag it if you need clarity on either in the meantime.
