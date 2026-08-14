---
title: 'ADR: Model-backend externalization: vendor vs external package'
description: Decision criteria for keeping model-backend integration in Trackers or consuming it from a separate package.
---

# Model-backend externalization: vendor vs external package

- Status: Accepted
- Date: 2026-08-14

## Context

Trackers needs learned-model backends without making them part of its core tracking API. The existing mask pipeline keeps its SAM and Cutie integration in `trackers.core.masks`: Trackers owns checkpoint discovery, preprocessing, inference lifecycle, and conversion to the mask protocols. The upstream model packages and weights remain optional dependencies; “vendor” here means owning the backend integration code, not copying upstream model source.

Appearance ReID has a different boundary. Trackers owns the `ReIDEncoder` protocol, feature bank, association helpers, and threshold-analysis tools. Model architectures, pretrained checkpoints, preprocessing, training, and gallery evaluation live in the standalone `reid` package, installed by the `reid` extra.

## Decision

Keep the ReID model backend in the external `reid` package. Trackers consumes it through the small `ReIDEncoder.extract_features(detections, frame)` contract and does not duplicate model-loading or model-catalog code.

Choose the ownership boundary for future learned backends using these criteria:

- Keep integration in Trackers when it is tightly coupled to tracker state or frame-to-frame lifecycle, and a small, stable set of backends implements a Trackers-owned protocol.
- Use an external package when model architectures, training, preprocessing, checkpoint catalogs, or release cadence form a substantial product surface independent of multi-object tracking.
- In both cases, import heavy optional dependencies lazily, expose a lightweight protocol from Trackers, and keep tracker behavior usable without the extra.
- Revisit an in-repo backend if its integration grows an independently useful model API; revisit an external backend if its boundary cannot express required tracker lifecycle semantics without leaking implementation details.

## Consequences

- `trackers` stays focused on tracking and association while `reid` can evolve models and training independently.
- Users install `trackers[reid]` for the supported ReID backend; custom encoders may implement `ReIDEncoder` without depending on that package.
- Compatibility is enforced at the protocol and optional-dependency version range, so cross-package changes require coordinated tests and releases.
- The masks and ReID layouts remain intentionally asymmetric: mask lifecycle adapters are Trackers-owned, while ReID model implementations are external.
