# MOT class configuration for preprocessing

## Description

Make MOT ground-truth class handling configurable during preprocessing. The default remains MOT17 behavior, while the MOT20 preset treats class 6 as a distractor. Custom scored and distractor class sets are also supported.

This change adds the shared class configuration and resolver, threads the resolved config through MOT preprocessing, exports the new evaluation types, and covers the default, MOT20, and custom-config cases with in-memory tests.

The evaluation API, CLI option, MOT20 warning, and external TrackEval/count validation are intentionally left for follow-up work.

## Validation

Static compilation and `git diff --check` pass. The focused MOT suite passes with 14 tests.
