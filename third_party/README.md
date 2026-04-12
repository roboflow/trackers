# Third-Party Integrations

Vendor or check out external calibration systems here so their code stays
isolated from the `trackers/` package.

Recommended layout:

- `third_party/pnlcalib`

The local wrappers in `trackers/calibration/providers/` should stay thin:
resolve paths, normalize outputs, and translate provider-specific camera or
homography data into the repo's shared `CalibrationFrame` schema.
