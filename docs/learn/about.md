---
description: Learn about Roboflow Trackers — who built it, how to cite it, where to get support, and how to contribute.
---

# About

## What is Roboflow Trackers?

Roboflow Trackers is an open-source Python library that provides clean-room implementations of leading multi-object tracking (MOT) algorithms: [SORT](../trackers/sort.md), [ByteTrack](../trackers/bytetrack.md), and [OC-SORT](../trackers/ocsort.md). The library is designed to plug into any object detection model through the [supervision](https://github.com/roboflow/supervision) library, giving you a single `tracker.update(detections)` call to add tracking to an existing detection pipeline.

Every algorithm is implemented from scratch following the original papers, with a shared interface and consistent parameter naming. The library ships with built-in evaluation tools for HOTA, IDF1, and MOTA metrics, a CLI for running trackers on video files, and dataset download utilities for standard MOT benchmarks.

## Who Built It?

Roboflow Trackers is built and maintained by the [Roboflow](https://roboflow.com) team. Roboflow builds tools and infrastructure for computer vision, from dataset management and model training to deployment and monitoring. The trackers library is part of the broader Roboflow open-source ecosystem that includes [supervision](https://github.com/roboflow/supervision), [inference](https://github.com/roboflow/inference), and [RF-DETR](https://github.com/roboflow/rf-detr).

A full list of contributors is available on the [GitHub contributors page](https://github.com/roboflow/trackers/graphs/contributors).

## Support

If you run into a bug, have a feature request, or need help with integration:

- **GitHub Issues** — open an issue on the [trackers repository](https://github.com/roboflow/trackers/issues) for bug reports and feature requests.
- **Discord** — join the [Roboflow Discord server](https://discord.gg/GbfgXGJ8Bk) for real-time help and community discussion.
- **Documentation** — browse the [Guides](install.md) and [API Reference](../api/trackers.md) sections of this site.

## Contributing

Contributions are welcome. To get started:

1. Fork the [repository](https://github.com/roboflow/trackers) and clone it locally.
2. Install development dependencies with `uv sync`.
3. Create a feature branch, make your changes, and add tests.
4. Run `pre-commit run --all-files` to check formatting, linting, and type checks.
5. Open a pull request against the `develop` branch.

See the repository's [contributing guidelines](https://github.com/roboflow/trackers/blob/main/CONTRIBUTING.md) for full details on code style, commit conventions, and the review process.

## License

Roboflow Trackers is released under the [Apache License 2.0](https://github.com/roboflow/trackers/blob/main/LICENSE). You are free to use, modify, and distribute the library in both commercial and non-commercial projects, subject to the terms of the license.

## Citing Roboflow Trackers

If you use Roboflow Trackers in academic work or publications, please cite the library:

```bibtex
@software{roboflow_trackers,
  author  = {{Roboflow}},
  title   = {Roboflow Trackers},
  url     = {https://github.com/roboflow/trackers},
  year    = {2025},
  license = {Apache-2.0}
}
```

For citations of specific tracking algorithms, see the Reference section on each tracker page: [SORT](../trackers/sort.md#reference), [ByteTrack](../trackers/bytetrack.md#reference), [OC-SORT](../trackers/ocsort.md#reference).

## Learn More About Roboflow

Visit [roboflow.com](https://roboflow.com) to learn about the full Roboflow platform for building, deploying, and managing computer vision applications.
