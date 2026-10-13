---
title: Install Roboflow Trackers — Python MOT Library
description: Install Roboflow Trackers with pip or uv. Add multi-object tracking to any detection pipeline with a single package — supports SORT, ByteTrack, OC-SORT, and BoT-SORT out of the box.
---

# Install Trackers

Add multi-object tracking to your detection pipeline with a single install. Works with any detector, any framework.

**What you'll learn:**

- Install `trackers` with `pip` or `uv`
- Set up a local development environment

!!! tip "Requirements"

    Python `3.10+`. Works on Linux, macOS, and Windows.

---

## Quickstart

Get started by installing the package.

=== "pip"

    ```bash
    pip install trackers
    ```

=== "uv"

    ```bash
    uv pip install trackers
    ```

    If your project uses `uv` for dependency management, add `trackers` directly to your project.

    ```bash
    uv add trackers
    ```

=== "From Source"

    Install the latest development version directly from GitHub.

    ```bash
    pip install git+https://github.com/roboflow/trackers.git
    ```

---

## Extras

The base install covers tracking only. Optional extras add functionality like built-in detection.

### Detection

The `detection` extra installs `inference-models`, enabling the CLI to run detection automatically so you don't need to bring your own model.

=== "pip"

    ```bash
    pip install "trackers[detection]"
    ```

=== "uv"

    ```bash
    uv pip install "trackers[detection]"
    ```

!!! tip "GPU Acceleration"

    For GPU support, ensure PyTorch is installed with CUDA or MPS.

### Tune

The `tune` extra installs `optuna`, enabling automated hyperparameter search — see the [tune guide](tune.md).

=== "pip"

    ```bash
    pip install "trackers[tune]"
    ```

=== "uv"

    ```bash
    uv pip install "trackers[tune]"
    ```

### Mask

The `mask` extra installs `torch`, `torchvision`, `rf-segment-anything`, and `rf-cutie`, enabling the SAM + Cutie mask pipeline used by McByte, the `sam`, `cutie`, `mask-manager` and `mcbyte` inspection commands, and mask-conditioned benchmarking — see the [inspect guide](inspect.md).

=== "pip"

    ```bash
    pip install "trackers[mask]"
    ```

=== "uv"

    ```bash
    uv pip install "trackers[mask]"
    ```

!!! tip "GPU Acceleration"

    For GPU support, ensure PyTorch is installed with CUDA or MPS.

### EdgeTAM

The `edgetam` extra installs `torch`, `torchvision`, `timm`, and `transformers`, enabling EdgeTAM mask creation and propagation through the shared mask interfaces and `trackers inspect edgetam` — see the [EdgeTAM inspection guide](inspect.md#inspect-edgetam).

=== "pip"

    ```bash
    pip install "trackers[edgetam]"
    ```

=== "uv"

    ```bash
    uv pip install "trackers[edgetam]"
    ```

To inspect both EdgeTAM and SAM/Cutie, install `trackers[edgetam,mask]`. In a local checkout, use `uv sync --extra edgetam --extra mask --group dev`. The default model revision is pinned and inference uses fp32. Automatic device selection uses CUDA when available, otherwise CPU; pass an explicit device to override it.

---

## Development Setup

Set up a local environment for contributing or modifying `trackers`.

=== "virtualenv"

    ```bash
    git clone --depth 1 -b develop https://github.com/roboflow/trackers.git
    cd trackers

    python3.10 -m venv venv
    source venv/bin/activate

    pip install --upgrade pip
    pip install -e "."
    ```

=== "uv"

    ```bash
    git clone --depth 1 -b develop https://github.com/roboflow/trackers.git
    cd trackers

    uv python pin 3.10
    uv sync
    uv pip install -e . --all-extras
    ```
