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
