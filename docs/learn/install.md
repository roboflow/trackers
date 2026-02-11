# Install `trackers`

Get up and running with `trackers` in minutes. Choose your preferred package manager and start tracking objects in video.

**What you'll learn:**

- Install `trackers` with `pip` or `uv`
- Set up a development environment

!!! tip "Requirements"

    Python `3.10` or higher is required.

---

## Quickstart

=== "pip"

    ```bash
    pip install trackers
    ```

=== "uv"

    ```bash
    uv pip install trackers
    ```

    For `uv`-managed projects:

    ```bash
    uv add trackers
    ```

=== "From Source"

    Install the latest development version:

    ```bash
    pip install https://github.com/roboflow/trackers/archive/refs/heads/develop.zip
    ```

**Verify installation:**

```bash
python -c "import trackers; print(trackers.__version__)"
```

---

## Development Setup

Set up a local environment for contributing or modifying `trackers`.

=== "virtualenv"

    ```bash
    # Clone and enter repository
    git clone --depth 1 -b develop https://github.com/roboflow/trackers.git
    cd trackers

    # Create and activate environment
    python3.10 -m venv venv
    source venv/bin/activate

    # Install in editable mode
    pip install --upgrade pip
    pip install -e "."
    ```

=== "uv"

    ```bash
    # Clone and enter repository
    git clone --depth 1 -b develop https://github.com/roboflow/trackers.git
    cd trackers

    # Set up environment
    uv python pin 3.10
    uv sync
    uv pip install -e . --all-extras
    ```

**Verify dev install:**

```bash
python -c "import trackers; print(trackers.__version__)"
```
