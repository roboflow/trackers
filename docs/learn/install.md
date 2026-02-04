# Install Trackers

Get up and running with Trackers in minutes. Choose your preferred package manager and start tracking objects in video.

**What you'll learn:**

- Install Trackers with pip or uv
- Set up a development environment
- Troubleshoot common issues

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

    For uv-managed projects:

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

Set up a local environment for contributing or modifying Trackers.

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

!!! note "Requirements"
    Python 3.10 or higher is required.

---

<details>
<summary><strong>Troubleshooting</strong></summary>

**Permission denied**

- Use `pip install --user trackers` or activate a virtual environment

**Dependency conflicts**

- Install in a fresh virtual environment or clean uv project

**Python version errors**

- Trackers requires Python 3.10+. Check with `python --version`

**Still stuck?**

- Open an issue on [GitHub](https://github.com/roboflow/trackers/issues)

</details>

---

## Next Steps

- [Quickstart](../index.md) — Run your first tracker
- [Evaluate](evaluate.md) — Benchmark tracking results
- [SORT](../trackers/sort.md) / [ByteTrack](../trackers/bytetrack.md) — Tracker guides