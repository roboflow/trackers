# Installation

Welcome to Trackers! This guide will help you install and set up Trackers for your projects. Whether you're a developer looking to contribute or an end-user ready to start using Trackers, we've got you covered.

## Installation Methods

Trackers supports several installation methods. [Python 3.10](https://devguide.python.org/versions/) or higher is required. Choose the option which best fits your workflow.

!!! example "Installation"

    === "pip (recommended)"
        The easiest way to install Trackers is using `pip`. This method is recommended for most users.

        ```bash
        pip install trackers
        ```

    === "uv"
        If you are using `uv`, you can install Trackers using the following command.

        ```bash
        uv pip install trackers
        ```

        For uv-managed projects, add Trackers as a project dependency.

        ```bash
        uv add trackers
        ```

    === "Source Archive"
        To install the latest development version of Trackers from source without cloning the full repository, run the command below.

        ```bash
        pip install https://github.com/roboflow/trackers/archive/refs/heads/develop.zip
        ```

## Dev Environment

If you plan to contribute to Trackers or modify the codebase locally, set up a local development environment using the steps below. We recommend using an isolated environment to avoid dependency conflicts.

!!! example "Development Setup"

    === "virtualenv"
        ```bash
        # Clone repository
        git clone --depth 1 -b develop https://github.com/roboflow/trackers.git
        cd trackers

        # Create virtual environment
        python3.10 -m venv venv

        # Activate environment
        source venv/bin/activate

        # Upgrade pip
        pip install --upgrade pip

        # Install in editable mode
        pip install -e "."
        ```

    === "uv"
        ```bash
        # Clone repository
        git clone --depth 1 -b develop https://github.com/roboflow/trackers.git
        cd trackers

        # Pin Python version
        uv python pin 3.10

        # Sync environment and install dependencies
        uv sync

        # Install in editable mode with extras
        uv pip install -e . --all-extras
        ```

## Troubleshooting

Installation issues usually fall into a few common categories.

- **Permission Issues**. Package install fails due to missing write access to system Python paths. Fix by running `pip install --user trackers` or by using an isolated environment.
- **Dependency Conflicts**. Errors appear after installing other Python packages. Resolve by installing Trackers inside a fresh virtual environment or a clean `uv` project.
- **Python Version**. Installation fails or runtime errors appear when using older Python releases. Trackers requires Python 3.10 or higher.

If problems persist, open an issue on the [GitHub repository](https://github.com/roboflow/trackers).

## See also

- [Evaluate guide](evaluate.md)
- [Evals API](../api-evals.md)
