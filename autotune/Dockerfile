# ------------------------------------------------------------------------
# Trackers — optimize sandbox image
# Used by /optimize run --docker to run metric/guard in an isolated
# container with a read-only project mount.
#
# Build (run from project root):
#   docker build -f autotune/Dockerfile -t trackers-optimize:latest .
#
# Runtime mounts (added automatically by the campaign framework):
#   -v $(pwd):/workspace:ro                    project source (read-only)
#   -v $(pwd)/.experiments:/workspace/.experiments:rw
#   -v $HOME/.cache/trackers:/root/.cache/trackers:ro  MOT17 data
#   --tmpfs /tmp:rw,size=256m
#   -w /workspace
# ------------------------------------------------------------------------

FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# System libraries required by opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        git \
        cmake \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the full project so uv can resolve and lock all dependencies.
# The venv is installed at /opt/venv (not /workspace) so that the
# runtime read-only project mount does not shadow the Linux venv.
COPY . /build/
WORKDIR /build

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --group optimize

# At runtime the live project source is mounted at /workspace.
# UV_PROJECT_ENVIRONMENT points uv to the pre-built Linux venv so
# `uv run python` uses it without re-syncing.
# UV_NO_SYNC=1 prevents uv from attempting to re-sync on every run.
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/workspace"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_NO_SYNC=1

WORKDIR /workspace
