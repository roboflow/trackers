"""
Metrics Submodule Initialization

This module dynamically discovers and registers available tracking metric classes
and provides helper functions for instantiation.
"""

from typing import Dict, List, Optional, Tuple, Type  # Added Tuple

from .base_tracking_metric import TrackingMetric
from .clear import CLEARMetric
from .count import CountMetric

# --- Available Metrics Registry ---
# Maps metric names (lowercase for consistency) to their class implementation.
# The name() method of the class should ideally return the key used here,
# but we use explicit string keys for robustness.
AVAILABLE_METRICS: Dict[str, Type[TrackingMetric]] = {
    "count": CountMetric,
    # "hota": HOTAMetric, # Example placeholder
    "clear": CLEARMetric,
    # Add other metrics here as they are implemented
}

# --- Placeholder Metrics ---
# Define common metric names often used as placeholders.
# These might not have implementations yet but can be requested.
PLACEHOLDER_METRIC_NAMES: set[str] = {"hota", "mota", "idf1", "identity"}


# --- Use Optional for older Python versions ---
def get_metric_instance(name: str) -> Optional[TrackingMetric]:
    """
    Helper function to get an instance of a metric by name (case-insensitive).

    Args:
        name: The name of the metric to instantiate.

    Returns:
        An instance of the requested TrackingMetric class, or None if not found.
    """
    metric_class = AVAILABLE_METRICS.get(name.lower())
    if metric_class:
        return metric_class()
    return None


__all__ = [
    "AVAILABLE_METRICS",
    "PLACEHOLDER_METRIC_NAMES",
    "CLEARMetric",  # Added CLEARMetric to __all__
    "CountMetric",
    "TrackingMetric",
    "get_metric_instance",
    "instantiate_metrics",  # Added instantiate_metrics to __all__
]


def instantiate_metrics(
    metrics: List[str],
) -> Tuple[Dict[str, TrackingMetric], List[str]]:
    """
    Instantiates requested metrics based on the available registry and placeholders.

    Args:
        metrics: A list of metric names (case-insensitive) to instantiate.

    Returns:
        A tuple containing:
        - A dictionary mapping instantiated metric names (from metric.name)
          to their instances.
        - A list of requested metric names that were identified as placeholders
          (using their original casing).
    """
    metrics_to_compute: Dict[str, TrackingMetric] = {}
    placeholder_metrics: List[str] = []
    requested_metrics_lower = {m.lower() for m in metrics}

    for metric_name_lower in requested_metrics_lower:
        metric_instance = get_metric_instance(metric_name_lower)
        if metric_instance:
            metrics_to_compute[metric_instance.name] = metric_instance
        elif metric_name_lower in PLACEHOLDER_METRIC_NAMES:
            original_casing_name = next(
                (m for m in metrics if m.lower() == metric_name_lower),
                metric_name_lower,
            )
            placeholder_metrics.append(original_casing_name)
            print(
                f"Info: Metric '{original_casing_name}' \
                    requested but not implemented. Will report placeholder values."
            )
        else:
            original_casing_name = next(
                (m for m in metrics if m.lower() == metric_name_lower),
                metric_name_lower,
            )
            print(
                f"Warning: Unknown metric '{original_casing_name}' \
                    requested and not found in registry or placeholders. Skipping."
            )

    return metrics_to_compute, placeholder_metrics
