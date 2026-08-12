"""Typed exceptions shared across evaluation modules."""


class AggregationIncompatibleError(ValueError):
    """Raised when metric payloads cannot be aggregated as requested."""
