from typing import Any, Dict, List, Optional, Protocol, Union

import supervision as sv


class TrackingMetric(Protocol):
    """Protocol defining the interface for a tracking metric calculator."""

    @property
    def name(self) -> str:
        """Returns the standard name of the metric (e.g., 'HOTA', 'CLEAR')."""
        ...

    def compute(
        self,
        ground_truth: sv.Detections,
        predictions: sv.Detections,
        sequence_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Computes the metric(s) for a single sequence.

        Args:
            ground_truth: Ground truth annotations for the sequence as an
                          sv.Detections object. Expected to have `tracker_id`
                          attribute and `data['frame_idx']`.
            predictions: Tracker predictions for the sequence as an sv.Detections
                         object. Expected to have `tracker_id` attribute and
                         `data['frame_idx']`.
            sequence_info: Optional dictionary with sequence metadata (e.g.,
                           frame rate, image dimensions) if needed by the metric.

        Returns:
            A dictionary where keys are specific metric components (e.g.,
            'MOTA', 'MOTP' for CLEAR) and values are the computed scores as floats.
        """
        ...

    def aggregate(
        self, per_sequence_results: List[Dict[str, float]]
    ) -> Dict[str, Union[float, str]]:
        """
        Aggregates results across multiple sequences.

        The default implementation performs simple averaging of numeric results.
        Metrics requiring specific aggregation logic (like summing counts before
        recalculating final scores, e.g., CLEAR) MUST override this method.

        Args:
            per_sequence_results: A list where each element is the dictionary
                                  returned by `compute` for a single sequence.

        Returns:
            A dictionary containing the overall aggregated metric values.
            Can include messages (str) for errors or information.
        """
        # Default implementation (optional): Simple averaging if results are numeric
        # Concrete classes should override this if specific logic is needed.
        aggregated: Dict[str, Union[float, str]] = {}
        if not per_sequence_results:
            return {"message": "No sequence results to aggregate"}

        # Assuming all sequence results have the same keys as the first one
        first_result = per_sequence_results[0]
        if not isinstance(first_result, dict):
            # Handle case where compute might return a single float (though discouraged)
            if all(isinstance(res, (int, float)) for res in per_sequence_results):
                values = [
                    res for res in per_sequence_results if isinstance(res, (int, float))
                ]  # type: ignore
                return (
                    {self.name: sum(values) / len(values)}
                    if values
                    else {self.name: 0.0}
                )
            else:
                return {
                    "error": "Cannot aggregate non-dict, non-numeric results by default"
                }

        keys_to_average = {
            k for k, v in first_result.items() if isinstance(v, (int, float))
        }
        if not keys_to_average:
            return {
                "message": "No numeric keys found in sequence results to \
                    average by default"
            }

        for key in keys_to_average:
            values = [
                seq_res.get(key)
                for seq_res in per_sequence_results
                if isinstance(seq_res, dict)
                and isinstance(seq_res.get(key), (int, float))
            ]
            if values:
                # Filter out None values and convert to list of floats before summing
                filtered_values = [float(v) for v in values if v is not None]
                aggregated[key] = (
                    sum(filtered_values) / len(filtered_values)
                    if filtered_values
                    else 0.0
                )
            else:
                # Key existed in first result but not others or wasn't numeric
                aggregated[key] = 0.0  # Or some other default/indicator

        return aggregated
