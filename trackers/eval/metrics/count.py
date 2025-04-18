from typing import Any, Dict, List, Optional, Set, Union

import supervision as sv

from trackers.eval.metrics.base_tracking_metric import TrackingMetric


class CountMetric(TrackingMetric):
    """
    A simple metric that counts ground truth vs predicted annotations and tracks.
    """

    @property
    def name(self) -> str:
        """Returns the standard name of the metric."""
        return "Count"

    def compute(
        self,
        ground_truth: sv.Detections,
        predictions: sv.Detections,
        sequence_info: Optional[Dict[str, Any]] = None,  # Not used by this metric
    ) -> Dict[str, float]:
        """
        Computes the counts for a single sequence.

        Args:
            ground_truth: Ground truth annotations as an sv.Detections object.
                          Expected to have `tracker_id` attribute.
            predictions: Predictions as an sv.Detections object.
                         Expected to have `tracker_id` attribute.
            sequence_info: Optional sequence metadata (unused by this metric).

        Returns:
            A dictionary containing:
            - 'GT_Annotations': Total number of ground truth annotations.
            - 'Pred_Annotations': Total number of predicted annotations.
            - 'GT_Tracks': Number of unique ground truth track IDs.
            - 'Pred_Tracks': Number of unique predicted track IDs.
        """
        # Get total annotation counts
        gt_annotations_count = len(ground_truth)
        pred_annotations_count = len(predictions)

        # Extract unique track IDs
        gt_track_ids: Set[int] = set()
        if ground_truth.tracker_id is not None:
            gt_track_ids = set(ground_truth.tracker_id.tolist())

        pred_track_ids: Set[int] = set()
        if predictions.tracker_id is not None:
            pred_track_ids = set(predictions.tracker_id.tolist())

        gt_tracks_count = len(gt_track_ids)
        pred_tracks_count = len(pred_track_ids)

        return {
            "GT_Annotations": float(gt_annotations_count),
            "Pred_Annotations": float(pred_annotations_count),
            "GT_Tracks": float(gt_tracks_count),
            "Pred_Tracks": float(pred_tracks_count),
        }

    def aggregate(
        self, per_sequence_results: List[Dict[str, float]]
    ) -> Dict[str, Union[float, str]]:
        """
        Aggregates CountMetric results by summing counts across sequences.

        Args:
            per_sequence_results: A list where each element is the dictionary
                                  returned by `compute` for a single sequence.

        Returns:
            A dictionary containing the summed counts across all sequences,
            or a message if aggregation is not possible.
        """
        if not per_sequence_results:
            return {"message": "No sequence results to aggregate for Count"}

        overall_counts: Dict[str, float] = {
            "GT_Annotations": 0.0,
            "Pred_Annotations": 0.0,
            "GT_Tracks": 0.0,
            "Pred_Tracks": 0.0,
        }

        valid_results_count = 0
        for seq_output in per_sequence_results:
            # Basic check if the result looks like a valid count dict
            if isinstance(seq_output, dict) and all(
                key in seq_output for key in overall_counts
            ):
                valid_results_count += 1
                for key in overall_counts:
                    # Ensure value is numeric before adding
                    value = seq_output.get(key, 0.0)
                    if isinstance(value, (int, float)):
                        overall_counts[key] += value
                    else:
                        # Log or handle non-numeric value if necessary
                        pass
            else:
                # Log or handle invalid sequence result format if necessary
                print(
                    f"Warning: Skipping invalid sequence result format during\
                        Count aggregation: {seq_output}"
                )

        if valid_results_count == 0:
            return {"message": "No valid sequence results found for Count aggregation"}

        # Return type needs to match protocol, ensure values are float or str
        final_results: Dict[str, Union[float, str]] = {
            k: v for k, v in overall_counts.items()
        }
        return final_results
