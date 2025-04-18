import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import supervision as sv

from trackers.core.base import BaseTracker
from trackers.dataset.core import Dataset, MOTChallengeDataset
from trackers.eval.metrics import (
    instantiate_metrics,
)
from trackers.eval.metrics.base_tracking_metric import TrackingMetric
from trackers.eval.utils.save_tracks import load_tracks_from_disk, save_tracks
from trackers.sort_tracker import SORTTracker


def _add_frame_metadata_to_detections(
    detections: sv.Detections, frame_idx: int, image_path: str
) -> sv.Detections:
    """
    Helper function to add 'frame_idx' and 'image_path' to detections.data.

    If detections.data is None, it initializes it. If detections is empty,
    it initializes data with empty arrays for these keys.

    Args:
        detections: The sv.Detections object to modify.
        frame_idx: The frame index to associate with these detections.
        image_path: The path to the image file for this frame.

    Returns:
        The modified sv.Detections object with metadata added to its `data` attribute.
    """
    if len(detections) > 0:
        if not detections.data:  # Initialize data if None
            detections.data = {}
        # Create arrays matching the number of detections
        frame_indices = np.full(len(detections), frame_idx, dtype=int)
        image_paths = np.array([image_path] * len(detections), dtype=object)
        detections.data["frame_idx"] = frame_indices
        detections.data["image_path"] = image_paths
    else:
        detections.data = {}
        detections.data["frame_idx"] = np.empty(0, dtype=int)
        detections.data["image_path"] = np.empty(0, dtype=object)
    return detections


def generate_tracks(
    dataset: Dataset,
    detection_source: Union[
        MOTChallengeDataset,
        sv.DetectionDataset,
        Callable[[Optional[np.ndarray], Dict[str, Any]], sv.Detections],
    ],
    tracker_source: Union[
        BaseTracker,
        Callable[[sv.Detections, Optional[np.ndarray], Dict[str, Any]], sv.Detections],
    ],
    output_dir: Optional[Union[str, Path]] = None,
    image_loader: Optional[Callable[[str], np.ndarray]] = None,
) -> Dict[str, sv.Detections]:
    """
    Generates tracking results for all sequences in a dataset.

    Iterates through each sequence and frame, obtains detections, runs the tracker,
    and collects the results. Optionally saves tracks to disk.

    Args:
        dataset: A Dataset object providing sequences and frames.
        detection_source: Source providing sv.Detections per frame. Can be:
                          - MOTChallengeDataset (uses loaded public detections).
                          - sv.DetectionDataset (uses annotations keyed by image path).
                          - A callback function: `(frame, frame_info) -> sv.Detections`.
                            The frame can be None if image loading fails.
        tracker_source: Source providing tracked sv.Detections. Can be:
                        - A BaseTracker object (implements update method).
                        - A callback function:
                          `(detections, frame, frame_info) -> sv.Detections`.
                          The frame can be None if image loading fails.
                          *Note: The callback function should handle resetting
                          the tracker state internally if needed (e.g., at frame 1).*
        output_dir: Optional directory path (str or Path) to save tracking results
                    as JSON files (one per sequence).
        image_loader: Optional custom image loading function `(path) -> np.ndarray`.
                      Defaults to using `cv2.imread` and converting to RGB.

    Returns:
        A dictionary mapping sequence names (str) to sv.Detections objects
        containing all merged tracks for that sequence. If a sequence has no tracks
        or encounters an error during processing, it will map to an empty
        sv.Detections object.
    """
    if image_loader is None:

        def default_image_loader(path):
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        image_loader = default_image_loader

    # Handle different types of detection sources
    def get_detections(
        frame: Optional[np.ndarray], frame_info: Dict[str, Union[str, int]]
    ) -> sv.Detections:
        image_path = frame_info.get("image_path")
        frame_idx = frame_info.get("frame_idx")

        if isinstance(detection_source, MOTChallengeDataset):
            if detection_source.has_public_detections:
                dets = detection_source.get_public_detections(image_path)
                if dets.class_id is not None:
                    print(f"Warning: Detected class_id is not None for {image_path}")
                    # Ensure class_id is None
                    dets.class_id = None
                return _add_frame_metadata_to_detections(dets, frame_idx, image_path)
            else:
                print(
                    "Warning: MOTChallengeDataset provided as detection_source, \
                        but public detections not loaded. Returning empty."
                )
                return _add_frame_metadata_to_detections(
                    sv.Detections.empty(), frame_idx, image_path
                )
        elif isinstance(detection_source, sv.DetectionDataset):
            if image_path in detection_source.annotations:
                dets = detection_source.annotations[image_path]
                return _add_frame_metadata_to_detections(dets, frame_idx, image_path)
            else:
                print(
                    f"Warning: No detections found for {image_path} in \
                        sv.DetectionDataset"
                )
                return _add_frame_metadata_to_detections(
                    sv.Detections.empty(), frame_idx, image_path
                )
        elif callable(detection_source):
            # It's a callback function
            dets = detection_source(frame, frame_info)
            if not isinstance(dets, sv.Detections):
                raise TypeError(
                    f"Detection source callback must return sv.Detections, \
                        but got {type(dets)}"
                )
            return _add_frame_metadata_to_detections(dets, frame_idx, image_path)
        else:
            raise TypeError(
                f"Unsupported detection_source type: {type(detection_source)}"
            )

    def _process_tracking(
        detections: sv.Detections,
        frame: Optional[np.ndarray],
        frame_info: Dict[str, Any],
    ) -> sv.Detections:
        sequence_name = frame_info.get("sequence_name")
        frame_idx = frame_info.get("frame_idx")

        if callable(tracker_source):
            # It's a callback function
            # Need a way to reset the tracker state.
            # Maybe we need to have the user do that?
            # Or we can make it explicit in the function signature
            # (e.g., reset_tracker=True)
            tracks = tracker_source(detections, frame, frame_info)
        elif isinstance(tracker_source, BaseTracker):
            # It's a Tracker object
            # Reset tracker at the start of each sequence
            #  if frame_idx == 1:
            #       tracker_source.reset()
            # Pass frame if required by the tracker
            #  if tracker_source.requires_frame_image:
            #       tracks = tracker_source.update(detections, frame)
            #  else:
            tracks = tracker_source.update(detections)
        else:
            raise TypeError(f"Unsupported tracker_source type: {type(tracker_source)}")

        # --- Check tracker output type ---
        if not isinstance(tracks, sv.Detections):
            raise TypeError(
                f"Tracker source (type: {type(tracker_source)}) must return an \
                    sv.Detections object, "
                f"but returned type {type(tracks)} for sequence {sequence_name}, \
                    frame {frame_idx}."
            )
        # --- End check ---

        # Ensure tracker_id is present
        if tracks.tracker_id is None and len(tracks) > 0:
            print(
                f"Warning: Tracker output for sequence {sequence_name}, \
                    frame {frame_idx} is missing 'tracker_id'. Evaluation might fail."
            )
            # Optionally assign dummy IDs or raise error depending on strictness needed
            tracks.tracker_id = np.full(len(tracks), -1)  # Example: assign dummy ID

        return tracks

    # Generate tracks for all sequences
    all_tracks: Dict[str, sv.Detections] = {}
    sequence_names = dataset.get_sequence_names()

    for seq_name in sequence_names:
        print(f"\n--- Generating tracks for sequence: {seq_name} ---")

        sequence_detections_list: List[
            sv.Detections
        ] = []  # Store sv.Detections per frame
        frame_iterator = dataset.get_frame_iterator(seq_name)

        for frame_info in frame_iterator:
            frame_idx = frame_info["frame_idx"]
            frame_info["sequence_name"] = seq_name  # Add sequence name for context

            # Load the frame image
            frame = None
            if "image_path" in frame_info:
                try:
                    frame = image_loader(frame_info["image_path"])
                except Exception as e:
                    print(
                        f"Warning: Failed to load image for frame {frame_idx}: {e}. \
                            Proceeding without image."
                    )
                    # Decide how to handle - skip frame? pass None to detector/tracker?
                    # Passing None might cause errors downstream if not handled.
                    # For now, we proceed, but detector/tracker must handle None frame.

            detections = get_detections(frame, frame_info)

            # Process tracking
            try:
                tracker_output = _process_tracking(detections, frame, frame_info)
            except TypeError as e:  # Catch the type error from process_tracking
                print(f"Error during tracking: {e}")

                print(
                    f"Skipping remaining frames for sequence \
                        {seq_name} due to tracker output error."
                )
                sequence_detections_list = []  # Clear tracks for this sequence
                break

            sequence_detections_list.append(tracker_output)

        # --- Merge detections for the sequence ---
        merged_detections = sv.Detections.merge(sequence_detections_list)
        # --- End merge ---

        # Only add sequence if processing didn't break early and tracks exist
        if merged_detections is not None and len(merged_detections) > 0:
            all_tracks[seq_name] = merged_detections

            # Save to disk if requested
            save_tracks(merged_detections, seq_name, output_dir)

        elif not sequence_detections_list and any(dataset.get_frame_iterator(seq_name)):
            # Sequence had frames but no tracks were generated (or error occurred)
            print(f"No tracks generated or saved for sequence {seq_name}.")
            all_tracks[seq_name] = (
                sv.Detections.empty()
            )  # Store empty detections object
        elif not any(dataset.get_frame_iterator(seq_name)):
            # Sequence was empty
            print(f"Sequence {seq_name} is empty.")
            all_tracks[seq_name] = sv.Detections.empty()

    return all_tracks


def _evaluate_single_sequence(
    seq_name: str,
    seq_tracks: sv.Detections,
    dataset: Dataset,
    metrics_to_compute: Dict[str, TrackingMetric],
    placeholder_metrics: List[str],
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Evaluates tracking metrics for a single sequence.

    Loads ground truth, validates tracks, computes requested metrics, and handles
    placeholders and errors.

    Args:
        seq_name: The name of the sequence being evaluated.
        seq_tracks: The sv.Detections object containing tracking result for the sequence
                    Expected to have `tracker_id` and `data['frame_idx']`.
        dataset: The Dataset object used to load ground truth and sequence info.
        metrics_to_compute: A dictionary mapping metric names to their instantiated
                            TrackingMetric objects.
        placeholder_metrics: A list of metric names requested but not implemented.

    Returns:
        A dictionary where keys are metric names (including placeholders) and
        values are dictionaries containing the computed metric results or error/info
        messages for this sequence.
    """
    print(f"\n--- Evaluating sequence: {seq_name} ---")

    # Load ground truth
    gt_data = dataset.load_ground_truth(seq_name)
    if gt_data is None:
        print(f"Warning: No ground truth for sequence {seq_name}, skipping evaluation")
        seq_results_error = {
            metric_name: {"error": "Ground truth not found"}
            for metric_name in metrics_to_compute
        }
        seq_results_error.update(
            {
                metric_name: {"error": "Ground truth not found"}
                for metric_name in placeholder_metrics
            }
        )
        return seq_results_error

    # Load sequence info
    seq_info = dataset.get_sequence_info(seq_name)

    # --- Validate sequence tracks before passing to metric ---
    if len(seq_tracks) > 0:
        if seq_tracks.tracker_id is None:
            print(
                f"Warning: Tracks for sequence {seq_name} are missing \
                    'tracker_id'. Evaluation might fail or be incorrect."
            )
        if "frame_idx" not in seq_tracks.data:
            print(
                f"Warning: Tracks for sequence {seq_name} are missing \
                    'frame_idx' in data. Evaluation might fail or be incorrect."
            )
    # --- End Validation ---

    # Compute metrics for this sequence
    seq_results_for_this_seq: Dict[str, Dict[str, Union[float, str]]] = {}
    for metric_name, metric_instance in metrics_to_compute.items():
        try:
            metric_output = metric_instance.compute(
                ground_truth=gt_data,
                predictions=seq_tracks,
                sequence_info=seq_info,
            )
            if not isinstance(metric_output, dict):
                print(
                    f"Warning: Metric '{metric_name}' compute returned non-dict: \
                        {metric_output}. Wrapping."
                )
                seq_results_for_this_seq[metric_name] = {metric_name: metric_output}
            else:
                seq_results_for_this_seq[metric_name] = metric_output
            print(f"  {metric_name}: {seq_results_for_this_seq[metric_name]}")
        except Exception as e:
            error_msg = (
                f"Error computing metric '{metric_name}' for sequence {seq_name}: {e}"
            )
            print(error_msg)
            seq_results_for_this_seq[metric_name] = {"error": str(e)}

    # Add placeholder results
    for metric_name in placeholder_metrics:
        seq_results_for_this_seq[metric_name] = {
            "value": 0.0,
            "message": "Placeholder - Not implemented",
        }

    return seq_results_for_this_seq


def _aggregate_results(
    metric_results_by_name: Dict[str, List[Dict[str, Union[float, str]]]],
    metrics_to_compute: Dict[str, TrackingMetric],
    placeholder_metrics: List[str],
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Aggregates metric results across all evaluated sequences.

    Calls the `aggregate` method of each TrackingMetric instance for implemented
    metrics and handles placeholders and errors.

    Args:
        metric_results_by_name: A dictionary mapping metric names to lists of
                                per-sequence result dictionaries.
        metrics_to_compute: A dictionary mapping metric names to their instantiated
                            TrackingMetric objects.
        placeholder_metrics: A list of placeholder metric names.

    Returns:
        A dictionary where keys are metric names (including placeholders) and
        values are dictionaries containing the aggregated metric results or
        error/info messages.
    """
    overall_results: Dict[str, Dict[str, Union[str, float]]] = {}

    for metric_name, metric_instance in metrics_to_compute.items():
        raw_seq_outputs = metric_results_by_name[metric_name]
        valid_seq_outputs: List[Dict[str, float]] = [
            res
            for res in raw_seq_outputs
            if isinstance(res, dict) and "error" not in res
        ]

        if not valid_seq_outputs:
            if any(isinstance(res, dict) and "error" in res for res in raw_seq_outputs):
                overall_results[metric_name] = {
                    "message": f"Aggregation skipped due to errors in sequence \
                        results for {metric_name}"
                }
            else:
                overall_results[metric_name] = {
                    "message": f"No valid sequence results to aggregate for \
                        {metric_name}"
                }
            print(overall_results[metric_name].get("message", "Aggregation issue"))
            continue

        try:
            aggregated_result = metric_instance.aggregate(valid_seq_outputs)
            overall_results[metric_name] = aggregated_result
        except Exception as e:
            print(f"Error during aggregation for metric '{metric_name}': {e}")
            overall_results[metric_name] = {"error": f"Aggregation failed: {e}"}

    for metric_name in placeholder_metrics:
        raw_seq_outputs = metric_results_by_name[metric_name]
        has_errors = any(
            isinstance(res, dict) and "error" in res for res in raw_seq_outputs
        )
        if has_errors:
            overall_results[metric_name] = {
                "value": 0.0,
                "message": f"Placeholder ({metric_name}) \
                    - Errors in sequence results",
            }
        else:
            overall_results[metric_name] = {
                "value": 0.0,
                "message": f"Placeholder ({metric_name}) - Not implemented",
            }

    return overall_results


def evaluate_tracks(
    dataset: Dataset,
    tracks: Optional[Dict[str, sv.Detections]] = None,
    tracks_path: Optional[Union[str, Path]] = None,
    metrics: List[str] = ["HOTA", "CLEAR", "Count"],
) -> Dict[str, Any]:
    """
    Evaluates tracking results against ground truth using specified metrics.

    Loads tracks either directly or from disk, evaluates each sequence, and then
    aggregates the results.

    Args:
        dataset: The Dataset object containing ground truth and sequence information.
        tracks: Optional dictionary mapping sequence names (str) to sv.Detections
                objects containing all tracks for the sequence. If provided,
                `tracks_path` is ignored.
        tracks_path: Optional path (str or Path) to a directory containing saved
                     tracking results (JSON files, one per sequence, generated by
                     `save_tracks`). Used if `tracks` is None.
        metrics: A list of tracking metric names (case-insensitive) to compute
                 (e.g., ["Count", "HOTA", "CLEAR"]).

    Returns:
        A dictionary containing evaluation results, structured as:
        {
            "per_sequence": {
                "seq_name_1": {"metric_1": {...}, "metric_2": {...}, ...},
                "seq_name_2": {...},
                ...
            },
            "overall": {
                "metric_1": {...}, "metric_2": {...}, ...
            }
        }
        Metric results dictionaries contain computed values or error/info messages.
    """
    if tracks is None and tracks_path is None:
        raise ValueError("Either tracks or tracks_path must be provided")

    # --- Metric Instantiation ---
    metrics_to_compute, placeholder_metrics = instantiate_metrics(metrics)

    # Load tracks from disk if not provided
    if tracks is None:
        sequence_names = dataset.get_sequence_names()
        if not sequence_names:
            print("Warning: No sequences found in the dataset to load tracks for.")
            return {"per_sequence": {}, "overall": {}}

        tracks = load_tracks_from_disk(tracks_path or ".", sequence_names)  # type: ignore

    # Compute metrics for each sequence
    results: Dict[str, Dict[str, Any]] = {
        "per_sequence": {},
        "overall": {},
    }
    metric_results_by_name: Dict[str, List[Dict[str, Union[float, str]]]] = defaultdict(
        list
    )

    if not tracks:
        print("Warning: No tracks loaded or provided for evaluation.")
        for metric_name in placeholder_metrics:
            results["overall"][metric_name] = {
                "value": 0.0,
                "message": f"Placeholder ({metric_name}) - No tracks evaluated",
            }
        for metric_name in metrics_to_compute:
            results["overall"][metric_name] = {
                "message": f"No tracks evaluated for {metric_name}"
            }
        return results

    for seq_name, seq_tracks in tracks.items():
        seq_results_for_this_seq = _evaluate_single_sequence(
            seq_name, seq_tracks, dataset, metrics_to_compute, placeholder_metrics
        )

        results["per_sequence"][seq_name] = seq_results_for_this_seq
        for metric_name, metric_output in seq_results_for_this_seq.items():
            metric_results_by_name[metric_name].append(metric_output)

    # --- Aggregate overall metrics ---
    if metric_results_by_name:
        results["overall"] = _aggregate_results(
            metric_results_by_name, metrics_to_compute, placeholder_metrics
        )

    return results


def evaluate_tracker(
    dataset: Dataset,
    detection_source: Union[
        MOTChallengeDataset,
        sv.DetectionDataset,
        Callable[[Optional[np.ndarray], Dict[str, Any]], sv.Detections],
    ],
    tracker_source: Union[
        BaseTracker,
        Callable[[sv.Detections, Optional[np.ndarray], Dict[str, Any]], sv.Detections],
    ],
    cache_tracks: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
    metrics: List[str] = ["HOTA", "CLEAR", "Count"],
    image_loader: Optional[Callable[[str], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    End-to-end tracker evaluation function.

    Combines track generation (`generate_tracks`) and evaluation (`evaluate_tracks`).

    Args:
        dataset: The Dataset object providing sequences, frames, and ground truth.
        detection_source: Source providing sv.Detections per frame. Can be:
                          - MOTChallengeDataset (uses loaded public detections).
                          - sv.DetectionDataset (uses annotations keyed by image path).
                          - A callback function: `(frame, frame_info) -> sv.Detections`.
                            The frame can be None if image loading fails.
        tracker_source: Source providing tracked sv.Detections. Can be:
                        - A BaseTracker object (implements update method).
                        - A callback function:
                          `(detections, frame, frame_info) -> sv.Detections`.
                          The frame can be None if image loading fails.
                          *Note: The callback function should handle resetting
                          the tracker state internally if needed (e.g., at frame 1).*
        cache_tracks: If True, saves the generated tracks to disk using `cache_dir`.
        cache_dir: Directory path (str or Path) to save tracking results if
                   `cache_tracks` is True.
        metrics: A list of tracking metric names (case-insensitive) to compute
                 (e.g., ["Count", "HOTA", "CLEAR"]).
        image_loader: Optional custom image loading function (passed to
                      `generate_tracks`).

    Returns:
        A dictionary containing the evaluation metrics, structured as returned by
        `evaluate_tracks`.
    """
    # Generate tracks
    output_dir = cache_dir if cache_tracks else None
    tracks = generate_tracks(
        dataset=dataset,
        detection_source=detection_source,
        tracker_source=tracker_source,
        output_dir=output_dir,
        image_loader=image_loader,
    )

    # Evaluate tracks
    results = evaluate_tracks(
        dataset=dataset,
        tracks=tracks,  # Pass the Dict[str, sv.Detections]
        metrics=metrics,
    )

    return results


# --- Example Usage (Illustrative) ---
if __name__ == "__main__":
    # 1. Instantiate a dataset object
    try:
        mot_dataset_path = Path("./data/mot/MOT17/train")  # Example path
        if not mot_dataset_path.exists():
            print(f"ERROR: Dataset path does not exist: {mot_dataset_path}")
            print("Please update the path in the __main__ block.")
            exit()
        mot_dataset = MOTChallengeDataset(dataset_path=mot_dataset_path)

        # --- Example: Accessing dataset info ---
        print("Available sequences:", mot_dataset.get_sequence_names())
        if mot_dataset.get_sequence_names():
            seq_name_example = mot_dataset.get_sequence_names()[0]
            print(f"\nInfo for sequence '{seq_name_example}':")
            print(mot_dataset.get_sequence_info(seq_name_example))

            print(f"\nLoading GT for '{seq_name_example}':")
            gt_data = mot_dataset.load_ground_truth(seq_name_example)
            if gt_data is not None:
                print(f"Loaded GT with {len(gt_data)} entries. First 5:")
                print(gt_data[:5])  # Print first 5 entries of the list
            else:
                print("GT data not loaded.")

            print(f"\nFirst 5 frames for '{seq_name_example}':")
            frame_count = 0
            for frame_info in mot_dataset.get_frame_iterator(seq_name_example):
                print(frame_info)
                frame_count += 1
                if frame_count >= 5:
                    break
        # --- End example access ---

        # 2. Load or create detection source
        # Example: Using MOTChallengeDataset with public detections
        mot_dataset.load_public_detections(min_confidence=0.1)  # Load detections

        # 3. Instantiate a tracker object
        tracker = SORTTracker()  # Use SORT tracker

        # Run evaluation
        print("\n--- Starting Evaluation ---")
        results = evaluate_tracker(
            dataset=mot_dataset,
            detection_source=mot_dataset,  # Using detections from MOTChallengeDataset
            tracker_source=tracker,
            metrics=["Count", "HOTA", "CLEAR"],  # Specify desired metrics
            cache_tracks=True,  # Example: cache the tracks
            cache_dir="./cached_tracks_sv",  # Example cache directory
        )
        print("\n--- Evaluation Results ---")
        print(json.dumps(results, indent=2))

        # Example of evaluating from cached tracks
        print("\n--- Evaluating Cached Tracks ---")
        cache_path = "./cached_tracks_sv_merged"
        if Path(cache_path).exists():
            results_from_cache = evaluate_tracks(
                dataset=mot_dataset,
                tracks_path=cache_path,  # Point to the cache directory
                metrics=["Count"],  # Evaluate only Count from cache for demo
            )
            print("\n--- Cached Evaluation Results (Count only) ---")
            print(json.dumps(results_from_cache, indent=2))
        else:
            print(
                f"Cache directory '{cache_path}' not found, skipping cached evaluation."
            )

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ImportError as e:
        print(f"Import Error: {e}. Make sure all dependencies are installed.")
    except Exception as e:
        import traceback

        print(f"An unexpected error occurred: {e}")
        print(traceback.format_exc())
