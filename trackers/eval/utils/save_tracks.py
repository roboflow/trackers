import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import supervision as sv


def _serialize_detections(detections: sv.Detections) -> Dict[str, Any]:
    """Helper to convert sv.Detections to a JSON-serializable dict."""
    data = {}
    # Add 'frame_idx' from data if it exists
    for attr in ["xyxy", "mask", "confidence", "class_id", "tracker_id"]:
        value = getattr(detections, attr)
        if value is not None:
            data[attr] = value.tolist()  # Convert numpy arrays to lists
        else:
            data[attr] = None

    # Handle custom data - ensure 'frame_idx' is included and convert numpy arrays
    data["data"] = {}
    if detections.data:
        for key, val in detections.data.items():
            if isinstance(val, np.ndarray):
                data["data"][key] = val.tolist()
            elif isinstance(val, list):  # Assume lists are already serializable
                data["data"][key] = val
            else:
                # Attempt to serialize other types, warn if not possible
                try:
                    json.dumps(val)  # Test serialization
                    data["data"][key] = val
                except TypeError:
                    print(
                        f"Warning: Skipping non-serializable type {type(val)} \
                            in detections.data['{key}']"
                    )
                    data["data"][key] = None  # Or handle appropriately
    # Ensure frame_idx is present if it was added during generation
    if "frame_idx" not in data["data"] and hasattr(detections, "_temp_frame_indices"):
        # This is a fallback if frame_idx wasn't added correctly before merge
        print("Warning: Adding frame_idx during serialization as fallback.")
        data["data"]["frame_idx"] = detections._temp_frame_indices.tolist()

    # Metadata is usually already serializable
    data["metadata"] = detections.metadata if detections.metadata else None

    return data


def _deserialize_detections(data: Dict[str, Any]) -> sv.Detections:
    """Helper to reconstruct sv.Detections from a deserialized dict."""
    kwargs = {}
    for attr in ["xyxy", "mask", "confidence", "class_id", "tracker_id"]:
        value = data.get(attr)
        if value is not None:
            # Ensure mask has boolean dtype if present
            dtype = bool if attr == "mask" else None
            kwargs[attr] = np.array(value, dtype=dtype)
        else:
            kwargs[attr] = None

    # Handle custom data - convert lists back to numpy arrays if needed
    # Crucially, retrieve 'frame_idx'
    deserialized_data = {}
    if data.get("data"):
        for key, val in data["data"].items():
            if isinstance(val, list):
                # Convert back to numpy array - assuming lists were originally arrays
                # Special handling might be needed if lists should remain lists
                try:
                    deserialized_data[key] = np.array(val)
                except ValueError:
                    print(
                        f"Warning: Could not convert list back to numpy array \
                            for data['{key}']. Keeping as list."
                    )
                    deserialized_data[key] = val
            else:
                deserialized_data[key] = val  # Keep other types as is
    kwargs["data"] = (
        deserialized_data if deserialized_data else {}
    )  # Ensure data is a dict

    kwargs["metadata"] = data.get("metadata")

    # Basic validation: Check if essential arrays have consistent lengths if they exist
    num_dets = None
    essential_arrays = ["xyxy", "tracker_id"]  # Add others if always expected
    if kwargs.get("xyxy") is not None:
        num_dets = len(kwargs["xyxy"])

    if num_dets is not None:
        for key in essential_arrays:
            arr = kwargs.get(key)
            if arr is not None and len(arr) != num_dets:
                print(
                    f"Warning: Deserialized array '{key}' length ({len(arr)}) \
                        mismatch with 'xyxy' length ({num_dets})."
                )
                # Handle mismatch? e.g., return empty detections or raise error
                # return sv.Detections.empty() # Example: return empty on error

    # Ensure frame_idx exists in data if detections are present
    if num_dets is not None and num_dets > 0 and "frame_idx" not in kwargs["data"]:
        print("Warning: Deserialized detections are missing 'frame_idx' in data.")
        # Handle missing frame_idx? Assign default? Raise error?

    return sv.Detections(**kwargs)


def save_tracks(merged_detections, seq_name, output_dir):
    """Save tracks to disk in the specified directory"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / f"{seq_name}_tracks.json"
        # Serialize the single sv.Detections object
        serializable_tracks = _serialize_detections(merged_detections)
        try:
            with open(output_path, "w") as f:
                json.dump(serializable_tracks, f, indent=2)
            return True
        except TypeError as e:
            print(f"Error serializing tracks for {seq_name} to JSON: {e}")
        except Exception as e:
            print(f"Unexpected error saving tracks for {seq_name}: {e}")
    return False


def load_tracks_from_disk(
    tracks_path: str, sequence_names: List[str]
) -> Dict[str, sv.Detections]:
    """
    Helper to load track files from disk

    Args:
        tracks_path: Directory containing JSON tracking results
        sequence_names: List of sequence names to load

    Returns:
        Dictionary mapping sequence names to loaded sv.Detections objects
    """
    tracks = {}
    print(f"Loading tracks from: {tracks_path}")
    for seq_name in sequence_names:
        track_file = Path(tracks_path) / f"{seq_name}_tracks.json"
        if track_file.exists():
            try:
                with open(track_file, "r") as f:
                    serializable_tracks = json.load(f)
                # Deserialize back to single sv.Detections object
                loaded_sequence_tracks = _deserialize_detections(serializable_tracks)
                tracks[seq_name] = loaded_sequence_tracks
                print(
                    f"  Loaded tracks for sequence: {seq_name} \
                        ({len(loaded_sequence_tracks)} detections)"
                )
            except json.JSONDecodeError as e:
                print(f"Error loading track file {track_file}: {e}")
            except Exception as e:
                print(f"Unexpected error loading or deserializing {track_file}: {e}")
        else:
            print(
                f"Warning: Track file not found for sequence \
                    {seq_name} at {track_file}"
            )
    return tracks
