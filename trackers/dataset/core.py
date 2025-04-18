import abc
import configparser
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import supervision as sv


# --- Base Dataset ---
class Dataset(abc.ABC):
    """Abstract base class for datasets used in tracking evaluation."""

    @abc.abstractmethod
    def load_ground_truth(self, sequence_name: str) -> Optional[sv.Detections]:
        """
        Loads ground truth data for a specific sequence.

        Args:
            sequence_name: The name of the sequence.

        Returns:
            An sv.Detections object containing ground truth annotations, or None
            if ground truth is not available or cannot be loaded. The Detections
            object should ideally include `tracker_id` and `data['frame_idx']`.
        """
        pass

    @abc.abstractmethod
    def get_sequence_names(self) -> List[str]:
        """Returns a list of sequence names available in the dataset."""
        pass

    @abc.abstractmethod
    def get_sequence_info(self, sequence_name: str) -> Dict[str, Any]:
        """
        Returns metadata for a specific sequence.

        Args:
            sequence_name: The name of the sequence.

        Returns:
            A dictionary containing sequence information (e.g., 'frame_rate',
            'seqLength', 'img_width', 'img_height', 'img_dir'). Keys and value
            types may vary depending on the dataset format.
        """
        pass

    @abc.abstractmethod
    def get_frame_iterator(self, sequence_name: str) -> Iterator[Dict[str, Any]]:
        """
        Returns an iterator over frame information dictionaries for a sequence.

        Args:
            sequence_name: The name of the sequence.

        Yields:
            Dictionaries, each representing a frame. Each dictionary should
            contain at least 'frame_idx' (int, typically 1-based) and
            'image_path' (str, absolute path recommended).
        """
        pass


class MOTChallengeDataset(Dataset):
    """
    Dataset class for loading sequences in the MOTChallenge format.
    Handles parsing `seqinfo.ini`, `gt/gt.txt`, and optionally `det/det.txt`.

    Expected directory structure:
        dataset_path/
            sequence_name_1/
                seqinfo.ini
                gt/
                    gt.txt      # Ground truth annotations
                img1/           # Image frames (directory name from seqinfo.ini)
                    000001.jpg
                    ...
                det/            # Optional, for public detections
                    det.txt     # Format: frame,id,bb_left,bb_top,w,h,conf,-1,-1,-1
            sequence_name_2/
                ...
    """

    def __init__(self, dataset_path: Union[str, Path]):
        """
        Initializes the MOTChallengeDataset.

        Args:
            dataset_path: Path (str or Path object) to the root directory of the
                          MOTChallenge dataset (e.g., `/path/to/MOT17/train`).

        Raises:
            FileNotFoundError: If the `dataset_path` does not exist or is not a
            directory.
        """
        self.root_path = Path(dataset_path)
        if not self.root_path.is_dir():
            raise FileNotFoundError(f"Dataset path not found: {self.root_path}")
        self._sequence_names = self._find_sequences()
        self._public_detections: Optional[Dict[str, sv.Detections]] = (
            None  # Cache for public detections
        )

    def _find_sequences(self) -> List[str]:
        """
        Finds valid sequence directories (containing seqinfo.ini) within
        the root path.
        """
        sequences = []
        for item in self.root_path.iterdir():
            # Check if it's a directory and contains seqinfo.ini
            if item.is_dir() and (item / "seqinfo.ini").exists():
                sequences.append(item.name)
        if not sequences:
            print(f"Warning: No valid MOTChallenge sequences found in {self.root_path}")
        return sorted(sequences)

    def _parse_mot_file(
        self, file_path: Path, min_confidence: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
        """
        Parses a MOT format file (gt.txt or det.txt) into structured dictionaries.

        Handles comma-separated values and converts bounding boxes from xywh to xyxy.

        Args:
            file_path: Path object pointing to the MOT format file.
            min_confidence: Optional minimum confidence threshold. Detections below
                            this threshold will be ignored.

        Returns:
            A tuple containing:
            - A list of dictionaries, where each dictionary represents a single
              detection/annotation line parsed from the file. Keys include
              'frame_idx', 'obj_id', 'xyxy', 'confidence', 'class_id'.
            - A dictionary mapping frame indices (int) to lists of detection/annotation
              dictionaries belonging to that frame.
            Returns ([], {}) if the file doesn't exist or an error occurs during parsing
        """
        if not file_path.exists():
            return [], {}

        all_detections = []
        frame_detections: Dict[int, List[Dict[str, Any]]] = {}

        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(",")
                    if len(parts) < 7:  # Need at least 7 columns
                        print(
                            f"Warning: Skipping malformed line in {file_path}: {line}"
                        )
                        continue

                    try:
                        frame_idx = int(parts[0])
                        obj_id = int(parts[1])
                        x = float(parts[2])
                        y = float(parts[3])
                        width = float(parts[4])
                        height = float(parts[5])
                        confidence = float(parts[6])
                        class_id = int(parts[7]) if len(parts) > 7 else -1

                        # Filter by confidence if specified
                        if min_confidence is not None and confidence < min_confidence:
                            continue

                        # Convert from xywh to xyxy format
                        detection = {
                            "frame_idx": frame_idx,
                            "obj_id": obj_id,
                            "xyxy": [x, y, x + width, y + height],
                            "confidence": confidence,
                            "class_id": class_id,
                        }

                        all_detections.append(detection)

                        # Group by frame
                        if frame_idx not in frame_detections:
                            frame_detections[frame_idx] = []
                        frame_detections[frame_idx].append(detection)

                    except ValueError as ve:
                        print(
                            f"Warning: Skipping line with invalid numeric data in \
                                {file_path}: {line} ({ve})"
                        )
                        continue

            return all_detections, frame_detections

        except Exception as e:
            print(f"Error parsing MOT file {file_path}: {e}")
            return [], {}

    def load_ground_truth(self, sequence_name: str) -> Optional[sv.Detections]:
        """
        Loads ground truth data for a specific sequence from the `gt/gt.txt` file.

        Parses the file and converts annotations into an sv.Detections object.
        Frame indices are stored in `sv.Detections.data['frame_idx']`.

        Args:
            sequence_name: The name of the sequence (e.g., 'MOT17-02-SDP').

        Returns:
            An sv.Detections object containing all ground truth annotations for the
            sequence, or None if the `gt.txt` file doesn't exist or an error occurs
            during loading/parsing. Returns `sv.Detections.empty()` if the file
            exists but contains no valid annotations.
        """
        gt_path = self.root_path / sequence_name / "gt" / "gt.txt"
        if not gt_path.exists():
            print(
                f"Warning: Ground truth file not found for sequence \
                    {sequence_name} at {gt_path}"
            )
            return None

        try:
            all_detections, _ = self._parse_mot_file(gt_path)

            if not all_detections:
                print(f"Warning: No valid annotations found in {gt_path}")
                return sv.Detections.empty()

            # Extract data from parsed detections
            boxes = [det["xyxy"] for det in all_detections]
            confidence_scores = [det["confidence"] for det in all_detections]
            class_ids = [det["class_id"] for det in all_detections]
            track_ids = [det["obj_id"] for det in all_detections]
            frame_indices = [det["frame_idx"] for det in all_detections]

            # Convert lists to numpy arrays
            xyxy = np.array(boxes, dtype=np.float32)
            confidence = np.array(confidence_scores, dtype=np.float32)
            class_id = np.array(class_ids, dtype=np.int32)
            tracker_id = np.array(track_ids, dtype=np.int32)
            frame_idx = np.array(frame_indices, dtype=np.int32)

            # Create sv.Detections object with frame indices in data
            return sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                tracker_id=tracker_id,
                data={"frame_idx": frame_idx},
            )

        except Exception as e:
            print(f"Error loading ground truth for {sequence_name}: {e}")
            return None

    def get_sequence_names(self) -> List[str]:
        """Returns a sorted list of sequence names found in the dataset directory."""
        return self._sequence_names

    def get_sequence_info(self, sequence_name: str) -> Dict[str, Any]:
        """
        Parses the `seqinfo.ini` file for a given sequence and returns its contents.

        Attempts to convert values to int or float where possible. Standardizes
        common keys like 'frame_rate', 'seqLength', 'img_width', 'img_height',
        'img_dir', 'name'.

        Args:
            sequence_name: The name of the sequence.

        Returns:
            A dictionary containing sequence information. Returns an empty dictionary
            if `seqinfo.ini` is not found, cannot be parsed, or lacks the
            '[Sequence]' section. The 'img_dir' value will be a Path object.
        """
        seq_info_path = self.root_path / sequence_name / "seqinfo.ini"
        if not seq_info_path.exists():
            print(f"Warning: seqinfo.ini not found for sequence {sequence_name}")
            return {}

        config = configparser.ConfigParser()
        try:
            config.read(seq_info_path)
            if "Sequence" in config:
                # Convert values to appropriate types (int, float, str)
                info: Dict[str, Union[int, float, str]] = {}
                for key, value in config["Sequence"].items():
                    try:
                        # Attempt to convert to int, then float, else keep as string
                        info[key] = int(value)
                    except ValueError:
                        try:
                            info[key] = float(value)
                        except ValueError:
                            info[key] = value
                # Ensure standard keys exist (case-insensitive matching)

                standard_info: Dict[str, Union[int, float, str, Path, None]] = {
                    "frame_rate": info.get("framerate"),
                    "seqLength": info.get("seqlength"),
                    "img_width": info.get("imwidth"),
                    "img_height": info.get("imheight"),
                    "img_dir": self.root_path
                    / sequence_name
                    / str(
                        info.get("imdir", "img1")
                    ),  # Convert to string to ensure Path compatibility
                    "name": info.get("name", sequence_name),
                }
                # Filter out None values if keys weren't present
                return {k: v for k, v in standard_info.items() if v is not None}
            else:
                print(f"Warning: '[Sequence]' section not found in {seq_info_path}")
                return {}
        except configparser.Error as e:
            print(f"Error parsing {seq_info_path}: {e}")
            return {}

    def get_frame_iterator(self, sequence_name: str) -> Iterator[Dict[str, Any]]:
        """
        Returns an iterator yielding information about each frame in a sequence.

        Determines frame count and image directory from `seqinfo.ini`. Infers the
        image file extension based on the first frame found.

        Args:
            sequence_name: The name of the sequence.

        Yields:
            Dictionaries, each containing:
            - 'frame_idx': The frame number (int, 1-based).
            - 'image_path': The absolute path to the image file (str).
            Yields nothing if sequence info is incomplete or image files
            cannot be found.
        """
        seq_info = self.get_sequence_info(sequence_name)
        num_frames = seq_info.get("seqLength")
        img_dir = seq_info.get(
            "img_dir"
        )  # Already a Path object from get_sequence_info

        if num_frames is None or img_dir is None or not img_dir.is_dir():
            print(
                f"Warning: Could not determine frame count or image directory for \
                    {sequence_name}. Check seqinfo.ini."
            )
            return  # Return empty iterator

        # Determine image file extension (common are .jpg, .png)
        # Look for the first file to determine the extension
        first_frame_pattern = f"{1:06d}.*"
        potential_files = list(img_dir.glob(first_frame_pattern))
        if not potential_files:
            print(
                f"Warning: No image files found matching pattern \
                    '{first_frame_pattern}' in {img_dir}"
            )
            # Try common extensions explicitly if glob fails
            if (img_dir / f"{1:06d}.jpg").exists():
                img_ext = ".jpg"
            elif (img_dir / f"{1:06d}.png").exists():
                img_ext = ".png"
            else:
                print(
                    f"Warning: Could not determine image extension for sequence \
                        {sequence_name}."
                )
                return  # Cannot proceed without knowing the extension
        else:
            img_ext = potential_files[0].suffix  # e.g., '.jpg'

        for i in range(1, num_frames + 1):
            frame_filename = f"{i:06d}{img_ext}"  # Use determined extension
            frame_path = img_dir / frame_filename
            if not frame_path.exists():
                print(f"Warning: Expected frame image not found: {frame_path}")
                # Decide whether to skip or raise error. Skipping for now.
                continue

            yield {
                "frame_idx": i,
                # --- Store absolute path ---
                "image_path": str(
                    frame_path.resolve()
                ),  # Use absolute path for reliable dict key
                # --- End change ---
                # Add other relevant info if needed, e.g., timestamp (if available)
            }

    # --- Methods for Public Detections ---

    def load_public_detections(self, min_confidence: Optional[float] = None) -> None:
        """
        Loads public detections from `det/det.txt` for all sequences into memory.

        Parses `det.txt` files and stores detections in an internal cache, keyed
        by the absolute image path. Detections are stored as sv.Detections objects.

        Args:
            min_confidence: Optional minimum detection confidence score to include.
                            If None, all detections are loaded.
        """
        print("Loading public detections...")
        self._public_detections = {}
        loaded_count = 0
        total_dets = 0

        for seq_name in self.get_sequence_names():
            det_path = self.root_path / seq_name / "det" / "det.txt"
            if not det_path.exists():
                print(f"  Info: No det.txt found for sequence {seq_name}")
                continue

            try:
                # Load detections using common parser
                _, frame_detections = self._parse_mot_file(det_path, min_confidence)

                if not frame_detections:
                    continue

                loaded_count += 1
                seq_total_dets = 0

                # Get frame iterator to map frame index to image path
                frame_map = {
                    info["frame_idx"]: info["image_path"]
                    for info in self.get_frame_iterator(seq_name)
                }

                for frame_idx, detections in frame_detections.items():
                    if frame_idx not in frame_map:
                        print(
                            f"  Warning: Detections found for frame {frame_idx} \
                                outside sequence length in {seq_name}. Skipping."
                        )
                        continue

                    image_path = frame_map[frame_idx]

                    # Prepare arrays for sv.Detections
                    xyxy = np.array([det["xyxy"] for det in detections])
                    confidence = np.array([det["confidence"] for det in detections])

                    self._public_detections[image_path] = sv.Detections(
                        xyxy=xyxy,
                        confidence=confidence,
                        class_id=None,  # MOT public detections don't have class IDs
                    )
                    seq_total_dets += len(detections)

                print(f"  Loaded {seq_total_dets} detections for sequence {seq_name}")
                total_dets += seq_total_dets

            except Exception as e:
                print(f"  Error loading detections for sequence {seq_name}: {e}")

        print(
            f"Finished loading public detections. Found {total_dets} \
                detections across {loaded_count} sequences."
        )
        if not self._public_detections:
            print("Warning: No public detections were loaded.")

    @property
    def has_public_detections(self) -> bool:
        """Returns True if public detections have been loaded via
        `load_public_detections`."""
        return self._public_detections is not None

    def get_public_detections(self, image_path: str) -> sv.Detections:
        """
        Retrieves the loaded public detections associated with a specific image path.

        Requires `load_public_detections()` to have been called first.

        Args:
            image_path: The absolute path (str) to the image file.

        Returns:
            An sv.Detections object containing the public detections for the given
            image path. Returns `sv.Detections.empty()` if no detections were loaded
            for this path or if `load_public_detections()` was not called.
        """
        if not self.has_public_detections:
            print(
                "Warning: Public detections requested but not loaded. \
            # Call load_public_detections() first."
            )
            return sv.Detections.empty()
        # Ensure consistent path format (absolute)
        abs_image_path = str(Path(image_path).resolve())
        # Fix the None check - use empty dict if _public_detections is None
        return (self._public_detections or {}).get(
            abs_image_path, sv.Detections.empty()
        )

    # --- End Methods for Public Detections ---
