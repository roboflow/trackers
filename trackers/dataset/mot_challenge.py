import configparser
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.dataset.base import EvaluationDataset
from trackers.dataset.utils import relabel_ids
from trackers.log import get_logger

# --- Define MOT Constants needed for preprocessing ---
MOT_PEDESTRIAN_ID = 1
MOT_DISTRACTOR_IDS = [
    2,
    7,
    8,
    12,
]  # person_on_vehicle, static_person, distractor, reflection
MOT_IGNORE_IDS = [2, 7, 8, 12, 13]  # Includes crowd (13) for ignore, adjust as needed
ZERO_MARKED_EPSILON = 1e-5


logger = get_logger(__name__)


class MOTChallengeDataset(EvaluationDataset):
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
        self._public_detections: Dict[str, sv.Detections] = {}
        self._frame_maps: Dict[str, Dict[int, str]] = {}

    def _find_sequences(self) -> List[str]:
        """
        Finds valid sequence directories (containing seqinfo.ini) within
        the root path.
        """
        sequences = []
        for item in self.root_path.iterdir():
            if item.is_dir() and (item / "seqinfo.ini").exists():
                sequences.append(item.name)
        if not sequences:
            logger.warning(f"No valid MOTChallenge sequences found in {self.root_path}")
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
                    if len(parts) < 7:
                        logger.warning(
                            f"Skipping malformed line in {file_path}: {line}"
                        )
                        continue

                    try:
                        frame_idx = int(parts[0])
                        obj_id = int(parts[1])
                        x = float(parts[2])
                        y = float(parts[3])
                        width = float(parts[4])
                        height = float(parts[5])
                        # --- Ensure 7th column (index 6) is assigned to confidence ---
                        # This value often represents visibility/ignore flag in gt.txt
                        # (0=ignore)
                        # or detection confidence in det.txt
                        confidence = float(parts[6])

                        class_id = int(parts[7]) if len(parts) > 7 else -1

                        if min_confidence is not None and confidence < min_confidence:
                            continue

                        detection = {
                            "frame_idx": frame_idx,
                            "obj_id": obj_id,
                            "xyxy": [x, y, x + width, y + height],
                            "confidence": confidence,
                            "class_id": class_id,
                        }

                        all_detections.append(detection)

                        if frame_idx not in frame_detections:
                            frame_detections[frame_idx] = []
                        frame_detections[frame_idx].append(detection)

                    except ValueError as ve:
                        logger.warning(
                            f"Skipping line with invalid numeric data in {file_path}: \
                                {line} ({ve})"
                        )
                        continue

            return all_detections, frame_detections

        except Exception as e:
            logger.error(f"Error parsing MOT file {file_path}: {e}")
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
            logger.warning(
                f"Ground truth file not found for sequence {sequence_name} at {gt_path}"
            )
            return None

        try:
            all_detections, _ = self._parse_mot_file(gt_path)

            if not all_detections:
                logger.warning(f"No valid annotations found in {gt_path}")
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
            logger.error(f"Error loading ground truth for {sequence_name}: {e}")
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
            logger.warning(f"seqinfo.ini not found for sequence {sequence_name}")
            return {}

        config = configparser.ConfigParser()
        try:
            config.read(seq_info_path)
            if "Sequence" in config:
                # Convert values to appropriate types (int, float, str)
                info: Dict[str, Union[int, float, str]] = {}
                for key, value in config["Sequence"].items():
                    try:
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
                logger.warning(f"'[Sequence]' section not found in {seq_info_path}")
                return {}
        except configparser.Error as e:
            logger.error(f"Error parsing {seq_info_path}: {e}")
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
        img_dir = seq_info.get("img_dir")

        if num_frames is None or img_dir is None or not img_dir.is_dir():
            logger.warning(
                f"Could not determine frame count or image directory for \
                    {sequence_name}. Check seqinfo.ini."
            )
            return  # Return empty iterator

        # Look for the first file to determine the extension
        first_frame_pattern = f"{1:06d}.*"
        potential_files = list(img_dir.glob(first_frame_pattern))
        if not potential_files:
            logger.warning(
                f"No image files found matching pattern '{first_frame_pattern}' \
                    in {img_dir}"
            )
            # Try common extensions explicitly if glob fails
            if (img_dir / f"{1:06d}.jpg").exists():
                img_ext = ".jpg"
            elif (img_dir / f"{1:06d}.png").exists():
                img_ext = ".png"
            else:
                logger.warning(
                    f"Could not determine image extension for sequence {sequence_name}."
                )
                return
        else:
            img_ext = potential_files[0].suffix

        for i in range(1, num_frames + 1):
            frame_filename = f"{i:06d}{img_ext}"
            frame_path = img_dir / frame_filename
            if not frame_path.exists():
                logger.warning(f"Expected frame image not found: {frame_path}")
                # Decide whether to skip or raise error. Skipping for now.
                continue

            yield {
                "frame_idx": i,
                "image_path": str(
                    frame_path.resolve()
                ),  # Use absolute path for reliable dict key
            }

    def load_public_detections(self, min_confidence: Optional[float] = None) -> None:
        """
        Loads public detections from `det/det.txt` for all sequences into memory.

        Parses `det.txt` files and stores detections in an internal cache, keyed
        by the absolute image path. Detections are stored as sv.Detections objects.

        Args:
            min_confidence: Optional minimum detection confidence score to include.
                            If None, all detections are loaded.
        """
        logger.info("Loading public detections...")
        self._public_detections = {}
        self._frame_maps = {}
        loaded_count = 0
        total_dets = 0

        for seq_name in self.get_sequence_names():
            det_path = self.root_path / seq_name / "det" / "det.txt"
            if not det_path.exists():
                logger.info(f"No det.txt found for sequence {seq_name}")
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

                self._frame_maps[seq_name] = frame_map

                for frame_idx, detections in frame_detections.items():
                    if frame_idx not in frame_map:
                        logger.warning(
                            f"Detections found for frame {frame_idx} outside sequence \
                                length in {seq_name}. Skipping."
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

                logger.info(
                    f"Loaded {seq_total_dets} detections for sequence {seq_name}"
                )
                total_dets += seq_total_dets

            except Exception as e:
                logger.error(f"Error loading detections for sequence {seq_name}: {e}")

        logger.info(
            f"Finished loading public detections. Found {total_dets} detections \
                across {loaded_count} sequences."
        )
        if not self._public_detections:
            logger.warning("No public detections were loaded.")

    @property
    def has_public_detections(self) -> bool:
        """Returns True if public detections have been loaded via
        `load_public_detections`."""
        return self._public_detections is not None

    def get_public_detections_from_image_path(self, image_path: str) -> sv.Detections:
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
            logger.warning(
                "Public detections requested but not loaded. \
                    Call load_public_detections() first."
            )
            return sv.Detections.empty()

        abs_image_path = str(Path(image_path).resolve())

        return (self._public_detections or {}).get(
            abs_image_path, sv.Detections.empty()
        )

    def get_public_detections_from_frame_index(
        self, sequence_name: str, frame_idx: int
    ) -> sv.Detections:
        """
        Retrieves the loaded public detections for a specific frame index in a
        sequence.
        Requires `load_public_detections()` to have been called first.
        Args:
            sequence_name: The name of the sequence (e.g., 'MOT17-02-SDP').
            frame_idx: The frame index (1-based).
        Returns:
            An sv.Detections object containing the public detections for the
            specified frame index. Returns `sv.Detections.empty()` if no detections
            were loaded for this frame or if `load_public_detections()` was not
            called.
        """
        if not self.has_public_detections:
            logger.warning(
                "Public detections requested but not loaded. \
                    Call load_public_detections() first."
            )
            return sv.Detections.empty()

        frame_map = self._frame_maps.get(sequence_name, {})
        abs_image_path = frame_map.get(frame_idx)

        if abs_image_path is None:
            logger.warning(
                f"No public detections found for sequence {sequence_name} at frame \
                {frame_idx}"
            )
            return sv.Detections.empty()

        return self.get_public_detections_from_image_path(abs_image_path)

    def preprocess(
        self,
        gt_dets: sv.Detections,
        pred_dets: sv.Detections,
        iou_threshold: float = 0.5,
        remove_distractor_matches: bool = True,
    ) -> Tuple[sv.Detections, sv.Detections]:
        """
        Applies MOT specific preprocessing based on TrackEval logic.

        1. Optionally removes tracker detections matched to ground truth distractors.
        2. Removes ground truth annotations marked as distractors or "zero-marked".
        3. Relabels ground truth and (processed) prediction `tracker_id`s to be
           contiguous and 0-based using the utility function.

        Requires GT detections to have `frame_idx`, `tracker_id`, `class_id`, and
        `confidence` (where confidence corresponds to MOT `gt.txt` column 7).
        Requires prediction detections to have `frame_idx` and `tracker_id`.

        Args:
            gt_dets (sv.Detections): Raw ground truth detections.
            pred_dets (sv.Detections): Raw prediction detections.
            iou_threshold (float): IoU threshold used for matching predictions to
                distractors. Defaults to 0.5.
            remove_distractor_matches (bool): If True, remove predictions matched to
                GT distractors/ignored classes. Defaults to True.

        Returns:
            Tuple[sv.Detections, sv.Detections]: A tuple containing the processed
            ground truth detections and processed prediction detections. Returns
            original inputs if required fields are missing.
        """
        gt_out_list = []
        pred_out_list = []

        if (
            gt_dets.data is None
            or "frame_idx" not in gt_dets.data
            or gt_dets.tracker_id is None
            or gt_dets.class_id is None
            or gt_dets.confidence is None
        ):
            logger.warning(
                "GT detections missing required fields "
                "(frame_idx, tracker_id, class_id, confidence) for "
                "MOT preprocessing. Skipping."
            )
            return gt_dets, pred_dets
        if (
            pred_dets.data is None
            or "frame_idx" not in pred_dets.data
            or pred_dets.tracker_id is None
        ):
            logger.warning(
                "Prediction detections missing required fields "
                "(frame_idx, tracker_id) for MOT preprocessing. Skipping."
            )
            return gt_dets, pred_dets

        all_frame_indices = sorted(
            list(set(gt_dets.data["frame_idx"]).union(set(pred_dets.data["frame_idx"])))
        )

        for frame_idx in all_frame_indices:
            gt_dets_t = gt_dets[gt_dets.data["frame_idx"] == frame_idx]
            pred_dets_t = pred_dets[pred_dets.data["frame_idx"] == frame_idx]

            pred_dets_t_filtered = pred_dets_t

            # Optionally remove tracker dets matching distractor GTs
            if remove_distractor_matches:
                to_remove_tracker_indices = np.array([], dtype=int)
                if len(gt_dets_t) > 0 and len(pred_dets_t) > 0:
                    # Match all preds against all GTs for this frame
                    similarity = sv.detection.utils.box_iou_batch(
                        gt_dets_t.xyxy, pred_dets_t.xyxy
                    )
                    match_scores = similarity.copy()
                    match_scores[
                        match_scores < iou_threshold - np.finfo("float").eps
                    ] = 0

                    match_rows, match_cols = linear_sum_assignment(
                        -match_scores
                    )  # Maximize score
                    valid_match_mask = (
                        match_scores[match_rows, match_cols] > 0 + np.finfo("float").eps
                    )
                    match_rows = match_rows[valid_match_mask]
                    match_cols = match_cols[valid_match_mask]

                    # Identify matches where GT is a distractor
                    matched_gt_classes = gt_dets_t.class_id[match_rows]
                    is_distractor_match = np.isin(
                        matched_gt_classes, MOT_DISTRACTOR_IDS
                    )
                    to_remove_tracker_indices = match_cols[is_distractor_match]

                # Filter tracker detections for the frame
                if len(to_remove_tracker_indices) > 0:
                    pred_keep_mask = np.ones(len(pred_dets_t), dtype=bool)
                    pred_keep_mask[to_remove_tracker_indices] = False
                    pred_dets_t_filtered = pred_dets_t[pred_keep_mask]

            # Remove unwanted GT dets
            gt_is_pedestrian = gt_dets_t.class_id == MOT_PEDESTRIAN_ID

            # Refined check for zero_marked: Check if confidence is very close to 0
            gt_is_effectively_zero = np.abs(gt_dets_t.confidence) < ZERO_MARKED_EPSILON
            # Also consider explicit ignore classes
            gt_is_ignore_class = np.isin(gt_dets_t.class_id, MOT_IGNORE_IDS)

            # Keep GT if it IS a pedestrian AND it is NOT effectively zero-marked
            # AND NOT an ignore class
            gt_keep_mask = (
                gt_is_pedestrian & ~gt_is_effectively_zero & ~gt_is_ignore_class
            )

            gt_dets_t_filtered = gt_dets_t[gt_keep_mask]

            # Append filtered detections for the frame
            if len(gt_dets_t_filtered) > 0:
                gt_out_list.append(gt_dets_t_filtered)
            if len(pred_dets_t_filtered) > 0:
                pred_out_list.append(pred_dets_t_filtered)

        # Merge filtered detections across all frames
        gt_processed = (
            sv.Detections.merge(gt_out_list) if gt_out_list else sv.Detections.empty()
        )
        pred_processed = (
            sv.Detections.merge(pred_out_list)
            if pred_out_list
            else sv.Detections.empty()
        )

        # Relabel IDs using the utility function
        gt_processed = relabel_ids(gt_processed)
        pred_processed = relabel_ids(pred_processed)

        return gt_processed, pred_processed
