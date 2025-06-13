import abc
from typing import Any, Dict, Iterator, List, Optional, Tuple

import supervision as sv


# --- Base Dataset ---
class EvaluationDataset(abc.ABC):
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

    @abc.abstractmethod
    def preprocess(
        self,
        ground_truth: sv.Detections,
        predictions: sv.Detections,
        iou_threshold: float = 0.5,
        remove_distractor_matches: bool = True,
    ) -> Tuple[sv.Detections, sv.Detections]:
        """
        Applies dataset-specific preprocessing steps to ground truth and predictions.

        This typically involves filtering unwanted annotations (e.g., distractors,
        zero-marked GTs) and potentially relabeling IDs.

        Args:
            ground_truth (sv.Detections): Raw ground truth detections for a sequence.
            predictions (sv.Detections): Raw prediction detections for a sequence.
            iou_threshold (float): IoU threshold used for matching during preprocessing
                                 (e.g., for removing predictions matching distractors).
            remove_distractor_matches (bool): Flag indicating whether to remove
                                              predictions matched to distractors.

        Returns:
            Tuple[sv.Detections, sv.Detections]: A tuple containing the processed
            ground truth detections and processed prediction detections.
        """
        pass
