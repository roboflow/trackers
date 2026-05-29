# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import torch

from trackers.core.mcbyte.masks.base import MaskGenerator, MaskOutput, TrackletSnapshot

logger = logging.getLogger(__name__)

SAM_CHECKPOINT_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

SAM_DEFAULT_CHECKPOINT_PATHS = {
    "vit_b": Path("models/sam/sam_vit_b_01ec64.pth"),
}


def _ensure_checkpoint_exists(
    checkpoint_path: Path,
    model_type: str,
) -> None:
    """Download the default SAM checkpoint if it is not available locally."""
    if checkpoint_path.exists():
        return

    checkpoint_url = SAM_CHECKPOINT_URLS.get(model_type)
    if checkpoint_url is None:
        raise ValueError(f"No default checkpoint URL for model_type={model_type!r}.")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading SAM checkpoint to %s", checkpoint_path)

    urlretrieve(
        checkpoint_url,
        checkpoint_path,
    )


class SAMBoxMaskGenerator(MaskGenerator):
    """Generate binary masks from tracklet bounding boxes using Segment Anything.

    The generator is stateless with respect to tracking: it receives a frame and a
    list of tracklet snapshots, prompts SAM with one box per tracklet, and returns
    one binary mask per input tracklet.

    Returned mask indices are local array indices, starting from zero. Persistent
    mapping to global mask/object IDs is intentionally left to MaskManager or a
    mask propagation component.

    Args:
        checkpoint_path: Optional path to the SAM checkpoint file. If not
            provided, the default checkpoint path for ``model_type`` is used.
        model_type: SAM model type. Currently only ``"vit_b"`` has a default
            checkpoint URL/path.
        device: Device used by SAM, for example ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_type: str = "vit_b",
        device: str = "cpu",
    ) -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            msg = (
                "SAM support requires Segment Anything. "
                "Install it via `pip install "
                "git+https://github.com/facebookresearch/segment-anything.git`."
            )
            raise ImportError(msg) from exc

        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else SAM_DEFAULT_CHECKPOINT_PATHS[model_type]
        )

        _ensure_checkpoint_exists(
            checkpoint_path=self.checkpoint_path,
            model_type=model_type,
        )
        self.device = torch.device(device)

        sam_model = sam_model_registry[model_type](checkpoint=str(self.checkpoint_path))
        sam_model.to(device=self.device)

        self.predictor = SamPredictor(sam_model)

    def generate(
        self,
        frame: np.ndarray,
        tracklets: list[TrackletSnapshot],
    ) -> MaskOutput:
        height, width = frame.shape[:2]
        """Generate one binary mask per tracklet bounding box.

        Args:
            frame: Current RGB frame with shape ``(H, W, 3)``.
            tracklets: Tracklet snapshots containing tracker IDs and ``xyxy``
                bounding boxes.

        Returns:
            MaskOutput containing masks with shape ``(N, H, W)``, where ``N`` is
            the number of input tracklets. ``tracklet_mask_dict`` maps each
            tracker ID to its local mask index in the returned mask array.
        """

        if len(tracklets) == 0:
            return MaskOutput(
                masks=np.zeros((0, height, width), dtype=bool),
                tracklet_mask_dict={},
                mask_avg_prob_dict=None,
            )

        boxes = np.array([tracklet.xyxy for tracklet in tracklets], dtype=np.float32)

        self.predictor.set_image(frame)

        box_tensor = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            box_tensor,
            frame.shape[:2],
        )

        # McByte expects one mask per box, hence multimask_output=False
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks_np = self._convert_masks(masks)
        tracklet_mask_dict = {tracklet.tracker_id: mask_index for mask_index, tracklet in enumerate(tracklets)}

        return MaskOutput(
            masks=masks_np,
            tracklet_mask_dict=tracklet_mask_dict,
            mask_avg_prob_dict=None,
        )

    def _convert_masks(self, masks: Any) -> np.ndarray:
        """Convert SAM mask tensor to McByte mask format.

        SAM returns masks with shape ``(N, C, H, W)``, where ``C`` is the number
        of candidate masks per prompt. With ``multimask_output=False``, ``C`` is
        expected to be 1. McByte keeps one binary mask per tracklet, so this method
        converts the tensor to a NumPy array with shape ``(N, H, W)``.
        """
        masks_np = masks.detach().cpu().numpy()

        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]

        return masks_np.astype(bool)
