# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import logging
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch

from trackers.core.masks.base import (
    MaskGenerator,
    MaskOutput,
    TrackletSnapshot,
    _resolve_auto_device,
)
from trackers.utils.downloader import _download_file

logger = logging.getLogger(__name__)

SAM_RELEASE_URL = "https://dl.fbaipublicfiles.com/segment_anything"


class SAMAsset(Enum):
    VIT_B = (
        "sam_vit_b_01ec64.pth",
        "01ec64d29a2fca3f0661936605ae66f8",
    )

    def __init__(self, filename: str, md5_hash: str) -> None:
        self.filename = filename
        self.md5_hash = md5_hash

    @property
    def url(self) -> str:
        return f"{SAM_RELEASE_URL}/{self.filename}"

    @property
    def default_path(self) -> Path:
        return Path("models/sam") / self.filename


SAM_ASSETS = {
    "vit_b": SAMAsset.VIT_B,
}


def _ensure_checkpoint_exists(
    checkpoint_path: Path,
    model_type: str,
) -> None:
    """Ensure default SAM checkpoint is available and passes checksum validation."""
    asset = SAM_ASSETS.get(model_type)
    if asset is None:
        raise ValueError(f"No default SAM asset for model_type={model_type!r}. Supported types: {sorted(SAM_ASSETS)}")

    parsed_url = urlparse(asset.url)
    if parsed_url.scheme != "https":
        raise ValueError(f"Unsupported checkpoint URL scheme: {parsed_url.scheme!r}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Ensuring SAM checkpoint exists at %s", checkpoint_path)
    _download_file(
        asset.url,
        checkpoint_path,
        md5=asset.md5_hash,
    )


def _autocast_context(use_amp: bool, device: torch.device) -> Any:
    """Return the appropriate autocast context for the current AMP setting.

    The autocast device type follows ``device`` rather than a hardcoded
    ``"cuda"`` so the helper is backend-generic. Callers gate ``use_amp`` to
    CUDA, so the returned context is a CUDA autocast on the default path and a
    no-op otherwise.

    Args:
        use_amp: Whether automatic mixed precision is requested.
        device: Device whose ``type`` selects the autocast backend.
    """
    if use_amp:
        if hasattr(torch, "amp"):
            return torch.amp.autocast(device.type, enabled=True)
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


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
        device: Device used by SAM, for example ``"cpu"``, ``"cuda"``, or
            ``"mps"``. The default ``"auto"`` resolves to CUDA when available,
            otherwise CPU. MPS is never auto-selected (measured ~an order of
            magnitude slower than CPU for this pipeline); pass ``"mps"``
            explicitly to use it.
        use_amp: Whether to use CUDA automatic mixed precision during SAM
            inference. Only takes effect when ``device`` is a CUDA device;
            it is ignored on CPU and MPS. Off by default so default runs stay
            fp32 on every backend.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_type: str = "vit_b",
        device: str = "auto",
        use_amp: bool = False,
    ) -> None:
        try:
            from segment_anything import (  # type: ignore[import-untyped]
                SamPredictor,
                sam_model_registry,
            )
        except ImportError as exc:
            msg = (
                "SAM support requires Segment Anything. "
                "Install it via `pip install "
                "git+https://github.com/facebookresearch/segment-anything.git`."
            )
            raise ImportError(msg) from exc

        asset = SAM_ASSETS.get(model_type)
        if asset is None:
            raise ValueError(f"Unsupported model_type={model_type!r}. Supported types: {sorted(SAM_ASSETS)}")
        default_checkpoint = asset.default_path
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else default_checkpoint

        _ensure_checkpoint_exists(
            checkpoint_path=self.checkpoint_path,
            model_type=model_type,
        )
        if device == "auto":
            device = _resolve_auto_device()
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"

        sam_model = sam_model_registry[model_type](checkpoint=str(self.checkpoint_path))
        sam_model.to(device=self.device)

        self.predictor = SamPredictor(sam_model)

    def generate(
        self,
        frame: np.ndarray,
        tracklets: list[TrackletSnapshot],
    ) -> MaskOutput:
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
        height, width = frame.shape[:2]

        if len(tracklets) == 0:
            return MaskOutput(
                masks=np.zeros((0, height, width), dtype=bool),
                tracklet_mask_dict={},
                mask_avg_prob_dict=None,
            )

        boxes = np.array([tracklet.xyxy for tracklet in tracklets], dtype=np.float32)

        box_tensor = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)

        with _autocast_context(self.use_amp, self.device):
            # set_image runs the SAM image encoder and predict_torch the mask
            # decoder; both are the dominant SAM cost and benefit from AMP on CUDA.
            self.predictor.set_image(frame)

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
