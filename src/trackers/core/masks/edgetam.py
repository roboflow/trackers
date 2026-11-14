# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Streaming EdgeTAM generation and propagation using one shared model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from trackers.core.masks.base import MaskGenerator, MaskOutput, MaskPropagator, TrackletSnapshot, _resolve_auto_device

if TYPE_CHECKING:
    import torch

DEFAULT_EDGETAM_MODEL = "yonigozlan/EdgeTAM-hf"
DEFAULT_EDGETAM_REVISION = "c266ce53b3fc00f0f495b583f6a116c4e57f53bb"


def _validate_frame(frame: np.ndarray) -> None:
    """Require the RGB pixel grid used by the shared mask interfaces."""
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) == 0 or frame.dtype != np.uint8:
        raise ValueError("EdgeTAM expects a nonempty RGB uint8 frame with shape (H, W, 3).")


def _unpack_masks(output: MaskOutput, frame: np.ndarray) -> dict[int, np.ndarray]:
    """Validate the complete row mapping before modifying temporal state."""
    masks = output.masks
    mapping = output.tracklet_mask_dict
    if masks is None:
        if mapping:
            raise ValueError("Mask mapping must be empty when masks are None.")
        return {}
    if masks.ndim != 3 or masks.shape[1:] != frame.shape[:2]:
        raise ValueError("Masks must have shape (N, H, W) at the frame resolution.")
    if (
        any(not isinstance(index, (int, np.integer)) or isinstance(index, bool) for index in mapping.values())
        or len(mapping) != len(masks)
        or sorted(mapping.values()) != list(range(len(masks)))
    ):
        raise ValueError("Mask mapping must contain each local mask index exactly once.")
    if any(not isinstance(key, (int, np.integer)) or isinstance(key, bool) for key in mapping):
        raise ValueError("Tracklet IDs must be integers.")
    return {int(key): masks[index].astype(bool, copy=True) for key, index in mapping.items()}


def _validate_predictions(
    masks: dict[int, np.ndarray], expected_ids: set[int], frame: np.ndarray
) -> dict[int, np.ndarray]:
    """Reject lost identities or a different pixel grid at the model boundary."""
    if set(masks) != expected_ids:
        raise RuntimeError(f"EdgeTAM returned object IDs {sorted(masks)}; expected {sorted(expected_ids)}.")
    if any(mask.shape != frame.shape[:2] for mask in masks.values()):
        raise RuntimeError("EdgeTAM returned masks at the wrong frame resolution.")
    return {key: masks[key].astype(bool, copy=True) for key in sorted(masks)}


def _pack_masks(masks: dict[int, np.ndarray], frame: np.ndarray) -> MaskOutput:
    """Return caller-owned mask arrays, without inventing confidence scores."""
    return MaskOutput(
        masks=np.stack(list(masks.values())) if masks else np.zeros((0, *frame.shape[:2]), dtype=bool),
        tracklet_mask_dict={key: index for index, key in enumerate(masks)},
    )


class EdgeTAMMaskBackend(MaskGenerator, MaskPropagator):
    """Create and propagate overlapping masks with one EdgeTAM model.

    Pass the same instance as both ``mask_generator`` and ``mask_propagator``
    to :class:`~trackers.core.masks.manager.MaskManager`. Generation uses a
    temporary session and cannot change temporal propagation state. All frames
    follow the shared mask interface's RGB convention.

    Sessions retain at most ``max_session_frames`` processed images, including
    the seed. Before accepting an image beyond that limit, the previous image
    and its masks seed a new session. This applies to every input source. The
    bound limits retained history, not total RAM/VRAM; resolution and object
    count also affect memory use. Empty masks transfer unchanged. Recovering an
    occluded object through a new box prompt is a separate tracker policy.

    Args:
        model_id: Transformers-compatible EdgeTAM checkpoint ID or local path.
        revision: Model and processor revision override. With the default model,
            ``None`` selects ``DEFAULT_EDGETAM_REVISION``; with a custom model it
            leaves revision selection to Transformers. Use a commit to pin it.
        device: ``auto`` selects CUDA when available, otherwise CPU. Explicit
            PyTorch devices such as ``mps`` are also accepted. Inference uses fp32.
        max_session_frames: Maximum images retained per temporal session, at
            least two. Defaults to 200. ``None`` disables rollover.
        mask_threshold: Logit threshold for binary masks, not a confidence gate.

    Raises:
        ImportError: If the ``edgetam`` extra is unavailable.
        ValueError: If the session bound or threshold is invalid.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_EDGETAM_MODEL,
        revision: str | None = None,
        device: str = "auto",
        max_session_frames: int | None = 200,
        mask_threshold: float = 0.0,
    ) -> None:
        """Validate configuration before loading the optional model once."""
        if max_session_frames is not None and (
            isinstance(max_session_frames, bool) or not isinstance(max_session_frames, int) or max_session_frames < 2
        ):
            raise ValueError("max_session_frames must be an integer >= 2 or None.")
        if not np.isfinite(mask_threshold):
            raise ValueError("mask_threshold must be finite.")
        self.max_session_frames = max_session_frames
        if revision is None and model_id == DEFAULT_EDGETAM_MODEL:
            revision = DEFAULT_EDGETAM_REVISION
        self._runtime = _EdgeTAMRuntime(
            model_id=model_id, revision=revision, device=device, mask_threshold=mask_threshold
        )
        self._session: Any = None
        self._last_frame: np.ndarray | None = None
        self._last_masks: dict[int, np.ndarray] = {}
        self._session_frame_count = 0
        self._session_index = 0

    @property
    def device(self) -> torch.device:
        """Resolved device of the loaded model; assigning metadata cannot move it."""
        return self._runtime.device

    @property
    def model_revision(self) -> str | None:
        """Resolved model commit, when available, for logging and configuration replay."""
        return self._runtime.model_revision

    @property
    def session_frame_count(self) -> int:
        """Number of stored images, including the seed; zero when inactive."""
        return self._session_frame_count

    @property
    def session_index(self) -> int:
        """Zero-based temporal session number since initialization or reset."""
        return self._session_index

    def reset(self) -> None:
        """Release the complete session, including frames and feature caches.

        Loaded weights and the processor remain reusable. Clearing only the upstream session's object tracking data
        would retain video history.
        """
        self._session = None
        self._last_frame = None
        self._last_masks = {}
        self._session_frame_count = 0
        self._session_index = 0

    def generate(self, frame: np.ndarray, tracklets: list[TrackletSnapshot]) -> MaskOutput:
        """Generate independent box masks on an RGB frame, without temporal effects.

        Args:
            frame: RGB uint8 image of shape ``(H, W, 3)``.
            tracklets: Unique integer IDs with finite positive-area xyxy boxes.

        Returns:
            Boolean ``(N, H, W)`` masks with overlap preserved and no confidence.
            Empty tracklets return an empty mask array without inference.
        """
        _validate_frame(frame)
        ids = [tracklet.tracker_id for tracklet in tracklets]
        if any(not isinstance(key, (int, np.integer)) or isinstance(key, bool) for key in ids):
            raise ValueError("Tracklet IDs must be integers.")
        if len(set(ids)) != len(ids):
            raise ValueError("Tracklet IDs must be unique.")
        for tracklet in tracklets:
            box = np.asarray(tracklet.xyxy)
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                raise ValueError("Tracklet boxes must be finite xyxy coordinates with positive area.")
        if not tracklets:
            return _pack_masks({}, frame)
        session = self._runtime.new_session(frame)
        self._runtime.prompt_boxes(session, tracklets)
        masks = _validate_predictions(self._runtime.infer(session, 0), set(ids), frame)
        return _pack_masks(masks, frame)

    def initialize(self, frame: np.ndarray, mask_output: MaskOutput) -> None:
        """Replace temporal history with masks on this RGB frame; empty masks reset."""
        _validate_frame(frame)
        masks = _unpack_masks(mask_output, frame)
        self.reset()
        if masks:
            self._seed_session(frame, masks)

    def _seed_session(self, frame: np.ndarray, masks: dict[int, np.ndarray]) -> None:
        """Transfer all masks in one evaluated batch and release the old session."""
        self._session = None
        try:
            session = self._runtime.new_session(frame)
            self._runtime.prompt_masks(session, 0, masks)
            seeded = _validate_predictions(self._runtime.infer(session, 0), set(masks), frame)
        except Exception:
            self.reset()
            raise
        self._session = session
        self._last_masks = seeded
        self._last_frame = frame.copy()
        self._session_frame_count = 1

    def propagate(self, frame: np.ndarray) -> MaskOutput | None:
        """Advance one accepted image, rolling over before the session exceeds its cap.

        Source filenames and timestamps do not determine session indices. The caller decides which images to process. A
        resolution change requires ``reset()``. An inference failure resets temporal state before raising.
        """
        _validate_frame(frame)
        if self._session is None or self._last_frame is None:
            return None
        if frame.shape != self._last_frame.shape:
            raise ValueError("Frame resolution changed; reset EdgeTAM before starting a new pixel grid.")
        if self.max_session_frames is not None and self._session_frame_count >= self.max_session_frames:
            self._seed_session(self._last_frame, self._last_masks)
            self._session_index += 1
        try:
            masks = self._runtime.infer(self._session, self._session_frame_count, frame)
            self._last_masks = _validate_predictions(masks, set(self._last_masks), frame)
        except Exception:
            self.reset()
            raise
        self._session_frame_count += 1
        self._last_frame = frame.copy()
        return _pack_masks(self._last_masks, frame)

    def add_masks(self, frame: np.ndarray, mask_output: MaskOutput) -> None:
        """Add new identities on the last processed RGB image without advancing time.

        The frame must match the image on which the latest masks were produced. Existing IDs are rejected. With no
        active session, this initializes one; an empty addition leaves temporal state untouched. The latest existing
        masks join the new masks in one prompt batch; earlier history is retained.
        """
        _validate_frame(frame)
        masks = _unpack_masks(mask_output, frame)
        if not masks:
            return
        if self._session is None:
            self.initialize(frame, mask_output)
            return
        if set(masks) & set(self._last_masks):
            raise ValueError("Cannot add tracklet IDs that already have EdgeTAM masks.")
        if self._last_frame is None or not np.array_equal(frame, self._last_frame):
            raise ValueError("Added masks must refer to the last processed frame.")
        # EdgeTAM's propagated and mask-prompt object-score tensors have different
        # ranks. Mixing them on a stored frame fails upstream concatenation, even
        # though we do not use these scores. Prompt every current mask together;
        # this rewrites this frame's outputs and memory, retaining earlier history.
        # Existing IDs reuse their latest masks, never a new detector box.
        combined = {**self._last_masks, **masks}
        self._apply_masks(combined, set(combined))

    def _apply_masks(self, masks: dict[int, np.ndarray], expected_ids: set[int]) -> None:
        """Consume a complete prompt batch before another call can replace it."""
        if self._last_frame is None:
            raise RuntimeError("No stored EdgeTAM frame is available for mask prompts.")
        try:
            index = self._session_frame_count - 1
            self._runtime.prompt_masks(self._session, index, masks)
            output = self._runtime.infer(self._session, index)
            self._last_masks = _validate_predictions(output, expected_ids, self._last_frame)
        except Exception:
            self.reset()
            raise

    def remove_masks(self, tracklet_ids: list[int]) -> None:
        """Remove identities by reseeding survivors; unknown IDs are ignored."""
        if any(not isinstance(key, (int, np.integer)) or isinstance(key, bool) for key in tracklet_ids):
            raise ValueError("Tracklet IDs must be integers.")
        removed = set(tracklet_ids) & set(self._last_masks)
        if not removed:
            return
        survivors = {key: mask for key, mask in self._last_masks.items() if key not in removed}
        if not survivors:
            self.reset()
            return
        # Tracking-data reset preserves stored frames/features. The session cap
        # still bounds those; full reset and all-remove discard them altogether.
        self._session.reset_tracking_data()
        self._apply_masks(survivors, set(survivors))


class _EdgeTAMRuntime:
    """Keep the optional Transformers API behind a small inference boundary."""

    def __init__(self, model_id: str, revision: str | None, device: str, mask_threshold: float) -> None:
        """Load one fp32 model and matching processor, with no temporal session."""
        try:
            import torch
            from transformers import EdgeTamVideoModel, Sam2VideoProcessor  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("EdgeTAM requires the edgetam extra: pip install 'trackers[edgetam]'") from exc
        from trackers.utils.device import _validate_device

        if device == "auto":
            device = _resolve_auto_device()
        self.device = torch.device(_validate_device(device, label="EdgeTAM"))
        self._torch = torch
        self._model = EdgeTamVideoModel.from_pretrained(model_id, revision=revision)
        self._model.to(device=self.device, dtype=torch.float32).eval()
        self._processor = Sam2VideoProcessor.from_pretrained(model_id, revision=revision)
        self._mask_threshold = mask_threshold
        self.model_revision = getattr(self._model.config, "_commit_hash", None)

    def new_session(self, frame: np.ndarray) -> Any:
        """Prepare a fresh session with exactly one boundary or prompt frame."""
        return self._processor.init_video_session(
            video=[frame],
            inference_device=self.device,
            inference_state_device=self.device,
            processing_device=self.device,
            video_storage_device="cpu",
            dtype=self._torch.float32,
        )

    def prompt_boxes(self, session: Any, tracklets: list[TrackletSnapshot]) -> None:
        """Queue all supplied boxes on frame zero of a temporary one-frame session.

        ``generate()`` creates this session for the supplied RGB image, independently of ongoing propagation. Submit the
        requested IDs and xyxy boxes together; ``infer(session, 0)`` then evaluates them to produce their masks.
        """
        self._processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=[int(tracklet.tracker_id) for tracklet in tracklets],
            input_boxes=[[tracklet.xyxy.tolist() for tracklet in tracklets]],
            original_size=(session.video_height, session.video_width),
        )

    def prompt_masks(self, session: Any, frame_idx: int, masks: dict[int, np.ndarray]) -> None:
        """Queue the supplied masks together on one stored session frame.

        Each processor call replaces ``session.obj_with_new_inputs`` instead of accumulating pending IDs. Submit every
        mask needed by this operation in one call, then evaluate that frame before submitting another prompt batch. This
        pending list is separate from the session's complete object registry.
        """
        ids = sorted(masks)
        self._processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=frame_idx,
            obj_ids=ids,
            input_masks=[masks[key] for key in ids],
        )

    def infer(self, session: Any, frame_idx: int, frame: np.ndarray | None = None) -> dict[int, np.ndarray]:
        """Evaluate a stored or streamed image, preserving original size and ID order."""
        kwargs: dict[str, Any] = {"inference_session": session, "frame_idx": frame_idx}
        with self._torch.inference_mode():
            if frame is not None:
                inputs = self._processor(images=frame, return_tensors="pt").to(self.device)
                kwargs["frame"] = inputs.pixel_values[0]
            output = self._model(**kwargs)
            processed = self._processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[session.video_height, session.video_width]],
                binarize=False,
            )[0]
        ids = [int(key) for key in output.object_ids]
        if len(ids) != len(processed) or len(set(ids)) != len(ids):
            raise RuntimeError("EdgeTAM returned inconsistent object IDs and masks.")
        return {
            key: mask[0].detach().cpu().numpy() > self._mask_threshold for key, mask in zip(ids, processed, strict=True)
        }
