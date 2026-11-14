# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""EdgeTAM box-mask generation and streaming propagation with shared model weights.

The backend adapts Hugging Face's video model to the shared mask interfaces. It owns temporary box-prompt sessions, a
persistent propagation session, late object insertion, mask replacement, object removal and bounded session rollover.
Tracker-level association, occlusion checks and refresh eligibility remain outside this module. All image inputs here
use RGB, not the tracker's BGR convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from trackers.core.masks.base import MaskGenerator, MaskOutput, MaskPropagator, TrackletSnapshot, _resolve_auto_device

if TYPE_CHECKING:
    import torch

DEFAULT_EDGETAM_MODEL = "yonigozlan/EdgeTAM-hf"
DEFAULT_EDGETAM_REVISION = "c266ce53b3fc00f0f495b583f6a116c4e57f53bb"


def _validate_frame(frame: np.ndarray) -> None:
    """Validate image shape and dtype before mask or session operations.

    Args:
        frame: RGB image supplied by the caller. Shape and dtype can be checked
            here, but the caller is responsible for the actual channel order.

    Raises:
        ValueError: If the image is not a nonempty ``uint8`` array of shape
            ``(H, W, 3)``. No dtype or color conversion is performed.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) == 0 or frame.dtype != np.uint8:
        raise ValueError("EdgeTAM expects a nonempty RGB uint8 frame with shape (H, W, 3).")


def _unpack_masks(output: MaskOutput, frame: np.ndarray) -> dict[int, np.ndarray]:
    """Validate mask ownership and copy each object's mask to a boolean array.

    The ID-to-row mapping must cover every row exactly once, with integer IDs
    and integer row indices. Confidence metadata is not used by this backend.
    Validation occurs before callers modify the propagation session.

    Args:
        output: Mask channels and their complete tracklet-ID mapping.
        frame: Image whose height and width define the required mask resolution.

    Returns:
        Independent boolean ``(H, W)`` arrays keyed by tracklet ID. No objects
        gives an empty dictionary; an all-false mask still retains its object ID.

    Raises:
        ValueError: If mask dimensions, integer IDs or the row mapping are invalid.
    """
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
    """Validate inferred object IDs and resolution, then copy masks in ID order.

    Args:
        masks: Model output keyed by object ID.
        expected_ids: Exact object-ID set required after the current operation.
        frame: Image defining the expected mask height and width.

    Returns:
        Boolean mask copies with dictionary entries ordered by ascending ID.

    Raises:
        RuntimeError: If IDs are missing or unexpected, or a mask has a different
            resolution. An all-false mask is valid and does not remove its ID.
    """
    if set(masks) != expected_ids:
        raise RuntimeError(f"EdgeTAM returned object IDs {sorted(masks)}; expected {sorted(expected_ids)}.")
    if any(mask.shape != frame.shape[:2] for mask in masks.values()):
        raise RuntimeError("EdgeTAM returned masks at the wrong frame resolution.")
    return {key: masks[key].astype(bool, copy=True) for key in sorted(masks)}


def _pack_masks(masks: dict[int, np.ndarray], frame: np.ndarray) -> MaskOutput:
    """Stack masks into caller-owned channels and construct their ID-to-row mapping.

    Preserves the dictionary's ID order and independent, possibly overlapping object masks. An empty dictionary produces
    shape ``(0, H, W)`` using the frame resolution. ``mask_avg_prob_dict`` remains ``None`` because this backend does
    not expose per-object mask confidence.
    """
    return MaskOutput(
        masks=np.stack(list(masks.values())) if masks else np.zeros((0, *frame.shape[:2]), dtype=bool),
        tracklet_mask_dict={key: index for index, key in enumerate(masks)},
    )


class EdgeTAMMaskBackend(MaskGenerator, MaskPropagator):
    """Create box-prompted masks and propagate them with one loaded EdgeTAM model.

    Pass this same instance as both ``mask_generator`` and ``mask_propagator``
    to :class:`~trackers.core.masks.manager.MaskManager`. ``generate`` creates
    box masks in an isolated, temporary one-image session. ``initialize`` starts
    the persistent video session from supplied masks, and ``propagate`` accepts
    later RGB images one at a time through the Hugging Face Transformers
    ``EdgeTamVideoModel`` and ``Sam2VideoProcessor`` streaming API. This selected
    implementation does not require future images or a final sequence length.
    Overlapping objects keep independent boolean channels; no winning-object
    pixel assignment or mask-confidence gate is applied.

    An EdgeTAM video session stores image data/features and temporal object
    memory separately from model weights. With a finite ``max_session_frames``,
    the backend replaces the entire session before accepting an image that would
    exceed the limit. This session rollover drops old history, loads the last
    processed image as frame zero of a fresh session, prompts all remaining
    object masks on it, evaluates those prompts, then processes the incoming
    image as frame one. The transferred image counts toward the new session's
    limit. Object IDs are preserved; loaded model weights and the processor
    are reused. The limit applies to files and live streams alike and bounds
    history length, not total RAM/VRAM, which also depends on object count and
    image resolution.

    Rollover submits every remaining object's latest mask, including all-false
    masks. The model evaluates these prompts; it does not simply copy their
    pixels into the returned predictions. No detection boxes are used during
    rollover. Deciding when an empty or displaced mask should be regenerated
    from a detector box is a separate tracker policy using ``generate`` and
    ``refresh_masks``.

    ``add_masks`` and ``refresh_masks`` modify the latest stored image without
    advancing the session's frame counter. They evaluate all current masks
    together, replacing that frame's outputs and memory while retaining earlier
    history. ``remove_masks`` clears object-tracking data and reseeds surviving
    objects on the latest image; stored frames/features remain until rollover
    or reset. Removing the last object or calling ``reset`` discards the whole
    session. These operations are distinct from regenerating masks from boxes.

    The backend propagates out-of-memory exceptions to its caller. When used
    by ``McBytePlusPlusTracker``, a recognized runtime out-of-memory error during
    a mask update causes the tracker to clear mask state, log the failure and
    continue association without masks. Three such failures without a successful
    mask result disable the mask manager until the tracker is reset. This is a
    tracker policy, not a fallback performed by the backend itself; it applies
    to catchable exceptions during updates, not model-loading failures.

    Args:
        model_id: Hugging Face repository ID or local directory containing a
            Transformers-compatible EdgeTAM model and processor. Defaults to
            ``yonigozlan/EdgeTAM-hf``; model files are not bundled with Trackers.
        revision: Hub branch, tag or commit ID passed to both loaders. With the
            default model and ``None``, Trackers substitutes
            ``DEFAULT_EDGETAM_REVISION``:
            ``c266ce53b3fc00f0f495b583f6a116c4e57f53bb``. With another Hub model
            and ``None``, no commit is pinned and Hugging Face uses ``main``.
            An explicit value overrides this selection; a commit ID fixes the
            version for reproducibility. A local model path loads files from
            that directory rather than selecting a Hub revision.
        device: ``auto`` selects CUDA when available, otherwise CPU. Explicit
            PyTorch devices such as ``mps`` are also accepted. Inference uses fp32.
        max_session_frames: Maximum processed images per persistent session,
            including its initial or transferred image. Must be an integer of
            at least two. Defaults to 200; ``None`` allows history to grow until
            reset or removal of all objects. Unbounded history can exhaust
            available host RAM or device memory, depending on sequence length,
            image resolution, object count and available memory. The backend
            propagates resulting exceptions; tracker-specific handling is
            described above. Temporary generation sessions and
            repeated mask prompts on a stored image do not advance this count.
        mask_threshold: Finite logit cutoff applied to each object's decoded
            mask. Pixels strictly above the cutoff are foreground. Defaults to
            0.0; this is not a per-object confidence threshold.

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
        """Validate settings, resolve the model revision and load model/processor.

        Temporal state starts inactive. Loading weights does not create a video session; ``generate`` and ``initialize``
        create their own sessions later.
        """
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
        """Return the loaded model's resolved PyTorch device as read-only metadata.

        An ``auto`` request has already resolved to CUDA or CPU. Accessing this property neither moves the model nor
        changes any session state.
        """
        return self._runtime.device

    @property
    def model_revision(self) -> str | None:
        """Return the loaded model's resolved Hub commit for run provenance.

        Reads the commit recorded in the model configuration, not the requested branch or tag. May be ``None`` when that
        metadata is absent, for example for some local checkpoints. This read-only property does not reload weights.
        """
        return self._runtime.model_revision

    @property
    def session_frame_count(self) -> int:
        """Return the number of processed images in the persistent video session.

        Includes the initial image, including one transferred during rollover. Initialization sets it to one;
        propagation increments it once per image. Addition, refresh and partial removal do not increment it. Zero means
        no active session; temporary generation sessions are not counted.
        """
        return self._session_frame_count

    @property
    def session_index(self) -> int:
        """Return the zero-based rollover index of the persistent video session.

        Starts at zero, increments after each successful session rollover and returns to zero on initialization or
        reset. Temporary generation, addition, refresh and partial object removal do not increment it.
        """
        return self._session_index

    def reset(self) -> None:
        """Discard all propagation state so another sequence starts without history.

        Drop the session, stored RGB image, object masks and session counters. Unlike the upstream session's
        ``reset_tracking_data``, this also drops the session's stored frames and feature caches. Loaded model weights,
        processor, device, revision and configuration remain reusable; they carry no object-propagation history into the
        next sequence.
        """
        self._session = None
        self._last_frame = None
        self._last_masks = {}
        self._session_frame_count = 0
        self._session_index = 0

    def generate(self, frame: np.ndarray, tracklets: list[TrackletSnapshot]) -> MaskOutput:
        """Generate masks from boxes in a temporary session for one RGB image.

        Validate all IDs and boxes before inference, create a separate session
        containing this image, submit every box together on frame zero and
        evaluate the prompt batch. The persistent propagation session, its
        objects and its counters are not modified. The same loaded model and
        processor are reused.

        Args:
            frame: RGB uint8 image of shape ``(H, W, 3)``.
            tracklets: Unique integer IDs with finite positive-area xyxy boxes.

        Returns:
            ``MaskOutput`` with boolean ``(N, H, W)`` masks at input resolution,
            a complete ID-to-channel mapping and no confidence metadata. Object
            overlap is preserved. An empty tracklet list returns zero channels
            without creating a session or running inference.

        Raises:
            ValueError: If the image, IDs or boxes violate the input contract.
            RuntimeError: If inferred object IDs or mask resolution are invalid.
                Inference exceptions otherwise propagate without
                changing the persistent session.
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
        """Replace propagation history with a new session seeded by supplied masks.

        Validate the image and complete mask mapping, then reset the old session
        and counters. If object IDs are supplied, submit their masks together on
        frame zero of a fresh session and evaluate that frame. A mask with no
        foreground pixels is still an object and is included. With no object IDs,
        reset leaves the backend inactive; this differs from an all-false mask.

        Args:
            frame: RGB ``uint8`` image of shape ``(H, W, 3)`` on which the masks
                were defined.
            mask_output: Object masks on that image and their complete ID-to-row
                mapping. Confidence metadata is ignored.

        Raises:
            ValueError: If the image or mask mapping is invalid; validation errors
                occur before the old session is reset. Seeding failures propagate
                after leaving the backend inactive.
        """
        _validate_frame(frame)
        masks = _unpack_masks(mask_output, frame)
        self.reset()
        if masks:
            self._seed_session(frame, masks)

    def _seed_session(self, frame: np.ndarray, masks: dict[int, np.ndarray]) -> None:
        """Build a fresh propagation session from one image and all remaining masks.

        Used by ``initialize`` and session rollover. Release the old session
        reference before creating the replacement, prompt every supplied object
        on frame zero, run inference and validate the returned ID set/resolution.
        Store those evaluated masks and a copy of the image; set the image count
        to one. All-false input masks are submitted too. Older image features
        and object history are not transferred, and no box prompts are generated.

        This helper does not increment ``session_index``; the rollover caller
        does so only after successful seeding. Any failure resets the backend
        to an inactive state and re-raises the exception.

        Args:
            frame: Validated RGB image used as frame zero of the new session.
            masks: Validated nonempty mapping of object IDs to masks on that image.
        """
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
        """Propagate every active object's mask to one newly supplied RGB image.

        Before processing this image, check the persistent session's image count.
        If it has reached ``max_session_frames``, perform session rollover:
        ``_seed_session`` creates a fresh session from the previous RGB image and
        all its latest masks, then ``session_index`` increments. Inference for
        the incoming image follows at index one of the new session. Otherwise
        inference appends it at the next index of the existing session.

        Validate the inferred IDs and resolution, store the resulting masks and
        an image copy, and increment the session image count. Source filenames
        and timestamps play no role here; the caller selects which images to
        submit. Repeating a call counts as another image, even if pixels match.

        Args:
            frame: New RGB ``uint8`` image of shape ``(H, W, 3)``, at the same
                resolution as the active session's initial image.

        Returns:
            Independent boolean masks and their ID mapping for this image, with
            overlap preserved and no confidence metadata. Returns ``None`` when
            no propagation session is active.

        Raises:
            ValueError: If the image is invalid or changes resolution. Reset
                before starting a different image grid; validation leaves the
                existing session intact.
            RuntimeError: If inference returns unexpected IDs or resolution.
                Seeding/inference failures reset temporal state and propagate
                their exception to the caller.
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
        """Insert new object IDs on the latest image of the propagation session.

        Merge the new masks with the latest masks of existing objects and evaluate
        them together on the last stored image. Prompting all objects in one batch
        avoids mixing incompatible propagated and mask-prompt object-score tensor
        shapes in the upstream model. This rewrites that image's outputs and
        memory but retains earlier session history. Existing objects reuse their
        latest masks; no new detector boxes are used for them.

        With an active session, neither image count nor session index advances.
        With no active session, delegate to ``initialize``. A validated input
        containing no object IDs is a no-op; an all-false object mask is not.

        Args:
            frame: RGB ``uint8`` image with pixels identical to the last processed
                image when a session is active, or the initial image otherwise.
            mask_output: New objects' masks at frame resolution, with a complete
                ID-to-row mapping. IDs must not already exist in the session.

        Raises:
            ValueError: If the image, mask mapping or IDs are invalid, or the
                supplied pixels differ from the last image. Prompt/inference
                failures reset propagation state and re-raise their exception.
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

    def refresh_masks(self, frame: np.ndarray, mask_output: MaskOutput) -> None:
        """Replace selected existing masks on the session's latest stored image.

        Merge the supplied replacements into the latest mask dictionary:
        replacement values override matching IDs, and every other object's
        latest mask is retained as a prompt. ``_apply_masks`` evaluates that
        complete object batch on the same stored frame, validating that the
        object-ID set has not changed. All objects' current-frame outputs and
        memory are recomputed; earlier history is retained, and neither session
        counter advances.

        This method does not decide which masks are bad or when refreshing is
        safe. The tracker chooses eligible objects and can call ``generate``
        with their detector boxes to obtain replacements first. A validated
        input with no object IDs is a no-op. All-false replacements for existing
        IDs are accepted and submitted rather than treated as no-ops.

        Args:
            frame: Last processed RGB uint8 image, with identical pixels.
            mask_output: Replacement masks at frame resolution, with a complete
                ID-to-row mapping for the subset of existing objects to replace.

        Raises:
            ValueError: If the frame, mapping or replacement IDs are invalid,
                or nonempty replacements are supplied without an active session.
                Prompt/inference failures reset propagation state and re-raise.
        """
        _validate_frame(frame)
        masks = _unpack_masks(mask_output, frame)
        if not masks:
            return
        if self._session is None or not set(masks).issubset(self._last_masks):
            raise ValueError("Refreshed masks must belong to existing EdgeTAM tracklet IDs.")
        if self._last_frame is None or not np.array_equal(frame, self._last_frame):
            raise ValueError("Refreshed masks must refer to the last processed frame.")
        # For duplicate dictionary keys, the later value wins.
        combined = {**self._last_masks, **masks}
        self._apply_masks(combined, set(self._last_masks))

    def _apply_masks(self, masks: dict[int, np.ndarray], expected_ids: set[int]) -> None:
        """Evaluate all remaining object masks on the last stored session frame.

        Queue one complete mask batch at ``session_frame_count - 1`` and run
        inference immediately, before another processor call can replace its
        pending object-ID list. Validate the returned ID set and pixel grid,
        then replace ``_last_masks`` with the evaluated output. The stored
        image and session counters are unchanged. This is used for addition,
        refresh and survivor reseeding after partial removal.

        Args:
            masks: Prompts for every object that should remain after this
                operation, including unchanged objects during addition/refresh.
            expected_ids: Exact set required in the inference result. Addition
                uses the union of old and new IDs, refresh uses the existing
                IDs, and removal uses only surviving IDs.

        Raises:
            RuntimeError: If no stored frame exists or inferred IDs/resolution
                are invalid. Once prompting begins, any exception resets the
                backend to an inactive state before propagating to the caller.
        """
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
        """Remove object IDs and rebuild tracking memory from surviving masks.

        Unknown IDs are ignored. If any objects survive, clear the upstream
        session's object-tracking data, then prompt and evaluate all survivors
        together on the last processed image using their latest masks. This
        rebuilds object memory for the survivors too; it does not regenerate
        masks from detector boxes. Stored frames/features and session counters
        remain, so the frame limit still triggers normal session rollover.

        If the last object is removed, ``reset`` instead discards the entire
        session, stored image/masks and counters. Loaded weights remain available.

        Args:
            tracklet_ids: Integer object IDs to remove. Duplicates have no extra
                effect; an empty list or only unknown IDs changes nothing.

        Raises:
            ValueError: If any supplied ID is not an integer. Failures while
                evaluating survivor prompts reset the backend and re-raise.
        """
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
    """Adapt the optional Hugging Face model and processor to single-image calls.

    Loads one float32 ``EdgeTamVideoModel`` and its ``Sam2VideoProcessor`` for use by temporary generation sessions and
    the persistent propagation session. Session lifecycle, object ownership validation and rollover policy belong to
    ``EdgeTAMMaskBackend``; this adapter creates sessions, submits prompts and decodes model outputs.
    """

    def __init__(self, model_id: str, revision: str | None, device: str, mask_threshold: float) -> None:
        """Load model/processor from the same checkpoint and configure inference.

        The backend has already resolved the default revision before this call.
        Resolve and validate the device, load both components with the supplied
        revision, and move the model to float32 evaluation mode. Record its
        resolved commit when available. No video session is created here.

        Args:
            model_id: Hugging Face repository ID or local model directory.
            revision: Revision selected by the backend, or ``None`` for an
                unpinned custom model.
            device: Requested PyTorch device or ``auto`` for CUDA/CPU selection.
            mask_threshold: Foreground logit cutoff used when decoding masks.

        Raises:
            ImportError: If torch or the required Transformers classes are absent.
            RuntimeError: If the requested device is unsupported or unavailable.
                Model and processor loading errors otherwise propagate unchanged.
        """
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
        """Create a video session containing exactly one RGB image at index zero.

        Used for temporary box generation, explicit initialization and rollover
        from the previous session's last image. Stored video data uses CPU;
        processing, inference and object state use the configured device in
        float32. No object prompts or inference are performed here, and no
        later images or final sequence length are needed.

        Args:
            frame: Initial RGB ``uint8`` image with shape ``(H, W, 3)``.

        Returns:
            A new processor-managed inference session with no prompted objects.
        """
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

        Args:
            session: Fresh one-image session created for box-mask generation.
            tracklets: Validated, unique IDs and boxes in original-image pixel
                coordinates. All boxes are submitted in one processor call.
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

        Args:
            session: Session receiving mask prompts for initialization, addition,
                refresh, removal or rollover.
            frame_idx: Stored frame index within this session, not a source-video
                frame number. Zero for seeding; the latest index for other updates.
            masks: Validated ID-to-mask mapping at the session's original image
                resolution. IDs are sorted to keep prompt/mask ordering consistent.
        """
        ids = sorted(masks)
        self._processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=frame_idx,
            obj_ids=ids,
            input_masks=[masks[key] for key in ids],
        )

    def infer(self, session: Any, frame_idx: int, frame: np.ndarray | None = None) -> dict[int, np.ndarray]:
        """Run one model inference and decode independent masks at original size.

        For a new image, preprocess its RGB pixels and pass its tensor to the
        model with the next session index. Without an image, evaluate the already
        stored frame at ``frame_idx``, for example immediately after prompting it.
        Both paths use torch inference mode with no gradient recording.

        Resize output mask logits to the session's original height/width without
        binarizing in the processor, then apply ``mask_threshold`` separately to
        each object. Return CPU NumPy boolean arrays keyed by the model's object
        IDs in returned order. Overlap is preserved; object scores are not used.

        Args:
            session: Processor-managed session with the objects to evaluate.
            frame_idx: Zero-based index within this session.
            frame: New RGB image to process, or ``None`` to evaluate a stored image.

        Returns:
            Object IDs mapped to boolean ``(H, W)`` arrays at original resolution.

        Raises:
            RuntimeError: If returned IDs are duplicated or their count differs
                from the number of mask channels. Model errors propagate; the
                owning backend decides whether the affected session must reset.
        """
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
