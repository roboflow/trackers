# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Reusable long-term identity joining and delivery of pending observations."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from trackers.reid.appearance import _l2_normalize, extract_detection_embeddings
from trackers.reid.encoder import ReIDEncoder
from trackers.reid.long_term_config import LongTermReIDConfig
from trackers.reid.occlusion import foreground_overlap
from trackers.utils.detections import default_confidences


@dataclass
class _Tracklet:
    first_frame: int
    first_step: int
    last_frame: int
    features: deque[np.ndarray]
    samples: int = 0
    last_feature_step: int | None = None
    public_id: int = -1
    paused_steps: int = 0
    pending: list[tuple[int, sv.Detections]] = field(default_factory=list)

    def descriptor(self) -> np.ndarray:
        return _l2_normalize(np.mean(np.stack(self.features), axis=0))


@dataclass(frozen=True)
class _GalleryIdentity:
    descriptor: np.ndarray
    last_frame: int


class LongTermReID:
    """Join new tracklets to terminated identities using mutual-best appearance.

    This component owns public identities and optional historical output; the
    host tracker retains motion state and internal tracklet IDs. A confirmed
    tracklet has an internal ID assigned by its host, but its public identity
    may remain pending while this component decides whether to reconnect it or
    assign a new identity. Temporarily lost tracklets retain ownership of their
    assigned public IDs. Only explicit termination offers an identity with
    enough extracted features to the gallery of reconnection candidates. The
    gallery has no automatic expiry.

    A join is accepted when the new tracklet and terminated identity are each
    other's highest-similarity eligible candidate, and their cosine similarity
    meets the threshold. Temporally overlapping pairs are excluded before
    selecting these candidates. Each public identity has at most one owner.

    ``update`` returns current observations, with public ID ``-1`` while a
    confirmed tracklet is awaiting a decision. ``pop_finalized_results`` returns
    only earlier pending observations whose identities have since resolved.
    It never repeats a current observation already returned with a final ID.
    Each tracklet releases its history independently. Call ``finish`` at a
    finite sequence's end to resolve the remaining pending identities.

    A host may explicitly pause pending decision ages while independently
    suppressing new appearance samples. Existing descriptors can still join;
    the ordinary sampling step and observation references keep advancing.
    This component does not interpret camera motion or measure image quality.
    By default, its own box-based subject-occlusion check suppresses new re-ID
    features and pauses the affected pending identity's decision deadline.
    Historical buffering retains pending observations until their IDs resolve.

    Args:
        model: Appearance encoder accepting BGR frames. Loaded weights are
            retained by ``reset``; no encoder is created by this component.
        config: Joining, extraction and buffering settings. Defaults to
            ``LongTermReIDConfig()``.
    """

    _frame_number: int
    _step: int
    _closed: bool
    _feature_dimension: int | None

    def __init__(self, model: ReIDEncoder, config: LongTermReIDConfig | None = None) -> None:
        if not callable(getattr(model, "extract_embeddings", None)):
            raise TypeError("model must implement extract_embeddings(detections, frame).")
        self.model = model
        self.config = config or LongTermReIDConfig()
        self.reset()

    @property
    def public_ids(self) -> dict[int, int]:
        """Return internal-to-public IDs for current and temporarily lost tracks."""
        return {internal_id: track.public_id for internal_id, track in self._tracks.items()}

    @property
    def frame_number(self) -> int:
        """Return the latest accepted processing-frame reference (one-based)."""
        return self._frame_number

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        *,
        tracklet_starts: Mapping[int, tuple[int, int]] | None = None,
        removed_tracklet_ids: Iterable[int] = (),
        reference_boxes: np.ndarray | None = None,
        frame_number: int | None = None,
        step: int | None = None,
        extract_features: bool = True,
        pause_reconnection: bool = False,
        tracklet_paused_steps: Mapping[int, int] | None = None,
    ) -> sv.Detections:
        """Resolve identities for currently observed, output-eligible tracklets.

        Two clocks have separate roles: ``frame_number`` identifies an accepted
        update for historical output, while ``step`` controls feature sampling
        and joining age. The frame reference need not equal the original video
        frame number. McByte++ advances only the reference on duplicate
        timestamps, even if the image is missing. Other accepted missing-image
        updates advance both clocks.

        Joining uses ``decision_age = current_step - birth_step - paused_steps``.
        A pending identity may reconnect while this age is at most
        ``maximum_reconnection_age``. Once it exceeds that limit, a currently
        observed pending tracklet receives a fresh public ID. Temporarily lost
        pending tracklets wait until observed again, terminated or finished.
        Tracklets born on processing frame 1 finalize immediately when eligible.

        Args:
            detections: Current matched observations. ``tracker_id`` contains
                the host's immutable internal IDs; negative values denote
                ordinary unassigned rows, which are never buffered.
            frame: Current BGR uint8 image or ``None``. Missing images prevent
                extraction but do not discard history or existing descriptors.
            tracklet_starts: Optional internal ID to ``(birth_frame, birth_step)``
                mapping. Hosts with delayed confirmation supply original birth
                references here; otherwise first appearance in this method is
                the birth. Birth frame 1 identities finalize immediately.
            removed_tracklet_ids: Truly terminated IDs, not temporarily lost
                ones. Termination finalizes any pending history before gallery
                matching. Repeated termination notifications are harmless.
            reference_boxes: All current detector boxes for optional occlusion
                gating, including unconfirmed and low-confidence detections.
                Defaults to ``detections.xyxy``. No previous-image boxes are used.
            frame_number: Strictly increasing processing reference. Defaults to
                the preceding value plus one. McByte++ supplies its accepted
                update count so references remain aligned after a failed update.
            step: Non-decreasing ordinary processing clock, defaulting to the
                preceding value plus one. A duplicate timestamp supplies the
                same step: no extraction occurs, but the observation still has
                its own frame reference. Non-duplicate missing-image calls
                advance one step; elapsed wall-clock time does not synthesize steps.
            extract_features: Whether this call may sample the supplied image.
                A host handling duplicate timestamps sets this to False even
                when its clock advanced during a preceding failed update.
                False also suppresses image-based occlusion pause requests;
                a host-requested ``pause_reconnection`` still applies.
            pause_reconnection: Discount this advancing step from pending
                decision ages, subject to the configured lifetime pause cap.
                Does not suppress extraction by itself or prevent matching
                with existing descriptors. Duplicate steps add no pause.
                A simultaneous configured occlusion pause counts only once.
                Termination and ``finish`` still finalize pending identities.
            tracklet_paused_steps: For delayed confirmation, initial pause
                counts by internal ID, strictly after original birth and
                including this registration update. Read only for new IDs;
                known IDs maintain their own counts. Each seed must be an
                integer from zero through raw age and is limited by the cap.
                Without a seed, only this call's eligible pause is counted.
                Hosts need no per-frame pause history.

        Returns:
            Current detections with public ``tracker_id`` and reserved row data
            ``tracklet_id`` (internal owner) and ``reid_pending`` (boolean).
            Input arrays and collection metadata are not modified.

        Raises:
            RuntimeError: After ``finish``, or on an identity ownership violation.
            ValueError: For invalid clocks, duplicate internal IDs, inconsistent
                birth references or invalid encoder output.
        """
        if self._closed:
            raise RuntimeError("Long-term ReID is finished; call reset() before update().")
        if not isinstance(pause_reconnection, bool):
            raise ValueError("pause_reconnection must be a boolean.")
        next_frame = self._frame_number + 1 if frame_number is None else frame_number
        next_step = self._step + 1 if step is None else step
        if isinstance(next_frame, bool) or not isinstance(next_frame, int) or next_frame <= self._frame_number:
            raise ValueError("frame_number must be a strictly increasing positive integer.")
        if isinstance(next_step, bool) or not isinstance(next_step, int) or next_step < max(1, self._step):
            raise ValueError("step must be a non-decreasing positive integer.")
        internal_ids: np.ndarray
        if detections.tracker_id is None:
            if len(detections):
                raise ValueError("detections must contain internal tracker_id values.")
            internal_ids = np.empty(0, dtype=int)
        else:
            internal_ids = np.asarray(detections.tracker_id, dtype=int)
        observed = [int(identity) for identity in internal_ids if identity >= 0]
        if len(set(observed)) != len(observed):
            raise ValueError("An internal tracklet ID occurs more than once in the current observations.")
        removed = {int(identity) for identity in removed_tracklet_ids if identity >= 0}
        if removed.intersection(observed) or self._retired.intersection(observed):
            raise ValueError("A terminated internal tracklet ID cannot be observed again; allocate a new internal ID.")
        advanced = next_step > self._step
        self._frame_number, self._step = next_frame, next_step
        for internal_id in sorted(removed):
            self._terminate(internal_id)
        occluded: set[int] = set()
        if self.config.pause_reconnection_on_occlusion and frame is not None and advanced and extract_features:
            boxes = detections.xyxy if reference_boxes is None else np.asarray(reference_boxes)
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError("reference_boxes must have shape (N, 4).")
            # Check every accepted image, not only scheduled feature samples.
            # Only observed targets can request an occlusion-specific pause.
            occluded = {
                int(identity)
                for identity, box in zip(internal_ids, detections.xyxy)
                if identity >= 0
                and (identity not in self._tracks or self._tracks[int(identity)].public_id < 0)
                and foreground_overlap(box, boxes) >= self.config.feature_occlusion_overlap_threshold
            }
        cap = self.config.maximum_reconnection_pause_steps
        if advanced and (pause_reconnection or occluded):
            for internal_id, track in self._tracks.items():
                if (
                    track.public_id < 0
                    and (pause_reconnection or internal_id in occluded)
                    and (cap is None or track.paused_steps < cap)
                ):
                    track.paused_steps += 1
        starts = tracklet_starts or {}
        seeds = tracklet_paused_steps or {}
        for internal_id in observed:
            if internal_id not in self._tracks:
                birth_frame, birth_step = starts.get(internal_id, (next_frame, next_step))
                if not 1 <= birth_frame <= next_frame or not 1 <= birth_step <= next_step:
                    raise ValueError("Tracklet birth references must precede or equal the current frame and step.")
                # The seed includes the registration step. Existing tracks were
                # aged above; applying that same pause again here would double it.
                paused = seeds.get(
                    internal_id,
                    int(advanced and (pause_reconnection or internal_id in occluded) and birth_step < next_step),
                )
                if isinstance(paused, bool) or not isinstance(paused, int) or not 0 <= paused <= next_step - birth_step:
                    raise ValueError("Tracklet paused steps must be a non-negative integer no greater than raw age.")
                self._tracks[internal_id] = _Tracklet(
                    birth_frame,
                    birth_step,
                    next_frame,
                    deque(maxlen=self.config.feature_history_size),
                    paused_steps=paused if cap is None else min(paused, cap),
                )
                if birth_frame == 1:
                    self._finalize(internal_id)
            self._tracks[internal_id].last_frame = next_frame
        if frame is not None and advanced and extract_features:
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("ReID expects a BGR uint8 frame with shape (H, W, 3).")
            boxes = detections.xyxy if reference_boxes is None else np.asarray(reference_boxes)
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError("reference_boxes must have shape (N, 4).")
            self._collect_features(detections, internal_ids, frame, boxes)
        self._match(observed)
        for internal_id in observed:
            track = self._tracks[internal_id]
            if track.public_id < 0 and self._decision_age(track) > self.config.maximum_reconnection_age:
                self._finalize(internal_id)

        result = detections[:]
        result.data = dict(detections.data)
        result.data["tracklet_id"] = internal_ids.copy()
        result.tracker_id = np.array(
            [self._tracks[int(identity)].public_id if identity >= 0 else -1 for identity in internal_ids], dtype=int
        )
        pending = (internal_ids >= 0) & (result.tracker_id < 0)
        result.data["reid_pending"] = pending
        self._validate_current(result)
        if self.config.buffer_results:
            for index in np.flatnonzero(pending):
                # Only observations need replay: never retain images, segmentation
                # masks or arbitrary collection-level metadata in the output queue.
                row = result[index]
                snapshot = sv.Detections(
                    xyxy=row.xyxy.copy(),
                    confidence=None if row.confidence is None else row.confidence.copy(),
                    class_id=None if row.class_id is None else row.class_id.copy(),
                    tracker_id=row.tracker_id.copy(),
                    data=deepcopy(row.data),
                )
                self._tracks[int(internal_ids[index])].pending.append((next_frame, snapshot))
        return result

    def _collect_features(
        self, detections: sv.Detections, internal_ids: np.ndarray, frame: np.ndarray, reference_boxes: np.ndarray
    ) -> None:
        """Sample eligible observed boxes and count only successful, nonzero descriptors."""
        config = self.config
        confidences = default_confidences(detections)
        height, width = frame.shape[:2]
        indices = []
        for index, internal_id in enumerate(internal_ids):
            if internal_id < 0:
                continue
            track = self._tracks[int(internal_id)]
            if (
                track.last_feature_step is not None
                and self._step - track.last_feature_step < config.minimum_feature_frame_gap
            ):
                continue
            box = detections.xyxy[index]
            box_width, box_height = box[2:] - box[:2]
            confidence = confidences[index]
            if (
                not np.isfinite(box).all()
                or not np.isfinite(confidence)
                or confidence < config.minimum_detection_confidence
                or box_width <= 0
                or box_height <= 0
                or box_width < config.minimum_box_width
                or box_height < config.minimum_box_height
                or box_width * box_height < config.minimum_box_area
                or box[0] <= config.border_margin
                or box[1] <= config.border_margin
                or box[2] >= width - config.border_margin
                or box[3] >= height - config.border_margin
            ):
                continue
            if config.enable_feature_occlusion_filter and (
                foreground_overlap(box, reference_boxes) >= config.feature_occlusion_overlap_threshold
            ):
                continue
            indices.append(index)
        if not indices:
            return
        features = extract_detection_embeddings(self.model, frame, detections.xyxy[indices])
        if features.shape[1] == 0:
            raise ValueError("ReID embeddings must have a non-empty feature dimension.")
        if self._feature_dimension is not None and features.shape[1] != self._feature_dimension:
            raise ValueError("ReID encoder changed its feature dimension during the sequence.")
        self._feature_dimension = features.shape[1]
        for index, feature in zip(indices, features):
            if not np.any(feature):
                continue
            track = self._tracks[int(internal_ids[index])]
            track.features.append(feature.copy())
            track.samples += 1
            track.last_feature_step = self._step

    def _decision_age(self, track: _Tracklet) -> int:
        """Return ``current_step - birth_step - paused_steps`` for matching and expiry."""
        return self._step - track.first_step - track.paused_steps

    def _match(self, observed: list[int]) -> None:
        """Reconnect eligible pending tracklets using temporally compatible, mutual-best matches."""
        candidates = sorted(
            internal_id
            for internal_id in observed
            if self._tracks[internal_id].public_id < 0
            and self._tracks[internal_id].samples >= self.config.minimum_feature_samples
            and self._decision_age(self._tracks[internal_id]) <= self.config.maximum_reconnection_age
        )
        gallery_ids = sorted(self._gallery)
        if not candidates or not gallery_ids:
            return
        current = np.stack([self._tracks[identity].descriptor() for identity in candidates])
        previous = np.stack([self._gallery[identity].descriptor for identity in gallery_ids])
        similarity = current @ previous.T
        # Reject impossible pairs before BOTH best-match selections. Otherwise
        # an overlapping trajectory could hide a valid, second-best identity.
        for row, internal_id in enumerate(candidates):
            for column, public_id in enumerate(gallery_ids):
                if (
                    public_id in self._owners
                    or self._gallery[public_id].last_frame >= self._tracks[internal_id].first_frame
                ):
                    similarity[row, column] = -np.inf
        best_previous = np.argmax(similarity, axis=1)
        best_current = np.argmax(similarity, axis=0)
        for row, column in enumerate(best_previous):
            if best_current[column] == row and similarity[row, column] >= self.config.minimum_cosine_similarity:
                self._finalize(candidates[row], gallery_ids[column])

    def _finalize(self, internal_id: int, public_id: int | None = None) -> None:
        """Assign a fresh ID or claim a gallery identity, then release buffered observations."""
        track = self._tracks[internal_id]
        if track.public_id >= 0:
            return
        if public_id is None:
            public_id = self._next_public_id
            self._next_public_id += 1
        elif public_id not in self._gallery:
            raise RuntimeError(f"Public identity {public_id} is not available in the terminated gallery.")
        if public_id in self._owners:
            raise RuntimeError(f"Public identity {public_id} already has a live or temporarily lost owner.")
        last = self._finalized_through.get(public_id, 0)
        if any(frame <= last for frame, _ in track.pending):
            raise RuntimeError(f"Historical observations overlap public identity {public_id}.")
        self._gallery.pop(public_id, None)
        self._owners[public_id] = internal_id
        track.public_id = public_id
        for frame, row in track.pending:
            self._register_observation(public_id, frame)
            row.tracker_id = np.array([public_id], dtype=int)
            row.data["reid_pending"] = np.array([False])
            self._ready.setdefault(frame, []).append(row)
        track.pending.clear()

    def _terminate(self, internal_id: int) -> None:
        """Finalize pending output, release identity ownership and archive a usable descriptor."""
        if internal_id not in self._tracks:
            return
        self._finalize(internal_id)
        track = self._tracks.pop(internal_id)
        if self._owners.pop(track.public_id, None) != internal_id:
            raise RuntimeError(f"Public identity {track.public_id} has inconsistent ownership at termination.")
        if track.samples >= self.config.minimum_feature_samples:
            self._gallery[track.public_id] = _GalleryIdentity(track.descriptor(), track.last_frame)
        self._retired.add(internal_id)

    def _register_observation(self, public_id: int, frame: int) -> None:
        # Every segment is chronological and joins require disjoint observation
        # intervals. A persistent high-water mark therefore detects duplicate
        # finalized (frame, ID) records without retaining every delivered row.
        if frame <= self._finalized_through.get(public_id, 0):
            raise RuntimeError(f"Duplicate or overlapping finalized identity {public_id} at processing frame {frame}.")
        self._finalized_through[public_id] = frame

    def _validate_current(self, detections: sv.Detections) -> None:
        ids = detections.tracker_id[detections.tracker_id >= 0]
        if len(np.unique(ids)) != len(ids):
            raise RuntimeError(f"Duplicate public identity at processing frame {self._frame_number}.")
        for public_id in ids:
            self._register_observation(int(public_id), self._frame_number)

    def pop_finalized_results(self) -> dict[int, sv.Detections]:
        """Drain resolved historical observations grouped by processing reference.

        Only previously pending rows appear here. A frame can appear in later
        pops again when another identity resolves, so consumers append rows
        rather than replace a previously received frame. Finalized rows have
        nonnegative IDs, no segmentation masks and empty collection metadata.
        Retrieval does not erase the identity ownership/uniqueness ledger.

        Raises:
            RuntimeError: If historical buffering is disabled, or queued rows
                violate identity uniqueness.
        """
        if not self.config.buffer_results:
            raise RuntimeError("Historical results require buffer_results=True.")
        result = {}
        for frame, rows in sorted(self._ready.items()):
            combined = sv.Detections.merge(rows)
            ids = combined.tracker_id
            if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
                raise RuntimeError(f"Invalid finalized identities at processing frame {frame}.")
            result[frame] = combined
        self._ready.clear()
        return result

    def finish(self) -> dict[int, sv.Detections]:
        """Close this sequence, assigning unused IDs to unresolved eligible tracks.

        Return unread resolved history, including the final pending tail. Without buffering, or on repeated calls,
        return an empty dictionary. Further ``update`` calls require ``reset`` in every mode.
        """
        if self._closed:
            return {}
        for internal_id in self._tracks:
            self._finalize(internal_id)
        self._closed = True
        return self.pop_finalized_results() if self.config.buffer_results else {}

    def reset(self) -> None:
        """Discard sequence state and unread history; retain model and settings."""
        self._frame_number = 0
        self._step = 0
        self._closed = False
        self._next_public_id = 0
        self._feature_dimension = None
        self._tracks: dict[int, _Tracklet] = {}
        self._gallery: dict[int, _GalleryIdentity] = {}
        self._owners: dict[int, int] = {}
        self._retired: set[int] = set()
        self._ready: dict[int, list[sv.Detections]] = {}
        self._finalized_through: dict[int, int] = {}
