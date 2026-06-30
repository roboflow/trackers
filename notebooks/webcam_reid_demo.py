#!/usr/bin/env python3
"""Live webcam demo: BoT-SORT + ReID track ID persistence.

Stand in front of the camera, note your track ID, leave the frame completely,
then walk back in. With ReID enabled the same number should reappear if the
lost-track buffer has not expired.

Re-ID uses OSNet x1.0 pretrained on MSMT17 (``osnet_x1_0_msmt17_combineall``).

Detector: **RF-DETR** via ``inference_models.AutoModel`` (default ``rfdetr-nano``).
Target classes are resolved from the model's ``class_names`` (default
``person,cup``). RF-DETR uses ``class_id=1`` for person, not COCO's 0,
because index 0 is a ``coco`` placeholder. ReID is pedestrian-trained;
appearance helps people more than objects like cups.

Requires:
  pip install 'trackers[detection,reid]' opencv-python supervision

Example:
  uv run python notebooks/webcam_reid_demo.py
  uv run python notebooks/webcam_reid_demo.py --camera 1
  uv run python notebooks/webcam_reid_demo.py --list-cameras
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass

import cv2

if not hasattr(cv2, "FONT_HERSHEY_SIMPLEX"):
    raise SystemExit(
        "OpenCV install is broken (empty cv2 namespace). "
        "This usually means opencv-python and opencv-python-headless "
        "were installed together. Fix with:\n"
        "  uv pip uninstall opencv-python opencv-python-headless\n"
        "  uv pip install opencv-python"
    )

import numpy as np
import supervision as sv

from trackers import BoTSORTTracker
from trackers.core.reid import ReIDModel
from trackers.core.reid.distance import appearance_similarity
from trackers.core.reid.extraction import extract_detection_embeddings
from trackers.core.reid.models.registry import DEFAULT_MODEL
from trackers.utils.iou import IoU

MSMT17_REID = DEFAULT_MODEL  # osnet_x1_0_msmt17_combineall
MIN_ACTIVATION_FRAMES = 3

WINDOW_NAME = "BoT-SORT + ReID webcam demo"
PANEL_WIDTH = 500
PANEL_BG = (24, 24, 24)
PANEL_TEXT = (235, 235, 235)
PANEL_MUTED = (120, 120, 120)
PANEL_LOST = (100, 170, 255)
PANEL_RULE = (55, 55, 55)
# BGR — higher similarity → greener.
PANEL_SIM_GOOD = (80, 200, 120)
PANEL_SIM_MED = (0, 220, 255)
PANEL_SIM_HIGH = (0, 140, 255)
PANEL_SIM_BAD = (80, 80, 240)
PANEL_VALUE_SCALE = 0.58
PANEL_VALUE_COL = 300
PANEL_DET_COL = 200
PANEL_DET_SPACING = 58


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the webcam demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        type=str,
        default="auto",
        help="Webcam index (0, 1, …) or 'auto' to pick the first working device.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rfdetr-nano",
        help="RF-DETR model id for inference-models (default: rfdetr-nano).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.20,
        help=(
            "High-confidence threshold for BoT-SORT (default: 0.20). "
            "Lower values keep partial re-entry detections in the high-conf pass."
        ),
    )
    parser.add_argument(
        "--lost-buffer",
        type=int,
        default=70,
        help=(
            "Lost-track buffer in frames before a track is deleted "
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Assumed camera FPS for scaling the lost-track buffer (default: 30).",
    )
    parser.add_argument(
        "--reid-source",
        type=str,
        default=MSMT17_REID,
        help=f"ReID checkpoint alias or path (default: {MSMT17_REID}).",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="person,cup",
        help=(
            "Comma-separated class names or ids to track "
            "(default: person,cup). Use --all-classes for every class."
        ),
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Track all detector classes (ignore --classes).",
    )
    parser.add_argument(
        "--no-debug-panel",
        action="store_true",
        help="Hide the side ReID distance debug panel.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe camera indices 0–4 and exit.",
    )
    parser.add_argument(
        "--no-cmc",
        action="store_true",
        help="Disable camera-motion compensation (faster on a static webcam).",
    )
    parser.add_argument(
        "--reid-emb-dist-threshold",
        type=float,
        default=0.40,
        help=(
            "Appearance distance gate θ_emb for BoT-SORT ReID fusion "
            "(default: 0.40; paper uses 0.25)."
        ),
    )
    parser.add_argument(
        "--reid-iou-dist-threshold",
        type=float,
        default=0.85,
        help=(
            "IoU distance gate θ_iou for active tracks — requires IoU > 1−θ "
            "(default: 0.85 → IoU > 0.15; paper uses 0.5 → IoU > 0.5)."
        ),
    )
    parser.add_argument(
        "--reid-iou-dist-threshold-lost",
        type=float,
        default=0.95,
        help=(
            "IoU distance gate θ_iou for lost tracks — requires IoU > 1−θ "
            "(default: 0.95 → IoU > 0.05; use 1.0 to skip IoU gating entirely)."
        ),
    )
    parser.add_argument(
        "--reid-emb-dist-threshold-lost",
        type=float,
        default=0.55,
        help=(
            "Appearance distance gate θ_emb for lost tracks (default: 0.55 → "
            "cos > 0.45; looser than active tracks for re-entry)."
        ),
    )
    return parser.parse_args()


def load_detector(model_id: str):
    """Load an RF-DETR detector via inference-models.

    Args:
        model_id: Model identifier accepted by ``AutoModel.from_pretrained``.

    Returns:
        Loaded detection model.

    Raises:
        SystemExit: If inference-models is not installed.
    """
    try:
        from inference_models import AutoModel
    except ImportError as exc:
        raise SystemExit(
            "inference-models is required.\n"
            "Install with: pip install 'trackers[detection,reid]'"
        ) from exc

    from trackers.utils.device import _best_device

    device = _best_device()
    print(f"Loading detector {model_id!r} on {device}...")
    return AutoModel.from_pretrained(model_id, device=device)


def resolve_class_filter(
    classes_arg: str | None,
    class_names: list[str],
) -> list[int] | None:
    """Resolve comma-separated class names or ids to integer class ids.

    Args:
        classes_arg: Comma-separated class names or numeric ids.
        class_names: Model class list where index equals class id.

    Returns:
        Resolved class ids, or ``None`` when ``classes_arg`` is empty.
    """
    if not classes_arg:
        return None

    name_to_id = {name: i for i, name in enumerate(class_names)}
    class_filter: list[int] = []
    for token in (part.strip() for part in classes_arg.split(",")):
        if not token:
            continue
        try:
            class_filter.append(int(token))
        except ValueError:
            class_id = name_to_id.get(token)
            if class_id is None:
                class_id = name_to_id.get(token.lower())
            if class_id is None:
                print(
                    f"Warning: class {token!r} not in model class list, skipping.",
                    file=sys.stderr,
                )
            else:
                class_filter.append(class_id)
    return class_filter if class_filter else None


def format_class_summary(class_ids: list[int] | None, class_names: list[str]) -> str:
    """Format tracked class ids for display.

    Args:
        class_ids: Class ids to include, or ``None`` for all classes.
        class_names: Model class list where index equals class id.

    Returns:
        Human-readable summary string.
    """
    if not class_ids:
        return "all classes"
    labels: list[str] = []
    for class_id in class_ids:
        if 0 <= class_id < len(class_names):
            labels.append(f"{class_names[class_id]}={class_id}")
        else:
            labels.append(str(class_id))
    return ", ".join(labels)


def run_detector(model, frame_bgr: np.ndarray) -> sv.Detections:
    """Run RF-DETR on a BGR webcam frame.

    Args:
        model: Loaded inference-models detector.
        frame_bgr: OpenCV BGR frame.

    Returns:
        Supervision detections for the frame.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    predictions = model(frame_rgb)
    if not predictions:
        return sv.Detections.empty()
    return predictions[0].to_supervision()


def filter_detections(
    detections: sv.Detections,
    *,
    class_filter: list[int] | None,
    min_confidence: float = 0.1,
) -> tuple[sv.Detections, int]:
    """Return class-filtered detections and raw count before class filter.

    Confidence is not cut at the high-confidence tracker threshold here.
    BoT-SORT needs sub-threshold boxes for low-confidence association and
    lost-track ReID recovery during partial re-entry.

    Args:
        detections: Raw detector output.
        class_filter: Optional class ids to keep, or ``None`` for all classes.
        min_confidence: Minimum detector confidence to retain.

    Returns:
        Tuple of filtered detections and the pre-filter detection count.
    """
    raw_count = len(detections)
    if len(detections) == 0:
        return detections, raw_count

    if detections.confidence is not None and min_confidence > 0.0:
        detections = detections[detections.confidence >= min_confidence]

    if class_filter is None or detections.class_id is None:
        return detections, raw_count

    allowed = np.isin(detections.class_id, class_filter)
    return detections[allowed], raw_count


@dataclass(frozen=True)
class PanelConfig:
    """Static tracker / detector settings shown in the side panel."""

    detector: str
    reid_model: str
    classes: str
    confidence: float
    min_activation_frames: int
    lost_buffer: int
    max_lost_frames: int
    emb_threshold: float
    emb_threshold_lost: float
    iou_threshold: float
    iou_threshold_lost: float
    cmc_enabled: bool


@dataclass
class PanelLive:
    """Per-frame stats for the side panel."""

    fps: float
    num_tracks: int
    num_dets: int
    num_high_dets: int


def format_track_labels(
    detections: sv.Detections,
    class_names: list[str],
) -> list[str]:
    """Build overlay labels from tracker ids and class names.

    Args:
        detections: Tracked detections with ``tracker_id`` assigned.
        class_names: Model class list where index equals class id.

    Returns:
        One label string per detection.
    """
    if detections.tracker_id is None:
        return []
    labels: list[str] = []
    for index, track_id in enumerate(detections.tracker_id):
        if track_id is None or int(track_id) < 0:
            labels.append("…")
            continue
        parts = [f"ID {int(track_id)}"]
        if detections.class_id is not None:
            class_id = int(detections.class_id[index])
            if 0 <= class_id < len(class_names):
                parts.append(class_names[class_id])
        labels.append(" ".join(parts))
    return labels


def _panel_text(
    panel: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    color: tuple[int, int, int] = PANEL_TEXT,
    scale: float = 0.42,
    thickness: int = 1,
) -> None:
    cv2.putText(
        panel,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _panel_rule(panel: np.ndarray, y: int, margin: int = 12) -> int:
    cv2.line(panel, (margin, y), (PANEL_WIDTH - margin, y), PANEL_RULE, 1, cv2.LINE_AA)
    return y + 14


def _similarity_color(value: float) -> tuple[int, int, int]:
    if value >= 0.80:
        return PANEL_SIM_GOOD
    if value >= 0.50:
        return PANEL_SIM_MED
    if value >= 0.25:
        return PANEL_SIM_HIGH
    return PANEL_SIM_BAD


def _det_column_labels(
    detections: sv.Detections,
    class_names: list[str],
) -> list[str]:
    labels: list[str] = []
    for index in range(len(detections)):
        class_part = ""
        if detections.class_id is not None:
            class_id = int(detections.class_id[index])
            if 0 <= class_id < len(class_names):
                class_part = class_names[class_id][:5]
        conf_part = ""
        if detections.confidence is not None:
            conf_part = f"{detections.confidence[index]:.2f}"
        if class_part and conf_part:
            labels.append(f"D{index} {class_part} {conf_part}")
        elif class_part:
            labels.append(f"D{index} {class_part}")
        else:
            labels.append(f"D{index}")
    return labels


def compute_iou_matrix(tracks, detections: sv.Detections) -> np.ndarray:
    """Compute IoU between predicted track boxes and detections.

    Args:
        tracks: Internal BoT-SORT tracklets.
        detections: Current-frame detections.

    Returns:
        IoU matrix of shape ``(num_tracks, num_dets)``.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)), dtype=np.float32)
    track_boxes = np.array([track.get_state_bbox() for track in tracks], dtype=np.float32)
    return IoU().compute(track_boxes, detections.xyxy)


def compute_reid_similarities(
    tracks,
    det_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between track and detection embeddings.

    Args:
        tracks: Internal BoT-SORT tracklets.
        det_embeddings: Detection embeddings of shape ``(num_dets, dim)``.

    Returns:
        Similarity matrix of shape ``(num_tracks, num_dets)``.
    """
    track_feats = [
        t.feature_bank.feature
        if t.feature_bank is not None and t.feature_bank.is_initialized
        else None
        for t in tracks
    ]
    return appearance_similarity(track_feats, det_embeddings)


def _draw_param_row(panel: np.ndarray, y: int, label: str, value: str) -> int:
    _panel_text(panel, label, 16, y, color=PANEL_MUTED, scale=0.40)
    _panel_text(panel, value, PANEL_VALUE_COL, y, color=PANEL_TEXT, scale=0.40)
    return y + 20


def _draw_params_section(panel: np.ndarray, y: int, config: PanelConfig, live: PanelLive) -> int:
    _panel_text(panel, "Parameters", 16, y, scale=0.48, thickness=1)
    y += 24
    y = _draw_param_row(panel, y, "Detector", config.detector)
    y = _draw_param_row(panel, y, "ReID", config.reid_model)
    y = _draw_param_row(panel, y, "Classes", config.classes)
    y = _draw_param_row(panel, y, "High conf", f">= {config.confidence:.2f}")
    y = _draw_param_row(panel, y, "Min activation", f"{config.min_activation_frames} frames")
    y = _draw_param_row(
        panel,
        y,
        "Lost buffer",
        f"{config.lost_buffer} ref frames (~{config.max_lost_frames} max)",
    )
    y = _draw_param_row(
        panel,
        y,
        "ReID active",
        f"cos > {1.0 - config.emb_threshold:.2f}, IoU > {1.0 - config.iou_threshold:.2f}",
    )
    y = _draw_param_row(
        panel,
        y,
        "ReID lost",
        f"cos > {1.0 - config.emb_threshold_lost:.2f}, IoU > {1.0 - config.iou_threshold_lost:.2f}",
    )
    y = _draw_param_row(panel, y, "CMC", "on" if config.cmc_enabled else "off")
    y = _panel_rule(panel, y)
    _panel_text(panel, "Live", 16, y, scale=0.48)
    y += 24
    y = _draw_param_row(panel, y, "FPS", f"{live.fps:.1f}")
    y = _draw_param_row(panel, y, "Tracks", str(live.num_tracks))
    y = _draw_param_row(panel, y, "Detections", f"{live.num_dets} ({live.num_high_dets} high conf)")
    y = _draw_param_row(panel, y, "Keys", "q quit  ·  r reset")
    return y


def _track_has_reid(track) -> bool:
    return track.feature_bank is not None and track.feature_bank.is_initialized


def _draw_similarity_section(
    panel: np.ndarray,
    y: int,
    *,
    title: str,
    track_rows: list[tuple[int, object]],
    det_labels: list[str],
    matrix: np.ndarray,
    frame_height: int,
    reid: bool = False,
) -> int:
    y = _panel_rule(panel, y + 4)
    _panel_text(panel, title, 16, y, scale=0.48)
    y += 22

    n_dets = len(det_labels)
    if n_dets == 0:
        _panel_text(panel, "No detections", 16, y, color=PANEL_MUTED, scale=0.40)
        return y + 22

    for col, label in enumerate(det_labels):
        _panel_text(
            panel,
            label,
            PANEL_DET_COL + col * PANEL_DET_SPACING,
            y,
            color=PANEL_MUTED,
            scale=0.34,
        )
    y += 20

    if not track_rows:
        _panel_text(panel, "No tracks", 16, y, color=PANEL_MUTED, scale=0.40)
        return y + 22

    for track_row, track in track_rows:
        if y > frame_height - 16:
            _panel_text(panel, "...", 16, y, color=PANEL_MUTED)
            return y + 20

        if track.tracker_id >= 0:
            id_label = f"ID {track.tracker_id}"
        else:
            id_label = "ID ?"

        if track.time_since_update == 0:
            state = "active"
            state_color = PANEL_TEXT
        else:
            state = f"lost {track.time_since_update}"
            state_color = PANEL_LOST

        _panel_text(panel, id_label, 16, y, color=state_color, scale=0.40, thickness=1)
        _panel_text(panel, state, 88, y, color=state_color, scale=0.36)

        for col in range(n_dets):
            if track_row >= matrix.shape[0]:
                cell = "—"
                color = PANEL_MUTED
            elif reid and not _track_has_reid(track):
                cell = "—"
                color = PANEL_MUTED
            else:
                value = float(matrix[track_row, col])
                cell = f"{value:.2f}"
                color = _similarity_color(value)
            _panel_text(
                panel,
                cell,
                PANEL_DET_COL + col * PANEL_DET_SPACING,
                y,
                color=color,
                scale=PANEL_VALUE_SCALE,
                thickness=2,
            )
        y += 26

    return y


def render_side_panel(
    *,
    frame_height: int,
    config: PanelConfig,
    live: PanelLive,
    tracks,
    detections: sv.Detections,
    iou_matrix: np.ndarray,
    reid_matrix: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    """Render the side panel with parameters and similarity tables.

    Args:
        frame_height: Height of the annotated video frame in pixels.
        config: Static demo configuration values.
        live: Per-frame runtime stats.
        tracks: Internal BoT-SORT tracklets.
        detections: Current-frame detections passed to the tracker.
        iou_matrix: IoU similarities of shape ``(num_tracks, num_dets)``.
        reid_matrix: ReID cosine similarities of the same shape.
        class_names: Model class list where index equals class id.

    Returns:
        BGR side-panel image.
    """
    panel = np.full((frame_height, PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)
    y = 28
    _panel_text(panel, "BoT-SORT + ReID", 16, y, scale=0.62, thickness=2)
    y += 28
    y = _draw_params_section(panel, y, config, live)

    det_labels = _det_column_labels(detections, class_names)
    track_rows = sorted(
        enumerate(tracks),
        key=lambda pair: (
            pair[1].tracker_id if pair[1].tracker_id >= 0 else 10_000,
            pair[1].time_since_update,
        ),
    )

    y = _draw_similarity_section(
        panel,
        y,
        title="cos ReID (track x det)",
        track_rows=track_rows,
        det_labels=det_labels,
        matrix=reid_matrix,
        frame_height=frame_height,
        reid=True,
    )
    y = _draw_similarity_section(
        panel,
        y,
        title="IoU (track x det)",
        track_rows=track_rows,
        det_labels=det_labels,
        matrix=iou_matrix,
        frame_height=frame_height,
    )

    legend_y = min(y + 12, frame_height - 18)
    _panel_text(panel, "color:", 16, legend_y, color=PANEL_MUTED, scale=0.34)
    legend_x = 62
    for label, color in (("0.80+", PANEL_SIM_GOOD), ("0.50", PANEL_SIM_MED), ("0.25", PANEL_SIM_HIGH), ("<0.25", PANEL_SIM_BAD)):
        _panel_text(panel, label, legend_x, legend_y, color=color, scale=0.34, thickness=2)
        legend_x += 48

    return panel


def attach_side_panel(frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
    """Concatenate the video frame and side panel horizontally.

    Args:
        frame: Annotated video frame.
        panel: Side-panel image.

    Returns:
        Combined BGR image.
    """
    height = frame.shape[0]
    if panel.shape[0] != height:
        panel = cv2.resize(panel, (PANEL_WIDTH, height))
    return np.hstack([frame, panel])


def parse_camera(value: str) -> int | None:
    if value.strip().lower() == "auto":
        return None
    return int(value)


def _host_app_hint() -> tuple[str, str]:
    """Return (display name, tccutil bundle id) for the app running Python."""
    exe = sys.executable.lower()
    if "cursor" in exe:
        return "Cursor", "com.todesktop.230313mzl4w4u92"
    if "iterm" in exe:
        return "iTerm", "com.googlecode.iterm2"
    return "Terminal", "com.apple.Terminal"


def read_frame_with_warmup(
    cap: cv2.VideoCapture,
    *,
    attempts: int = 40,
    delay_s: float = 0.05,
) -> np.ndarray | None:
    """Read until a non-empty frame arrives (macOS webcams often need warmup)."""
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return frame
        time.sleep(delay_s)
    return None


def open_webcam(index: int) -> cv2.VideoCapture:
    """Open a webcam, preferring AVFoundation on macOS."""
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)
    return cap


def webcam_error_message(tried_indices: list[int]) -> str:
    indices = ", ".join(str(i) for i in tried_indices) or "none"
    msg = f"No working webcam among indices tried: {indices}."
    if platform.system() == "Darwin":
        app_name, bundle_id = _host_app_hint()
        msg += (
            "\n\nmacOS camera checklist:"
            f"\n  1. System Settings → Privacy & Security → Camera → enable {app_name}"
            f"\n  2. Quit {app_name} completely (Cmd+Q), reopen, run again, click Allow"
            f"\n  3. If you denied access: tccutil reset Camera {bundle_id}"
            "\n  4. Index 0 is often iPhone Continuity Camera (needs phone unlocked)."
            " Built-in FaceTime camera is usually index 1:"
            "\n       uv run python notebooks/webcam_reid_demo.py --camera 1"
            "\n  5. Probe devices: --list-cameras"
        )
    return msg


def acquire_webcam(preferred: int | None) -> tuple[cv2.VideoCapture, int, np.ndarray]:
    """Open the first camera that delivers frames."""
    if preferred is not None:
        order = [preferred] + [i for i in range(6) if i != preferred]
    else:
        order = list(range(6))

    tried: list[int] = []
    for index in order:
        tried.append(index)
        cap = open_webcam(index)
        if not cap.isOpened():
            cap.release()
            continue
        frame = read_frame_with_warmup(cap)
        if frame is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap, index, frame
        cap.release()

    raise SystemExit(webcam_error_message(tried))


def list_cameras(max_index: int = 5) -> None:
    """Probe camera indices and print readability results."""
    print("Probing camera indices (macOS needs Privacy → Camera enabled)...")
    any_readable = False
    for index in range(max_index + 1):
        cap = open_webcam(index)
        if not cap.isOpened():
            print(f"  index {index}: not available")
            cap.release()
            continue
        frame = read_frame_with_warmup(cap, attempts=20)
        if frame is not None:
            h, w = frame.shape[:2]
            print(f"  index {index}: readable ({w}x{h})")
            any_readable = True
        else:
            print(f"  index {index}: opened but no frames (Continuity Camera?)")
        cap.release()
    if not any_readable:
        print(webcam_error_message(list(range(max_index + 1))))


def main() -> None:
    """Run the live webcam BoT-SORT + ReID demo."""
    args = parse_args()

    if args.list_cameras:
        list_cameras()
        return

    preferred = parse_camera(args.camera)
    cap, camera_index, probe_frame = acquire_webcam(preferred)
    print(
        f"Webcam index {camera_index} OK "
        f"({probe_frame.shape[1]}x{probe_frame.shape[0]}). Loading models..."
    )
    detector = load_detector(args.model)
    class_names = list(getattr(detector, "class_names", None) or [])
    class_filter = None if args.all_classes else resolve_class_filter(args.classes, class_names)
    class_summary = format_class_summary(class_filter, class_names)
    print(
        f"Detector ready: {args.model!r} (RF-DETR via inference-models, tracking {class_summary})"
    )

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.9, text_thickness=2)

    print(f"Loading ReID model ({args.reid_source})...")
    reid_model = ReIDModel.from_pretrained(args.reid_source)

    tracker = BoTSORTTracker(
        lost_track_buffer=args.lost_buffer,
        frame_rate=args.fps,
        track_activation_threshold=args.confidence,
        high_conf_det_threshold=args.confidence,
        minimum_iou_threshold_first_assoc=0.08,
        minimum_consecutive_frames=MIN_ACTIVATION_FRAMES,
        instant_first_frame_activation=True,
        enable_cmc=not args.no_cmc,
        reid_model=reid_model,
        reid_emb_dist_threshold=args.reid_emb_dist_threshold,
        reid_iou_dist_threshold=args.reid_iou_dist_threshold,
        reid_iou_dist_threshold_lost=args.reid_iou_dist_threshold_lost,
        reid_emb_dist_threshold_lost=args.reid_emb_dist_threshold_lost,
    )
    max_lost_frames = tracker.maximum_frames_without_update

    panel_config = PanelConfig(
        detector=args.model,
        reid_model=args.reid_source,
        classes=class_summary,
        confidence=args.confidence,
        min_activation_frames=MIN_ACTIVATION_FRAMES,
        lost_buffer=args.lost_buffer,
        max_lost_frames=max_lost_frames,
        emb_threshold=args.reid_emb_dist_threshold,
        emb_threshold_lost=args.reid_emb_dist_threshold_lost,
        iou_threshold=args.reid_iou_dist_threshold,
        iou_threshold_lost=args.reid_iou_dist_threshold_lost,
        cmc_enabled=not args.no_cmc,
    )

    print(f"Open {WINDOW_NAME}. Walk out of frame and back to test ReID.")
    frame_times: list[float] = []
    t_prev = time.perf_counter()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("Webcam read failed — exiting.")
            break

        raw = run_detector(detector, frame_bgr)
        filtered_dets, _raw_count = filter_detections(
            raw,
            class_filter=class_filter,
        )

        det_embeddings = np.empty((0, 0), dtype=np.float32)
        if len(filtered_dets) > 0:
            det_embeddings = extract_detection_embeddings(
                reid_model, frame_bgr, filtered_dets.xyxy
            )

        tracked = tracker.update(filtered_dets, frame=frame_bgr)

        annotated = box_annotator.annotate(frame_bgr.copy(), tracked)
        annotated = label_annotator.annotate(
            annotated,
            tracked,
            labels=format_track_labels(tracked, class_names),
        )

        t_now = time.perf_counter()
        frame_times.append(t_now - t_prev)
        t_prev = t_now
        if len(frame_times) > 30:
            frame_times.pop(0)
        display_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

        high_conf_mask = (
            filtered_dets.confidence >= args.confidence
            if filtered_dets.confidence is not None and len(filtered_dets) > 0
            else np.zeros(len(filtered_dets), dtype=bool)
        )
        num_high_dets = int(np.sum(high_conf_mask)) if len(filtered_dets) else 0

        if not args.no_debug_panel:
            iou_matrix = compute_iou_matrix(tracker.tracks, filtered_dets)
            reid_matrix = compute_reid_similarities(tracker.tracks, det_embeddings)
            side_panel = render_side_panel(
                frame_height=annotated.shape[0],
                config=panel_config,
                live=PanelLive(
                    fps=display_fps,
                    num_tracks=len(tracker.tracks),
                    num_dets=len(filtered_dets),
                    num_high_dets=num_high_dets,
                ),
                tracks=tracker.tracks,
                detections=filtered_dets,
                iou_matrix=iou_matrix,
                reid_matrix=reid_matrix,
                class_names=class_names,
            )
            annotated = attach_side_panel(annotated, side_panel)

        cv2.imshow(WINDOW_NAME, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            tracker.reset()
            print("Tracker reset — next detection gets a new ID.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
