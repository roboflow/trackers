# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from trackers.calibration.base import PitchCalibrator
from trackers.calibration.projection import invert_homography
from trackers.calibration.smoothing import fill_calibration_gaps
from trackers.calibration.types import CalibrationFrame, PitchDimensions


class PnLCalibProvider(PitchCalibrator):
    """Adapter for the upstream PnLCalib inference pipeline."""

    def __init__(
        self,
        *,
        pitch_dimensions: PitchDimensions | None = None,
        config_path: str | Path | None = None,
        config_data: dict[str, object] | None = None,
        upstream_root: str | Path = "third_party/pnlcalib",
    ) -> None:
        super().__init__(pitch_dimensions=pitch_dimensions)
        self.config_path = None if config_path is None else Path(config_path)
        self.config_data = config_data or self._load_config_data()
        self.upstream_root = Path(upstream_root)
        self._runtime: dict[str, Any] | None = None
        self._models: tuple[Any, Any] | None = None

    @property
    def name(self) -> str:
        return "pnlcalib"

    def is_available(self) -> bool:
        return (
            self.upstream_root.exists()
            and self._weights_kp_path().exists()
            and self._weights_line_path().exists()
            and importlib.util.find_spec("ellipse") is not None
        )

    def availability_hint(self) -> str | None:
        if self.is_available():
            return None
        if not self.upstream_root.exists():
            return (
                "Expected an upstream checkout at "
                f"{self.upstream_root}. Clone or vendor PnLCalib there first."
            )
        if importlib.util.find_spec("ellipse") is None:
            return (
                "PnLCalib needs the `lsq-ellipse` package in the current "
                "environment. Install it with `pip install lsq-ellipse==2.2.1`."
            )
        missing_weights = []
        if not self._weights_kp_path().exists():
            missing_weights.append(str(self._weights_kp_path()))
        if not self._weights_line_path().exists():
            missing_weights.append(str(self._weights_line_path()))
        if missing_weights:
            return (
                "PnLCalib weights are missing. Expected: "
                + ", ".join(missing_weights)
            )
        return (
            "PnLCalib is not ready in the current environment. Check the "
            "upstream checkout, weights, and Python dependencies."
        )

    def describe(self) -> dict[str, object]:
        data = super().describe()
        data.update(
            {
                "config_path": None if self.config_path is None else str(self.config_path),
                "upstream_root": str(self.upstream_root),
                "weights_kp": str(self._weights_kp_path()),
                "weights_line": str(self._weights_line_path()),
            }
        )
        return data

    def calibrate_video(
        self,
        source: str | Path,
        output_dir: str | Path,
    ) -> list[CalibrationFrame]:
        runtime = self._ensure_runtime()
        model_kp, model_line = self._ensure_models()
        source_path = Path(source)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video source: {source_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sampling_every_n = max(1, int(self._sampling_config().get("every_n_frames", 1)))
        cam = runtime["FramebyFrameCalib"](
            iwidth=frame_width,
            iheight=frame_height,
            denormalize=True,
        )

        frames: list[CalibrationFrame] = []
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            timestamp_s = (frame_number - 1) / fps

            if (frame_number - 1) % sampling_every_n == 0:
                calibration = self._run_frame_inference(
                    frame=frame,
                    frame_idx=frame_number,
                    timestamp_s=timestamp_s,
                    cam=cam,
                    runtime=runtime,
                    model_kp=model_kp,
                    model_line=model_line,
                )
            else:
                calibration = CalibrationFrame(
                    frame_idx=frame_number,
                    timestamp_s=timestamp_s,
                    confidence=0.0,
                    provider=self.name,
                    pitch_dimensions=self.pitch_dimensions,
                    diagnostics={"sampled": False},
                )
            frames.append(calibration)

        cap.release()

        smoothing_config = self._smoothing_config()
        if bool(smoothing_config.get("enabled", True)):
            frames = fill_calibration_gaps(
                frames,
                max_gap_frames=int(smoothing_config.get("max_gap_frames", 15)),
                min_confidence=float(smoothing_config.get("min_confidence", 0.4)),
                mode=str(smoothing_config.get("mode", "hold")),
                edge_strategy=str(smoothing_config.get("edge_strategy", "hold")),
            )

        overlay_config = self._overlay_config()
        if bool(overlay_config.get("enabled", False)):
            overlay_name = str(overlay_config.get("path", "overlay.mp4"))
            self._render_overlay_video(
                source=source_path,
                frames=frames,
                fps=fps,
                frame_width=frame_width,
                frame_height=frame_height,
                output_path=output_path / overlay_name,
                line_color_bgr=tuple(overlay_config.get("line_color_bgr", [255, 0, 0])),
                line_thickness=int(overlay_config.get("line_thickness", 2)),
            )

        if total_frames and len(frames) != total_frames:
            diagnostics_path = output_path / "provider_diagnostics.json"
            diagnostics_path.write_text(
                (
                    "{\n"
                    f'  "expected_frames": {total_frames},\n'
                    f'  "produced_frames": {len(frames)}\n'
                    "}\n"
                ),
                encoding="utf-8",
            )

        return frames

    def _load_config_data(self) -> dict[str, object]:
        if self.config_path is None or not self.config_path.exists():
            return {}
        return tomllib.loads(self.config_path.read_text(encoding="utf-8"))

    def _provider_config(self) -> dict[str, Any]:
        section = self.config_data.get("provider", {})
        return section if isinstance(section, dict) else {}

    def _weights_config(self) -> dict[str, Any]:
        section = self.config_data.get("weights", {})
        return section if isinstance(section, dict) else {}

    def _threshold_config(self) -> dict[str, Any]:
        section = self.config_data.get("thresholds", {})
        return section if isinstance(section, dict) else {}

    def _sampling_config(self) -> dict[str, Any]:
        section = self.config_data.get("sampling", {})
        return section if isinstance(section, dict) else {}

    def _smoothing_config(self) -> dict[str, Any]:
        section = self.config_data.get("smoothing", {})
        return section if isinstance(section, dict) else {}

    def _overlay_config(self) -> dict[str, Any]:
        section = self.config_data.get("overlay", {})
        return section if isinstance(section, dict) else {}

    def _inference_config(self) -> dict[str, Any]:
        section = self.config_data.get("inference", {})
        return section if isinstance(section, dict) else {}

    def _solve_mode(self) -> str:
        mode = str(self._inference_config().get("mode", "camera"))
        if mode not in {"camera", "homography"}:
            raise ValueError(f"Unsupported PnLCalib solve mode: {mode}")
        return mode

    def _resolve_relative_path(self, value: str | Path, *, root: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return root / path

    def _weights_kp_path(self) -> Path:
        return self._resolve_relative_path(
            self._weights_config().get("keypoints", "weights/SV_kp"),
            root=self.upstream_root,
        )

    def _weights_line_path(self) -> Path:
        return self._resolve_relative_path(
            self._weights_config().get("lines", "weights/SV_lines"),
            root=self.upstream_root,
        )

    def _kp_threshold(self) -> float:
        return float(self._threshold_config().get("keypoints", 0.3434))

    def _line_threshold(self) -> float:
        return float(self._threshold_config().get("lines", 0.7867))

    def _max_reproj_error(self) -> float:
        return float(self._threshold_config().get("max_reproj_error", 20.0))

    def _vote_threshold(self) -> float:
        return float(self._threshold_config().get("vote_threshold", 5.0))

    def _pnl_refine(self) -> bool:
        return bool(self._inference_config().get("pnl_refine", True))

    def _ensure_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime

        os.environ.setdefault(
            "XDG_CACHE_HOME",
            str(self.upstream_root.parent.parent / ".cache"),
        )
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str((self.upstream_root.parent.parent / ".cache" / "matplotlib")),
        )

        upstream_path = str(self.upstream_root.resolve())
        if upstream_path not in sys.path:
            sys.path.insert(0, upstream_path)

        import torch
        import torchvision.transforms as T
        import torchvision.transforms.functional as f
        import yaml
        from PIL import Image

        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l
        from utils.utils_calib import FramebyFrameCalib
        from utils.utils_heatmap import (
            complete_keypoints,
            coords_to_dict,
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
        )

        device_name = str(self._provider_config().get("device", "auto"))
        if device_name == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_name)

        cfg = yaml.safe_load(
            (self.upstream_root / "config" / "hrnetv2_w48.yaml").read_text(encoding="utf-8")
        )
        cfg_l = yaml.safe_load(
            (self.upstream_root / "config" / "hrnetv2_w48_l.yaml").read_text(
                encoding="utf-8"
            )
        )

        self._runtime = {
            "Image": Image,
            "FramebyFrameCalib": FramebyFrameCalib,
            "T": T,
            "f": f,
            "cfg": cfg,
            "cfg_l": cfg_l,
            "device": device,
            "torch": torch,
            "get_cls_net": get_cls_net,
            "get_cls_net_l": get_cls_net_l,
            "get_keypoints_from_heatmap_batch_maxpool": get_keypoints_from_heatmap_batch_maxpool,
            "get_keypoints_from_heatmap_batch_maxpool_l": get_keypoints_from_heatmap_batch_maxpool_l,
            "coords_to_dict": coords_to_dict,
            "complete_keypoints": complete_keypoints,
            "transform": T.Resize((540, 960)),
        }
        return self._runtime

    def _ensure_models(self) -> tuple[Any, Any]:
        if self._models is not None:
            return self._models

        runtime = self._ensure_runtime()
        torch = runtime["torch"]
        device = runtime["device"]
        get_cls_net = runtime["get_cls_net"]
        get_cls_net_l = runtime["get_cls_net_l"]

        loaded_state = torch.load(self._weights_kp_path(), map_location=device)
        model_kp = get_cls_net(runtime["cfg"])
        model_kp.load_state_dict(loaded_state)
        model_kp.to(device)
        model_kp.eval()

        loaded_state_l = torch.load(self._weights_line_path(), map_location=device)
        model_line = get_cls_net_l(runtime["cfg_l"])
        model_line.load_state_dict(loaded_state_l)
        model_line.to(device)
        model_line.eval()

        self._models = (model_kp, model_line)
        return self._models

    def _run_frame_inference(
        self,
        *,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        cam: Any,
        runtime: dict[str, Any],
        model_kp: Any,
        model_line: Any,
    ) -> CalibrationFrame:
        torch = runtime["torch"]
        image = runtime["Image"].fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = runtime["f"].to_tensor(image).float().unsqueeze(0)
        if frame_tensor.size()[-1] != 960:
            frame_tensor = runtime["transform"](frame_tensor)
        frame_tensor = frame_tensor.to(runtime["device"])
        _, _, h, w = frame_tensor.size()

        with torch.no_grad():
            heatmaps = model_kp(frame_tensor)
            heatmaps_l = model_line(frame_tensor)

        kp_coords = runtime["get_keypoints_from_heatmap_batch_maxpool"](
            heatmaps[:, :-1, :, :]
        )
        line_coords = runtime["get_keypoints_from_heatmap_batch_maxpool_l"](
            heatmaps_l[:, :-1, :, :]
        )
        kp_dict = runtime["coords_to_dict"](kp_coords, threshold=self._kp_threshold())
        lines_dict = runtime["coords_to_dict"](
            line_coords, threshold=self._line_threshold()
        )
        kp_dict, lines_dict = runtime["complete_keypoints"](
            kp_dict[0],
            lines_dict[0],
            w=w,
            h=h,
            normalize=True,
        )

        cam.update(kp_dict, lines_dict)
        if self._solve_mode() == "homography":
            return self._solve_homography_only(
                cam=cam,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
            )

        result = cam.heuristic_voting(
            refine=False,
            refine_lines=self._pnl_refine(),
            th=self._vote_threshold(),
        )
        if result is None:
            return CalibrationFrame(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                confidence=0.0,
                provider=self.name,
                pitch_dimensions=self.pitch_dimensions,
                diagnostics={"sampled": True, "reason": "no_solution"},
            )

        rep_err = float(result["rep_err"])
        if rep_err > self._max_reproj_error():
            return CalibrationFrame(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                confidence=0.0,
                provider=self.name,
                pitch_dimensions=self.pitch_dimensions,
                diagnostics={
                    "sampled": True,
                    "reason": "reprojection_error_too_high",
                    "rep_err": rep_err,
                    "mode": result.get("mode"),
                    "use_ransac": result.get("use_ransac"),
                    "calib_plane": result.get("calib_plane"),
                },
            )

        camera_params = dict(result["cam_params"])
        pitch_to_image = self._pitch_to_image_homography(camera_params)
        image_to_pitch = invert_homography(pitch_to_image)
        confidence = max(
            0.0,
            min(1.0, 1.0 - (rep_err / self._max_reproj_error())),
        )
        diagnostics = {
            "sampled": True,
            "rep_err": rep_err,
            "mode": result.get("mode"),
            "use_ransac": result.get("use_ransac"),
            "calib_plane": result.get("calib_plane"),
        }
        camera_parameters = {
            **camera_params,
            "world_origin": "pitch_center",
            "position_meters_top_left_origin": [
                float(camera_params["position_meters"][0] + (self.pitch_dimensions.length_m / 2.0)),
                float(camera_params["position_meters"][1] + (self.pitch_dimensions.width_m / 2.0)),
                float(camera_params["position_meters"][2]),
            ],
        }
        return CalibrationFrame(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            image_to_pitch=image_to_pitch,
            pitch_to_image=pitch_to_image,
            confidence=confidence,
            provider=self.name,
            pitch_dimensions=self.pitch_dimensions,
            camera_parameters=camera_parameters,
            diagnostics=diagnostics,
        )

    def _solve_homography_only(
        self,
        *,
        cam: Any,
        frame_idx: int,
        timestamp_s: float,
    ) -> CalibrationFrame:
        result = cam.heuristic_voting_ground(
            refine_lines=self._pnl_refine(),
            th=self._vote_threshold(),
        )
        if result is None:
            return CalibrationFrame(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                confidence=0.0,
                provider=self.name,
                pitch_dimensions=self.pitch_dimensions,
                diagnostics={
                    "sampled": True,
                    "reason": "no_solution",
                    "solve_mode": "homography",
                },
            )

        rep_err = float(result["rep_err"])
        if rep_err > self._max_reproj_error():
            return CalibrationFrame(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                confidence=0.0,
                provider=self.name,
                pitch_dimensions=self.pitch_dimensions,
                diagnostics={
                    "sampled": True,
                    "reason": "reprojection_error_too_high",
                    "solve_mode": "homography",
                    "rep_err": rep_err,
                    "use_ransac": result.get("use_ransac"),
                },
            )

        centered_image_to_pitch = np.asarray(result["homography"], dtype=np.float64)
        centered_to_canonical = np.array(
            [
                [1.0, 0.0, self.pitch_dimensions.length_m / 2.0],
                [0.0, 1.0, self.pitch_dimensions.width_m / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        image_to_pitch = centered_to_canonical @ centered_image_to_pitch
        image_to_pitch /= image_to_pitch[-1, -1]
        pitch_to_image = invert_homography(image_to_pitch)
        confidence = max(
            0.0,
            min(1.0, 1.0 - (rep_err / self._max_reproj_error())),
        )
        return CalibrationFrame(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            image_to_pitch=image_to_pitch,
            pitch_to_image=pitch_to_image,
            confidence=confidence,
            provider=self.name,
            pitch_dimensions=self.pitch_dimensions,
            camera_parameters={},
            diagnostics={
                "sampled": True,
                "solve_mode": "homography",
                "rep_err": rep_err,
                "use_ransac": result.get("use_ransac"),
            },
        )

    def _pitch_to_image_homography(self, camera_params: dict[str, Any]) -> np.ndarray:
        x_focal_length = float(camera_params["x_focal_length"])
        y_focal_length = float(camera_params["y_focal_length"])
        principal_point = np.asarray(camera_params["principal_point"], dtype=np.float64)
        position = np.asarray(camera_params["position_meters"], dtype=np.float64)
        rotation = np.asarray(camera_params["rotation_matrix"], dtype=np.float64)

        intrinsic = np.array(
            [
                [x_focal_length, 0.0, principal_point[0]],
                [0.0, y_focal_length, principal_point[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        it_matrix = np.eye(4, dtype=np.float64)[:-1]
        it_matrix[:, -1] = -position
        projection = intrinsic @ (rotation @ it_matrix)
        centered_pitch_to_image = projection[:, [0, 1, 3]]

        canonical_to_centered = np.array(
            [
                [1.0, 0.0, -(self.pitch_dimensions.length_m / 2.0)],
                [0.0, 1.0, -(self.pitch_dimensions.width_m / 2.0)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        homography = centered_pitch_to_image @ canonical_to_centered
        return homography / homography[-1, -1]

    def _render_overlay_video(
        self,
        *,
        source: Path,
        frames: list[CalibrationFrame],
        fps: float,
        frame_width: int,
        frame_height: int,
        output_path: Path,
        line_color_bgr: tuple[int, int, int],
        line_thickness: int,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(source))
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

        frame_idx = 0
        polylines = self._pitch_overlay_polylines()
        while cap.isOpened() and frame_idx < len(frames):
            ret, frame = cap.read()
            if not ret:
                break
            calibration = frames[frame_idx]
            if calibration.pitch_to_image is not None:
                overlay = frame.copy()
                for polyline in polylines:
                    projected = self._project_polyline(polyline, calibration.pitch_to_image)
                    if projected.shape[0] < 2:
                        continue
                    overlay = cv2.polylines(
                        overlay,
                        [projected.astype(np.int32)],
                        isClosed=False,
                        color=line_color_bgr,
                        thickness=line_thickness,
                    )
                writer.write(overlay)
            else:
                writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

    def _project_polyline(self, polyline: np.ndarray, homography: np.ndarray) -> np.ndarray:
        homogeneous = np.concatenate(
            [polyline, np.ones((polyline.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        projected = homogeneous @ homography.T
        valid_scale = np.abs(projected[:, 2]) > 1e-9
        projected = projected[valid_scale]
        if projected.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        projected = projected[:, :2] / projected[:, 2:3]
        finite = np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1])
        return projected[finite]

    def _pitch_overlay_polylines(self) -> list[np.ndarray]:
        def rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
            return np.array(
                [
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1],
                    [x0, y0],
                ],
                dtype=np.float64,
            )

        def circle(cx: float, cy: float, radius: float, samples: int = 180) -> np.ndarray:
            angles = np.linspace(0.0, 2.0 * np.pi, samples)
            return np.column_stack(
                [cx + (radius * np.cos(angles)), cy + (radius * np.sin(angles))]
            ).astype(np.float64)

        return [
            rect(0.0, 0.0, 105.0, 68.0),
            np.array([[52.5, 0.0], [52.5, 68.0]], dtype=np.float64),
            rect(0.0, 13.84, 16.5, 54.16),
            rect(88.5, 13.84, 105.0, 54.16),
            rect(0.0, 24.84, 5.5, 43.16),
            rect(99.5, 24.84, 105.0, 43.16),
            circle(52.5, 34.0, 9.15),
        ]
