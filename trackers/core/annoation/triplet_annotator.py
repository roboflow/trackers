from __future__ import annotations

import validators
import torch.nn as nn
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from trackers.utils.torch_utils import load_torch_checkpoint, parse_device_spec
from trackers.utils.downloader import download_file


class TripletAnnotator:
    def __init__(self, model: nn.Module, video_path: str, device: str = "auto") -> None:
        self.model = model
        self.video_path = video_path
        self.device = parse_device_spec(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        video_path: str,
        checkpoint_file_or_url: str,
        config_file_or_url: str,
        device: str = "auto",
        hydra_overrides_extra: list[str] = [],
        apply_postprocessing: bool = True,
    ) -> TripletAnnotator:
        if validators.url(checkpoint_file_or_url):
            checkpoint_file_or_url = download_file(checkpoint_file_or_url)
        if validators.url(config_file_or_url):
            config_file_or_url = download_file(config_file_or_url)
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
        ]
        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                "++model.binarize_mask_from_pts_for_mem_enc=true",
                "++model.fill_hole_area=8",
            ]
        hydra_overrides.extend(hydra_overrides_extra)
        config = compose(config_name=config_file_or_url, overrides=hydra_overrides)
        OmegaConf.resolve(config)
        model = instantiate(config.model, _recursive_=True)
        load_torch_checkpoint(checkpoint_file_or_url, model)
        return cls(model, video_path, device)
