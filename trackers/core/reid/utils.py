from typing import Optional

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from trackers.log import get_logger
from trackers.utils.torch_utils import load_safetensors_checkpoint

logger = get_logger(__name__)



def initialize_reid_model_from_timm(
    cls,
    model_name_or_checkpoint_path: str,
    device: Optional[str] = "auto",
    get_pooled_features: bool = True,
    **kwargs,
):
    if model_name_or_checkpoint_path not in timm.list_models(
        filter=model_name_or_checkpoint_path, pretrained=True
    ):
        probable_model_name_list = timm.list_models(
            f"*{model_name_or_checkpoint_path}*", pretrained=True
        )
        if len(probable_model_name_list) == 0:
            raise ValueError(
                f"Model {model_name_or_checkpoint_path} not found in timm. "
                + "Please check the model name and try again."
            )
        logger.warning(
            f"Model {model_name_or_checkpoint_path} not found in timm. "
            + f"Using {probable_model_name_list[0]} instead."
        )
        model_name_or_checkpoint_path = probable_model_name_list[0]
    if not get_pooled_features:
        kwargs["global_pool"] = ""
    model = timm.create_model(
        model_name_or_checkpoint_path, pretrained=True, num_classes=0, **kwargs
    )
    config = resolve_data_config(model.pretrained_cfg)
    transforms = create_transform(**config)
    model_metadata = {
        "model_source": "timm",
        "model_name_or_checkpoint_path": model_name_or_checkpoint_path,
        "get_pooled_features": get_pooled_features,
        "kwargs": kwargs,
    }
    return cls(model, device, transforms, model_metadata)


def initialize_reid_model_from_checkpoint(cls, checkpoint_path: str):
    state_dict, config = load_safetensors_checkpoint(checkpoint_path)
    reid_model_instance = initialize_reid_model_from_timm(
        cls, **config["model_metadata"]
    )
    if config["projection_dimension"]:
        reid_model_instance._add_projection_layer(
            projection_dimension=config["projection_dimension"]
        )
    for k, v in state_dict.items():
        state_dict[k].to(reid_model_instance.device)
    reid_model_instance.backbone_model.load_state_dict(state_dict)
    return reid_model_instance
