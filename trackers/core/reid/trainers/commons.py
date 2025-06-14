import json
import os
from typing import Any

import torch.nn as nn
from safetensors.torch import save_file

from trackers.core.reid.trainers.callbacks import BaseCallback


def save_checkpoint(
    model: nn.Module,
    checkpoint_interval: int,
    epoch: int,
    log_dir: str,
    config: dict[str, Any],
    callbacks: list[BaseCallback],
):
    if checkpoint_interval is not None and (epoch + 1) % checkpoint_interval == 0:
        state_dict = model.state_dict()
        checkpoint_path = os.path.join(
            log_dir, "checkpoints", f"reid_model_{epoch + 1}.safetensors"
        )
        save_file(
            state_dict,
            checkpoint_path,
            metadata={"config": json.dumps(config), "format": "pt"},
        )
        if callbacks:
            for callback in callbacks:
                callback.on_checkpoint_save(checkpoint_path, epoch + 1)
