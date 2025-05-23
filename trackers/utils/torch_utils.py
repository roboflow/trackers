import json
import re
from typing import Any, Tuple, Union

import torch
from safetensors import safe_open


def parse_device_spec(device_spec: Union[str, torch.device]) -> torch.device:
    """
    Convert a string or torch.device into a valid torch.device. Allowed strings:
    `'auto'`, `'cpu'`, `'cuda'`, `'cuda:N'` (e.g. `'cuda:0'`), or `'mps'`.
    This function raises ValueError if the input is unrecognized or the GPU
    index is out of range.

    Args:
        device_spec (Union[str, torch.device]): A specification for the device.
            This can be a valid `torch.device` object or one of the recognized
            strings described above.

    Returns:
        torch.device: The corresponding `torch.device` object.

    Raises:
        ValueError: If the device specification is unrecognized or the provided GPU
            index exceeds the available devices.
    """
    if isinstance(device_spec, torch.device):
        return device_spec

    device_str = device_spec.lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_str == "cpu":
        return torch.device("cpu")
    elif device_str == "cuda":
        return torch.device("cuda")
    elif device_str == "mps":
        return torch.device("mps")
    else:
        match = re.match(r"^cuda:(\d+)$", device_str)
        if match:
            index = int(match.group(1))
            if index < 0:
                raise ValueError(f"GPU index must be non-negative, got {index}.")
            if index >= torch.cuda.device_count():
                raise ValueError(
                    f"Requested cuda:{index} but only {torch.cuda.device_count()}"
                    + " GPU(s) are available."
                )
            return torch.device(f"cuda:{index}")

        raise ValueError(f"Unrecognized device spec: {device_spec}")


def load_safetensors_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> Tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Load a safetensors checkpoint into a dictionary of tensors and a dictionary
    of metadata.

    Args:
        checkpoint_path (str): The path to the safetensors checkpoint.
        device (str): The device to load the checkpoint on.

    Returns:
        Tuple[dict[str, torch.Tensor], dict[str, Any]]: A tuple containing the
            state_dict and the config.
    """
    state_dict = {}
    with safe_open(checkpoint_path, framework="pt", device=device) as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
        metadata = f.metadata()
        config = json.loads(metadata["config"]) if "config" in metadata else {}
    model_metadata = config.pop("model_metadata") if "model_metadata" in config else {}
    if "kwargs" in model_metadata:
        kwargs = model_metadata.pop("kwargs")
        model_metadata = {**kwargs, **model_metadata}
    config["model_metadata"] = model_metadata
    return state_dict, config
