import torch
import torch.nn as nn


class SiameseNetworkModel(nn.Module):
    """
    Siamese network model for feature extraction.

    References:
        - [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
        - [https://github.com/nwojke/deep_sort/blob/master/tools/freeze_model.py](https://github.com/nwojke/deep_sort/blob/master/tools/freeze_model.py)
        - [https://github.com/abhyantrika/nanonets_object_tracking/blob/master/siamese_net.py](https://github.com/abhyantrika/nanonets_object_tracking/blob/master/siamese_net.py)

    Args:
        backbone_model (nn.Module): The backbone model to use for feature extraction.
    """

    def __init__(self, backbone_model: nn.Module):
        super(SiameseNetworkModel, self).__init__()
        self.backbone_model = backbone_model

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on a single input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor.
        """
        output = self.backbone_model(input_tensor)
        output = torch.squeeze(output)
        return output
