import torch
import torch.nn as nn


class DeepSORTBackbone(nn.Module):
    """
    Backbone model for DeepSORT's feature extractor.

    References:
        - [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
        - [https://github.com/nwojke/deep_sort/blob/master/tools/freeze_model.py](https://github.com/nwojke/deep_sort/blob/master/tools/freeze_model.py)
        - [https://github.com/abhyantrika/nanonets_object_tracking/blob/master/siamese_net.py](https://github.com/abhyantrika/nanonets_object_tracking/blob/master/siamese_net.py)
    """

    def __init__(self):
        super(DeepSORTBackbone, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on a single input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor.
        """
        return self.network(input_tensor)
