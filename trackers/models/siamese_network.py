from typing import Optional

import torch
import torch.nn as nn


class SiameseNetworkModel(nn.Module):
    def __init__(self, backbone_model: nn.Module):
        super(SiameseNetworkModel, self).__init__()
        self.backbone_model = backbone_model

    def forward_on_single_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.backbone_model(input_tensor)
        output = torch.squeeze(output)
        return output

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        input3: Optional[torch.Tensor] = None,
    ):
        output1 = self.forward_on_single_input(input1)
        output2 = self.forward_on_single_input(input2)

        if input3 is not None:
            output3 = self.forward_on_single_input(input3)
            return output1, output2, output3

        return output1, output2
