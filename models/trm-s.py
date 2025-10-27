from dataclasses import dataclass
import torch
from typing import Dict


@dataclass
class InnerCarryData:
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class CarryData:
    inner_carry: InnerCarryData
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]










class OuterReasoningModel(nn.Module):
    def __init(self, config_dict: dict):
        super().__init__()
        self.config = config_dict
        self.inner = InnerReasoningModel(config_dict)

    def forward():
        pass