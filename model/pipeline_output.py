from dataclasses import dataclass

import torch
from diffusers.utils import BaseOutput


@dataclass
class BasePipelineOutput(BaseOutput):
    frames: torch.Tensor
