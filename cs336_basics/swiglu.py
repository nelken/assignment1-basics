import torch
import math
from torch import Tensor
from jaxtyping import Float
from cs336_basics.linear import Linear  


def silu_activation(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1 = self.w1(x)
        silu = silu_activation(a1)
        return self.w2(silu * self.w3(x))   

class SiLU(torch.nn.Module):
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu = silu_activation(x)   
        return silu
