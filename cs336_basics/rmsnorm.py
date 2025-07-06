import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the L2 norm over the last dimension (d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm = x.norm(2, dim=-1, keepdim=True)  # shape: (batch, seq_len, 1)
        rms = norm / (self.d_model ** 0.5)
        x_normed = x / (rms + self.eps)
        result = self.weight * x_normed
        return result.to(in_dtype)
    