import torch
import math
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE."
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        # Compute frequencies: shape (d_k // 2,)
        freqs = theta ** (
            -torch.arange(0, d_k, 2, dtype=torch.float32) / d_k
        ).to(device)
        
        # Compute positions: shape (max_seq_len, 1)
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1).to(device)

        # Compute angles: shape (max_seq_len, d_k // 2)
        angles = positions * freqs  # Broadcasting: (max_seq_len, d_k // 2)

        # Precompute cos and sin, interleaved to match [even, odd, even, odd, ...]
        cos = torch.cos(angles).repeat_interleave(2, dim=1)  # (max_seq_len, d_k)
        sin = torch.sin(angles).repeat_interleave(2, dim=1)  # (max_seq_len, d_k)

        self.register_buffer("cos", cos, persistent=False)  # (max_seq_len, d_k)
        self.register_buffer("sin", sin, persistent=False)  # (max_seq_len, d_k)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (..., seq_len, d_k)
        token_positions: Tensor of shape (..., seq_len)
        """
        *batch_dims, seq_len, d_k = x.shape
        assert d_k == self.d_k, f"Expected d_k={self.d_k}, got {d_k}"

        # Get cos/sin for positions: (..., seq_len, d_k)
        cos = self.cos[token_positions]  # Advanced indexing
        sin = self.sin[token_positions]

        # Split last dimension into even and odd parts
        x1 = x[..., ::2]  # (..., seq_len, d_k//2)
        x2 = x[..., 1::2] # (..., seq_len, d_k//2)

        # Apply RoPE rotation: interleaved [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        cos = cos[..., ::2]
        sin = sin[..., ::2]
        x_rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)  # (..., seq_len, d_k//2, 2)

        return x_rotated.flatten(-2)  # (..., seq_len, d_k)