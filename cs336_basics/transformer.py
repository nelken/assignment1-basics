import torch
import torch.nn as nn
from cs336_basics import rope
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.attention import CausalMultiheadSelfAttention
from cs336_basics.linear import Linear
from cs336_basics.swiglu import SwiGLU, SiLU
from cs336_basics.rope import RotaryPositionalEmbedding



class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding, device=None, dtype=None):
        super().__init__()

        self.rope = rope
        self.rms1 = RMSNorm(d_model, device = device, dtype = dtype)
        self.self_attn = CausalMultiheadSelfAttention(d_model, num_heads, rope = rope, device = device, dtype = dtype)
        self.rms2 = RMSNorm(d_model, device=None, dtype=None)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        attn_out = self.self_attn(self.rms1(x))
        x = x + attn_out  # Residual

        # Pre-norm feed-forward
        ffn_out = self.ffn(self.rms2(x))
        x = x + ffn_out  # Residual

        return x