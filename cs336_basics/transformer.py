import torch
import torch.nn as nn
from cs336_basics import rope
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.attention import CausalMultiheadSelfAttention
from cs336_basics.linear import Linear
from cs336_basics.swiglu import SwiGLU, SiLU
from cs336_basics.rope import RotaryPositionalEmbedding



class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding | None, device=None, dtype=None):
        super().__init__()

        self.ln1 = RMSNorm(d_model, eps=1e-6, device = device, dtype = dtype)        
        self.attn = CausalMultiheadSelfAttention(d_model, num_heads, rope = rope, device = device, dtype = dtype)
        self.ln2 = RMSNorm(d_model, eps=1e-6, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype) 

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        attn_out = self.attn(self.ln1(x), token_positions)
        x = x + attn_out  # Residual

        # Pre-norm feed-forward
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out  # Residual

        return x