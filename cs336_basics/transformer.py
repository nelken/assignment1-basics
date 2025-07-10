import torch
import torch.nn as nn
from cs336_basics import rope
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.attention import CausalMultiheadSelfAttention
from cs336_basics.linear import Linear
from cs336_basics.swiglu import SwiGLU, SiLU
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.embedding import Embedding


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float | None, device=None, dtype=None):
        super().__init__()

        self.ln1 = RMSNorm(d_model, eps=1e-6, device = device, dtype = dtype)        
        self.attn = CausalMultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device = device, dtype = dtype)
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


        
class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 num_layers: int,
                 theta: float,
                 device=None,
                 dtype=None):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device=device)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch_size, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        token_positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)  # (batch, seq_len)

        # Token embeddings
        x = self.token_embeddings(input_ids)  # (batch, seq_len, d_model)

        # Pass through transformer blocks with RoPE
        for block in self.layers:
            x = block(x, token_positions)

        # Final normalization and output projection
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        return logits