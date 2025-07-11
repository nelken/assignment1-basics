import torch
import torch.nn.functional as F
import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    query: (batch_size, ..., seq_len_q, d_k)
    key:   (batch_size, ..., seq_len_k, d_k)
    value: (batch_size, ..., seq_len_k, d_v)
    mask:  optional (seq_len_q, seq_len_k) boolean mask (True = keep, False = mask out)
    
    Returns: output of shape (batch_size, ..., seq_len_q, d_v)
    """
    d_k = query.size(-1)

    # Compute scaled dot product
    scores = torch.matmul(query, key.transpose(-2, -1)) / d_k**0.5
    # scores: (batch_size, ..., seq_len_q, seq_len_k)

    if mask is not None:
        # Expand mask to broadcast over batch and head dims
        while mask.ndim < scores.ndim:
            mask = mask.unsqueeze(0)
        scores = scores.masked_fill(~mask, float('-inf'))

    # Compute attention weights
    attn = F.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attn, value)
    return output

class CausalMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=d_model // num_heads,
                max_seq_len=max_seq_len,
                device=device,
                dtype=dtype
            )
        else:
            self.rope = None

        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch_size, seq_len, d_model)
        """
        B, seq_len, _ = x.shape

        # Project inputs and split heads
        def project_and_reshape(proj, x):
            return proj(x).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        q = project_and_reshape(self.q_proj, x)  # (B, H, S, d_k)
        k = project_and_reshape(self.k_proj, x)
        v = project_and_reshape(self.v_proj, x)

        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Create causal mask of shape (S, S)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # Apply attention
        attn_output = scaled_dot_product_attention(q, k, v, mask=mask)

        # Combine heads: (B, H, S, d_k) → (B, S, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)

        # Final linear projection
        return self.output_proj(attn_output)