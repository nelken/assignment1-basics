from torch import nn
import torch
import math

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        # num_embeddings: int Size of the vocabulary
        # embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reset_parameters()

    def reset_parameters(self):
        # Truncated normal initialization with std = sqrt(2 / (in + out))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

        
    
    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # token_ids: shape (batch_size, seq_len)
        # returns: shape (batch_size, seq_len, d_model)
        return self.weight[token_ids]