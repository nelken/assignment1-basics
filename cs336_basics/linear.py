from torch import nn
import torch
from torch.nn import init
import math



class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):        
        #in_features: int final dimension of the input
        #out_features: int final dimension of the output
        #device: torch.device | None = None Device to store the parameters on
        #dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight as learnable parameters
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # Initialize parameters using a uniform distribution like PyTorch nn.Linear
        self.reset_parameters()


    def reset_parameters(self):
        # Truncated normal initialization with std = sqrt(2 / (in + out))
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch..., in_features]
        # weight: [out_features, in_features]
        # result: [batch..., out_features]
        return torch.einsum('...si,oi->...so', x, self.weight)
        #return x @ self.weight.T

        