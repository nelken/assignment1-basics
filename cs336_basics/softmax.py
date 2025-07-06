import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Subtract the max along the specified dimension for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp