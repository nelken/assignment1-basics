import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss given logits and integer targets.
    
    logits: Tensor of shape (..., vocab_size)
    targets: Tensor of shape (...) with integer indices of true class

    Returns: scalar tensor representing average cross-entropy loss
    """
    # Subtract max for numerical stability (broadcast over vocab dim)
    logits = logits - logits.amax(dim=-1, keepdim=True)

    # Compute log-softmax efficiently (avoid separate softmax + log)
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    # Gather the log-probability of the true class at each position
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Return average negative log likelihood
    return nll.mean()