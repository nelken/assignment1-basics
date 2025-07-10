import torch
import math
  
def xxxlearning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    """
    t: current step (int or torch.Tensor)
    alpha_max: float
    alpha_min: float
    T_w: int, number of warmup steps
    T_c: int, number of cosine decay steps
    """
    t = torch.as_tensor(t, dtype=torch.float32)

    # Warmup phase
    warmup = (t < T_w)
    lr_warmup = alpha_max * t / T_w

    # Cosine decay phase
    cosine = (t >= T_w) & (t <= T_w + T_c)
    #decay_step = (t - T_w).clamp(min=0, max=T_c)
    decay_step = (t - T_w).clamp(min=0) / (T_c - T_w)
    cosine_decay = 0.5 * (1 + torch.cos(math.pi * decay_step))
    lr_cosine = alpha_min + (alpha_max - alpha_min) * cosine_decay

    # Post-decay phase
    lr_constant = torch.full_like(t, alpha_min)

    # Combine
    return torch.where(warmup, lr_warmup,
           torch.where(cosine, lr_cosine, lr_constant))


def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    t = torch.as_tensor(t, dtype=torch.float64)

    # Warmup phase
    warmup = (t < T_w)
    lr_warmup = alpha_max * t / T_w

    # Cosine decay phase (corrected)
    cosine = (t >= T_w) & (t <= T_c)
    decay_step = (t - T_w)
    decay_range = T_c - T_w
    cosine_decay = 0.5 * (1 + torch.cos(math.pi * decay_step / decay_range))
    lr_cosine = alpha_min + (alpha_max - alpha_min) * cosine_decay

    # Post-decay phase
    lr_constant = torch.full_like(t, alpha_min)

    # Combine
    return torch.where(warmup, lr_warmup,
           torch.where(cosine, lr_cosine, lr_constant))