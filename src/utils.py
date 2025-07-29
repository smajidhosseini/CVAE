# src/utils.py

import torch                        # PyTorch core
import numpy as np                  # NumPy for arrays
import random                       # Python RNG

def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    torch.manual_seed(seed)         # CPU RNG
    if torch.cuda.is_available():  # If using GPU
        torch.cuda.manual_seed_all(seed)  # GPU RNG
    np.random.seed(seed)            # NumPy RNG
    random.seed(seed)               # Python RNG

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick: sample z ~ N(mu, var)
    Args:
        mu: tensor of means
        logvar: tensor of log-variances
    Returns:
        z: sampled latent
    """
    std = torch.exp(0.5 * logvar)   # compute standard deviation
    eps = torch.randn_like(std)     # sample epsilon
    return mu + eps * std           # return mu + eps * sigma

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(N(mu,var) || N(0,1)) per sample, averaged.
    Args:
        mu: tensor of means
        logvar: tensor of log-variances
    Returns:
        mean KL divergence
    """
    # formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def get_beta(epoch: int, cycle: int = 10, max_beta: float = 0.01) -> float:
    """
    Cyclic annealing schedule for beta.
    Args:
        epoch: current epoch (1-based)
        cycle: number of epochs per cycle
        max_beta: maximum beta value
    Returns:
        beta for this epoch
    """
    pos = (epoch - 1) % cycle + 1    # position in current cycle
    return max_beta * min(1.0, pos / (cycle / 2))  # linear ramp
