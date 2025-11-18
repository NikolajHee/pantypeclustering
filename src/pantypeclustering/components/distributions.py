import torch
from torch.distributions import Distribution


class ReparameterizedDiagonalGaussian(Distribution):
    """
    CHANGE TO TORCH

    A distribution `N(y | mu, sigma I)` compatible with
    the reparameterization trick given `epsilon ~ N(0, 1)`.
    """

    def __init__(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        assert mu.shape == log_sigma.shape
        f"torch.Tensors `mu` : {mu.shape} and"
        f"`log_sigma` : {log_sigma.shape} must be of the same shape"

        self.mu = mu
        # Add epsilon for numerical stability to prevent sigma from being too small
        self.sigma = log_sigma.exp().clamp(min=1e-6)

    def sample_epsilon(self) -> torch.Tensor:
        """`eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()

    def sample(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> torch.Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick)"""
        return self.mu + self.sigma * self.sample_epsilon()

    def log_prob(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """return the log probability: log `p(z)`"""

        return torch.distributions.Normal(self.mu, self.sigma).log_prob(z)
