from abc import abstractmethod

import lightning
import torch
import torch.nn.functional as f
from pantypeclustering.models.distributions import ReparameterizedDiagonalGaussian
from torch import Tensor, nn


def log_normal_diag(z: Tensor, means: Tensor, logvars: Tensor) -> Tensor:
    """
    Compute log probability of diagonal Gaussian distribution.

    Args:
        z: Input tensor of shape (batch_size, latent_dim)
        means: Mean tensor of shape (num_clusters, batch_size, latent_dim)
        logvars: Log variance tensor of shape (num_clusters, batch_size, latent_dim)

    Returns:
        Log probabilities of shape (num_clusters, batch_size)
    """
    diff = z - means
    squared_diff = diff.pow(2)

    log_prob_per_dim = -0.5 * (
        torch.log(torch.tensor(2 * torch.pi)) + logvars + squared_diff * torch.exp(-logvars)
    )

    log_prob = log_prob_per_dim.sum(dim=0)  # Shape: (batch_size, latent_dimz)

    return log_prob


class BasePrior(lightning.LightningModule):
    @abstractmethod
    def rsample(self, batch_size: int) -> Tensor: ...

    @abstractmethod
    def log_prob(self, z: Tensor) -> Tensor: ...


class MixtureOfGaussian(BasePrior):
    def __init__(self, latent_dim: int, num_clusters: int, batch_size: int):
        super().__init__()

        self.L = latent_dim
        self.K = num_clusters
        self.batch_size = batch_size

        self.means = nn.Parameter(torch.randn(num_clusters, latent_dim)).to("mps:0")
        self.logvars = nn.Parameter(torch.zeros(num_clusters, 1)).to("mps:0")

        self.w = nn.Parameter(torch.zeros(num_clusters, 1, 1)).to("mps:0")

    def rsample(self, batch_size: int) -> Tensor:
        w = f.softmax(self.w, dim=0)
        w = w.squeeze()

        indexes = torch.multinomial(w, batch_size, replacement=True)

        eps = torch.randn(batch_size, self.L)

        z = torch.cat(
            [
                self.means[[indx]] + eps[[indx]] * torch.exp(self.logvars[[indx]])
                for indx in indexes
            ],
            dim=0,
        )
        return z

    def log_prob(self, z: Tensor) -> Tensor:
        w = f.softmax(self.w, dim=0)

        z = z.unsqueeze(0)
        means = self.means.unsqueeze(1)
        logvars = self.logvars.unsqueeze(1)

        log_p = log_normal_diag(z, means, logvars) + torch.log(w)
        log_prob = torch.logsumexp(log_p, dim=0, keepdim=False)

        return log_prob


class UnivariateGaussian(BasePrior):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()

        self.dist = ReparameterizedDiagonalGaussian(
            mu=torch.zeros(latent_dim).to("mps:0"),
            log_sigma=torch.zeros(latent_dim).to("mps:0"),
        )

    def rsample(self, batch_size: int) -> Tensor:
        return self.dist.rsample()

    def log_prob(self, z: Tensor) -> Tensor:
        return self.dist.log_prob(z)
