from abc import abstractmethod

import lightning
from pantypeclustering.models.priors import BasePrior
from torch import Tensor, nn
from torch.distributions import Distribution


class BaseModel(lightning.LightningModule):
    """Base VAE"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        prior: BasePrior,
        input_shape: tuple[int, int, int],  # (C,W,H)
        latent_features: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.prior = prior

    @abstractmethod
    def posterior(self, x: Tensor) -> Distribution: ...

    @abstractmethod
    def observation_model(self, z: Tensor) -> Distribution: ...

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Distribution, Distribution, Tensor]: ...
