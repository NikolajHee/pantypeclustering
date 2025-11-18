from abc import abstractmethod
from typing import Any

import lightning
from torch import Tensor, nn

from pantypeclustering.components.priors import BasePrior


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
        self.batch_size = batch_size

    @abstractmethod
    def posterior(self, x: Tensor) -> Any: ...

    @abstractmethod
    def observation_model(self, z: Tensor) -> Any: ...

    @abstractmethod
    def forward(self, x: Tensor) -> Any: ...


class BaseVAE(lightning.LightningModule): ...
