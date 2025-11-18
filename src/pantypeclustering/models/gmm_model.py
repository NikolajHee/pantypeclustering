from typing import Any, Mapping, Union

import torch
from loguru import logger
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.distributions.categorical import Categorical

from pantypeclustering.components.distributions import ReparameterizedDiagonalGaussian
from pantypeclustering.components.priors import BasePrior
from pantypeclustering.models.basemodel import BaseModel, BaseVAE
from pantypeclustering.models.utils import reduce


class ModelVAEGMM(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        prior: BasePrior,
        input_shape: tuple[int, int, int],  # (C,W,H)
        latent_features: int,
        batch_size: int,
        k: int,
        clamp: bool = False,
    ) -> None:
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            input_shape=input_shape,
            latent_features=latent_features,
            prior=prior,
            batch_size=batch_size,
        )
        self.clamp = clamp
        self.k = k
        if self.logger:
            self.logger.log_hyperparams(
                {
                    "batch_size": batch_size,
                    "latent_features": latent_features,
                },
            )

    def pz2(self) -> Distribution:
        """N(0,I)"""
        mu, log_sigma = torch.zeros(self.latent_features), torch.zeros(self.latent_features)

        return ReparameterizedDiagonalGaussian(mu=mu, log_sigma=log_sigma)

    def py(self) -> Distribution:
        """Mult(K)"""
        probs = torch.ones(self.k) / self.k

        return Categorical(probs=probs)

    def pz_y_z2(self, y: Tensor, z2: Tensor) -> Distribution:
        """GMM"""
        pass

    def p_x_z1(self, z: Tensor) -> Distribution:
        """p(x|z) = N(x | mu(z), sigma(z))"""

        px_logits = self.decoder(z)
        mu, log_sigma = px_logits.chunk(2, dim=1)

        # Apply sigmoid to mu to constrain to [0, 1] for MNIST pixel values
        mu = torch.sigmoid(mu)

        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def forward(self, x: Tensor) -> tuple[Distribution, Distribution, Tensor]:
        qz = self.posterior(x)

        z = qz.rsample()

        px = self.observation_model(z)

        return (px, qz, z)


class VariationalAutoencoderGMM(BaseVAE):
    def __init__(
        self,
        model: BaseModel,
        input_shape: tuple[int, int, int],
        beta: float,
        learning_rate: float,
        val_num_images: int = 5,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_shape = input_shape
        self.beta = beta
        self.val_num_images = val_num_images
        self.learning_rate = learning_rate
        if self.logger:
            self.logger.log_hyperparams(
                {
                    "beta": self.beta,
                    "learning_rate": self.learning_rate,
                },
            )

    def training_step(
        self,
        train_batch: Tensor,
        batch_idx: int,
    ) -> Union[Tensor, Mapping[str, Any]]:
        # forward pass through the model
        x, _ = train_batch

        # unpack outputs
        px, qz, z = self.model.forward(x)

        log_px = reduce(px.log_prob(x))
        log_pz = reduce(self.model.prior.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        kl = log_qz - log_pz

        beta_elbo = (log_px) - (self.beta * kl)

        # loss
        loss = -beta_elbo.mean()

        # Check for NaN/Inf and replace with a large but finite value
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"{loss} detected.")
            loss = torch.tensor(1e6, device=loss.device, dtype=loss.dtype, requires_grad=True)

        self.log("train_loss", loss, prog_bar=True)
        if batch_idx == 0:
            reconstructed_images = px.sample()
            reconstructed_images = reconstructed_images.view(-1, *self.input_shape)

            original_images = x.view(-1, *self.input_shape)

            train_images = torch.vstack((original_images, reconstructed_images))

            self.logger.experiment.add_images(  # type: ignore
                "train_images",
                train_images,
                self.global_step,
            )

        return loss
