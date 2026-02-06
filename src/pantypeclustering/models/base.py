import math
from typing import Any, Mapping, Union

import matplotlib.pyplot as plt
import torch
from loguru import logger
from numpy.typing import NDArray
from sklearn.manifold import TSNE
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor, nn
from torch.optim import Optimizer

pi = torch.tensor([math.pi]).to("mps")

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
        self.log_sigma = log_sigma
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


class CNNEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=2 * latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encoder(x)

        h = self.fc1(z)

        mean, log_var = h.chunk(2, dim=1)

        return mean, log_var


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.to_decoder = nn.Linear(in_features=latent_dim, out_features=64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

        # lineært lag
        self.final_layer = nn.Linear(28*28*1, 28*28*1*2)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        x = self.to_decoder(z)

        x = x.reshape(-1, 64, 7, 7)

        x = self.decoder(x)

        x = x.reshape(-1, 28 * 28)

        x = self.final_layer(x)

        mean, log_var = x.chunk(2, dim=1)

        mean = mean.reshape(-1, 1, 28, 28)
        log_var = log_var.reshape(-1, 1, 28, 28)

        return (mean, log_var)






class ModelVAEKingma(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],  # (C,W,H)
        latent_features: int,
        batch_size: int,
        clamp: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = CNNEncoder(latent_features)
        self.decoder = CNNDecoder(latent_features)

        self.prior =  ReparameterizedDiagonalGaussian(
            mu=torch.zeros(latent_features).to("mps:0"),
            log_sigma=torch.zeros(latent_features).to("mps:0"),
        )

        self.clamp = clamp

    def posterior(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """q(z|x)"""

        mu, log_sigma = self.encoder(x)

        if self.clamp:
            log_sigma = torch.clamp(log_sigma, min=-10, max=2)

        return mu, log_sigma

    @staticmethod
    def reparameterize(mean: Tensor, log_var: Tensor):
        mu, sigma = mean, torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        return z

    def observation_model(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """p(x|z) = N(x | mu(z), sigma(z))"""

        mu, log_sigma = self.decoder(z)

        # Constrain log_sigma to reasonable range to prevent extreme values
        if self.clamp:
            log_sigma = torch.clamp(log_sigma, min=-10, max=0)

        return mu, log_sigma

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        mu, log_sigma = self.posterior(x)

        z = self.reparameterize(mu, log_sigma)

        out_mu, out_log_sigma = self.observation_model(z)

        return {
            "mean": mu,
            "log_sigma": log_sigma,
            "z": z,
            "out_mean": out_mu,
            "out_log_sigma": out_log_sigma,
        }


class VariationalAutoencoderKingma(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_theta(x | z) = B(x | g_theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_phi(z|x) = N(z | mu(x), sigma(x))`
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        learning_rate: float,
        val_num_images: int = 5,
    ) -> None:
        super().__init__()
        self.model = ModelVAEKingma(
            input_shape=input_shape,
            latent_features=64,
            batch_size=128,
            clamp=True,
        )
        self.input_shape = input_shape
        self.val_num_images = val_num_images
        self.learning_rate = learning_rate
        self.reset_save()

    def compute_elbo(
        self,
        x: Tensor,
        out_mean: Tensor,
        out_logvar: Tensor,
        mean: Tensor,
        log_var: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        reconst = -0.5 * torch.sum(
            torch.log(2 * pi) + out_logvar + (x - out_mean).pow(2) / out_logvar.exp(),
        )
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        elbo = (reconst - kl) / len(x)

        return -elbo, reconst, kl

    def forward(
        self,
        train_batch: Tensor,
    ) -> Tensor:
        # forward pass through the model
        x, _ = train_batch

        # unpack outputs
        data = self.model.forward(x)

        mean = data["mean"]
        log_sigma = data["log_sigma"]
        out_mean = data["out_mean"]
        out_log_sigma = data["out_log_sigma"]

        loss, reconst, kl = self.compute_elbo(
            x=x,
            mean=mean,
            log_var=log_sigma,
            out_mean=out_mean,
            out_logvar=out_log_sigma,
        )

        # Check for NaN/Inf and replace with a large but finite value
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"{loss} detected.")
            loss = torch.tensor(1e6, device=loss.device, dtype=loss.dtype, requires_grad=True)

        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        return optimizer



    def on_validation_epoch_end(self) -> None:
        if len(self.val_latents) > 30:
            latent_points = torch.cat(self.val_latents, dim=0)
            latent_points_np = latent_points.numpy()
            labels = torch.cat(self.val_labels, dim=0)
            labels_np = labels.numpy()

            metrics = {}

            # labelled stuff
            self._tsne(latent_points, labels)

            db_sc_labelled = davies_bouldin_score(
                X=latent_points_np,
                labels=labels_np,
            )
            silhouette_sc_labelled = float(
                silhouette_score(
                    X=latent_points_np,
                    labels=labels_np,
                ),
            )

            metrics["davies_bouldin_score_labelled"] = db_sc_labelled
            metrics["silhouette_sc_labelled"] = silhouette_sc_labelled

            self.logger.log_metrics(metrics=metrics, step=self.current_epoch)

    def _tsne(self, latent_points: Tensor, labels: Tensor):
        def fit_tsne(_latents_np: NDArray[Any]):
            return TSNE(n_components=2, random_state=42).fit_transform(_latents_np)  # pyright: ignore[reportUnknownArgumentType]

        latents_np = latent_points.numpy()
        labels_np = labels.numpy()

        tsne_result = fit_tsne(latents_np)

        fig = tsne_plot(tsne_result, labels_np)

        self.logger.experiment.log_figure(
            self.logger.run_id,
            fig,
            "tsne/tsne.png",
        )
        plt.close("all")
