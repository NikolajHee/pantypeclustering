import math
from typing import Any, Mapping, Sequence, Union

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.utilities.types import LRSchedulerConfig
from loguru import logger
from numpy.typing import NDArray
from sklearn.manifold import TSNE
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor, nn
from torch.optim import Optimizer, lr_scheduler

from pantypeclustering.components.priors import BasePrior
from pantypeclustering.models.basemodel import BaseModel, BaseVAE
from pantypeclustering.utils import tsne_plot

pi = torch.tensor([math.pi]).to("mps")


class ModelVAEKingma(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        prior: BasePrior,
        input_shape: tuple[int, int, int],  # (C,W,H)
        latent_features: int,
        batch_size: int,
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


class VariationalAutoencoderKingma(BaseVAE):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_theta(x | z) = B(x | g_theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_phi(z|x) = N(z | mu(x), sigma(x))`
    """

    def __init__(
        self,
        model: BaseModel,
        input_shape: tuple[int, int, int],
        learning_rate: float,
        val_num_images: int = 5,
    ) -> None:
        super().__init__()
        self.model = model
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
    ) -> Tensor:
        reconst = -0.5 * torch.sum(
            torch.log(2 * pi) + out_logvar + (x - out_mean).pow(2) / out_logvar.exp(),
        )
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        elbo = (reconst - kl) / len(x)

        return -elbo

    def training_step(
        self,
        train_batch: Tensor,
        batch_idx: int,
    ) -> Union[Tensor, Mapping[str, Any]]:
        # forward pass through the model
        x, _ = train_batch

        # unpack outputs
        data = self.model.forward(x)

        mean = data["mean"]
        log_sigma = data["log_sigma"]
        out_mean = data["out_mean"]
        out_log_sigma = data["out_log_sigma"]

        loss = -self.compute_elbo(
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

        self.log("train_loss", loss, prog_bar=True)
        if (batch_idx == 0) and (self.current_epoch % 10 == 0):
            reconstructed_images = out_mean
            reconstructed_images = reconstructed_images.view(-1, *self.input_shape)

            original_images = x.view(-1, *self.input_shape)

            for i, (org_img, recon_img) in enumerate(
                zip(
                    original_images,
                    reconstructed_images,
                    strict=False,
                ),
            ):
                fig, ax = plt.subplots(1, 2)  # pyright: ignore[reportUnknownMemberType]

                ax[0].imshow(org_img.detach().cpu().squeeze(), cmap="gray")
                ax[1].imshow(recon_img.detach().cpu().squeeze(), cmap="gray")
                ax[0].axis("off")
                ax[1].axis("off")

                self.logger.experiment.log_figure(
                    self.logger.run_id,  # type: ignore
                    fig,
                    f"training/train_img_e{self.current_epoch}_n{i}.png",
                )
                plt.close("all")

                if i >= self.val_num_images:
                    break

        return loss

    def configure_optimizers(self) -> tuple[Sequence[Optimizer], Sequence[LRSchedulerConfig]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler.ReduceLROnPlateau(optimizer),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "test_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }
        return [optimizer], [LRSchedulerConfig(**lr_scheduler_config)]

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        x, _ = batch

        data = self.model.forward(x)

        mean = data["mean"]
        log_sigma = data["log_sigma"]
        z = data["z"]
        out_mean = data["out_mean"]
        out_log_sigma = data["out_log_sigma"]

        loss = -self.compute_elbo(
            x=x,
            mean=mean,
            log_var=log_sigma,
            out_mean=out_mean,
            out_logvar=out_log_sigma,
        )

        self.log("test_loss", loss, prog_bar=True)

        # Store latents and labels for TSNE visualization
        self.val_labels.append(batch[1].cpu())
        self.val_latents.append(z.detach().cpu())

        if (batch_idx <= self.val_num_images) and (self.current_epoch % 10 == 0):
            reconstructed_images = out_mean
            reconstructed_images = reconstructed_images.view(-1, *self.input_shape)

            original_images = batch[0]
            original_images = original_images.view(-1, *self.input_shape)

            for i, (org_img, recon_img) in enumerate(
                zip(
                    original_images,
                    reconstructed_images,
                    strict=False,
                ),
            ):
                fig, ax = plt.subplots(1, 2)

                ax[0].imshow(org_img.cpu().squeeze(), cmap="gray")
                ax[1].imshow(recon_img.cpu().squeeze(), cmap="gray")
                ax[0].axis("off")
                ax[1].axis("off")

                self.logger.experiment.log_figure(
                    self.logger.run_id,
                    fig,
                    f"validation/validation_img_e{self.current_epoch}_n{i}.png",
                )
                plt.close("all")

                if i > self.val_num_images:
                    break

    def reset_save(self) -> None:
        self.val_latents: list[Tensor] = []
        self.val_labels: list[Tensor] = []

    def on_validation_epoch_start(self) -> None:
        self.reset_save()

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

            self.logger.log_metrics(metrics=metrics)

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
