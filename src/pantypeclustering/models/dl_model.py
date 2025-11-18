import math
from typing import Any, Mapping, Union

import matplotlib.pyplot as plt
import torch
from loguru import logger
from numpy.typing import NDArray
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor, nn
from torch.distributions import Distribution

from pantypeclustering.components.distributions import ReparameterizedDiagonalGaussian
from pantypeclustering.components.priors import BasePrior, MixtureOfGaussian
from pantypeclustering.models.basemodel import BaseModel, BaseVAE
from pantypeclustering.models.utils import reduce
from pantypeclustering.utils import tsne_plot

pi = torch.tensor([math.pi]).to("mps")


class ModelVAEDL(BaseModel):
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
        if self.logger:
            self.logger.log_hyperparams(
                {
                    "batch_size": batch_size,
                    "latent_features": latent_features,
                },
            )

    def posterior(self, x: Tensor) -> Distribution:
        """q(z|x)"""

        mu, log_sigma = self.encoder(x)

        # Constrain log_sigma to prevent numerical instability
        # Clamp to reasonable range: exp(-10) ≈ 4.5e-5, exp(2) ≈ 7.4

        if self.clamp:
            log_sigma = torch.clamp(log_sigma, min=-10, max=2)

        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def observation_model(self, z: Tensor) -> Distribution:
        """p(x|z) = N(x | mu(z), sigma(z))"""

        mu, log_sigma = self.decoder(z)

        # Constrain log_sigma to reasonable range to prevent extreme values
        if self.clamp:
            log_sigma = torch.clamp(log_sigma, min=-10, max=0)

        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def forward(self, x: Tensor) -> tuple[Distribution, Distribution, Tensor]:
        """
        determine distribution of z | x by using q(z|x),
        sample z from q(z|x)
        and return the distribution p(x|z) (decoder)
        """

        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)

        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()

        # define the observation model p(x|z) = N
        px = self.observation_model(z)

        return (px, qz, z)


class VariationalAutoencoderDL(BaseVAE):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_theta(x | z) = B(x | g_theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_phi(z|x) = N(z | mu(x), sigma(x))`
    """

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

        self.reset_save()

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

        loss = -beta_elbo.mean()

        # Check for NaN/Inf and replace with a large but finite value
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"{loss} detected.")
            loss = torch.tensor(1e6, device=loss.device, dtype=loss.dtype, requires_grad=True)

        self.log("train_loss", loss, prog_bar=True)
        if (batch_idx == 0) and (self.current_epoch % 10 == 0):
            reconstructed_images = px.sample()
            reconstructed_images = reconstructed_images.view(-1, *self.input_shape)

            original_images = x.view(-1, *self.input_shape)

            for i, (org_img, recon_img) in enumerate(
                zip(
                    original_images,
                    reconstructed_images,
                    strict=False,
                ),
            ):
                fig, ax = plt.subplots(1, 2)

                ax[0].imshow(org_img.detach().cpu().squeeze(), cmap="gray")
                ax[1].imshow(recon_img.detach().cpu().squeeze(), cmap="gray")
                ax[0].axis("off")
                ax[1].axis("off")

                self.logger.experiment.log_figure(
                    self.logger.run_id,
                    fig,
                    f"training/train_img_e{self.current_epoch}_n{i}.png",
                )
                plt.close("all")

                if i >= self.val_num_images:
                    break

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        x, _ = batch

        px, qz, z = self.model.forward(x)

        log_px = reduce(px.log_prob(x))
        log_pz = reduce(self.model.prior.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        kl = log_qz - log_pz

        beta_elbo = (log_px) - (self.beta * kl)

        loss = -beta_elbo.mean()

        self.log("test_loss", loss, prog_bar=True)

        # Store latents and labels for TSNE visualization
        self.val_labels.append(batch[1].cpu())
        self.val_latents.append(z.detach().cpu())

        if (batch_idx <= self.val_num_images) and (self.current_epoch % 10 == 0):
            reconstructed_images = px.sample()
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

        if type(self.model.prior) is MixtureOfGaussian:
            prototypes = self.model.prior.means
            means: list[Tensor] = []
            for i in range(prototypes.size(0)):
                px = self.model.observation_model(prototypes[i])
                means_image = px.sample()
                means.append(means_image)

            test_images = torch.vstack(means)
            self.logger.experiment.add_images(  # type: ignore
                "GaussianMixtureMeans",
                test_images,
                self.global_step,
            )

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

            # UNLABELLED STUFF
            if self._calculate_unlabelled_stuff and self._clustering:
                assignments = self._determine_assignments(latent_points)

                db_sc_unlabelled = davies_bouldin_score(
                    X=latent_points_np,
                    labels=assignments,
                )

                silhouette_sc_unlabelled = float(
                    silhouette_score(
                        X=latent_points_np,
                        labels=assignments,
                    ),
                )

                metrics["davies_bouldin_score_unlabelled"] = db_sc_unlabelled
                metrics["silhouette_sc_unlabelled"] = silhouette_sc_unlabelled

                if self._calculate_labelled_stuff:
                    rand_score = adjusted_rand_score(labels_np, assignments)

                    metrics["rand_score"] = rand_score

            # LABELLED STUFF
            if self._calculate_labelled_stuff:
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

        if type(self.model.prior) is MixtureOfGaussian:
            prototypes = self.prototypes

            latents_np = torch.vstack(
                (
                    latent_points,
                    prototypes,
                ),
            )

        latents_np = latent_points.numpy()
        labels_np = labels.numpy()

        tsne_result = fit_tsne(latents_np)

        tsne_means = None
        if type(self.model.prior) is MixtureOfGaussian:
            tsne_result, tsne_means = tsne_result[:1000], tsne_result[1000:]

        fig = tsne_plot(tsne_result, labels_np, means=tsne_means)

        self.logger.experiment.log_figure(
            self.logger.run_id,
            fig,
            "tsne/tsne.png",
        )
        plt.close("all")

    def _determine_assignments(self, latent_points: Tensor) -> Tensor:
        def _similarity_score(latent_points: Tensor, cluster_centers: Tensor):
            return torch.cdist(latent_points, cluster_centers)

        similarities = _similarity_score(latent_points, self.prototypes)

        return torch.argmin(similarities, dim=1)

    @property
    def prototypes(self) -> Tensor:
        return self.model.prior.means.cpu().detach()  # type: ignore
