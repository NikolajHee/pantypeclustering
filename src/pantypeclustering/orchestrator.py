from typing import Any

from pantypeclustering.components.priors import BasePrior, MixtureOfGaussian, UnivariateGaussian
from pantypeclustering.config import TrainParameters
from pantypeclustering.models import (
    BaseModel,
    BaseVAE,
    ModelVAEDL,
    ModelVAEGMM,
    ModelVAEKingma,
    VariationalAutoencoderDL,
    VariationalAutoencoderGMM,
    VariationalAutoencoderKingma,
)
from pantypeclustering.models.architectures import (
    BaseDecoder,
    BaseEncoder,
    CNNDecoder,
    CNNEncoder,
    FFDecoderMNIST,
    FFEncoderMNIST,
)


class Orchestrator:
    def __init__(self, cfg: TrainParameters):
        self.encoder, self.decoder = self._choose_architectures(
            cfg.chosen_architecture,
            **cfg.model_dump(),
        )

        self.prior = self._choose_prior(cfg.chosen_prior, **cfg.model_dump())

        self.model, self.vae = self._choose_model(
            cfg.chosen_framework,
            encoder=self.encoder,
            decoder=self.decoder,
            prior=self.prior,
            **cfg.model_dump(),
        )

    def _choose_model(
        self,
        model_name: str,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        prior: BasePrior,
        **kwargs: Any,
    ) -> tuple[BaseModel, BaseVAE]:
        match model_name:
            case "kingma_model":
                args = {
                    k: kwargs[k]
                    for k in [
                        "input_shape",
                        "latent_dim",
                        "batch_size",
                        "clamp",
                        "learning_rate",
                        "val_num_images",
                    ]
                    if k in kwargs
                }

                model = ModelVAEKingma(
                    encoder=encoder,
                    decoder=decoder,
                    prior=prior,
                    input_shape=args["input_shape"],
                    latent_features=args["latent_dim"],
                    batch_size=args["batch_size"],
                    clamp=args["clamp"],
                )

                vae = VariationalAutoencoderKingma(
                    model=model,
                    input_shape=args["input_shape"],
                    learning_rate=args["learning_rate"],
                    val_num_images=args["val_num_images"],
                )

                return (model, vae)
            case "dl_model":
                args = {
                    k: kwargs[k]
                    for k in [
                        "input_shape",
                        "latent_dim",
                        "batch_size",
                        "clamp",
                        "beta",
                        "learning_rate",
                        "val_num_images",
                    ]
                    if k in kwargs
                }

                model = ModelVAEDL(
                    encoder=encoder,
                    decoder=decoder,
                    prior=prior,
                    input_shape=args["input_shape"],
                    latent_features=args["latent_dim"],
                    batch_size=args["batch_size"],
                    clamp=args["clamp"],
                )

                vae = VariationalAutoencoderDL(
                    model=model,
                    beta=args["beta"],
                    input_shape=args["input_shape"],
                    learning_rate=args["learning_rate"],
                    val_num_images=args["val_num_images"],
                )

                return (model, vae)

            case "gmm_model":
                args = {
                    k: kwargs[k]
                    for k in [
                        "input_shape",
                        "latent_dim",
                        "batch_size",
                        "clamp",
                        "learning_rate",
                        "val_num_images",
                        "k",
                        "beta",
                    ]
                    if k in kwargs
                }

                model = ModelVAEGMM(
                    encoder=encoder,
                    decoder=decoder,
                    prior=prior,
                    input_shape=args["input_shape"],
                    latent_features=args["latent_dim"],
                    batch_size=args["batch_size"],
                    clamp=args["clamp"],
                    k=args["k"],
                )

                vae = VariationalAutoencoderGMM(
                    model=model,
                    beta=args["beta"],
                    input_shape=args["input_shape"],
                    learning_rate=args["learning_rate"],
                    val_num_images=args["val_num_images"],
                )

                return (model, vae)

            case _:
                raise ValueError(f"{model_name} is not a valid model.")

    def _choose_prior(self, prior_name: str, **kwargs: Any) -> BasePrior:
        match prior_name:
            case "MixtureOfGaussians":
                args = {
                    k: kwargs[k]
                    for k in [
                        "latent_dim",
                        "num_clusters",
                        "batch_size",
                    ]
                    if k in kwargs
                }
                return MixtureOfGaussian(**args)
            case "UnivariateGaussian":
                args = {k: kwargs[k] for k in ["latent_dim"] if k in kwargs}
                return UnivariateGaussian(**args)
            case _:
                raise ValueError(f"{prior_name} is not a valid prior.")

    def _choose_architectures(
        self,
        architecture_name: str,
        **kwargs: Any,
    ) -> tuple[BaseEncoder, BaseDecoder]:
        match architecture_name:
            case "CNN":
                args = {k: kwargs[k] for k in ["latent_dim"] if k in kwargs}

                return (CNNEncoder(**args), CNNDecoder(**args))
            case "FFN":
                args = {
                    k: kwargs[k] for k in ["latent_dim", "input_size", "hidden_dim"] if k in kwargs
                }

                return (FFEncoderMNIST(**args), FFDecoderMNIST(**args))
            case _:
                raise ValueError(f"{architecture_name} is not a valid architecture.")
