import os
import time

import lightning
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from pantypeclustering.config import get_training_parameters
from pantypeclustering.models.architectures import CNNDecoder, CNNEncoder
from pantypeclustering.models.model import ModelVAE, VariationalAutoencoder
from pantypeclustering.models.priors import MixtureOfGaussian

curr_time = time.strftime("%H:%M:%S", time.localtime())
curr_date = time.strftime("%Y:%m:%d", time.localtime())


def main() -> None:
    cfg = get_training_parameters()

    train_dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    test_dataset = MNIST(os.getcwd(), download=True, transform=ToTensor(), train=False)

    train_dataloader = DataLoader(  # type: ignore
        train_dataset,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=True,
    )

    test_dataloader = DataLoader(  # type: ignore
        test_dataset,
        num_workers=cfg.num_workers,
        persistent_workers=True,
    )

    logger.info(f"len of train: {len(train_dataset)}")
    logger.info(f"len of test: {len(test_dataset)}")

    prior = MixtureOfGaussian(
        latent_dim=cfg.latent_dim,
        num_clusters=cfg.num_of_clusters,
        batch_size=cfg.batch_size,
    )
    # prior = UnivariateGaussian(cfg.latent_dim)

    model_vae = ModelVAE(
        CNNEncoder(cfg.latent_dim),
        CNNDecoder(cfg.latent_dim),
        prior=prior,
        input_shape=cfg.input_shape,
        latent_features=cfg.latent_dim,
        batch_size=cfg.batch_size,
    )

    vae = VariationalAutoencoder(
        model_vae,
        input_shape=cfg.input_shape,
        beta=cfg.beta,
        learning_rate=cfg.learning_rate,
    )

    _logger = TensorBoardLogger("", version=f"{cfg.version}_{curr_time}_{curr_date}")

    trainer = lightning.Trainer(
        limit_train_batches=cfg.num_train_batches,
        limit_val_batches=cfg.num_test_batches,
        max_epochs=cfg.max_epochs,
        logger=_logger,
        gradient_clip_val=cfg.gradient_clipping_value,
    )
    trainer.fit(
        model=vae,
        train_dataloaders=train_dataloader,  # pyright: ignore[reportUnknownArgumentType]
        val_dataloaders=test_dataloader,  # pyright: ignore[reportUnknownArgumentType]
    )


if __name__ == "__main__":
    main()
