"""GMVAE training entrypoint for MNIST clustering."""

import time

import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
from pantypeclustering.config import get_training_parameters
from pantypeclustering.dataloader import get_mnist_dataloaders
from pantypeclustering.model_pl import GMVAE

curr_time = time.strftime("%H:%M:%S", time.localtime())
curr_date = time.strftime("%Y:%m:%d", time.localtime())


def main(seed: int | None = None) -> None:
    """Train the VAE model on MNIST dataset."""
    try:
        cfg = get_training_parameters()
        if seed is not None:
            torch.manual_seed(seed)
        else:
            torch.manual_seed(cfg.seed)

        torch.set_float32_matmul_precision(precision="medium")

        # mlf_logger = MLFlowLogger(
        #     experiment_name="lightning_logs",
        #     run_name=f"{curr_date}_{curr_time}",
        #     tracking_uri="http://127.0.0.1:5000",
        # )
        mlf_logger = WandbLogger(
            project="pantypeclustering",
            name=f"alpha_{curr_date}_{curr_time}_seed_{seed}",
            # log_model="all",
        )

        train_loader, test_loader = get_mnist_dataloaders(
            batch_size=cfg.batch_size,
            binarize=False,
            seed=0,
            num_workers=cfg.num_workers,
        )
        model = GMVAE(
            learning_rate=cfg.learning_rate,
            x_size=cfg.x_size,
            z1_size=cfg.z1_size,
            z2_size=cfg.z2_size,
            hidden_size=cfg.hidden_size,
            number_of_mixtures=cfg.number_of_mixtures,
            mc=cfg.mc,
            continuous=cfg.continuous,
            lambda_threshold=cfg.lambda_threshold,
            N=len(test_loader.dataset),
            seed=seed,
            alpha=80.0,
        )

        trainer = pl.Trainer(
            max_epochs=100,
            logger=mlf_logger,
            enable_progress_bar=False,
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=test_loader,
        )
        # save model
        torch.save(model.state_dict(), "alpha_newest_model.pth")
    finally:
        wandb.finish()


if __name__ == "__main__":
    for i in [1, 2, 3, 4, 5]:
        main(seed=i)
