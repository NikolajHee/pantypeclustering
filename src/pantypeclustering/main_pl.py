"""GMVAE training entrypoint for MNIST clustering."""
import time

import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner.tuning import Tuner

from pantypeclustering.config import get_training_parameters
from pantypeclustering.dataloader import get_mnist_dataloaders
from pantypeclustering.model_pl import GMVAE

curr_time = time.strftime("%H:%M:%S", time.localtime())
curr_date = time.strftime("%Y:%m:%d", time.localtime())

def main(seed: int | None = None) -> None:
    """Train the VAE model on MNIST dataset."""
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
        name=f"{curr_date}_{curr_time}",
        #log_model="all",
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
        N = len(test_loader.dataset),
        seed=seed,
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
    torch.save(model.state_dict(), "model.pth")



if __name__ == "__main__":
    main(seed=67)
