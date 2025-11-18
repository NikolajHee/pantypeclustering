import os
import time

import lightning
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from pantypeclustering.config import get_training_parameters
from pantypeclustering.orchestrator import Orchestrator

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
        batch_size=cfg.val_batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=True,
    )

    logger.info(f"len of train: {len(train_dataset)}")
    logger.info(f"len of test: {len(test_dataset)}")

    orchestrator = Orchestrator(cfg)

    mlf_logger = MLFlowLogger(
        experiment_name="lightning_logs",
        run_name=f"{curr_date}_{curr_time}",
        tracking_uri="http://127.0.0.1:5000",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    mlf_logger.log_hyperparams(cfg.model_dump())

    trainer = lightning.Trainer(
        limit_train_batches=cfg.num_train_batches,
        limit_val_batches=cfg.num_test_batches,
        max_epochs=cfg.max_epochs,
        logger=mlf_logger,
        callbacks=[lr_monitor],
    )
    trainer.fit(
        model=orchestrator.vae,
        train_dataloaders=train_dataloader,  # pyright: ignore[reportUnknownArgumentType]
        val_dataloaders=test_dataloader,  # pyright: ignore[reportUnknownArgumentType]
    )


if __name__ == "__main__":
    main()
