import multiprocessing as mp

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import wandb
from pantypeclustering.config import get_training_parameters
from pantypeclustering.dataloader import get_mnist_dataloaders
from pantypeclustering.model_pl import GMVAE


def objective(trial: optuna.Trial):
    try:
        mp.set_start_method("spawn", force=True)

        cfg = get_training_parameters()

        #* insert seed
        torch.manual_seed(cfg.seed)

        # ---- OPTUNA SEARCH SPACE ----
        #lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        #batch_size = trial.suggest_categorical("batch_size", [32, 64])
        alpha = trial.suggest_float("alpha", 0.0, 100.0)

        print(f"Trial {trial.number}: alpha={alpha}")


        # ---- W&B LOGGER (optional but recommended) ----
        wandb_logger = WandbLogger(
            project="final_results",
            name=f"sweep-{trial.number}-alpha-{alpha:.4f}",
            log_model=False,
        )

        wandb_logger.experiment.config.update(
            trial.params,
            allow_val_change=True,
        )

        # ---- MODEL & DATA ----
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
            seed=cfg.seed,
            alpha=alpha,
        )

        # ---- PRUNING CALLBACK ----

        pruning_callback = optuna.integration.PyTorchLightningPruningCallback(
            trial,
            monitor="val_loss",
        )

        trainer = pl.Trainer(
            max_epochs=50,
            logger=wandb_logger,
            callbacks=[
                pruning_callback,
                EarlyStopping(monitor="val_loss", patience=5),
            ],
            accelerator="gpu",
            devices=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )

        trainer.fit(model,
                    train_dataloaders=train_loader,
                    val_dataloaders=test_loader
        )

        return trainer.callback_metrics["val_loss"].item()
    finally:
        wandb.finish()



if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.NopPruner(),
        storage="sqlite:///optuna_study.db",
        study_name="alpha_experiment",
    )

    study.optimize(objective, n_trials=50, n_jobs=1)

    print("Best trial:")
    print(study.best_trial.params)
