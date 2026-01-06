from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class TrainParameters(BaseSettings):
    chosen_architecture: str = Field(
        default="gmm",
        description="Current chosen model.",
    )
    chosen_framework: str = Field(
        default="gmm_model",
        description="The chosen framework.",
    )

    chosen_prior: str = Field(
        default="gmm",
        description="Current chosen prior.",
    )

    version: str = Field(
        default="Development",
        description="Name used in naming of experiments",
    )
    seed: int = 67

    # MODEL RELATED PARAMS
    latent_dim: int = Field(
        default=8,
        description="Dimension of the latent space",
    )
    beta: float = Field(
        default=1.0,  # Changed from 1e-1 to 1.0 for standard VAE (beta-VAE uses lower values)
        description="Weight of KL-divergence term",
        deprecated=True,
    )

    batch_size: int = 50
    val_batch_size: int = 5
    max_epochs: int = 100
    input_shape: tuple[int, int, int] = (1, 28, 28)

    learning_rate: float = 1e-4

    # DATA RELATED PARAM
    num_train_batches: int = 3750
    num_test_batches: int = 1000

    num_workers: int = Field(
        default=9,
        description="Number of workers in dataloader.",
    )

    # GMM
    y_size: int = 28*28
    continuous: bool = True
    hidden_size: int = 500
    x_size: int = 200
    w_size: int = 150
    number_of_mixtures: int = 10
    mc: int = 5
    lambda_threshold: float = 0.5


@lru_cache(maxsize=1)
def get_training_parameters() -> TrainParameters:
    return TrainParameters()
