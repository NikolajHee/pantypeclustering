from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class TrainParameters(BaseSettings):
    version: str = Field(
        default="Development",
        description="Name used in naming of experiments",
    )
    latent_dim: int = Field(
        default=16,
        description="Dimension of the latent space",
    )
    beta: float = Field(
        default=1.0,  # Changed from 1e-1 to 1.0 for standard VAE (beta-VAE uses lower values)
        description="Weight of KL-divergence term",
    )
    num_workers: int = Field(
        default=9,
        description="Number of workers in dataloader.",
    )

    batch_size: int = 32
    max_epochs: int = 20
    input_shape: tuple[int, int, int] = (1, 28, 28)

    learning_rate: float = 1e-4
    num_of_clusters: int = 10

    num_train_batches: int = 3750
    num_test_batches: int = 1000

    gradient_clipping_value: float = 1.0


@lru_cache(maxsize=1)
def get_training_parameters() -> TrainParameters:
    return TrainParameters()
