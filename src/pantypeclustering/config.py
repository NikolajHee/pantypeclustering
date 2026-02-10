"""Training configuration loaded from env / dotenv."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class TrainParameters(BaseSettings):
    """GMVAE training and model hyperparameters."""

    seed: int = 67

    batch_size: int = 50
    max_epochs: int = 100
    input_shape: tuple[int, int, int] = (1, 28, 28)
    learning_rate: float = 1e-4

    num_workers: int = Field(
        default=4,
        description="Number of workers in dataloader.",
    )

    # GMM
    x_size: int = 28*28
    continuous: bool = True
    hidden_size: int = 500
    z1_size: int = 200
    z2_size: int = 150
    number_of_mixtures: int = 10
    mc: int = 5
    lambda_threshold: float = 0.5


@lru_cache(maxsize=1)
def get_training_parameters() -> TrainParameters:
    """Return cached training parameters (read once from env)."""
    return TrainParameters()
