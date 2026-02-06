from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class TrainParameters(BaseSettings):
    version: str = Field(
        default="Development1",
        description="Name used in naming of experiments",
    )
    seed: int | None = None
    results_dir: str = "./results/"

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
    x_size: int = 28 * 28
    continuous: bool = True
    hidden_size: int = 500
    z1_size: int = 200
    z2_size: int = 150
    number_of_mixtures: int = 10
    mc: int = 5
    lambda_threshold: float = 0.5


@lru_cache(maxsize=1)
def get_training_parameters() -> TrainParameters:
    return TrainParameters()
