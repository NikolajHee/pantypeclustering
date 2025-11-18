# define the models, evaluator and optimizer
import math

import torch
from torch import Tensor, nn


class BaseEncoder(nn.Module): ...


class BaseDecoder(nn.Module): ...


class FFEncoderMNIST(BaseEncoder):
    """Feedforward encoder for MNIST"""

    input_shape = (1, 28, 28)

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.input_size = math.prod(self.input_shape)

        self.fc1 = nn.Linear(self.input_size, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.view(-1, self.input_size)
        h1 = torch.tanh(self.fc1(x))
        mean, log_var = self.fc21(h1), self.fc22(h1)
        return mean, log_var


class FFDecoderMNIST(BaseDecoder):
    """Feedforward decoder for MNIST"""

    input_shape = (1, 28, 28)

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.input_size = math.prod(self.input_shape)

        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc41 = nn.Linear(hidden_dim, self.input_size)
        self.fc42 = nn.Linear(hidden_dim, self.input_size)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        h3 = torch.tanh(self.fc3(z))

        mean, log_var = torch.sigmoid(self.fc41(h3)), self.fc42(h3)

        mean = mean.view(-1, *self.input_shape)
        log_var = log_var.view(-1, *self.input_shape)

        return (mean, log_var)


class CNNEncoder(BaseEncoder):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=2 * latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encoder(x)

        h = torch.tanh(self.fc1(z))

        mean, log_var = h.chunk(2, dim=1)

        return mean, log_var


class CNNDecoder(BaseDecoder):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.to_decoder = nn.Linear(in_features=latent_dim, out_features=64 * 7 * 7)
        final_conv = nn.ConvTranspose2d(32, 2 * 1, 4, 2, 1)

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            final_conv,
        )

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        x = self.to_decoder(z)
        x = x.reshape(-1, 64, 7, 7)  # TODO: remove hardcoding?
        x = self.decoder(x)
        mean, log_var = x.chunk(2, dim=1)
        return (mean, log_var)
