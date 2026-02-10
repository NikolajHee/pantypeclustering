"""Convolutional encoder, decoder, and GMM prior for GMVAE."""
from torch import Tensor, nn


class Encoder(nn.Module):
<<<<<<< HEAD
    """CNN encoder mapping images to (z1, z2) latent parameters."""

    num_channels = 1
    num_filters = 16

    def __init__(
            self,
            hidden_size: int,
            x_size: int,
            w_size: int,
=======
    # Recogniser
    num_channels = 1  #  MNIST images are grayscale
    #  Number of filters for the convolutional layers

    def __init__(
        self,
        hidden_size: int,
        z1_size: int,
        z2_size: int,
        num_filters: int = 16,
>>>>>>> master
    ):
        super().__init__()  # type: ignore

        self.main_network = nn.Sequential(
            nn.Conv2d(self.num_channels, num_filters, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, 2 * num_filters, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(2 * num_filters),
            nn.ReLU(),
            nn.Conv2d(2 * num_filters, 4 * num_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * num_filters),
            nn.ReLU(),
            nn.Conv2d(4 * num_filters, hidden_size, kernel_size=9),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )

        self.mean_z1 = nn.Linear(hidden_size, z1_size)
        self.logVar_z1 = nn.Linear(hidden_size, z1_size)
        self.mean_z2 = nn.Linear(hidden_size, z2_size)
        self.logVar_z2 = nn.Linear(hidden_size, z2_size)

    def forward(self, x: Tensor) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        batch_size, _, _, _ = x.shape

        hidden = self.main_network(x)

        hidden = hidden.reshape(batch_size, -1)

        mean_z1 = self.mean_z1(hidden)
        logvar_z1 = self.logVar_z1(hidden)

        mean_z2 = self.mean_z2(hidden)
        logvar_z2 = self.logVar_z2(hidden)

        return (mean_z1, logvar_z1), (mean_z2, logvar_z2)


class Decoder(nn.Module):
<<<<<<< HEAD
    """Transposed CNN decoder mapping latent z1 to images."""

=======
    # XGenerator
>>>>>>> master
    num_channels = 1

<<<<<<< HEAD
    def __init__(self, input_size: int, hidden_size: int):
=======
    def __init__(
        self,
        z1_size: int,
        hidden_size: int,
        num_filters: int = 16,
    ):
>>>>>>> master
        super().__init__()  # type: ignore

        self.hidden_size = hidden_size

        self.projection_layer = nn.Sequential(
            nn.Linear(z1_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self.main_network = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, 4 * num_filters, kernel_size=9),
            nn.BatchNorm2d(4 * num_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(
                4 * num_filters,
                2 * num_filters,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(2 * num_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * num_filters, num_filters, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, self.num_channels, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(self.num_channels),
            nn.Sigmoid(),
        )

    def forward(self, z1: Tensor) -> Tensor:
        hidden = self.projection_layer(z1)
        hidden = hidden.view(-1, self.hidden_size, 1, 1)
        x = self.main_network(hidden)
        return x


class PriorGenerator(nn.Module):
    """Gaussian mixture prior p(z1|z2) with K components."""

    def __init__(
        self,
        z2_size: int,
        hidden_size: int,
        z1_size: int,
        number_of_mixtures: int,
    ):
        super().__init__()  # type: ignore
        self.projection_layer = nn.Sequential(
            nn.Linear(z2_size, hidden_size),
            nn.Tanh(),
        )

        self.mixture_means = nn.ModuleList(
            [nn.Linear(hidden_size, z1_size) for _ in range(number_of_mixtures)],
        )
        self.mixture_logvars = nn.ModuleList(
            [nn.Linear(hidden_size, z1_size) for _ in range(number_of_mixtures)],
        )

    def forward(self, input_tensor: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        hidden = self.projection_layer(input_tensor)

        means = [linear(hidden) for linear in self.mixture_means]
        logvars = [linear(hidden) for linear in self.mixture_logvars]

        return (means, logvars)
