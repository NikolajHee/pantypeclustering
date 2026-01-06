from torch import Tensor, nn


class Recogniser(nn.Module):
    num_channels = 1
    num_filters = 16

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            x_size: int,
            w_size: int,
            number_of_mixtures: int,
    ):
        super().__init__()  # type: ignore

        self.main_network = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_filters, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(),
            nn.Conv2d(self.num_filters, 2*self.num_filters, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(2*self.num_filters),
            nn.ReLU(),
            nn.Conv2d(2*self.num_filters, 4*self.num_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4*self.num_filters),
            nn.ReLU(),
            nn.Conv2d(4*self.num_filters, hidden_size, kernel_size=9),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )

        self.mean_x = nn.Linear(hidden_size, x_size)
        self.logVar_x = nn.Linear(hidden_size, x_size)
        self.mean_w = nn.Linear(hidden_size, w_size)
        self.logVar_w = nn.Linear(hidden_size, w_size)

    def forward(self, x: Tensor) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        batch_size, _, _, _ = x.shape

        hidden = self.main_network(x)

        hidden = hidden.reshape(batch_size, -1)

        mean_x = self.mean_x(hidden)
        logvar_x = self.logVar_x(hidden)

        mean_w = self.mean_w(hidden)
        logvar_w = self.logVar_w(hidden)

        return (mean_x, logvar_x), (mean_w, logvar_w)


class YGenerator(nn.Module):
    num_channels = 1
    num_filters = 16

    def __init__(self, input_size: int, hidden_size: int, output_size: int, continuous: bool):
        super().__init__()  # type: ignore

        self.hidden_size = hidden_size

        self.projection_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.main_network = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, 4*self.num_filters, kernel_size=9),
            nn.BatchNorm2d(4*self.num_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(4*self.num_filters, 2*self.num_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*self.num_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(2*self.num_filters, self.num_filters, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_filters, self.num_channels, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(self.num_channels),
            nn.Sigmoid(),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        hidden = self.projection_layer(input_tensor)
        hidden = hidden.view(-1, self.hidden_size, 1, 1)
        y = self.main_network(hidden)
        return y


class PriorGenerator(nn.Module):
    def __init__(
            self, input_size: int, hidden_size: int, output_size: int, number_of_mixtures: int
    ):
        super().__init__()  # type: ignore
        self.projection_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
        )

        self.mixture_means = nn.ModuleList([
            nn.Linear(hidden_size, output_size) for _ in range(number_of_mixtures)
        ])
        self.mixture_logvars = nn.ModuleList([
            nn.Linear(hidden_size, output_size) for _ in range(number_of_mixtures)
        ])

    def forward(self, input_tensor: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        hidden = self.projection_layer(input_tensor)

        means = [linear(hidden) for linear in self.mixture_means]
        logvars = [linear(hidden) for linear in self.mixture_logvars]

        return (means, logvars)
