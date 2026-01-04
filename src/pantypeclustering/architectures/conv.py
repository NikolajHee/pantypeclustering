


from torch import Tensor, nn
import torch

class Recogniser(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            x_size: int,
            w_size: int,
            number_of_mixtures: int,
    ):
        super().__init__()
        nChannels = 1
        nFilters = 16

        self.main_network = nn.Sequential(
            nn.Conv2d(nChannels, nFilters, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(nFilters),
            nn.ReLU(),
            nn.Conv2d(nFilters, 2*nFilters, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(2*nFilters),
            nn.ReLU(),
            nn.Conv2d(2*nFilters, 4*nFilters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4*nFilters),
            nn.ReLU(),
            nn.Conv2d(4*nFilters, hidden_size, kernel_size=9),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )

        self.mean_x = nn.Linear(hidden_size, x_size)
        self.logVar_x = nn.Linear(hidden_size, x_size)
        self.mean_w = nn.Linear(hidden_size, w_size)
        self.logVar_w = nn.Linear(hidden_size, w_size)

    def forward(self, x: Tensor):
        B, C, W, H = x.shape
        hidden = self.main_network(x)

        hidden = hidden.reshape(B, -1)

        mean_x = self.mean_x(hidden)
        logVar_x = self.logVar_x(hidden)

        mean_w = self.mean_w(hidden)
        logVar_w = self.logVar_w(hidden)

        return (mean_x, logVar_x), (mean_w, logVar_w)


class YGenerator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, continuous: bool):
        super().__init__()
        nChannels = 1
        nFilters = 16
        self.hidden_size = hidden_size

        self.projection_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.main_network = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, 4*nFilters, kernel_size=9),
            nn.BatchNorm2d(4*nFilters),
            nn.ReLU(),
            nn.ConvTranspose2d(4*nFilters, 2*nFilters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*nFilters),
            nn.ReLU(),
            nn.ConvTranspose2d(2*nFilters, nFilters, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(nFilters),
            nn.ReLU(),
            nn.ConvTranspose2d(nFilters, nChannels, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(nChannels),
            nn.Sigmoid(),
        )

    def forward(self, inputTensor: Tensor):
        hidden = self.projection_layer(inputTensor)
        hidden = hidden.view(-1, self.hidden_size, 1, 1)
        y = self.main_network(hidden)
        return y

class PriorGenerator(nn.Module):
    def __init__(
            self, input_size: int, hidden_size: int, output_size: int, number_of_mixtures: int
    ):
        super().__init__()
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

    def forward(self, inputTensor: Tensor):
        hidden = self.projection_layer(inputTensor)
        means = [l(hidden) for l in self.mixture_means]
        logvars = [l(hidden) for l in self.mixture_logvars]

        return (means, logvars)



