import argparse
from dataclasses import dataclass

from torch import Tensor


@dataclass
class ForwardOutput:
    mean_z1: Tensor
    logvar_z1: Tensor
    z1: Tensor
    mean_z2: Tensor
    logvar_z2: Tensor
    x_recon: Tensor
    prior_means: Tensor
    prior_logvars: Tensor

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training options")

    parser.add_argument(
        "-dataSet",
        type=str,
        default="mnist",
        help="Dataset used",
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=1,
        help="random seed",
    )
    parser.add_argument(
        "-learningRate",
        type=float,
        default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "-batchSize",
        type=int,
        default=50,
        help="Batch Size",
    )
    parser.add_argument(
        "-optimiser",
        type=str,
        default="adam",
        help="Optimser",
    )
    parser.add_argument(
        "-gpu",
        type=int,
        default=0,
        help="Using Cuda, 1 to enable",
    )
    parser.add_argument(
        "-epoch",
        type=int,
        default=100,
        help="Number of Epoch",
    )
    parser.add_argument(
        "-visualise2D",
        type=int,
        default=0,
        help="Save data for visualisation, 1 to enable",
    )
    parser.add_argument(
        "-xSize",
        type=int,
        default=200,
        help="Size of x variable",
    )
    parser.add_argument(
        "-wSize",
        type=int,
        default=150,
        help="Size of w variable",
    )
    parser.add_argument(
        "-K",
        type=int,
        default=15,
        help="Number of clusters",
    )
    parser.add_argument(
        "-hiddenSize",
        type=int,
        default=500,
        help="Size of the hidden layer",
    )
    parser.add_argument(
        "-zPriorWeight",
        type=float,
        default=1,
        help="Weight on the z prior term",
    )
    parser.add_argument(
        "-ACC",
        type=int,
        default=0,
        help="Report Clustering accuracy",
    )
    parser.add_argument(
        "-visualGen",
        type=int,
        default=0,
        help="Visualise the generation samples at every [input] epochs (0 to disable)",
    )
    parser.add_argument(
        "-continuous",
        type=int,
        default=0,
        help="Data is continous use Gaussian Criterion (1), Data is discrete use BCE Criterion (0)",
    )
    parser.add_argument(
        "-inputDimension",
        type=int,
        default=1,
        help="Dimension of the input vector into the network (e.g. 2 for height x width)",
    )
    parser.add_argument(
        "-network",
        type=str,
        default="fc",
        help="Network architecture use: fc (FullyConnected) / conv (Convolutional AE)",
    )
    parser.add_argument(
        "-nChannels",
        type=int,
        default=1,
        help="Number of Input channels",
    )
    parser.add_argument(
        "-nFilters",
        type=int,
        default=16,
        help="Number of Convolutional Filters in first layer",
    )
    parser.add_argument(
        "-nMC",
        type=int,
        default=1,
        help="Number of monte-carlo sample",
    )
    parser.add_argument(
        "-gpuID",
        type=int,
        default=1,
        help="Set GPU id to use",
    )
    parser.add_argument(
        "-cvWeight",
        type=float,
        default=1,
        help="Weight of the information theoretic cost term",
    )
    parser.add_argument(
        "-_id",
        type=str,
        default="",
        help="Experiment Path",
    )
    parser.add_argument(
        "-saveModel",
        type=int,
        default=0,
        help="Save model in experiments folder after finish training",
    )
    parser.add_argument(
        "-reportLowerBound",
        type=int,
        default=1,
        help="Save lowerbound while training",
    )
    parser.add_argument(
        "-lambda",
        type=float,
        default=0.5,
        help="Free bits threshold",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
