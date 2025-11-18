from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray


def tsne_plot(
    tsne_results: NDArray[Any],
    labels: NDArray[Any],
    means: NDArray[Any] | None = None,
) -> Figure:
    fig = plt.figure(figsize=(10, 8))  # pyright: ignore[reportUnknownMemberType]
    scatter = plt.scatter(  # pyright: ignore[reportUnknownMemberType]
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
    )
    if means is not None:
        _ = plt.scatter(  # pyright: ignore[reportUnknownMemberType]
            means[:, 0],
            means[:, 1],
            marker="x",
        )

    plt.colorbar(scatter)  # pyright: ignore[reportUnknownMemberType]
    plt.title("t-SNE Visualization of Latent Space")  # pyright: ignore[reportUnknownMemberType]
    plt.xlabel("t-SNE 1")  # pyright: ignore[reportUnknownMemberType]
    plt.ylabel("t-SNE 2")  # pyright: ignore[reportUnknownMemberType]

    plt.tight_layout()
    return fig
