"""MNIST dataloader helper using PyTorch.

Provides `get_mnist_dataloaders` which returns train and test
`torch.utils.data.DataLoader` objects with sensible defaults.

Dependencies:
  pip install torch torchvision

Usage example:
  from dataloader import get_mnist_dataloaders
  train_loader, test_loader = get_mnist_dataloaders(batch_size=128)

The module also contains a small CLI example when run as a script.
"""

from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def _binarize(t: torch.Tensor) -> torch.Tensor:
	return (t > 0.5).float()

def get_mnist_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = False,
    download: bool = True,
    seed: Optional[int] = None,
    shuffle_train: bool = True,
    binarize: bool = True,
) -> Tuple[DataLoader, DataLoader]:
	"""Create and return MNIST train and test dataloaders.

	Args:
		data_dir: Directory where MNIST will be downloaded.
		batch_size: Batch size for both loaders.
		num_workers: Number of DataLoader worker processes.
		pin_memory: Whether to use pinned memory (recommended for GPU).
		download: If True, download the dataset if not present.
		seed: Optional random seed for reproducibility (affects shuffle).
		shuffle_train: Whether to shuffle the training dataset.

	Returns:
		(train_loader, test_loader): Two PyTorch DataLoader objects.
	"""

	if seed is not None:
		torch.manual_seed(seed)


	# Build transform pipeline. If `binarize=True` we convert to 0/1
	# pixel values (optionally stochastically) and skip normalization
	# to preserve strict binary values.
	transform_list = [transforms.ToTensor()]

	if binarize:

		transform_list.append(transforms.Lambda(_binarize))
	else:
		pass
		#transform_list.append(transforms.Normalize(mnist_mean, mnist_std))

	common_transforms = transforms.Compose(transform_list)

	train_dataset = datasets.MNIST(
		root=data_dir, train=True, transform=common_transforms, download=download
	)

	test_dataset = datasets.MNIST(
		root=data_dir, train=False, transform=common_transforms, download=download
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=shuffle_train,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)

	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)

	return train_loader, test_loader


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Create MNIST dataloaders (example)")
	parser.add_argument("--data-dir", default="./data", help="Directory to store MNIST data")
	parser.add_argument("--batch-size", type=int, default=64, help="Batch size for loaders")
	parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes")
	parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false",
						help="Disable pin_memory (useful on CPU-only machines)")
	parser.add_argument("--no-download", dest="download", action="store_false",
						help="Don't download the dataset")
	parser.add_argument("--seed", type=int, default=None, help="Optional random seed")

	args = parser.parse_args()

	train_loader, test_loader = get_mnist_dataloaders(
		data_dir=args.data_dir,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_memory,
		download=args.download,
		seed=args.seed,
	)

	print(f"Train batches: {len(train_loader)}; Test batches: {len(test_loader)}")
	# Print dataset sizes
	print(f"Train samples: {len(train_loader.dataset)}; Test samples: {len(test_loader.dataset)}")

	# Inspect one batch
	batch = next(iter(train_loader))
	images, labels = batch
	print(f"Batch images shape: {images.shape}; labels shape: {labels.shape}")

	# Plot a 10 by 10 grid of images
	import matplotlib.pyplot as plt

	fig, axes = plt.subplots(8, 8, figsize=(8, 8))
	for i in range(8):
		for j in range(8):
			axes[i, j].imshow(images[i * 8 + j].squeeze(), cmap="gray")
			axes[i, j].axis("off")
	plt.show()

	print(images.min(), images.max(), images.unique())  # Check binarization
