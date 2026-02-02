from typing import List, Sequence, Tuple, cast

import numpy as np
import torch

def set_seed(seed: int) -> None:
	"""Set random seeds for reproducibility."""
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)


def create_dummy_dataset(
		weights: Sequence[float],
		num_samples: int = 100,
		bias: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
	"""Create a dummy linear dataset with optional bias."""
	rng = np.random.default_rng()

	X = rng.standard_normal((num_samples, len(weights)))
	y = X @ np.asarray(weights)

	bias_w = rng.standard_normal() if bias else 0.0
	y = y + bias_w

	return X, y, bias_w


def create_datasets(
		weights_list: Sequence[Sequence[float]],
		num_samples: int = 100,
) -> List[Tuple[np.ndarray, np.ndarray]]:
	"""Create multiple datasets, one per weight vector."""
	datasets: List[Tuple[np.ndarray, np.ndarray]] = []

	for weights in weights_list:
		X, y = create_dummy_dataset(weights, num_samples)
		datasets.append((X, y))

	return datasets
