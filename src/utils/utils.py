from typing import List, Tuple

import numpy as np
import torch


def set_seed(seed):
	"""Set random seeds for reproducibility"""
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)


def create_dummy_dataset(weights: List[float], num_samples: int = 100, bias: bool = False) -> Tuple[
	np.ndarray, np.ndarray]:
	"""Create a dummy dataset with given weights"""
	X = np.random.randn(num_samples, len(weights))
	y = X @ weights
	if bias:
		y += np.random.randn(num_samples)
	return X, y


def create_datasets(weights_list: List[List[float]], num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
	"""Create a list of datasets with given weights"""
	datasets = []
	for weights in weights_list:
		X, y = create_dummy_dataset(weights, num_samples)
		datasets.append((X, y))
	return datasets
