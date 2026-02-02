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


def create_multiplication_dataset(
    num_samples: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a dataset where y = a * b + c with a, b, c ~ N(0, 1).

    Args:
            num_samples: Number of samples to generate.
            seed: Random seed for reproducibility.

    Returns:
            Tuple of (X, y) where X has shape (num_samples, 3) with columns [a, b, c]
            and y = a * b + c.
    """
    rng = np.random.default_rng(seed)

    # Generate a, b, c ~ N(0, 1)
    a = rng.standard_normal(num_samples)
    b = rng.standard_normal(num_samples)
    c = rng.standard_normal(num_samples)

    # Compute y = a * b + c
    y = a * b + c

    # Stack into feature matrix: X = [a, b, c]
    X = np.stack([a, b, c], axis=1).astype(np.float32)

    return X, y.astype(np.float32)


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
