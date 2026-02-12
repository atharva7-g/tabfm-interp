"""Synthetic datasets for TabPFN."""

import numpy as np
from typing import Dict, Callable, Tuple

DATASET_REGISTRY: Dict[str, Callable] = {}


def register_dataset(name: str):
    """Decorator to register a dataset."""

    def decorator(func: Callable) -> Callable:
        DATASET_REGISTRY[name] = func
        return func

    return decorator


@register_dataset("multiplication")
def create_multiplication_dataset(
    num_samples: int = 1000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """y = a * b + c"""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(num_samples)
    b = rng.standard_normal(num_samples)
    c = rng.standard_normal(num_samples)
    y = a * b + c
    X = np.stack([a, b, c], axis=1).astype(np.float32)
    return X, y.astype(np.float32)


@register_dataset("quadratic")
def create_quadratic_dataset(
    num_samples: int = 1000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """y = a² + b² + c"""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(num_samples)
    b = rng.standard_normal(num_samples)
    c = rng.standard_normal(num_samples)
    y = a**2 + b**2 + c
    X = np.stack([a, b, c], axis=1).astype(np.float32)
    return X, y.astype(np.float32)


def create_dataset(dataset_type: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Factory to create dataset by type."""
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_type}. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_type](**kwargs)


def list_datasets() -> list:
    """List available datasets."""
    return list(DATASET_REGISTRY.keys())


def get_dataset_formula(dataset_type: str) -> str:
    """Get formula for dataset type."""
    formulas = {
        "multiplication": "y = a × b + c",
        "quadratic": "y = a² + b² + c",
    }
    return formulas.get(dataset_type, "Unknown")
