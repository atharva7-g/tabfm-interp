#!/usr/bin/env python3
# WIP: Embedding-level activation patching for TabPFN

from pathlib import Path
import numpy as np
import torch
from typing import Dict, Callable
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

from src.datasets import create_dataset, get_dataset_formula
from src.utils.utils import set_seed

DATASET_TYPE = "multiplication"
CORRUPT_IDX = 1
NOISE_STD = 1.0
SEED = 42
N_SAMPLES = 1000
TEST_SIZE = 0.5
DEVICE = None
OUTPUT_DIR = Path(__file__).parent / "output"


def create_corrupted_input(
    X_clean: np.ndarray, corrupt_idx: int = 1, noise_std: float = 1.0, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X_corrupt = X_clean.copy()
    num_samples = X_clean.shape[0]
    X_corrupt[:, corrupt_idx] = rng.normal(0.0, noise_std, num_samples)
    return X_corrupt


def create_cache_hook(cached_activation: Dict[str, torch.Tensor], key: str) -> Callable:
    def hook(module, inputs, output):
        if isinstance(output, (tuple, list)):
            output_tensor = output[0]
        else:
            output_tensor = output
        cached_activation[key] = output_tensor.detach().clone()

    return hook


def create_patch_hook(cached_activation: torch.Tensor) -> Callable:
    def hook(module, inputs, output):
        if isinstance(output, (tuple, list)):
            output_list = [cached_activation.clone()]
            output_list.extend(output[1:])
            return tuple(output_list)
        cached_activation.to(output.device)
        return cached_activation.clone()

    return hook


def run_embedding_patching_experiment(
    regressor: TabPFNRegressor,
    X_clean: np.ndarray,
    X_corrupt: np.ndarray,
) -> Dict:
    model = regressor.model_
    embedding_layer = model.feature_positional_embedding_embeddings
    cached_embedding = {}
    cache_hook_fn = create_cache_hook(cached_embedding, "embeddings")
    cache_handle = embedding_layer.register_forward_hook(cache_hook_fn)
    with torch.no_grad():
        y_clean = regressor.predict(X_clean)
    cache_handle.remove()
    clean_embedding = cached_embedding["embeddings"]
    with torch.no_grad():
        y_corrupt = regressor.predict(X_corrupt)
    patch_hook_fn = create_patch_hook(clean_embedding)
    patch_handle = embedding_layer.register_forward_hook(patch_hook_fn)
    with torch.no_grad():
        y_patched = regressor.predict(X_corrupt)
    patch_handle.remove()
    y_clean_val = float(y_clean[0]) if len(y_clean.shape) > 0 else float(y_clean)
    y_corrupt_val = (
        float(y_corrupt[0]) if len(y_corrupt.shape) > 0 else float(y_corrupt)
    )
    y_patched_val = (
        float(y_patched[0]) if len(y_patched.shape) > 0 else float(y_patched)
    )
    restoration = y_patched_val - y_corrupt_val
    clean_corrupt_diff = y_clean_val - y_corrupt_val
    recovery_ratio = (
        restoration / clean_corrupt_diff if abs(clean_corrupt_diff) > 1e-10 else 0.0
    )
    return {
        "y_clean": y_clean_val,
        "y_corrupt": y_corrupt_val,
        "y_patched": y_patched_val,
        "restoration": restoration,
        "recovery_ratio": recovery_ratio,
        "clean_corrupt_diff": clean_corrupt_diff,
    }


def main():
    print("=" * 60)
    print("EMBEDDING-LEVEL PATCHING EXPERIMENT (WIP)")
    print("=" * 60)
    set_seed(SEED)
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {DATASET_TYPE}")
    print(f"Formula: {get_dataset_formula(DATASET_TYPE)}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nCreating dataset with {N_SAMPLES} samples...")
    X, y = create_dataset(DATASET_TYPE, num_samples=N_SAMPLES, seed=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    X_clean = X_test[0:1]
    print(
        f"Test sample: a={X_clean[0, 0]:.4f}, b={X_clean[0, 1]:.4f}, c={X_clean[0, 2]:.4f}"
    )
    X_corrupt = create_corrupted_input(X_clean, CORRUPT_IDX, NOISE_STD, SEED)
    corrupt_feature = ["a", "b", "c"][CORRUPT_IDX]
    print(f"Corrupted {corrupt_feature}: {X_corrupt[0, CORRUPT_IDX]:.4f} (noise)")
    print("\nTraining TabPFN...")
    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)
    print("Model trained")
    print("\n" + "=" * 60)
    print("RUNNING EMBEDDING PATCHING")
    print("=" * 60)
    result = run_embedding_patching_experiment(regressor, X_clean, X_corrupt)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Clean output:      {result['y_clean']:.6f}")
    print(f"Corrupted output:  {result['y_corrupt']:.6f}")
    print(f"Patched output:    {result['y_patched']:.6f}")
    print(f"\nRestoration:       {result['restoration']:.6f}")
    print(f"Target (clean-corrupt diff): {result['clean_corrupt_diff']:.6f}")
    print(f"Recovery ratio:    {result['recovery_ratio'] * 100:.2f}%")
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if result["recovery_ratio"] > 0.95:
        print("Near-complete recovery (95%+)")
        print("  The model relies heavily on the embedded input representation.")
        print("  Patching embeddings alone is sufficient to recover clean behavior.")
    elif result["recovery_ratio"] > 0.5:
        print("Partial recovery (50-95%)")
        print("  The embedding contains significant information, but")
        print("  later layers also process the corrupted feature.")
    else:
        print("Low recovery (<50%)")
        print("  The embedding layer alone doesn't contain enough information.")
        print("  The corruption affects processing in deeper layers.")
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    import json

    summary = {
        "experiment_type": "embedding_patching",
        "dataset_type": DATASET_TYPE,
        "corrupt_idx": CORRUPT_IDX,
        "noise_std": NOISE_STD,
        "seed": SEED,
        "n_samples": N_SAMPLES,
        **result,
    }
    json_path = OUTPUT_DIR / f"embedding_patching_{DATASET_TYPE}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {json_path}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
