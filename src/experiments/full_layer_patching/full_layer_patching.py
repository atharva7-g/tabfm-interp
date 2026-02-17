from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Tuple, Any
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
from tqdm import tqdm

from src.datasets import create_dataset, get_dataset_formula
from src.utils.utils import set_seed

DATASET_TYPE = "multiplication"
CORRUPT_IDX = 1
NOISE_STD = 1.0
SEED = 42
N_SAMPLES = 1000
TEST_SIZE = 0.5
N_BATCH_SAMPLES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent / "output"


def create_corrupted_input(
    X_clean: np.ndarray, corrupt_idx: int = 1, noise_std: float = 1.0, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X_corrupt = X_clean.copy()
    num_samples = X_clean.shape[0]
    X_corrupt[:, corrupt_idx] = rng.normal(0.0, noise_std, num_samples)
    return X_corrupt


def create_cache_hook(
    cached_activation: Dict[str, torch.Tensor], layer_name: str
) -> Callable:
    def hook(module, inputs, output):
        if isinstance(output, (tuple, list)):
            output_tensor = output[0]
        else:
            output_tensor = output
        cached_activation[layer_name] = output_tensor.detach().clone()

    return hook


def create_full_layer_patch_hook(cached_activation: torch.Tensor) -> Callable:
    def hook(module, inputs, output):
        if isinstance(output, (tuple, list)):
            output_list = [cached_activation.clone()]
            output_list.extend(output[1:])
            return tuple(output_list)
        return cached_activation.clone()

    return hook


def run_single_layer_patching(
    regressor: TabPFNRegressor,
    X_clean: np.ndarray,
    X_corrupt: np.ndarray,
    layer_idx: int,
) -> Dict:
    model = regressor.model_
    layer_name = f"layer_{layer_idx}"
    cached_activation = {}
    layer = model.transformer_encoder.layers[layer_idx]
    attention_module = layer.self_attn_between_features
    cache_hook_fn = create_cache_hook(cached_activation, layer_name)
    cache_handle = attention_module.register_forward_hook(cache_hook_fn)
    with torch.no_grad():
        y_clean = regressor.predict(X_clean)
    cache_handle.remove()
    clean_activation = cached_activation[layer_name]
    with torch.no_grad():
        y_corrupt = regressor.predict(X_corrupt)
    patch_hook_fn = create_full_layer_patch_hook(clean_activation)
    patch_handle = attention_module.register_forward_hook(patch_hook_fn)
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
    EPS = 1e-3
    denom = np.sign(clean_corrupt_diff) * max(abs(clean_corrupt_diff), EPS)
    recovery_ratio = restoration / denom
    return {
        "y_clean": y_clean_val,
        "y_corrupt": y_corrupt_val,
        "y_patched": y_patched_val,
        "restoration": restoration,
        "recovery_ratio": recovery_ratio,
        "layer_idx": layer_idx,
    }


def sweep_all_layers(
    regressor: TabPFNRegressor, X_clean: np.ndarray, X_corrupt: np.ndarray
) -> List[Dict]:
    model = regressor.model_
    total_layers = len(model.transformer_encoder.layers)
    results = []
    for layer_idx in range(total_layers):
        result = run_single_layer_patching(regressor, X_clean, X_corrupt, layer_idx)
        results.append(result)
    return results


def average_results(all_results: List[List[Dict]]) -> Tuple[List[Dict], Dict]:
    """
    Average results across multiple samples.

    Args:
        all_results: List of results, where each result is a list of layer dicts

    Returns:
        averaged_results: List of dicts with mean values per layer
        stats: Dict with mean and std per layer
    """
    n_samples = len(all_results)
    n_layers = len(all_results[0])

    averaged_results = []
    stats: Dict[str, list[Any] | float | dict[str, np.floating[Any] | int]] = {
        "per_layer": []
    }

    for layer_idx in range(n_layers):
        # Collect values across all samples for this layer
        recovery_ratios = [
            all_results[i][layer_idx]["recovery_ratio"] for i in range(n_samples)
        ]
        restorations = [
            all_results[i][layer_idx]["restoration"] for i in range(n_samples)
        ]
        y_cleans = [all_results[i][layer_idx]["y_clean"] for i in range(n_samples)]
        y_corrupts = [all_results[i][layer_idx]["y_corrupt"] for i in range(n_samples)]

        # Compute statistics
        avg_dict = {
            "layer_idx": layer_idx,
            "y_clean": np.mean(y_cleans),
            "y_corrupt": np.mean(y_corrupts),
            "restoration": np.mean(restorations),
            "recovery_ratio": np.mean(recovery_ratios),
        }
        averaged_results.append(avg_dict)

        # Store detailed stats
        stats["per_layer"].append(
            {
                "layer_idx": layer_idx,
                "recovery_ratio_mean": np.mean(recovery_ratios),
                "recovery_ratio_std": np.std(recovery_ratios),
                "restoration_mean": np.mean(restorations),
                "restoration_std": np.std(restorations),
                "n_samples": n_samples,
            }
        )

    # Overall stats
    all_recovery_ratios = [
        r["recovery_ratio"] for sample in all_results for r in sample
    ]
    stats["overall"] = {
        "mean_recovery": np.mean(all_recovery_ratios),
        "std_recovery": np.std(all_recovery_ratios),
        "n_samples": n_samples,
        "n_layers": n_layers,
    }

    return averaged_results, stats


def plot_results(results: List[Dict], save_path: Path, stats: Dict = None):
    layer_indices = [r["layer_idx"] for r in results]
    restorations = [r["restoration"] for r in results]
    recovery_ratios = [r["recovery_ratio"] for r in results]
    y_clean = results[0]["y_clean"]
    y_corrupt = results[0]["y_corrupt"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot restorations with error bars if stats available
    if stats:
        restoration_stds = [s["restoration_std"] for s in stats["per_layer"]]
        ax1.errorbar(
            layer_indices,
            restorations,
            yerr=restoration_stds,
            fmt="o-",
            linewidth=2,
            markersize=8,
            capsize=5,
        )
    else:
        ax1.plot(layer_indices, restorations, "o-", linewidth=2, markersize=8)

    ax1.axhline(
        y=y_clean - y_corrupt,
        color="r",
        linestyle="--",
        label=f"Target ({y_clean - y_corrupt:.4f})",
    )
    ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Restoration")
    ax1.set_title("Full Layer Patching: Restoration by Layer")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    recovery_pcts = [r * 100 for r in recovery_ratios]
    if stats:
        recovery_stds = [s["recovery_ratio_std"] * 100 for s in stats["per_layer"]]
        ax2.errorbar(
            layer_indices,
            recovery_pcts,
            yerr=recovery_stds,
            fmt="o-",
            linewidth=2,
            markersize=8,
            color="green",
            capsize=5,
        )
    else:
        ax2.plot(
            layer_indices, recovery_pcts, "o-", linewidth=2, markersize=8, color="green"
        )

    ax2.axhline(y=100, color="r", linestyle="--", label="Full Recovery")
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Recovery %")
    ax2.set_title("Full Layer Patching: Recovery Ratio")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")


def main():
    set_seed(SEED)
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {DATASET_TYPE}")
    print(f"Formula: {get_dataset_formula(DATASET_TYPE)}")
    print(f"Batch size: {N_BATCH_SAMPLES} samples")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nCreating dataset with {N_SAMPLES} samples...")
    X, y = create_dataset(DATASET_TYPE, num_samples=N_SAMPLES, seed=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # Select batch of test samples
    n_available = len(X_test)
    n_to_use = min(N_BATCH_SAMPLES, n_available)
    X_test_batch = X_test[:n_to_use]
    print(
        f"Using {n_to_use} test samples for batching (out of {n_available} available)"
    )

    print("\nTraining TabPFN...")
    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)
    print("Model trained")

    # Run experiment on each sample
    all_results = []
    for sample_idx in tqdm(range(n_to_use), desc="Processing samples"):
        X_clean = X_test_batch[sample_idx : sample_idx + 1]

        # Use different seed for each sample's corruption
        X_corrupt = create_corrupted_input(
            X_clean, CORRUPT_IDX, NOISE_STD, SEED + sample_idx
        )

        results = sweep_all_layers(regressor, X_clean, X_corrupt)
        all_results.append(results)

    averaged_results, stats = average_results(all_results)

    # Print summary
    print("\nOverall statistics:")
    print(f"  Mean recovery: {stats['overall']['mean_recovery'] * 100:.2f}%")
    print(f"  Std recovery: {stats['overall']['std_recovery'] * 100:.2f}%")
    print(f"  Samples: {stats['overall']['n_samples']}")

    print("\nLayer-by-layer (mean ± std):")
    print(f"{'Layer':<8} {'Recovery %':<20}")
    print("-" * 30)
    for s in stats["per_layer"]:
        mean_pct = s["recovery_ratio_mean"] * 100
        std_pct = s["recovery_ratio_std"] * 100
        print(f"{s['layer_idx']:<8} {mean_pct:.2f} ± {std_pct:.2f}")

    # Find best layer by mean recovery
    best_layer_idx = max(
        range(len(stats["per_layer"])),
        key=lambda i: abs(stats["per_layer"][i]["recovery_ratio_mean"]),
    )
    best_stats = stats["per_layer"][best_layer_idx]
    print(
        f"\nBest layer: {best_layer_idx} "
        f"({best_stats['recovery_ratio_mean'] * 100:.2f}% ± "
        f"{best_stats['recovery_ratio_std'] * 100:.2f}% recovery)"
    )

    # Save results
    print(f"\n{'=' * 60}")
    print("SAVING RESULTS")
    print(f"{'=' * 60}")
    plot_path = OUTPUT_DIR / f"restoration_{DATASET_TYPE}_batch{n_to_use}.png"
    plot_results(averaged_results, plot_path, stats)

    import json

    summary = {
        "dataset_type": DATASET_TYPE,
        "corrupt_idx": CORRUPT_IDX,
        "noise_std": NOISE_STD,
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "n_batch_samples": n_to_use,
        "y_clean": averaged_results[0]["y_clean"],
        "y_corrupt": averaged_results[0]["y_corrupt"],
        "best_layer": best_layer_idx,
        "best_recovery_mean": best_stats["recovery_ratio_mean"],
        "best_recovery_std": best_stats["recovery_ratio_std"],
        "overall_stats": stats["overall"],
        "per_layer_stats": stats["per_layer"],
        "averaged_results": averaged_results,
        "all_results": all_results,
    }
    json_path = OUTPUT_DIR / f"summary_{DATASET_TYPE}_batch{n_to_use}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {json_path}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
