from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Callable
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
    EPS = 1e-3  # choose relative to target scale

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
        print(f"Processing layer {layer_idx}/{total_layers - 1}...")
        result = run_single_layer_patching(regressor, X_clean, X_corrupt, layer_idx)
        results.append(result)
    return results


def plot_results(results: List[Dict], save_path: Path):
    layer_indices = [r["layer_idx"] for r in results]
    restorations = [r["restoration"] for r in results]
    recovery_ratios = [r["recovery_ratio"] for r in results]
    y_clean = results[0]["y_clean"]
    y_corrupt = results[0]["y_corrupt"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
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
    ax2.plot(
        layer_indices,
        [r * 100 for r in recovery_ratios],
        "o-",
        linewidth=2,
        markersize=8,
        color="green",
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
    # print("Model trained")
    # print("\n" + "=" * 60)
    # print("RUNNING FULL LAYER PATCHING")
    # print("=" * 60)
    results = sweep_all_layers(regressor, X_clean, X_corrupt)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Clean output:     {results[0]['y_clean']:.6f}")
    print(f"Corrupted output: {results[0]['y_corrupt']:.6f}")
    print("\nLayer-by-layer:")
    print(f"{'Layer':<8} {'Recovery %':<12}")
    print("-" * 20)
    for r in results:
        print(f"{r['layer_idx']:<8} {r['recovery_ratio'] * 100:<12.2f}")
    best = max(results, key=lambda x: abs(x["recovery_ratio"]))
    print(
        f"\nBest layer: {best['layer_idx']} ({best['recovery_ratio'] * 100:.2f}% recovery)"
    )
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    plot_path = OUTPUT_DIR / f"restoration_{DATASET_TYPE}.png"
    plot_results(results, plot_path)
    import json

    summary = {
        "dataset_type": DATASET_TYPE,
        "corrupt_idx": CORRUPT_IDX,
        "noise_std": NOISE_STD,
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "y_clean": results[0]["y_clean"],
        "y_corrupt": results[0]["y_corrupt"],
        "best_layer": best["layer_idx"],
        "best_recovery": best["recovery_ratio"],
        "results": results,
    }
    json_path = OUTPUT_DIR / f"summary_{DATASET_TYPE}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {json_path}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
