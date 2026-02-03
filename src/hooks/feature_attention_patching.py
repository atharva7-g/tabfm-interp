from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor


def create_corrupted_input(
    X_clean: np.ndarray,
    corrupt_idx: int = 1,
    noise_std: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X_corrupt = X_clean.copy()
    num_samples = X_clean.shape[0]
    X_corrupt[:, corrupt_idx] = rng.normal(0.0, noise_std, num_samples)
    return X_corrupt


def create_cache_hook(
    cached_activation: Dict[str, torch.Tensor],
    layer_name: str,
) -> Callable:
    def hook(output):
        if isinstance(output, (tuple, list)):
            output_tensor = output[0]
        else:
            output_tensor = output
        cached_activation[layer_name] = output_tensor.detach().clone()

    return hook


def create_patch_hook(
    cached_activation: torch.Tensor,
    feature_idx: int,
    sample: int = 3000
) -> Callable:
    def hook(output):
        if isinstance(output, (tuple, list)):
            output_tensor = output[0]
        else:
            output_tensor = output
        modified_output = output_tensor.clone()
        modified_output[:, sample, feature_idx, :] = cached_activation[:, sample, feature_idx, :]
        return modified_output

    return hook


def sweep_layers(
    regressor: TabPFNRegressor,
    X_clean: np.ndarray,
    X_corrupt: np.ndarray,
    corrupt_idx: int,
    max_layers: Optional[int] = None,
) -> List[Dict[str, float]]:
    model = regressor.model_
    total_layers = len(model.transformer_encoder.layers)
    num_layers = max_layers if max_layers is not None else total_layers
    num_layers = min(num_layers, total_layers)
    results = []
    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}/{num_layers - 1}...")
        result = run_single_layer_patching(
            regressor, X_clean, X_corrupt, corrupt_idx, layer_idx
        )
        results.append(result)
    return results


def run_single_layer_patching(
    regressor: TabPFNRegressor,
    X_clean: np.ndarray,
    X_corrupt: np.ndarray,
    corrupt_idx: int,
    layer_idx: int,
) -> Dict[str, float]:
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
    patch_hook_fn = create_patch_hook(clean_activation, corrupt_idx)
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
    if abs(clean_corrupt_diff) > 1e-10:
        recovery_ratio = restoration / clean_corrupt_diff
    else:
        recovery_ratio = 0.0
    return {
        "y_clean": y_clean_val,
        "y_corrupt": y_corrupt_val,
        "y_patched": y_patched_val,
        "restoration": restoration,
        "recovery_ratio": recovery_ratio,
        "layer_idx": layer_idx,
    }


def plot_restoration_results(
    results: List[Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
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
        label=f"Target (y_clean - y_corrupt = {y_clean - y_corrupt:.4f})",
    )
    ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Restoration (y_patched - y_corrupt)")
    ax1.set_title("Activation Patching: Restoration by Layer")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(
        layer_indices, recovery_ratios, "o-", linewidth=2, markersize=8, color="green"
    )
    ax2.axhline(y=1.0, color="r", linestyle="--", label="Full Recovery (100%)")
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Recovery Ratio")
    ax2.set_title("Activation Patching: Recovery Ratio by Layer")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def run_feature_attention_causal_patching_experiment(
    regressor: TabPFNRegressor,
    X_clean: np.ndarray,
    corrupt_idx: int,
    noise_std: float = 1.0,
    noise_seed: int = 42,
    max_layers: Optional[int] = None,
    plot: bool = True,
    save_path: Optional[str] = None,
) -> List[Dict[str, float]]:
    X_corrupt = create_corrupted_input(X_clean, corrupt_idx, noise_std, noise_seed)
    print(
        f"Clean input: a={X_clean[0, 0]:.4f}, b={X_clean[0, 1]:.4f}, c={X_clean[0, 2]:.4f}"
    )
    print(
        f"Corrupted input: a={X_corrupt[0, 0]:.4f}, b={X_corrupt[0, 1]:.4f} (noise), c={X_corrupt[0, 2]:.4f}"
    )
    results = sweep_layers(regressor, X_clean, X_corrupt, corrupt_idx, max_layers)
    print("\n" + "=" * 60)
    print("CAUSAL PATCHING RESULTS SUMMARY")
    print("=" * 60)
    print(f"Clean output:     y_clean   = {results[0]['y_clean']:.6f}")
    print(f"Corrupted output: y_corrupt = {results[0]['y_corrupt']:.6f}")
    print(
        f"Target: y_clean - y_corrupt = {results[0]['y_clean'] - results[0]['y_corrupt']:.6f}"
    )
    print("\nLayer-by-layer restoration:")
    print(f"{'Layer':<8} {'y_patched':<12} {'Restoration':<14} {'Recovery %':<12}")
    print("-" * 60)
    for r in results:
        recovery_pct = r["recovery_ratio"] * 100
        print(
            f"{r['layer_idx']:<8} {r['y_patched']:<12.6f} {r['restoration']:<14.6f} {recovery_pct:<12.2f}%"
        )
    best_layer = max(results, key=lambda x: abs(x["recovery_ratio"]))
    print(f"\nBest restoration at layer {best_layer['layer_idx']}:")
    print(f"  Recovery: {best_layer['recovery_ratio'] * 100:.2f}%")
    if plot:
        plot_restoration_results(results, save_path)
    return results


if __name__ == "__main__":
    from src.utils.utils import create_multiplication_dataset, set_seed
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("Feature Attention Activation Patching Demo")
    print("=" * 60)
    set_seed(42)
    X, y = create_multiplication_dataset(num_samples=1000, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    X_clean = X_test[0:1]
    print(
        f"\nTest sample: a={X_clean[0, 0]:.4f}, b={X_clean[0, 1]:.4f}, c={X_clean[0, 2]:.4f}"
    )
    print(
        f"Expected output (a*b + c): {X_clean[0, 0] * X_clean[0, 1] + X_clean[0, 2]:.4f}"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)
    print("Model fitted successfully")
    results = run_feature_attention_causal_patching_experiment(
        regressor=regressor,
        X_clean=X_clean,
        corrupt_idx=1,
        noise_std=1.0,
        noise_seed=42,
        max_layers=None,
        plot=True,
    )
    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)
