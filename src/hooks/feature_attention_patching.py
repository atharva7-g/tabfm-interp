"""
Causal Verification via Activation Patching

This module implements activation patching for feature attention in TabPFN
to test whether feature attention output is causally responsible for
the a * b computation that affects the test-row label token.

The procedure follows the standard activation patching workflow:
1. Run clean forward pass and cache activation at feature attention output
2. Run corrupted forward pass (with b replaced by noise)
3. Run corrupted forward pass with clean activation patched in
4. Compare outputs to determine causal sufficiency
"""

from typing import Dict, List, Optional, Tuple
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
    """Create corrupted input by replacing a feature with Gaussian noise.

    Args:
        X_clean: Clean input data of shape (num_samples, num_features)
        corrupt_idx: Index of feature to corrupt (default 1 for 'b' in [a, b, c])
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducible noise

    Returns:
        Corrupted input with specified feature replaced by noise
    """
    rng = np.random.default_rng(seed)
    X_corrupt = X_clean.copy()

    # Replace specified feature with Gaussian noise
    num_samples = X_clean.shape[0]
    X_corrupt[:, corrupt_idx] = rng.normal(0.0, noise_std, num_samples)

    return X_corrupt


def create_cache_hook(
    cached_activation: Dict[str, torch.Tensor],
    layer_name: str,
    token_idx: int = -1,
) -> callable:
    """Create a hook function to cache activation at test label token.

    Args:
        cached_activation: Dictionary to store the cached activation
        layer_name: Name of the layer being hooked
        token_idx: Index of token to cache (default -1 for last token)

    Returns:
        Hook function for register_forward_hook
    """

    def hook(module, inputs, output):
        """Hook that caches activation at specified token position.

        Args:
            module: The layer module
            inputs: Input tensors to the layer
            output: Output tensor from the layer (shape: [batch, seq_len, d_model])
        """
        # Extract output tensor (handle tuple/list outputs)
        if isinstance(output, (tuple, list)):
            output_tensor = output[0]
        else:
            output_tensor = output

        # Cache the full output, detached and cloned
        cached_activation[layer_name] = output_tensor.detach().clone()

    return hook


def create_patch_hook(
    cached_activation: torch.Tensor,
    token_idx: int = -1,
) -> callable:
    """Create a hook function to patch corrupted activation with clean.

    Args:
        cached_activation: The cached clean activation tensor to patch in
        token_idx: Index of token to patch (default -1 for last token)

    Returns:
        Hook function for register_forward_hook
    """

    def hook(module, inputs, output):
        """Hook that patches corrupted activation with cached clean activation.

        Args:
            module: The layer module
            inputs: Input tensors to the layer
            output: Output tensor from the layer (shape: [batch, seq_len, d_model])

        Returns:
            Modified output with cached activation patched in at token_idx
        """
        # Extract output tensor
        if isinstance(output, (tuple, list)):
            output_tensor = output[0]
        else:
            output_tensor = output

        # Clone the output for modification
        modified_output = output_tensor.clone()

        # Patch the cached clean activation at the test label token position
        # cached_activation shape: [batch, seq_len, d_model]
        # We patch: modified_output[0, token_idx, :] = cached_activation[0, token_idx, :]
        modified_output[0, token_idx, :] = cached_activation[0, token_idx, :]

        return modified_output

    return hook


def run_single_layer_patching(
    regressor: TabPFNRegressor,
    X_clean: np.ndarray,
    X_corrupt: np.ndarray,
    layer_idx: int,
    token_idx: int = -1,
) -> Dict[str, float]:
    """Run activation patching experiment on a single layer.

    Performs three forward passes:
    1. Clean run: normal prediction on X_clean
    2. Corrupted run: prediction on X_corrupt
    3. Patched run: prediction on X_corrupt with clean activation patched

    Args:
        regressor: Fitted TabPFNRegressor instance
        X_clean: Clean input data (at least 1 sample)
        X_corrupt: Corrupted input data (same shape as X_clean)
        layer_idx: Index of layer to patch
        token_idx: Index of test label token (default -1 for last)

    Returns:
        Dictionary with keys:
        - 'y_clean': prediction on clean input
        - 'y_corrupt': prediction on corrupted input
        - 'y_patched': prediction on corrupted input with patch
        - 'restoration': y_patched - y_corrupt (absolute recovery)
        - 'recovery_ratio': (y_patched - y_corrupt) / (y_clean - y_corrupt)
    """
    model = regressor.model_
    layer_name = f"layer_{layer_idx}"

    # Step 1: Clean run - cache the activation
    cached_activation = {}
    layer = model.transformer_encoder.layers[layer_idx]
    attention_module = layer.self_attn_between_features

    cache_hook_fn = create_cache_hook(cached_activation, layer_name, token_idx)
    cache_handle = attention_module.register_forward_hook(cache_hook_fn)

    with torch.no_grad():
        y_clean = regressor.predict(X_clean)

    cache_handle.remove()

    # Get the cached activation for patching
    clean_activation = cached_activation[layer_name]

    # Step 2: Corrupted run - no hooks
    with torch.no_grad():
        y_corrupt = regressor.predict(X_corrupt)

    # Step 3: Patched run - patch the cached activation
    patch_hook_fn = create_patch_hook(clean_activation, token_idx)
    patch_handle = attention_module.register_forward_hook(patch_hook_fn)

    with torch.no_grad():
        y_patched = regressor.predict(X_corrupt)

    patch_handle.remove()

    # Extract scalar values (assuming single test sample)
    y_clean_val = float(y_clean[0]) if len(y_clean.shape) > 0 else float(y_clean)
    y_corrupt_val = (
        float(y_corrupt[0]) if len(y_corrupt.shape) > 0 else float(y_corrupt)
    )
    y_patched_val = (
        float(y_patched[0]) if len(y_patched.shape) > 0 else float(y_patched)
    )

    # Compute metrics
    restoration = y_patched_val - y_corrupt_val

    # Avoid division by zero
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


def sweep_layers(
    regressor: TabPFNRegressor,
    X_clean: np.ndarray,
    X_corrupt: np.ndarray,
    max_layers: Optional[int] = None,
    token_idx: int = -1,
) -> List[Dict[str, float]]:
    """Sweep activation patching across all layers.

    Args:
        regressor: Fitted TabPFNRegressor instance
        X_clean: Clean input data
        X_corrupt: Corrupted input data
        max_layers: Maximum number of layers to sweep (None = all layers)
        token_idx: Index of test label token (default -1 for last)

    Returns:
        List of result dictionaries, one per layer
    """
    model = regressor.model_
    total_layers = len(model.transformer_encoder.layers)

    # Determine number of layers to sweep
    num_layers = max_layers if max_layers is not None else total_layers
    num_layers = min(num_layers, total_layers)

    results = []
    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}/{num_layers - 1}...")
        result = run_single_layer_patching(
            regressor, X_clean, X_corrupt, layer_idx, token_idx
        )
        results.append(result)

    return results


def plot_restoration_results(
    results: List[Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot restoration and recovery ratio across layers.

    Creates a 2-panel plot showing:
    - Top: Restoration (y_patched - y_corrupt) vs layer index
    - Bottom: Recovery ratio vs layer index

    Args:
        results: List of result dictionaries from sweep_layers
        save_path: Optional path to save the figure
    """
    # Extract data
    layer_indices = [r["layer_idx"] for r in results]
    restorations = [r["restoration"] for r in results]
    recovery_ratios = [r["recovery_ratio"] for r in results]
    y_clean = results[0]["y_clean"]
    y_corrupt = results[0]["y_corrupt"]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Restoration
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

    # Plot 2: Recovery Ratio
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
    corrupt_idx: int = 1,
    noise_std: float = 1.0,
    noise_seed: int = 42,
    max_layers: Optional[int] = None,
    token_idx: int = -1,
    plot: bool = True,
    save_path: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Run complete feature attention causal patching experiment.

    This is a convenience function that:
    1. Creates corrupted input
    2. Sweeps all layers
    3. Plots results
    4. Returns results

    Args:
        regressor: Fitted TabPFNRegressor instance
        X_clean: Clean input data
        corrupt_idx: Index of feature to corrupt (default 1 for 'b')
        noise_std: Standard deviation of Gaussian noise
        noise_seed: Random seed for noise generation
        max_layers: Maximum layers to sweep (None = all)
        token_idx: Index of test label token (default -1)
        plot: Whether to generate plots
        save_path: Optional path to save plot

    Returns:
        List of result dictionaries from sweep
    """
    # Create corrupted input
    X_corrupt = create_corrupted_input(X_clean, corrupt_idx, noise_std, noise_seed)

    print(
        f"Clean input: a={X_clean[0, 0]:.4f}, b={X_clean[0, 1]:.4f}, c={X_clean[0, 2]:.4f}"
    )
    print(
        f"Corrupted input: a={X_corrupt[0, 0]:.4f}, b={X_corrupt[0, 1]:.4f} (noise), c={X_corrupt[0, 2]:.4f}"
    )

    # Run sweep
    results = sweep_layers(regressor, X_clean, X_corrupt, max_layers, token_idx)

    # Print summary
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

    # Find best layer
    best_layer = max(results, key=lambda x: abs(x["recovery_ratio"]))
    print(f"\nBest restoration at layer {best_layer['layer_idx']}:")
    print(f"  Recovery: {best_layer['recovery_ratio'] * 100:.2f}%")

    # Plot if requested
    if plot:
        plot_restoration_results(results, save_path)

    return results


if __name__ == "__main__":
    # Demo usage
    from src.utils.utils import create_multiplication_dataset, set_seed
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("Feature Attention Activation Patching Demo")
    print("=" * 60)

    # Set seeds for reproducibility
    set_seed(42)

    # Create dataset
    X, y = create_multiplication_dataset(num_samples=1000, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # Select single test sample
    X_clean = X_test[0:1]  # Shape: (1, 3)

    print(
        f"\nTest sample: a={X_clean[0, 0]:.4f}, b={X_clean[0, 1]:.4f}, c={X_clean[0, 2]:.4f}"
    )
    print(
        f"Expected output (a*b + c): {X_clean[0, 0] * X_clean[0, 1] + X_clean[0, 2]:.4f}"
    )

    # Initialize and fit model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)
    print("Model fitted successfully")

    # Run causal patching experiment
    results = run_feature_attention_causal_patching_experiment(
        regressor=regressor,
        X_clean=X_clean,
        corrupt_idx=1,  # Corrupt 'b'
        noise_std=1.0,
        noise_seed=42,
        max_layers=None,  # Sweep all layers
        plot=True,
    )

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)
