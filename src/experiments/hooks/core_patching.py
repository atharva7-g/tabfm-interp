from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor
from src.utils.model_inspector import ModelInspector
from src.utils.shape_inspector import ShapeInspector


def inspect_regressor_model(
    regressor: TabPFNRegressor,
    name: str = "tabpfn_model",
    max_depth: Optional[int] = None,
    print_summary: bool = False,
) -> ModelInspector:
    model = regressor.model_
    inspector = ModelInspector(name, max_depth)

    with inspector:
        for mod_name, module in model.named_modules():
            inspector.record_module(mod_name, module)

    if print_summary:
        inspector._print_summary()

    return inspector


def create_corrupted_input(
    X_clean: np.ndarray,
    corrupt_idx: Union[int, List[int]] = 1,
    noise_std: float = 1.0,
    seed: int = 42,
    corruption_mode: str = "gaussian_replace",
    corruption_strength: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X_corrupt = X_clean.copy()
    num_samples = X_clean.shape[0]

    if isinstance(corrupt_idx, int):
        indices = [corrupt_idx]
    else:
        indices = list(corrupt_idx)

    if len(indices) == 0:
        raise ValueError("corrupt_idx list cannot be empty")

    n_features = X_clean.shape[1]
    if not all(0 <= idx < n_features for idx in indices):
        raise ValueError(
            f"corrupt_idx must be in [0, {n_features - 1}] for input with {n_features} features"
        )

    if corruption_strength < 0:
        raise ValueError("corruption_strength must be >= 0")

    cols = np.array(indices, dtype=int)
    selected = X_corrupt[:, cols]

    if corruption_mode == "gaussian_replace":
        X_corrupt[:, cols] = rng.normal(
            0.0,
            noise_std * corruption_strength,
            (num_samples, len(indices)),
        )
    elif corruption_mode == "gaussian_add":
        X_corrupt[:, cols] = selected + rng.normal(
            0.0,
            noise_std * corruption_strength,
            (num_samples, len(indices)),
        )
    elif corruption_mode == "mean_shift":
        X_corrupt[:, cols] = selected + (noise_std * corruption_strength)
    elif corruption_mode == "scale":
        X_corrupt[:, cols] = selected * corruption_strength
    elif corruption_mode == "sign_flip":
        X_corrupt[:, cols] = -selected * corruption_strength
    elif corruption_mode == "fixed":
        X_corrupt[:, cols] = corruption_strength
    elif corruption_mode == "zero":
        X_corrupt[:, cols] = 0.0
    elif corruption_mode == "permute":
        permutation = rng.permutation(num_samples)
        X_corrupt[:, cols] = selected[permutation]
    else:
        raise ValueError(
            "Unknown corruption_mode. "
            "Use one of: gaussian_replace, gaussian_add, mean_shift, scale, "
            "sign_flip, fixed, zero, permute"
        )

    return X_corrupt


def create_cache_hook(
    cached_activation: Dict[str, torch.Tensor],
    layer_name: str,
) -> Callable:
    inspector = ShapeInspector(f"cache_hook_{layer_name}")

    def hook(module, inputs, output):
        if isinstance(output, (tuple, list)):
            output_tensor = output[0]
        else:
            output_tensor = output
        cached_activation[layer_name] = output_tensor.detach().clone()

    return hook


def create_patch_hook(
    cached_activation: torch.Tensor,
    patch_indices: Union[int, List[int]],
    patch_dim: Optional[int] = None,
) -> Callable:
    """Create a hook function that patches activations from clean to corrupted runs."""
    if patch_dim is None:
        print("Patch dimension is null. Patching full layer.")

        def full_layer_hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output_list = [cached_activation.clone()]
                output_list.extend(output[1:])
                return tuple(output_list)
            return cached_activation.clone()

        return full_layer_hook

    # patch_dim: 1=tokens (565), 2=attention heads (4)
    # patch_indices: which token(s) or head(s) to patch (int or list of ints)

    # Normalize to list for consistent handling
    if isinstance(patch_indices, int):
        indices_list = [patch_indices]
    else:
        indices_list = list(patch_indices)

    # Validate all indices are within bounds
    dim_size = cached_activation.shape[patch_dim]
    for idx in indices_list:
        if idx < 0 or idx >= dim_size:
            raise ValueError(
                f"patch_indices must all be in range [0, {dim_size - 1}], got {idx}"
            )

    inspector = ShapeInspector(f"patch_hook_dim{patch_dim}_idx{indices_list}")

    def hook(module, inputs, output):
        if isinstance(output, (tuple, list)):
            output_tensor = output[0]
        else:
            output_tensor = output
        # inspector.record(output_tensor)
        modified_output = output_tensor.clone()
        if patch_dim == 1:  # Patch tokens across all heads
            for idx in indices_list:
                modified_output[:, idx, :, :] = cached_activation[:, idx, :, :]
        elif patch_dim == 2:  # Patch heads across all tokens
            for idx in indices_list:
                modified_output[:, :, idx, :] = cached_activation[:, :, idx, :]
        return modified_output

    return hook


def sweep_layers(
    regressor: TabPFNRegressor,
    X_clean: np.ndarray,
    X_corrupt: np.ndarray,
    corrupt_idx: Union[int, List[int]],
    n_train_samples: int,
    patch_indices: Union[int, List[int]],
    patch_dim: Optional[int] = 2,
    max_layers: Optional[int] = None,
    ratio_epsilon: float = 0.05,
    ratio_threshold: Optional[float] = None,
    y_scale: Optional[float] = None,
    metric_mode: str = "regime",
) -> List[Dict[str, float]]:
    model = regressor.model_
    total_layers = len(model.transformer_encoder.layers)  # type: ignore
    num_layers = max_layers if max_layers is not None else total_layers
    num_layers = min(num_layers, total_layers)
    results = []
    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}/{num_layers - 1}...")
        result = run_single_layer_patching(
            regressor,
            X_clean,
            X_corrupt,
            corrupt_idx,
            layer_idx,
            n_train_samples,
            patch_indices,
            patch_dim,
            ratio_epsilon,
            ratio_threshold,
            y_scale,
            metric_mode,
        )
        results.append(result)
    return results


def run_single_layer_patching(
    regressor: TabPFNRegressor,
    X_clean: np.ndarray,
    X_corrupt: np.ndarray,
    corrupt_idx: Union[int, List[int]],
    layer_idx: int,
    n_train_samples: int,
    patch_indices: Union[int, List[int]],
    patch_dim: Optional[int] = 2,
    ratio_epsilon: float = 0.05,
    ratio_threshold: Optional[float] = None,
    y_scale: Optional[float] = None,
    metric_mode: str = "regime",
) -> Dict[str, float]:
    model = regressor.model_
    layer_name = f"layer_{layer_idx}"
    cached_activation = {}
    layer = model.transformer_encoder.layers[layer_idx]  # type: ignore
    attention_module = layer.self_attn_between_features  # type: ignore
    cache_hook_fn = create_cache_hook(cached_activation, layer_name)
    cache_handle = attention_module.register_forward_hook(cache_hook_fn)
    with torch.no_grad():
        y_clean = regressor.predict(X_clean)
    cache_handle.remove()
    clean_activation = cached_activation[layer_name]
    with torch.no_grad():
        y_corrupt = regressor.predict(X_corrupt)
    patch_hook_fn = create_patch_hook(clean_activation, patch_indices, patch_dim)
    patch_handle = attention_module.register_forward_hook(patch_hook_fn)
    with torch.no_grad():
        y_patched = regressor.predict(X_corrupt)
    patch_handle.remove()
    y_clean_arr = np.asarray(y_clean, dtype=np.float64).reshape(-1)
    y_corrupt_arr = np.asarray(y_corrupt, dtype=np.float64).reshape(-1)
    y_patched_arr = np.asarray(y_patched, dtype=np.float64).reshape(-1)

    delta_gap_arr = y_clean_arr - y_corrupt_arr
    delta_restoration_arr = y_patched_arr - y_corrupt_arr
    delta_residual_arr = y_clean_arr - y_patched_arr

    y_clean_val = float(np.mean(y_clean_arr))
    y_corrupt_val = float(np.mean(y_corrupt_arr))
    y_patched_val = float(np.mean(y_patched_arr))

    restoration = float(np.mean(delta_restoration_arr))
    restoration_abs_mean = float(np.mean(np.abs(delta_restoration_arr)))
    clean_corrupt_diff = float(np.mean(delta_gap_arr))
    clean_corrupt_gap_abs_mean = float(np.mean(np.abs(delta_gap_arr)))
    residual_abs_mean = float(np.mean(np.abs(delta_residual_arr)))

    if abs(clean_corrupt_diff) > 1e-10:
        recovery_ratio = restoration / clean_corrupt_diff
    else:
        recovery_ratio = 0.0

    ratio_threshold_val = (
        float(ratio_threshold) if ratio_threshold is not None else float(ratio_epsilon)
    )
    ratio_valid_signed = abs(clean_corrupt_diff) >= ratio_threshold_val
    ratio_valid_abs = clean_corrupt_gap_abs_mean >= ratio_threshold_val

    if ratio_valid_signed:
        recovery_fractional_signed = restoration / clean_corrupt_diff
    else:
        recovery_fractional_signed = None

    if ratio_valid_abs:
        recovery_fractional_abs = restoration_abs_mean / clean_corrupt_gap_abs_mean
    else:
        recovery_fractional_abs = None

    safe_denominator = max(abs(clean_corrupt_diff), ratio_epsilon)
    recovery_ratio_stable = restoration / safe_denominator
    recovery_score = 1.0 - (abs(y_patched_val - y_clean_val) / safe_denominator)
    recovery_score = float(np.clip(recovery_score, -1.0, 1.0))

    if y_scale is not None:
        y_scale_val = float(max(float(y_scale), 1e-12))
    else:
        y_scale_val = float(max(np.std(y_clean_arr), 1e-12))

    restoration_sigma = restoration_abs_mean / y_scale_val
    residual_sigma = residual_abs_mean / y_scale_val

    low_gap_regime = not ratio_valid_signed
    if ratio_valid_signed and recovery_fractional_signed is not None:
        recovery_score_regime = recovery_fractional_signed
    else:
        recovery_score_regime = restoration_sigma

    if metric_mode == "legacy":
        recovery_primary = recovery_score
        recovery_primary_metric = "recovery_score"
    else:
        recovery_primary = recovery_score_regime
        recovery_primary_metric = (
            "restoration_sigma" if low_gap_regime else "recovery_fractional_signed"
        )

    return {
        "y_clean": y_clean_val,
        "y_corrupt": y_corrupt_val,
        "y_patched": y_patched_val,
        "restoration": restoration,
        "restoration_abs_mean": restoration_abs_mean,
        "clean_corrupt_gap_abs_mean": clean_corrupt_gap_abs_mean,
        "residual_abs_mean": residual_abs_mean,
        "recovery_ratio": recovery_ratio,
        "recovery_ratio_stable": recovery_ratio_stable,
        "recovery_score": recovery_score,
        "recovery_primary": recovery_primary,
        "recovery_primary_metric": recovery_primary_metric,
        "recovery_score_regime": recovery_score_regime,
        "recovery_fractional_signed": recovery_fractional_signed,
        "recovery_fractional_abs": recovery_fractional_abs,
        "ratio_threshold": ratio_threshold_val,
        "ratio_valid_signed": ratio_valid_signed,
        "ratio_valid_abs": ratio_valid_abs,
        "low_gap_regime": low_gap_regime,
        "restoration_sigma": restoration_sigma,
        "residual_sigma": residual_sigma,
        "y_scale": y_scale_val,
        "clean_corrupt_diff": clean_corrupt_diff,
        "safe_denominator": safe_denominator,
        "n_eval_samples": int(y_clean_arr.size),
        "layer_idx": layer_idx,
    }


def create_steer_hook(
    direction: torch.Tensor,
    steer_indices: Union[int, List[int]],
    steer_dim: Optional[int] = None,
    alpha: float = 1.0,
) -> Callable:
    """Create a hook function that adds a direction vector to activations.

    Args:
        direction: The direction tensor to add (shape matches activation)
        steer_indices: Which indices (heads/tokens) to steer
        steer_dim: Dimension to steer (1=tokens, 2=heads, None=full layer)
        alpha: Scaling factor for the steering direction
    """
    if steer_dim is None:
        print("Steer dimension is None. Steering full layer.")

        def full_layer_steer_hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output_tensor = output[0]
            else:
                output_tensor = output
            return output_tensor + alpha * direction

        return full_layer_steer_hook

    if isinstance(steer_indices, int):
        indices_list = [steer_indices]
    else:
        indices_list = list(steer_indices)

    dim_size = direction.shape[steer_dim]
    for idx in indices_list:
        if idx < 0 or idx >= dim_size:
            raise ValueError(
                f"steer_indices must all be in range [0, {dim_size - 1}], got {idx}"
            )

    def hook(module, inputs, output):
        if isinstance(output, (tuple, list)):
            output_tensor = output[0]
        else:
            output_tensor = output
        modified_output = output_tensor.clone()

        if steer_dim == 1:
            for idx in indices_list:
                modified_output[:, idx, :, :] += alpha * direction[:, idx, :, :]
        elif steer_dim == 2:
            for idx in indices_list:
                modified_output[:, :, idx, :] += alpha * direction[:, :, idx, :]

        return modified_output

    return hook


def run_single_layer_steering(
    regressor: TabPFNRegressor,
    X: np.ndarray,
    layer_idx: int,
    direction: torch.Tensor,
    steer_indices: Union[int, List[int]],
    steer_dim: Optional[int] = 2,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """Run steering on a single layer and return the effect."""
    model = regressor.model_
    layer = model.transformer_encoder.layers[layer_idx]
    attention_module = layer.self_attn_between_features

    steer_hook_fn = create_steer_hook(direction, steer_indices, steer_dim, alpha)
    steer_handle = attention_module.register_forward_hook(steer_hook_fn)

    with torch.no_grad():
        y_steered = regressor.predict(X)

    steer_handle.remove()

    with torch.no_grad():
        y_normal = regressor.predict(X)

    y_steered_val = (
        float(y_steered[0]) if len(y_steered.shape) > 0 else float(y_steered)
    )
    y_normal_val = float(y_normal[0]) if len(y_normal.shape) > 0 else float(y_normal)

    steering_effect = y_steered_val - y_normal_val

    return {
        "y_normal": y_normal_val,
        "y_steered": y_steered_val,
        "steering_effect": steering_effect,
        "alpha": alpha,
        "layer_idx": layer_idx,
    }


def sweep_steering_layers(
    regressor: TabPFNRegressor,
    X: np.ndarray,
    direction: torch.Tensor,
    steer_indices: Union[int, List[int]],
    steer_dim: Optional[int] = 2,
    alpha: float = 1.0,
    max_layers: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Sweep through layers, applying steering at each."""
    model = regressor.model_
    total_layers = len(model.transformer_encoder.layers)
    num_layers = max_layers if max_layers is not None else total_layers
    num_layers = min(num_layers, total_layers)
    results = []

    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}/{num_layers - 1}...")
        result = run_single_layer_steering(
            regressor,
            X,
            layer_idx,
            direction,
            steer_indices,
            steer_dim,
            alpha,
        )
        results.append(result)

    return results


def create_direction_from_difference(
    activation_high: torch.Tensor,
    activation_low: torch.Tensor,
    normalize: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a direction vector from the difference between two activation states.

    Args:
        activation_high: Activation from "high" state (e.g., large input values)
        activation_low: Activation from "low" state (e.g., small input values)
        normalize: Whether to normalize the direction tensor
        device: Device to place the direction tensor on

    Returns:
        Direction tensor (difference or normalized difference)
    """
    direction = activation_high - activation_low

    if normalize:
        direction = direction / (direction.norm() + 1e-8)

    if device is not None:
        direction = direction.to(device)

    return direction


def create_random_direction(
    shape: Tuple[int, ...],
    seed: int = 42,
    normalize: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a random direction vector.

    Args:
        shape: Shape of the direction tensor
        seed: Random seed
        normalize: Whether to normalize the direction tensor
        device: Device to place the direction tensor on

    Returns:
        Random direction tensor
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    direction = torch.randn(shape, generator=rng)

    if normalize:
        direction = direction / (direction.norm() + 1e-8)

    if device is not None:
        direction = direction.to(device)

    return direction


def plot_steering_results(
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
    n_train_samples: int,
    patch_indices: Union[int, List[int]],
    patch_dim: Optional[int] = 2,
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
    results = sweep_layers(
        regressor=regressor,
        X_clean=X_clean,
        X_corrupt=X_corrupt,
        corrupt_idx=corrupt_idx,
        n_train_samples=n_train_samples,
        patch_indices=patch_indices,
        patch_dim=patch_dim,
        max_layers=max_layers,
    )
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
        f"\nTest sample: a={X_clean[0, 0]:.4f}, b={X_clean[0, 1]:.4f}, c={X_clean[0, 2]:.4f}"  # type: ignore
    )
    print(
        f"Expected output (a*b + c): {X_clean[0, 0] * X_clean[0, 1] + X_clean[0, 2]:.4f}"  # type: ignore
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)
    print("Model fitted successfully")
    print("\n" + "=" * 60)
    print("Inspecting Model Structure")
    print("=" * 60)
    # # # # inspector = inspect_regressor_model(regressor, max_depth=4)
    # # # # hookable = inspector.get_hookable_modules()
    # # # print(f"\nFound {len(hookable)} hookable modules")
    # # attn_modules = inspector.get_modules_by_type("SelfAttention")
    # print(f"Found {len(attn_modules)} SelfAttention modules")
    # print("\n" + "=" * 60)
    print("Running Activation Patching Experiment")
    print("=" * 60)
    results = run_feature_attention_causal_patching_experiment(
        regressor=regressor,
        X_clean=X_clean,  # type: ignore
        corrupt_idx=1,
        n_train_samples=len(X_train),
        patch_indices=[0, 1, 2],  # Patch multiple attention heads
        patch_dim=2,
        noise_std=1.0,
        noise_seed=42,
        max_layers=None,
        plot=True,
    )
    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)
