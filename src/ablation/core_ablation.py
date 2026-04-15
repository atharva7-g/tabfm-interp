from typing import Callable, Dict, List, Optional, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor


def create_ablate_hook(
    ablate_indices: Union[int, List[int]],
    ablate_dim: Optional[int] = None,
    ablation_type: str = "zero",
) -> Callable:
    """Create a hook function that ablates (zeros or replaces with mean) activations."""
    if ablate_dim is None:
        print("Ablation dimension is null. Ablating full layer.")

        def full_layer_hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output_tensor = output[0]
            else:
                output_tensor = output
            if ablation_type == "zero":
                return torch.zeros_like(output_tensor)
            else:
                return torch.full_like(output_tensor, output_tensor.mean())

        return full_layer_hook

    if isinstance(ablate_indices, int):
        indices_list = [ablate_indices]
    else:
        indices_list = list(ablate_indices)

    def hook(module, inputs, output):
        if isinstance(output, (tuple, list)):
            output_tensor = output[0]
        else:
            output_tensor = output
        modified_output = output_tensor.clone()

        if ablate_dim == 1:
            for idx in indices_list:
                if ablation_type == "zero":
                    modified_output[:, idx, :, :] = 0.0
                else:
                    mean_val = output_tensor[:, idx, :, :].mean()
                    modified_output[:, idx, :, :] = mean_val
        elif ablate_dim == 2:
            for idx in indices_list:
                if ablation_type == "zero":
                    modified_output[:, :, idx, :] = 0.0
                else:
                    mean_val = output_tensor[:, :, idx, :].mean()
                    modified_output[:, :, idx, :] = mean_val

        return modified_output

    return hook


def sweep_layers_for_ablation(
    regressor: TabPFNRegressor,
    X: np.ndarray,
    ablate_indices: Union[int, List[int]],
    ablate_dim: Optional[int] = 2,
    ablation_type: str = "zero",
    max_layers: Optional[int] = None,
    ratio_epsilon: float = 0.05,
) -> List[Dict[str, float]]:
    model = regressor.model_
    total_layers = len(model.transformer_encoder.layers)
    num_layers = max_layers if max_layers is not None else total_layers
    num_layers = min(num_layers, total_layers)
    results = []

    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}/{num_layers - 1}...")
        result = run_single_layer_ablation(
            regressor,
            X,
            layer_idx,
            ablate_indices,
            ablate_dim,
            ablation_type,
            ratio_epsilon,
        )
        results.append(result)
    return results


def run_single_layer_ablation(
    regressor: TabPFNRegressor,
    X: np.ndarray,
    layer_idx: int,
    ablate_indices: Union[int, List[int]],
    ablate_dim: Optional[int] = 2,
    ablation_type: str = "zero",
    ratio_epsilon: float = 0.05,
) -> Dict[str, float]:
    model = regressor.model_
    layer = model.transformer_encoder.layers[layer_idx]
    attention_module = layer.self_attn_between_features

    ablate_hook_fn = create_ablate_hook(ablate_indices, ablate_dim, ablation_type)
    ablate_handle = attention_module.register_forward_hook(ablate_hook_fn)

    with torch.no_grad():
        y_ablated = regressor.predict(X)

    ablate_handle.remove()

    with torch.no_grad():
        y_normal = regressor.predict(X)

    y_ablated_arr = np.asarray(y_ablated, dtype=np.float64).reshape(-1)
    y_normal_arr = np.asarray(y_normal, dtype=np.float64).reshape(-1)

    y_ablated_val = float(np.mean(y_ablated_arr))
    y_normal_val = float(np.mean(y_normal_arr))

    delta_arr = y_normal_arr - y_ablated_arr
    ablation_effect = float(np.mean(delta_arr))
    ablation_effect_abs_mean = float(np.mean(np.abs(delta_arr)))

    if abs(y_normal_val) > 1e-10:
        ablation_ratio = ablation_effect / abs(y_normal_val)
    else:
        ablation_ratio = 0.0

    normal_scale = float(np.mean(np.abs(y_normal_arr)))
    stable_denominator = max(normal_scale, ratio_epsilon)
    ablation_ratio_stable = ablation_effect / stable_denominator
    ablation_ratio_stable_abs = ablation_effect_abs_mean / stable_denominator

    return {
        "y_normal": y_normal_val,
        "y_ablated": y_ablated_val,
        "ablation_effect": ablation_effect,
        "ablation_effect_abs_mean": ablation_effect_abs_mean,
        "ablation_ratio": ablation_ratio,
        "ablation_ratio_stable": ablation_ratio_stable,
        "ablation_ratio_stable_abs": ablation_ratio_stable_abs,
        "normal_scale": normal_scale,
        "ratio_epsilon": ratio_epsilon,
        "stable_denominator": stable_denominator,
        "layer_idx": layer_idx,
        "n_eval_samples": int(y_normal_arr.size),
    }


def plot_ablation_results(
    results: List[Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    layer_indices = [r["layer_idx"] for r in results]
    ablation_effects = [r["ablation_effect"] for r in results]
    ablation_ratios = [r["ablation_ratio"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(layer_indices, ablation_effects, "o-", linewidth=2, markersize=8)
    ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Ablation Effect")
    ax1.set_title("Ablation: Effect by Layer")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        layer_indices, ablation_ratios, "o-", linewidth=2, markersize=8, color="orange"
    )
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Ablation Ratio")
    ax2.set_title("Ablation: Ratio by Layer")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    from src.utils.utils import create_multiplication_dataset, set_seed
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("Feature Attention Ablation Demo")
    print("=" * 60)
    set_seed(42)
    X, y = create_multiplication_dataset(num_samples=1000, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    X_test_sample = X_test[0:1]
    print(
        f"\nTest sample: a={X_test_sample[0, 0]:.4f}, b={X_test_sample[0, 1]:.4f}, c={X_test_sample[0, 2]:.4f}"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)
    print("Model fitted successfully")
    print("\n" + "=" * 60)
    print("Running Ablation Experiment")
    print("=" * 60)
    results = sweep_layers_for_ablation(
        regressor=regressor,
        X=X_test_sample,
        ablate_indices=[0, 1, 2],
        ablate_dim=2,
        ablation_type="zero",
        max_layers=None,
    )
    plot_ablation_results(results)
    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)
