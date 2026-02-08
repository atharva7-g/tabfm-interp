from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn import TabPFNRegressor


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_simple_dataset(
    num_samples: int = 4000, noise_std: float = 0.0, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Create simple dataset with a + b expression.
    X = [a, b], y = a + b
    Returns: X, y
    """
    rng = np.random.RandomState(seed)
    a = rng.randn(num_samples)
    b = rng.randn(num_samples)
    y = a + b
    X = np.stack([a, b], axis=1).astype(np.float32)

    if noise_std > 0:
        y = y + rng.randn(num_samples) * noise_std

    return X, y.astype(np.float32)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@torch.no_grad()
def get_layer_names(model) -> List[str]:
    return [f"layer_{i}" for i, _ in enumerate(model.transformer_encoder.layers)]


@torch.no_grad()
def run_and_cache_activations(
    regressor: TabPFNRegressor,
    model,
    X_np: np.ndarray,
    target_layers: List[str],
) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    """Capture activations while returning the corresponding predictions."""
    activations: Dict[str, torch.Tensor] = {k: None for k in target_layers}

    def make_hook(name: str):
        def hook(module, inputs, output):
            test_activations = output
            activations[name] = test_activations.detach()

        return hook

    handles = []
    for i, layer in enumerate(model.transformer_encoder.layers):
        name = f"layer_{i}"
        if name in target_layers:
            handles.append(layer.register_forward_hook(make_hook(name)))

    y_pred = regressor.predict(X_np)

    for h in handles:
        h.remove()

    return activations, y_pred


def apply_activation_patching(
    regressor: TabPFNRegressor,
    model,
    X_np: np.ndarray,
    patch_dict: Dict[str, torch.Tensor],
    patch_position: int = 3000,
) -> np.ndarray:
    """Patch activations from patch_dict into the forward pass."""

    def make_patch_hook(name: str):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            patched_activation = patch_dict[name].to(out.device, dtype=out.dtype)
            out_patched = out.clone()
            out_patched[:, patch_position, :-1, :] = patched_activation[:, :, :-1, :]
            return out_patched

        return hook

    handles = []
    for name in patch_dict.keys():
        idx = int(name.split("_")[1])
        handles.append(
            model.transformer_encoder.layers[idx].register_forward_hook(
                make_patch_hook(name)
            )
        )

    y_pred = regressor.predict(X_np)

    for h in handles:
        h.remove()

    return y_pred


def plot_prediction_drift(
    results: List[Dict],
    y_anchor_base: float,
    y_anchor_true: float,
    save_path: Optional[str] = None,
) -> None:
    """Plot how predictions drift from anchor base toward source values across layers."""
    layer_names = sorted(set(r["layer"] for r in results))
    layer_indices = [int(name.split("_")[1]) for name in layer_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by source
    sources = sorted(set(r["source"] for r in results))
    for source in sources:
        source_results = [r for r in results if r["source"] == source]
        source_results = sorted(
            source_results, key=lambda x: int(x["layer"].split("_")[1])
        )

        y_patched = [r["y_anchor_patched"] for r in source_results]
        y_source_true = source_results[0]["y_source_true"]

        ax.plot(
            layer_indices,
            y_patched,
            "o-",
            linewidth=2,
            markersize=6,
            label=f"{source} (target={y_source_true:.3f})",
        )

    # Add reference lines
    ax.axhline(
        y=y_anchor_base,
        color="k",
        linestyle="--",
        alpha=0.5,
        label=f"Anchor base ({y_anchor_base:.3f})",
    )
    ax.axhline(
        y=y_anchor_true,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Anchor true ({y_anchor_true:.3f})",
    )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Predicted Value")
    ax.set_title("Prediction Drift Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(
            save_path.replace(".png", "_drift.png"), dpi=150, bbox_inches="tight"
        )
    plt.show()


def plot_layer_influence(results: List[Dict], save_path: Optional[str] = None) -> None:
    """Plot layer influence (average absolute prediction change)."""
    layer_names = sorted(set(r["layer"] for r in results))
    layer_indices = [int(name.split("_")[1]) for name in layer_names]

    avg_abs_delta = []
    avg_alignment = []

    for name in layer_names:
        layer_results = [r for r in results if r["layer"] == name]
        avg_abs_delta.append(np.mean([abs(r["pred_change"]) for r in layer_results]))
        avg_alignment.append(np.mean([r["alignment"] for r in layer_results]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Influence (|Δ|)
    ax1.plot(
        layer_indices, avg_abs_delta, "o-", linewidth=2, markersize=8, color="blue"
    )
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Average |Δ|")
    ax1.set_title("Layer Influence (Average Absolute Prediction Change)")
    ax1.grid(True, alpha=0.3)

    # Alignment
    colors = ["green" if a > 0 else "red" for a in avg_alignment]
    ax2.bar(layer_indices, avg_alignment, color=colors, alpha=0.7)
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Average Alignment")
    ax2.set_title("Layer Alignment (Positive = Toward Source)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(
            save_path.replace(".png", "_influence.png"), dpi=150, bbox_inches="tight"
        )
    plt.show()


def plot_summary_dashboard(
    results: List[Dict],
    y_anchor_base: float,
    y_anchor_true: float,
    mse: float,
    r2: float,
    save_path: Optional[str] = None,
) -> None:
    """Create a comprehensive dashboard with all metrics."""
    layer_names = sorted(set(r["layer"] for r in results))
    layer_indices = [int(name.split("_")[1]) for name in layer_names]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Prediction drift (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    sources = sorted(set(r["source"] for r in results))
    for source in sources:
        source_results = [r for r in results if r["source"] == source]
        source_results = sorted(
            source_results, key=lambda x: int(x["layer"].split("_")[1])
        )
        y_patched = [r["y_anchor_patched"] for r in source_results]
        y_source_true = source_results[0]["y_source_true"]
        ax1.plot(
            layer_indices,
            y_patched,
            "o-",
            linewidth=2,
            markersize=5,
            label=f"{source} (target={y_source_true:.3f})",
        )
    ax1.axhline(
        y=y_anchor_base, color="k", linestyle="--", alpha=0.5, label="Anchor base"
    )
    ax1.axhline(
        y=y_anchor_true, color="r", linestyle="--", alpha=0.5, label="Anchor true"
    )
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Predicted Value")
    ax1.set_title("Prediction Drift Across Layers")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # 2. Layer influence (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    avg_abs_delta = []
    for name in layer_names:
        layer_results = [r for r in results if r["layer"] == name]
        avg_abs_delta.append(np.mean([abs(r["pred_change"]) for r in layer_results]))
    ax2.plot(
        layer_indices, avg_abs_delta, "o-", linewidth=2, markersize=8, color="blue"
    )
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Average |Δ|")
    ax2.set_title("Layer Influence")
    ax2.grid(True, alpha=0.3)

    # 3. Layer alignment (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    avg_alignment = []
    for name in layer_names:
        layer_results = [r for r in results if r["layer"] == name]
        avg_alignment.append(np.mean([r["alignment"] for r in layer_results]))
    colors = ["green" if a > 0 else "red" for a in avg_alignment]
    ax3.bar(layer_indices, avg_alignment, color=colors, alpha=0.7)
    ax3.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax3.set_xlabel("Layer Index")
    ax3.set_ylabel("Average Alignment")
    ax3.set_title("Layer Alignment")
    ax3.grid(True, alpha=0.3)

    # 4. Error delta distribution (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    for name in layer_names:
        layer_results = [r for r in results if r["layer"] == name]
        error_deltas = [r["error_delta"] for r in layer_results]
        ax4.scatter(
            [int(name.split("_")[1])] * len(error_deltas), error_deltas, alpha=0.5
        )
    ax4.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Layer Index")
    ax4.set_ylabel("Error Delta")
    ax4.set_title("Error Change (Negative = Better)")
    ax4.grid(True, alpha=0.3)

    # 5. Summary stats (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    # Calculate summary statistics
    all_deltas = [abs(r["pred_change"]) for r in results]
    all_alignments = [r["alignment"] for r in results]
    positive_align = sum(1 for a in all_alignments if a > 0)

    summary_text = f"""
    SUMMARY STATISTICS
    =================
    
    Model Performance:
      MSE: {mse:.4f}
      R²:  {r2:.4f}
    
    Anchor Sample:
      True:      {y_anchor_true:.4f}
      Predicted: {y_anchor_base:.4f}
    
    Patching Results:
      Avg |Δ|:     {np.mean(all_deltas):.5f}
      Max |Δ|:     {np.max(all_deltas):.5f}
      Avg Align:   {np.mean(all_alignments):.5f}
      Pos Align:   {positive_align}/{len(all_alignments)} ({100 * positive_align / len(all_alignments):.1f}%)
    """

    ax5.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Dashboard saved to {save_path}")
    plt.show()


def run_activation_patching_experiment(
    num_samples: int = 6000,
    noise_std: float = 0.0,
    num_sources: int = 5,
    seed: int = 100,
    patch_position: int = 3000,
    plot: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[List[Dict], float, float]:
    """
    Run activation patching experiment with visualization.

    Args:
        num_samples: Total number of samples to generate
        noise_std: Standard deviation of noise to add
        num_sources: Number of source samples to patch
        seed: Random seed
        patch_position: Position in sequence to patch (default: 3000)
        plot: Whether to generate plots
        save_path: Path to save plots (optional)

    Returns:
        Tuple of (all_results, mse, r2_score)
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Train TabPFN on simple a + b dataset
    X, y = create_simple_dataset(num_samples=num_samples, noise_std=noise_std, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"TabPFN MSE on held-out split: {mse:.4f}, R²: {r2:.4f}")

    model = regressor.model_
    layer_names = get_layer_names(model)

    # 2) Select anchor and source samples
    anchor_idx = 0
    source_indices = np.arange(1, num_sources + 1)

    X_anchor = X_test[anchor_idx : anchor_idx + 1].astype(np.float32)
    y_anchor_true = y_test[anchor_idx]

    X_sources = X_test[source_indices].astype(np.float32)
    y_sources_true = y_test[source_indices]

    X_eval = np.concatenate([X_anchor, X_sources], axis=0)

    # 3) Cache activations
    print("\n" + "=" * 50)
    print("CACHING ACTIVATIONS")
    print("=" * 50)
    acts_batch, y_eval = run_and_cache_activations(
        regressor, model, X_eval, layer_names
    )

    y_anchor_base = float(y_eval[0])
    print(f"\nAnchor sample: a={X_anchor[0, 0]:.4f}, b={X_anchor[0, 1]:.4f}")
    print(f"  True y: {y_anchor_true:.4f}, Predicted: {y_anchor_base:.4f}")

    # 4) Run patching sweeps
    print("\n" + "=" * 50)
    print("PATCHING SWEEPS")
    print("=" * 50)

    all_results = []

    for source_idx, source_num in enumerate(source_indices):
        print(f"\n--- Patching Source {source_num} into Anchor ---")
        source_label = f"Source_{source_num}"
        y_source_true_val = y_sources_true[source_idx]

        # Extract activations for this source
        acts_source_single = {}
        for layer_name in layer_names:
            all_activations_at_layer = acts_batch[layer_name]
            batch_source_idx = source_idx + 1
            acts_source_single[layer_name] = all_activations_at_layer[
                :,
                patch_position + batch_source_idx : patch_position
                + batch_source_idx
                + 1,
                ...,
            ]

        # Patch at each layer
        for name in layer_names:
            y_anchor_patched = apply_activation_patching(
                regressor,
                model,
                X_eval,
                {name: acts_source_single[name]},
                patch_position,
            )
            y_anchor_patched_val = y_anchor_patched[0]

            pred_change = y_anchor_patched_val - y_anchor_base
            target_change = y_source_true_val - y_anchor_base
            alignment = pred_change * target_change

            error_anchor = np.abs(y_anchor_patched_val - y_anchor_true)
            error_base = np.abs(y_anchor_base - y_anchor_true)
            error_delta = error_anchor - error_base

            all_results.append(
                {
                    "layer": name,
                    "source": source_label,
                    "y_anchor_base": float(y_anchor_base),
                    "y_anchor_patched": float(y_anchor_patched_val),
                    "y_source_true": float(y_source_true_val),
                    "pred_change": float(pred_change),
                    "target_change": float(target_change),
                    "alignment": float(alignment),
                    "error_delta": float(error_delta),
                }
            )

    # 5) Generate visualizations
    if plot:
        print("\n" + "=" * 50)
        print("GENERATING VISUALIZATIONS")
        print("=" * 50)

        # Individual plots
        plot_prediction_drift(all_results, y_anchor_base, y_anchor_true, save_path)
        plot_layer_influence(all_results, save_path)

        # Combined dashboard
        plot_summary_dashboard(
            all_results, y_anchor_base, y_anchor_true, mse, r2, save_path
        )

    # 6) Print summary
    print("\n" + "=" * 50)
    print("SUMMARY ACROSS ALL SOURCES")
    print("=" * 50)
    print(
        f"{'Layer':<10} {'Avg |Δ|':<12} {'Max |Δ|':<12} {'Avg Align':<12} {'Pos Align':<10}"
    )
    print("-" * 60)
    for name in layer_names:
        layer_results = [r for r in all_results if r["layer"] == name]
        avg_abs_delta = np.mean([abs(r["pred_change"]) for r in layer_results])
        max_abs_delta = np.max([abs(r["pred_change"]) for r in layer_results])
        avg_align = np.mean([r["alignment"] for r in layer_results])
        pos_align_count = sum(1 for r in layer_results if r["alignment"] > 0)
        print(
            f"{name:<10} {avg_abs_delta:<12.5f} {max_abs_delta:<12.5f} {avg_align:<12.5f} {pos_align_count:<10}"
        )

    return all_results, mse, r2


if __name__ == "__main__":
    print("=" * 60)
    print("Activation Patching Regression with Visualization")
    print("=" * 60)

    results, mse, r2 = run_activation_patching_experiment(
        num_samples=6000,
        noise_std=0.0,
        num_sources=5,
        seed=100,
        patch_position=3000,
        plot=True,
        save_path=None,
    )

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)
