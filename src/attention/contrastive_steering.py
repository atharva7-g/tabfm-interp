#!/usr/bin/env python3
"""
Contrastive steering at the Head 2 / Layer 0 site.

On the Multiplication dataset (y = a*b + c), construct matched batches:
  - Multiplicative: b ~ N(0,1)  (full a*b + c computation active)
  - Additive:       b = 0        (task degenerates to y = c)

Extract the mean Head 2 MHA output at Layer 0 for each condition and compute
delta = mean(h2_mult) - mean(h2_add).  Then steer additive runs by injecting
alpha * delta at the Head 2 MHA output, sweeping alpha in [0, 10].

If steering shifts predictions toward a*b + c, it demonstrates that the
computation Head 2 performs at Layer 0 encodes steerable concept information.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.synthetic import create_dataset

SEED = 42
N_EVAL = 32
ALPHA_MIN = 0.0
ALPHA_MAX = 10.0
ALPHA_STEP = 0.5
HEAD_IDX = 2
TARGET_LAYER = 0
CONTROL_LAYERS = [1, 12]
DPI = 200


def _fit_model(device: str) -> Tuple[TabPFNRegressor, np.ndarray, np.ndarray]:
    X, y = create_dataset("multiplication", num_samples=1000, seed=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=SEED
    )
    model = TabPFNRegressor(device=device, n_estimators=1)
    model.fit(X_train, y_train)
    return model, X_test[:N_EVAL], y_test[:N_EVAL]


def _make_additive_batch(X_mult: np.ndarray) -> np.ndarray:
    X_add = X_mult.copy()
    X_add[:, 1] = 0.0
    return X_add


def _get_mha(model: TabPFNRegressor, layer_idx: int) -> MultiHeadAttention:
    return model.model_.transformer_encoder.layers[layer_idx].self_attn_between_features


def _get_num_layers(model: TabPFNRegressor) -> int:
    return len(model.model_.transformer_encoder.layers)


def _capture_head_outputs(
    model: TabPFNRegressor,
    X: np.ndarray,
    device: str,
    layer_idx: int,
    head_idx: int,
) -> torch.Tensor:
    """Run forward pass, capture head_idx MHA output at layer_idx.

    Returns tensor of shape [batch, seq, d_v] (averaged over items).
    """
    mha = _get_mha(model, layer_idx)
    original_compute = mha._compute
    captured = {}

    def capturing_compute(x, x_kv, k_cache, v_cache, kv_cache, **kwargs):
        q, k, v, kv, qkv = mha.compute_qkv(
            x, x_kv, k_cache, v_cache, kv_cache,
            cache_kv=kwargs.get("cache_kv", False),
            use_cached_kv=kwargs.get("use_cached_kv", False),
            reuse_first_head_kv=kwargs.get("reuse_first_head_kv", False),
        )
        head_outputs = MultiHeadAttention.compute_attention_heads(
            q, k, v, kv, qkv, mha.dropout_p, mha.softmax_scale
        )
        captured["head_outputs"] = head_outputs.detach().clone()
        captured["input"] = x.detach().clone()
        return torch.einsum("... h d, h d s -> ... s", head_outputs, mha._w_out)

    mha._compute = capturing_compute
    with torch.no_grad():
        model.predict(X)
    mha._compute = original_compute

    ho = captured["head_outputs"]
    return ho[:, :, head_idx, :]


def _compute_delta(
    model: TabPFNRegressor,
    X_mult: np.ndarray,
    X_add: np.ndarray,
    device: str,
    layer_idx: int,
    head_idx: int,
) -> torch.Tensor:
    """Compute steering direction: mean(h_mult) - mean(h_add)."""
    h_mult = _capture_head_outputs(model, X_mult, device, layer_idx, head_idx)
    h_add = _capture_head_outputs(model, X_add, device, layer_idx, head_idx)
    delta = h_mult.mean(dim=0) - h_add.mean(dim=0)
    return delta


def _run_with_steering(
    model: TabPFNRegressor,
    X: np.ndarray,
    device: str,
    layer_idx: int,
    head_idx: int,
    delta: torch.Tensor,
    alpha: float,
) -> np.ndarray:
    """Run forward pass with alpha * delta injected into head_idx at layer_idx."""
    if alpha == 0.0:
        with torch.no_grad():
            return model.predict(X)

    mha = _get_mha(model, layer_idx)
    original_compute = mha._compute

    def steering_compute(x, x_kv, k_cache, v_cache, kv_cache, **kwargs):
        q, k, v, kv, qkv = mha.compute_qkv(
            x, x_kv, k_cache, v_cache, kv_cache,
            cache_kv=kwargs.get("cache_kv", False),
            use_cached_kv=kwargs.get("use_cached_kv", False),
            reuse_first_head_kv=kwargs.get("reuse_first_head_kv", False),
        )
        head_outputs = MultiHeadAttention.compute_attention_heads(
            q, k, v, kv, qkv, mha.dropout_p, mha.softmax_scale
        )
        head_outputs[:, :, head_idx, :] = (
            head_outputs[:, :, head_idx, :] + alpha * delta.to(head_outputs.device)
        )
        return torch.einsum("... h d, h d s -> ... s", head_outputs, mha._w_out)

    mha._compute = steering_compute
    with torch.no_grad():
        preds = model.predict(X)
    mha._compute = original_compute
    return preds


def _compute_recovery(
    steered_y: np.ndarray,
    target_y: np.ndarray,
    baseline_y: np.ndarray,
) -> Tuple[float, float]:
    """Compute mean recovery fraction and std across samples."""
    gaps = target_y - baseline_y
    shifts = steered_y - baseline_y
    with np.errstate(divide="ignore", invalid="ignore"):
        per_sample = np.where(
            np.abs(gaps) > 1e-6,
            shifts / gaps,
            np.nan,
        )
    valid = per_sample[~np.isnan(per_sample)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    return float(np.mean(valid)), float(np.std(valid))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Fitting model on multiplication dataset...")
    model, X_test, y_test = _fit_model(device)
    X_mult = X_test[:N_EVAL]
    X_add = _make_additive_batch(X_mult)

    a = X_mult[:, 0]
    b = X_mult[:, 1]
    c = X_mult[:, 2]
    y_target = a * b + c
    y_baseline = c

    print(f"N={N_EVAL} samples, head={HEAD_IDX}, layers to test: {[TARGET_LAYER] + CONTROL_LAYERS}")

    with torch.no_grad():
        y_pred_mult = model.predict(X_mult)
        y_pred_add = model.predict(X_add)
    print(f"Multiplicative predictions: mean={y_pred_mult.mean():.4f}")
    print(f"Additive predictions:       mean={y_pred_add.mean():.4f}")
    print(f"Target (a*b+c):             mean={y_target.mean():.4f}")
    print(f"Baseline (c):               mean={y_baseline.mean():.4f}")

    alphas = np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_STEP / 2, ALPHA_STEP)
    layers = [TARGET_LAYER] + CONTROL_LAYERS

    results: Dict[str, Dict] = {}
    for layer_idx in layers:
        label = f"L{layer_idx}"
        print(f"\n{'='*60}")
        print(f"Computing delta at {label}...")
        delta = _compute_delta(model, X_mult, X_add, device, layer_idx, HEAD_IDX)
        delta_norm = float(torch.norm(delta).item())
        print(f"  delta norm: {delta_norm:.4f}")

        layer_results = []
        for alpha in alphas:
            steered = _run_with_steering(
                model, X_add, device, layer_idx, HEAD_IDX, delta, alpha
            )
            mean_rec, std_rec = _compute_recovery(steered, y_target, y_baseline)
            mse = float(np.mean((steered - y_target) ** 2))
            layer_results.append({
                "alpha": float(alpha),
                "mean_recovery": mean_rec,
                "std_recovery": std_rec,
                "mse_vs_target": mse,
                "mean_pred": float(steered.mean()),
            })
            if alpha % 2.0 == 0:
                print(f"  alpha={alpha:5.1f}  recovery={mean_rec*100:6.1f}%  "
                      f"std={std_rec*100:5.1f}%  pred_mean={steered.mean():.4f}")

        results[label] = layer_results

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results_contrastive_steering")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"steering_results_{timestamp}.json", "w") as f:
        json.dump({
            "alphas": alphas.tolist(),
            "results": results,
            "config": {
                "n_eval": N_EVAL,
                "head_idx": HEAD_IDX,
                "target_layer": TARGET_LAYER,
                "control_layers": CONTROL_LAYERS,
                "seed": SEED,
            },
        }, f, indent=2)
    print(f"\nResults saved to {output_dir}/steering_results_{timestamp}.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = {"L0": "red", "L1": "blue", "L12": "green"}
    for label, layer_results in results.items():
        al = [r["alpha"] for r in layer_results]
        rec = [r["mean_recovery"] * 100 for r in layer_results]
        mse = [r["mse_vs_target"] for r in layer_results]
        color = colors.get(label, "gray")
        ax1.plot(al, rec, "o-", color=color, label=label, linewidth=2, markersize=4)
        ax2.plot(al, mse, "o-", color=color, label=label, linewidth=2, markersize=4)

    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.axhline(y=100, color="k", linestyle="--", alpha=0.3, label="Full recovery")
    ax1.set_xlabel("Steering coefficient α")
    ax1.set_ylabel("Recovery %")
    ax1.set_title("Recovery: (steered − baseline) / (target − baseline)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    baseline_mse = float(np.mean((y_baseline - y_target) ** 2))
    ax2.axhline(y=baseline_mse, color="k", linestyle="--", alpha=0.3, label="Baseline MSE")
    ax2.set_xlabel("Steering coefficient α")
    ax2.set_ylabel("MSE vs target")
    ax2.set_title("MSE: steered output vs a·b + c")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Contrastive Steering: Head {HEAD_IDX} at Layer 0 vs Controls\n"
        f"Multiplication dataset (y = a·b + c), N={N_EVAL}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    fig_path = Path("docs/attention_head_patching") / "contrastive_steering.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Figure saved to {fig_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, layer_results in results.items():
        best = max(layer_results, key=lambda r: r["mean_recovery"] if not np.isnan(r["mean_recovery"]) else -999)
        print(f"{label}: best recovery = {best['mean_recovery']*100:.1f}% at alpha={best['alpha']:.1f}")


if __name__ == "__main__":
    main()
