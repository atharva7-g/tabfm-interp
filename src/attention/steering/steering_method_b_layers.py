#!/usr/bin/env python3
"""Method B at multiple layers: test if α=1.0 is layer-specific."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.datasets.synthetic import create_dataset

SEED = 42
N_EVAL = 32
ALPHA_MIN = 0.0
ALPHA_MAX = 10.0
ALPHA_STEP = 0.5
LAYERS = [0, 6, 12, 17]
DPI = 200


def fit_model(device: str):
    X, y = create_dataset("multiplication", num_samples=1000, seed=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=SEED
    )
    model = TabPFNRegressor(device=device, n_estimators=1)
    model.fit(X_train, y_train)
    return model, X_test[:N_EVAL], y_test[:N_EVAL]


def make_additive_batch(X_mult: np.ndarray) -> np.ndarray:
    X_add = X_mult.copy()
    X_add[:, 1] = 0.0
    return X_add


def capture_layer_output(model: TabPFNRegressor, X: np.ndarray, layer_idx: int) -> torch.Tensor:
    layer = model.model_.transformer_encoder.layers[layer_idx]
    original_forward = layer.forward
    captured = {}

    def patched_forward(
        state,
        single_eval_pos,
        *,
        save_peak_mem_factor,
        cache_trainset_representation=False,
        att_src=None,
    ):
        output = original_forward(
            state,
            single_eval_pos,
            save_peak_mem_factor=save_peak_mem_factor,
            cache_trainset_representation=cache_trainset_representation,
            att_src=att_src,
        )
        if "output" not in captured:
            captured["output"] = output.detach().clone()
        return output

    layer.forward = patched_forward
    with torch.no_grad():
        model.predict(X)
    layer.forward = original_forward

    if "output" not in captured:
        raise RuntimeError(f"Failed to capture layer {layer_idx} output.")
    return captured["output"]


def compute_delta(model, X_mult, X_add, layer_idx):
    out_mult = capture_layer_output(model, X_mult, layer_idx)
    out_add = capture_layer_output(model, X_add, layer_idx)
    return out_mult.mean(dim=0) - out_add.mean(dim=0)


def run_with_steering(model, X, delta, layer_idx, alpha):
    if alpha == 0.0:
        with torch.no_grad():
            return model.predict(X)

    layer = model.model_.transformer_encoder.layers[layer_idx]
    original_forward = layer.forward

    def patched_forward(
        state,
        single_eval_pos,
        *,
        save_peak_mem_factor,
        cache_trainset_representation=False,
        att_src=None,
    ):
        output = original_forward(
            state,
            single_eval_pos,
            save_peak_mem_factor=save_peak_mem_factor,
            cache_trainset_representation=cache_trainset_representation,
            att_src=att_src,
        )
        delta_local = delta.to(device=output.device, dtype=output.dtype)
        return output + alpha * delta_local.unsqueeze(0)

    layer.forward = patched_forward
    with torch.no_grad():
        preds = model.predict(X)
    layer.forward = original_forward
    return preds


def compute_recovery(steered_y, target_y, baseline_y):
    gaps = target_y - baseline_y
    shifts = steered_y - baseline_y
    with np.errstate(divide="ignore", invalid="ignore"):
        per_sample = np.where(np.abs(gaps) > 1e-6, shifts / gaps, np.nan)
    valid = per_sample[~np.isnan(per_sample)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    return float(np.mean(valid)), float(np.std(valid))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, X_test, _ = fit_model(device)
    X_mult = X_test[:N_EVAL]
    X_add = make_additive_batch(X_mult)

    a = X_mult[:, 0]
    b = X_mult[:, 1]
    c = X_mult[:, 2]
    y_target = a * b + c
    y_baseline = c

    alphas = np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_STEP / 2, ALPHA_STEP)

    all_results = {}
    layer_delta_norms = {}

    for layer_idx in LAYERS:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}: computing delta...")
        delta = compute_delta(model, X_mult, X_add, layer_idx)
        delta_norm = float(torch.norm(delta).item())
        layer_delta_norms[layer_idx] = delta_norm
        print(f"  Delta norm: {delta_norm:.4f}")

        layer_results = []
        for alpha in alphas:
            steered = run_with_steering(model, X_add, delta, layer_idx, float(alpha))
            mean_rec, std_rec = compute_recovery(steered, y_target, y_baseline)
            mse = float(np.mean((steered - y_target) ** 2))
            layer_results.append({
                "alpha": float(alpha),
                "mean_recovery": mean_rec,
                "std_recovery": std_rec,
                "mse_vs_target": mse,
                "mean_pred": float(steered.mean()),
            })
            if alpha % 2.0 == 0:
                print(f"  alpha={alpha:5.1f}  rec={mean_rec*100:6.1f}%  mse={mse:.4f}")

        all_results[layer_idx] = layer_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results_contrastive_steering_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"steering_method_b_layers_{timestamp}.json", "w") as f:
        json.dump({
            "config": {
                "method": "B_layers",
                "layers": LAYERS,
                "n_eval": N_EVAL,
                "seed": SEED,
            },
            "delta_norms": {str(k): v for k, v in layer_delta_norms.items()},
            "results": {str(k): v for k, v in all_results.items()},
        }, f, indent=2)

    # Plot
    colors = {0: "red", 6: "blue", 12: "green", 17: "purple"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for layer_idx in LAYERS:
        al = [r["alpha"] for r in all_results[layer_idx]]
        rec = [r["mean_recovery"] * 100 for r in all_results[layer_idx]]
        mse = [r["mse_vs_target"] for r in all_results[layer_idx]]
        color = colors.get(layer_idx, "gray")
        ax1.plot(al, rec, "o-", color=color, label=f"L{layer_idx} (δ={layer_delta_norms[layer_idx]:.1f})", linewidth=2, markersize=4)
        ax2.plot(al, mse, "o-", color=color, label=f"L{layer_idx}", linewidth=2, markersize=4)

    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.axhline(y=100, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Steering coefficient α")
    ax1.set_ylabel("Recovery %")
    ax1.set_title("Recovery vs steering strength")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    baseline_mse = float(np.mean((y_baseline - y_target) ** 2))
    ax2.axhline(y=baseline_mse, color="k", linestyle="--", alpha=0.3, label="Baseline")
    ax2.set_xlabel("Steering coefficient α")
    ax2.set_ylabel("MSE vs target")
    ax2.set_title("MSE vs steering strength")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Method B: Full Residual Stream Steering at Multiple Layers", fontsize=11, fontweight="bold")
    plt.tight_layout()

    fig_path = Path("docs/attention_head_patching/steering_method_b_layers.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nFigure: {fig_path}")
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for layer_idx in LAYERS:
        best_mse = min(all_results[layer_idx], key=lambda r: r["mse_vs_target"])
        best_rec = max(all_results[layer_idx], key=lambda r: r["mean_recovery"] if not np.isnan(r["mean_recovery"]) else -999)
        print(f"L{layer_idx}: δ={layer_delta_norms[layer_idx]:.2f}, best MSE={best_mse['mse_vs_target']:.4f} @ α={best_mse['alpha']:.1f}, best rec={best_rec['mean_recovery']*100:.1f}% @ α={best_rec['alpha']:.1f}")


if __name__ == "__main__":
    main()
