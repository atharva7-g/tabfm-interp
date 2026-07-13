#!/usr/bin/env python3
"""Null test: Method B with random direction (norm ≈90)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

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
LAYER_IDX = 0
DPI = 200


def fit_model(device: str):
    X, y = create_dataset("multiplication", num_samples=1000, seed=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=SEED
    )
    model = TabPFNRegressor(device=device, n_estimators=1)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def make_additive_batch(X_mult: np.ndarray) -> np.ndarray:
    X_add = X_mult.copy()
    X_add[:, 1] = 0.0
    return X_add


def capture_layer_output(model: TabPFNRegressor, X: np.ndarray) -> torch.Tensor:
    layer = model.model_.transformer_encoder.layers[LAYER_IDX]
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
        raise RuntimeError("Failed to capture layer output.")
    return captured["output"]


def run_with_steering(model, X, delta, alpha):
    if alpha == 0.0:
        with torch.no_grad():
            return model.predict(X)

    layer = model.model_.transformer_encoder.layers[LAYER_IDX]
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

    # 1. Real delta (same-samples)
    print(f"\n{'='*60}")
    print("Real delta (same-samples, leaky)")
    out_mult = capture_layer_output(model, X_mult)
    out_add = capture_layer_output(model, X_add)
    real_delta = out_mult.mean(dim=0) - out_add.mean(dim=0)
    real_norm = float(torch.norm(real_delta).item())
    print(f"Real delta norm: {real_norm:.4f}")

    # 2. Random delta (norm-matched)
    print(f"\n{'='*60}")
    print(f"Random delta (norm ≈{real_norm})")
    # state shape [1, items, 4, 192]
    rand_delta = torch.randn_like(real_delta)
    rand_delta = rand_delta / torch.norm(rand_delta) * real_norm
    rand_norm = float(torch.norm(rand_delta).item())
    print(f"Random delta norm: {rand_norm:.4f}")

    # 3. Random delta (norm ≈0.1, like Method A)
    print(f"\n{'='*60}")
    print("Random delta (norm ≈0.1, matching Method A)")
    small_rand_delta = torch.randn_like(real_delta)
    small_rand_delta = small_rand_delta / torch.norm(small_rand_delta) * 0.1
    small_norm = float(torch.norm(small_rand_delta).item())
    print(f"Small random delta norm: {small_norm:.4f}")

    results = {}
    deltas = {
        "real": real_delta,
        "rand_90": rand_delta,
        "rand_01": small_rand_delta,
    }

    for delta_name, delta in deltas.items():
        print(f"\n--- Running {delta_name} ---")
        delta_norm = float(torch.norm(delta).item())
        delta_results = []
        for alpha in alphas:
            steered = run_with_steering(model, X_add, delta, float(alpha))
            mean_rec, std_rec = compute_recovery(steered, y_target, y_baseline)
            mse = float(np.mean((steered - y_target) ** 2))
            delta_results.append({
                "alpha": float(alpha),
                "mean_recovery": mean_rec,
                "std_recovery": std_rec,
                "mse_vs_target": mse,
                "mean_pred": float(steered.mean()),
            })
            if alpha % 2.0 == 0:
                print(f"  alpha={alpha:5.1f}  rec={mean_rec*100:6.1f}%  mse={mse:.4f}")
        results[delta_name] = {
            "delta_norm": delta_norm,
            "results": delta_results,
        }

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results_contrastive_skeptics")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"null_test_l0_{timestamp}.json", "w") as f:
        json.dump({
            "config": {
                "method": "null_test",
                "layer": LAYER_IDX,
                "n_eval": N_EVAL,
                "seed": SEED,
            },
            "results": results,
        }, f, indent=2)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for name in ["real", "rand_90", "rand_01"]:
        info = results[name]
        best_mse = min(info["results"], key=lambda r: r["mse_vs_target"])
        best_rec = max(info["results"], key=lambda r: r["mean_recovery"] if not np.isnan(r["mean_recovery"]) else -999)
        print(f"{name:10s} | δ={info['delta_norm']:6.2f} | best MSE={best_mse['mse_vs_target']:8.4f} @ α={best_mse['alpha']:5.2f} | best rec={best_rec['mean_recovery']*100:6.1f}% @ α={best_rec['alpha']:5.2f}")

    # Plot
    import matplotlib.pyplot as plt
    colors = {"real": "red", "rand_90": "blue", "rand_01": "green"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, color in colors.items():
        info = results[name]
        al = [r["alpha"] for r in info["results"]]
        mse = [r["mse_vs_target"] for r in info["results"]]
        rec = [r["mean_recovery"] * 100 for r in info["results"]]
        label = f"{name} (δ={info['delta_norm']:.1f})"
        ax1.plot(al, rec, "o-", color=color, label=label, linewidth=2, markersize=4)
        ax2.plot(al, mse, "o-", color=color, label=label, linewidth=2, markersize=4)

    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.axhline(y=100, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Steering coefficient α")
    ax1.set_ylabel("Recovery %")
    ax1.set_title("Recovery vs steering strength (null test)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.axhline(y=float(np.mean((y_baseline - y_target) ** 2)), color="k", linestyle="--", alpha=0.3, label="Baseline")
    ax2.set_xlabel("Steering coefficient α")
    ax2.set_ylabel("MSE vs target")
    ax2.set_title("MSE vs steering strength (null test)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Null Test: Random Directions vs Real Delta (Layer 0)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    fig_path = Path("docs/attention_head_patching/null_test_l0.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {fig_path}")


if __name__ == "__main__":
    main()