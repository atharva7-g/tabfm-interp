#!/usr/bin/env python3
"""Held-out test: compute δ on one batch, evaluate on independent batch."""

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
N_HOLDOUT = N_EVAL  # second independent batch
ALPHA_MIN = 0.0
ALPHA_MAX = 10.0
ALPHA_STEP = 0.5
LAYER_IDX = 0
DPI = 200


def fit_model(device: str):
    X, y = create_dataset("multiplication", num_samples=2000, seed=SEED)
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


def compute_delta(model, X_mult, X_add):
    out_mult = capture_layer_output(model, X_mult)
    out_add = capture_layer_output(model, X_add)
    return out_mult.mean(dim=0) - out_add.mean(dim=0)


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
    alphas = np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_STEP / 2, ALPHA_STEP)

    # Batch 1 (compute delta)
    X_mult1 = X_test[:N_EVAL]
    X_add1 = make_additive_batch(X_mult1)
    print(f"\n{'='*60}")
    print("Batch 1: computing delta...")
    delta = compute_delta(model, X_mult1, X_add1)
    delta_norm = float(torch.norm(delta).item())
    print(f"Delta norm: {delta_norm:.4f}")

    # Batch 2 (held-out evaluation)
    X_mult2 = X_test[N_EVAL:N_EVAL + N_HOLDOUT]
    X_add2 = make_additive_batch(X_mult2)
    a2 = X_mult2[:, 0]
    b2 = X_mult2[:, 1]
    c2 = X_mult2[:, 2]
    y_target = a2 * b2 + c2
    y_baseline = c2

    baseline_mse = float(np.mean((y_baseline - y_target) ** 2))
    print(f"Batch 2 baseline MSE: {baseline_mse:.4f}")
    print(f"Batch 2 N samples: {len(y_target)}")

    results = []
    for alpha in alphas:
        steered = run_with_steering(model, X_add2, delta, float(alpha))
        mean_rec, std_rec = compute_recovery(steered, y_target, y_baseline)
        mse = float(np.mean((steered - y_target) ** 2))
        results.append({
            "alpha": float(alpha),
            "mean_recovery": mean_rec,
            "std_recovery": std_rec,
            "mse_vs_target": mse,
            "mean_pred": float(steered.mean()),
        })
        if alpha % 2.0 == 0:
            print(f"  alpha={alpha:5.1f}  rec={mean_rec*100:6.1f}%  mse={mse:.4f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results_contrastive_skeptics")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"heldout_test_{timestamp}.json", "w") as f:
        json.dump({
            "config": {
                "method": "heldout",
                "layer": LAYER_IDX,
                "compute_batch_size": N_EVAL,
                "eval_batch_size": N_HOLDOUT,
                "seed": SEED,
                "delta_norm": delta_norm,
                "baseline_mse": baseline_mse,
            },
            "results": results,
        }, f, indent=2)

    best_mse = min(results, key=lambda r: r["mse_vs_target"])
    best_rec = max(results, key=lambda r: r["mean_recovery"] if not np.isnan(r["mean_recovery"]) else -999)

    print(f"\n{'='*80}")
    print("HELD-OUT RESULTS (delta batch ≠ steer batch)")
    print(f"{'='*80}")
    print(f"Best MSE: {best_mse['mse_vs_target']:.4f} @ α={best_mse['alpha']:.2f}")
    print(f"Best rec: {best_rec['mean_recovery']*100:.1f}% @ α={best_rec['alpha']:.2f}")

    # Compare with same-sample
    X_mult_same = X_test[:N_HOLDOUT]
    X_add_same = make_additive_batch(X_mult_same)
    a_s = X_mult_same[:, 0]
    b_s = X_mult_same[:, 1]
    c_s = X_mult_same[:, 2]
    y_target_same = a_s * b_s + c_s
    y_baseline_same = c_s
    delta_same = compute_delta(model, X_mult_same, X_add_same)
    same_results = []
    for alpha in [0.0, 1.0, 2.0]:
        steered = run_with_steering(model, X_add_same, delta_same, float(alpha))
        mse = float(np.mean((steered - y_target_same) ** 2))
        same_results.append((alpha, mse))

    print(f"\nSame-batch (for reference):")
    for alpha, mse in same_results:
        print(f"  α={alpha:5.1f} MSE={mse:.4f}")


if __name__ == "__main__":
    main()