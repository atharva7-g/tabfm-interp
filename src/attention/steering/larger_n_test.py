#!/usr/bin/env python3
"""Test with larger N."""

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
N_EVAL = 256
N_HOLDOUT = N_EVAL
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

    # Same-sample test (leaky, N=256)
    X_mult = X_test[:N_EVAL]
    X_add = make_additive_batch(X_mult)
    a = X_mult[:, 0]
    b = X_mult[:, 1]
    c = X_mult[:, 2]
    y_target = a * b + c
    y_baseline = c

    print(f"\n{'='*60}")
    print(f"Same-sample test (N={N_EVAL})")
    delta = compute_delta(model, X_mult, X_add)
    delta_norm = float(torch.norm(delta).item())
    print(f"Delta norm: {delta_norm:.4f}")

    same_results = []
    for alpha in alphas:
        steered = run_with_steering(model, X_add, delta, float(alpha))
        mean_rec, std_rec = compute_recovery(steered, y_target, y_baseline)
        mse = float(np.mean((steered - y_target) ** 2))
        same_results.append({
            "alpha": float(alpha),
            "mean_recovery": mean_rec,
            "std_recovery": std_rec,
            "mse_vs_target": mse,
        })
        if alpha % 2.0 == 0:
            print(f"  α={alpha:5.1f} rec={mean_rec*100:6.1f}% mse={mse:.4f}")

    # Held-out test (N=256 compute, N=256 eval)
    print(f"\n{'='*60}")
    print(f"Held-out test (compute N={N_EVAL}, eval N={N_HOLDOUT})")
    X_mult_comp = X_test[N_EVAL:N_EVAL + N_EVAL]
    X_add_comp = make_additive_batch(X_mult_comp)
    delta_held = compute_delta(model, X_mult_comp, X_add_comp)

    X_eval = X_test[N_EVAL + N_EVAL:N_EVAL + N_EVAL + N_HOLDOUT]
    X_add_eval = make_additive_batch(X_eval)
    a_e = X_eval[:, 0]
    b_e = X_eval[:, 1]
    c_e = X_eval[:, 2]
    y_target_e = a_e * b_e + c_e
    y_baseline_e = c_e

    held_results = []
    for alpha in alphas:
        steered = run_with_steering(model, X_add_eval, delta_held, float(alpha))
        mean_rec, std_rec = compute_recovery(steered, y_target_e, y_baseline_e)
        mse = float(np.mean((steered - y_target_e) ** 2))
        held_results.append({
            "alpha": float(alpha),
            "mean_recovery": mean_rec,
            "std_recovery": std_rec,
            "mse_vs_target": mse,
        })
        if alpha % 2.0 == 0:
            print(f"  α={alpha:5.1f} rec={mean_rec*100:6.1f}% mse={mse:.4f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results_contrastive_skeptics")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"larger_n_test_{timestamp}.json", "w") as f:
        json.dump({
            "config": {
                "n_eval": N_EVAL,
                "n_holdout": N_HOLDOUT,
                "seed": SEED,
            },
            "same_sample": same_results,
            "held_out": held_results,
        }, f, indent=2)

    best_same = min(same_results, key=lambda r: r["mse_vs_target"])
    best_held = min(held_results, key=lambda r: r["mse_vs_target"])
    baseline_mse_same = same_results[0]["mse_vs_target"]
    baseline_mse_held = held_results[0]["mse_vs_target"]

    print(f"\n{'='*80}")
    print("SUMMARY (N=256)")
    print(f"{'='*80}")
    print(f"Same-sample: baseline MSE={baseline_mse_same:.4f}, best={best_same['mse_vs_target']:.4f} @ α={best_same['alpha']:.1f}")
    print(f"Held-out:    baseline MSE={baseline_mse_held:.4f}, best={best_held['mse_vs_target']:.4f} @ α={best_held['alpha']:.1f}")


if __name__ == "__main__":
    main()