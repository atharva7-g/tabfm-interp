#!/usr/bin/env python3
"""Method B with different seeds: test if α=1.0 is seed-specific."""

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

SEEDS = [42, 123, 456, 789, 1024]
N_EVAL = 32
ALPHA_MIN = 0.0
ALPHA_MAX = 10.0
ALPHA_STEP = 0.5
LAYER_IDX = 0
DPI = 200


def fit_model(device: str, seed: int):
    X, y = create_dataset("multiplication", num_samples=1000, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=seed
    )
    model = TabPFNRegressor(device=device, n_estimators=1)
    model.fit(X_train, y_train)
    return model, X_test[:N_EVAL], y_test[:N_EVAL]


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

    alphas = np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_STEP / 2, ALPHA_STEP)
    all_results = {}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}: fitting model...")
        model, X_test, _ = fit_model(device, seed)
        X_mult = X_test[:N_EVAL]
        X_add = make_additive_batch(X_mult)

        a = X_mult[:, 0]
        b = X_mult[:, 1]
        c = X_mult[:, 2]
        y_target = a * b + c
        y_baseline = c

        delta = compute_delta(model, X_mult, X_add)
        delta_norm = float(torch.norm(delta).item())
        print(f"  Delta norm: {delta_norm:.4f}")

        seed_results = []
        for alpha in alphas:
            steered = run_with_steering(model, X_add, delta, float(alpha))
            mean_rec, std_rec = compute_recovery(steered, y_target, y_baseline)
            mse = float(np.mean((steered - y_target) ** 2))
            seed_results.append({
                "alpha": float(alpha),
                "mean_recovery": mean_rec,
                "std_recovery": std_rec,
                "mse_vs_target": mse,
                "mean_pred": float(steered.mean()),
            })
            if alpha % 2.0 == 0:
                print(f"  alpha={alpha:5.1f}  rec={mean_rec*100:6.1f}%  mse={mse:.4f}")

        best_mse = min(seed_results, key=lambda r: r["mse_vs_target"])
        print(f"  Best MSE: {best_mse['mse_vs_target']:.4f} @ alpha={best_mse['alpha']:.1f}")

        all_results[seed] = {
            "delta_norm": delta_norm,
            "results": seed_results,
        }

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results_contrastive_steering_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"steering_method_b_seeds_{timestamp}.json", "w") as f:
        json.dump({
            "config": {
                "method": "B_seeds",
                "layer": LAYER_IDX,
                "n_eval": N_EVAL,
                "seeds": SEEDS,
            },
            "results": {str(k): v for k, v in all_results.items()},
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for seed in SEEDS:
        info = all_results[seed]
        best_mse = min(info["results"], key=lambda r: r["mse_vs_target"])
        print(f"Seed {seed:4d}: δ={info['delta_norm']:.2f}, best MSE={best_mse['mse_vs_target']:.4f} @ α={best_mse['alpha']:.1f}")


if __name__ == "__main__":
    main()
