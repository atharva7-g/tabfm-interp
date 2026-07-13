#!/usr/bin/env python3
"""Option C: steer full 192-d features MHA output at Layer 6."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.attention.steering import (
    N_EVAL,
    alpha_grid,
    capture_mha_output,
    fit_model,
    make_additive_batch,
    make_mha_compute_patch,
    plot_results,
    print_results_summary,
    run_alpha_sweep,
    save_results,
    temporary_patch,
)

METHOD_NAME = "full_mha_layer6"
LAYER_IDX = 6


def _get_mha(model: TabPFNRegressor) -> MultiHeadAttention:
    return model.model_.transformer_encoder.layers[LAYER_IDX].self_attn_between_features


def _compute_delta(
    model: TabPFNRegressor,
    X_mult: np.ndarray,
    X_add: np.ndarray,
) -> torch.Tensor:
    mha = _get_mha(model)
    out_mult = capture_mha_output(model, X_mult, mha)
    out_add = capture_mha_output(model, X_add, mha)
    return out_mult.mean(dim=0) - out_add.mean(dim=0)


def _run_with_steering(
    model: TabPFNRegressor,
    X: np.ndarray,
    delta: torch.Tensor,
    alpha: float,
) -> np.ndarray:
    if alpha == 0.0:
        with torch.no_grad():
            return model.predict(X)

    mha = _get_mha(model)
    original_compute = mha._compute

    def _transform(output: torch.Tensor, _x: torch.Tensor, _kwargs: dict) -> torch.Tensor:
        delta_local = delta.to(device=output.device, dtype=output.dtype)
        return output + alpha * delta_local

    patched_compute = make_mha_compute_patch(mha, original_compute, _transform)
    with temporary_patch(mha, "_compute", patched_compute):
        with torch.no_grad():
            return model.predict(X)


def main() -> None:
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

    print("Computing delta...")
    delta = _compute_delta(model, X_mult, X_add)
    delta_norm = float(torch.norm(delta).item())
    print(f"Delta norm: {delta_norm:.4f}")

    alphas = alpha_grid()
    results = run_alpha_sweep(
        model=model,
        X_add=X_add,
        delta=delta,
        inject_fn=_run_with_steering,
        alphas=alphas,
        target_y=y_target,
        baseline_y=y_baseline,
    )

    baseline_mse = float(np.mean((y_baseline - y_target) ** 2))
    fig_path = Path("docs/attention_head_patching/steering_full_mha_layer6.png")
    plot_results(
        results,
        "Steering C: Full 192-d Features MHA Output at Layer 6",
        fig_path,
        baseline_mse,
    )

    config = {
        "method": "C",
        "method_name": METHOD_NAME,
        "hook": "layer_6.self_attn_between_features._compute",
        "n_eval": int(N_EVAL),
        "layer_idx": LAYER_IDX,
        "delta_norm": delta_norm,
        "alphas": alphas.tolist(),
        "figure_path": str(fig_path),
    }
    json_path = save_results(METHOD_NAME, config, results)

    print_results_summary("C", results)
    print(f"Figure: {fig_path}")
    print(f"JSON:   {json_path}")


if __name__ == "__main__":
    main()
