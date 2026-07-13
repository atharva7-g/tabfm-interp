#!/usr/bin/env python3
"""Option D: steer only target token residual slice at Layer 0."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from tabpfn import TabPFNRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.attention.steering import (
    N_EVAL,
    alpha_grid,
    fit_model,
    make_additive_batch,
    plot_results,
    print_results_summary,
    run_alpha_sweep,
    save_results,
    temporary_patch,
)

METHOD_NAME = "target_token"
LAYER_IDX = 0
TARGET_TOKEN_IDX = 3


def _get_layer(model: TabPFNRegressor):
    return model.model_.transformer_encoder.layers[LAYER_IDX]


def _capture_layer_output(model: TabPFNRegressor, X: np.ndarray) -> torch.Tensor:
    layer = _get_layer(model)
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

    with temporary_patch(layer, "forward", patched_forward):
        with torch.no_grad():
            model.predict(X)

    if "output" not in captured:
        raise RuntimeError("Failed to capture layer output.")
    return captured["output"]


def _compute_delta(
    model: TabPFNRegressor,
    X_mult: np.ndarray,
    X_add: np.ndarray,
) -> torch.Tensor:
    out_mult = _capture_layer_output(model, X_mult)
    out_add = _capture_layer_output(model, X_add)
    if out_mult.shape[2] <= TARGET_TOKEN_IDX:
        raise RuntimeError(
            f"Target token index {TARGET_TOKEN_IDX} out of bounds for shape {tuple(out_mult.shape)}"
        )
    target_mult = out_mult[:, :, TARGET_TOKEN_IDX, :]
    target_add = out_add[:, :, TARGET_TOKEN_IDX, :]
    return target_mult.mean(dim=0) - target_add.mean(dim=0)


def _run_with_steering(
    model: TabPFNRegressor,
    X: np.ndarray,
    delta: torch.Tensor,
    alpha: float,
) -> np.ndarray:
    if alpha == 0.0:
        with torch.no_grad():
            return model.predict(X)

    layer = _get_layer(model)
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
        if output.shape[2] <= TARGET_TOKEN_IDX:
            raise RuntimeError(
                f"Target token index {TARGET_TOKEN_IDX} out of bounds for shape {tuple(output.shape)}"
            )
        delta_local = delta.to(device=output.device, dtype=output.dtype)
        steered = output.clone()
        steered[:, :, TARGET_TOKEN_IDX, :] += alpha * delta_local
        return steered

    with temporary_patch(layer, "forward", patched_forward):
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
    fig_path = Path("docs/attention_head_patching/steering_target_token.png")
    plot_results(
        results,
        "Steering D: Target Token Residual Slice at Layer 0",
        fig_path,
        baseline_mse,
    )

    config = {
        "method": "D",
        "method_name": METHOD_NAME,
        "hook": "layer_0.forward[target_token]",
        "target_token_idx": TARGET_TOKEN_IDX,
        "n_eval": int(N_EVAL),
        "layer_idx": LAYER_IDX,
        "delta_norm": delta_norm,
        "alphas": alphas.tolist(),
        "figure_path": str(fig_path),
    }
    json_path = save_results(METHOD_NAME, config, results)

    print_results_summary("D", results)
    print(f"Figure: {fig_path}")
    print(f"JSON:   {json_path}")


if __name__ == "__main__":
    main()
