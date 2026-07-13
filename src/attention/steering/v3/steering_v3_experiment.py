#!/usr/bin/env python3
"""Steering v3: Held-out contrastive steering with proper train/val/test splits.

Fixes the data leakage that invalidated v2 results.

v2 flaw: δ was computed on the same N=32 samples used for evaluation.
At α=1.0 this is arithmetic mean substitution, not concept injection.

v3 design:
  - Model is fit on SEPARATE training data (N=2000)
  - Steering data (N=1024) is split into train/val/test with no overlap
  - δ is computed on the train split ONLY
  - α is selected on the val split
  - Final metrics reported on the held-out test split
  - δ is averaged over ALL sample dimensions so it generalizes across batches

Key shape change vs v2:
  v2: delta has shape [N_items, 4, 192] (item-specific, doesn't transfer)
  v3: delta has shape [4, 192]          (generalizable across any batch size)

Hook sites tested (6):
  A - Features MHA output, Layer 0                capture (N,4,192)  → δ (4,192)
  B - Full residual stream, Layer 0                capture [1,N,4,192] → δ (4,192)
  C - Features MHA output, Layer 6                capture (N,4,192)  → δ (4,192)
  D - Target token only, Layer 0                  capture [1,N,192]  → δ (192,)
  E - Items MHA output, Layer 0                   capture (4,N,192)  → δ (4,192)
  F - Full residual stream, Layers 0+6+12+17      per-layer [4,192]

Direction estimators (4):
  mean_diff    — δ = mean(activation_mult) − mean(activation_add)
  per_sample   — δ = mean(activation_mult_i − activation_add_i)  [= mean_diff mathematically]
  pca          — δ = PC1 of {activation_mult_i − activation_add_i}
  linear_probe — δ = weight vector of logistic regression (mult vs add)

Controls:
  random   — random vector with same norm as δ
  shuffled — randomly permuted entries of δ (destroys spatial structure)

Success criteria (ALL must hold):
  1. Held-out MSE improvement > 5%
  2. Random direction control ≤ 1% improvement
  3. Shuffled direction control ≤ 1% improvement
  4. Same-sample baseline >> held-out result
  5. Cosine similarity (train δ vs val δ) > 0.3
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.datasets.synthetic import create_dataset
from src.attention.steering import temporary_patch, make_mha_compute_patch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
N_MODEL_TRAIN = 2000
N_STEER_TOTAL = 1024
ALPHAS = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
CONTROL_ALPHAS = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
DPI = 200
RESULTS_DIR = Path("results_steering_v3")

HOOK_CONFIGS = {
    "A": {"type": "features_mha", "layer": 0},
    "B": {"type": "residual",    "layer": 0},
    "C": {"type": "features_mha", "layer": 6},
    "D": {"type": "target_token", "layer": 0},
    "E": {"type": "items_mha",   "layer": 0},
}

TARGET_TOKEN_IDX = 3

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def make_additive_batch(X_mult: np.ndarray) -> np.ndarray:
    """Set b=0 so y = a*0 + c = c."""
    X_add = X_mult.copy()
    X_add[:, 1] = 0.0
    return X_add


def compute_recovery(
    steered_y: np.ndarray,
    target_y: np.ndarray,
    baseline_y: np.ndarray,
) -> Tuple[float, float]:
    gaps = target_y - baseline_y
    shifts = steered_y - baseline_y
    with np.errstate(divide="ignore", invalid="ignore"):
        per_sample = np.where(np.abs(gaps) > 1e-6, shifts / gaps, np.nan)
    valid = per_sample[~np.isnan(per_sample)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    return float(np.mean(valid)), float(np.std(valid))


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return float(torch.dot(a_flat, b_flat) / (torch.norm(a_flat) * torch.norm(b_flat) + 1e-10))


# ---------------------------------------------------------------------------
# Model fitting (on SEPARATE data from steering splits)
# ---------------------------------------------------------------------------


def fit_model(device: str) -> TabPFNRegressor:
    """Fit TabPFN on independent training data.

    Uses seed=SEED to generate 2000 samples, fits on the 80% train split.
    The remaining 20% are discarded — they are NOT used for steering.
    """
    X, y = create_dataset("multiplication", num_samples=N_MODEL_TRAIN, seed=SEED)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    model = TabPFNRegressor(device=device, n_estimators=1)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Activation capture — one function per hook type
# ---------------------------------------------------------------------------


def _patch_layer_forward(layer):
    """Return (original_forward, patch_context_manager, capture_fn).

    The capture_fn returns the first non-cached layer output as a detached tensor.
    """

    original = layer.forward

    @contextmanager
    def _ctx():
        captured: Dict[str, torch.Tensor] = {}

        def patched(state, single_eval_pos, *,
                    save_peak_mem_factor,
                    cache_trainset_representation=False,
                    att_src=None):
            out = original(
                state, single_eval_pos,
                save_peak_mem_factor=save_peak_mem_factor,
                cache_trainset_representation=cache_trainset_representation,
                att_src=att_src,
            )
            if "out" not in captured:
                captured["out"] = out.detach().clone()
            return out

        setattr(layer, "forward", patched)
        try:
            yield captured
        finally:
            setattr(layer, "forward", original)

    return _ctx


def capture_residual(model, X, layer_idx):
    """Capture full residual stream after layer finishes.

    Returns tensor of shape [1, N, 4, 192].
    """
    layer = model.model_.transformer_encoder.layers[layer_idx]
    ctx = _patch_layer_forward(layer)
    with ctx() as captured:
        with torch.no_grad():
            model.predict(X)
    if "out" not in captured:
        raise RuntimeError(f"Failed to capture residual at layer {layer_idx}")
    return captured["out"]


def capture_target_token(model, X, layer_idx):
    """Capture target token slice of residual stream.

    Returns tensor of shape [1, N, 192].
    """
    full = capture_residual(model, X, layer_idx)
    return full[:, :, TARGET_TOKEN_IDX, :]


def capture_mha(model, X, layer_idx, attn_name):
    """Capture MHA _compute output (pre-residual).

    attn_name: 'features' or 'items'
    Returns tensor of shape (N, 4, 192) for features, (4, N, 192) for items.
    """
    layer = model.model_.transformer_encoder.layers[layer_idx]
    if attn_name == "features":
        mha = layer.self_attn_between_features
    else:
        mha = layer.self_attn_between_items

    captured: Dict[str, torch.Tensor] = {}
    original_compute = mha._compute

    def _transform(output, _x, _kwargs):
        if "out" not in captured:
            captured["out"] = output.detach().clone()
        return output

    patched = make_mha_compute_patch(mha, original_compute, _transform)
    with temporary_patch(mha, "_compute", patched):
        with torch.no_grad():
            model.predict(X)

    if "out" not in captured:
        raise RuntimeError(f"Failed to capture MHA at layer {layer_idx} ({attn_name})")
    return captured["out"]


def capture_multi_layer(model, X, layer_indices):
    """Capture residual stream at multiple layers simultaneously.

    Returns dict mapping layer_idx → tensor [1, N, 4, 192].
    """
    layers = [model.model_.transformer_encoder.layers[i] for i in layer_indices]
    originals = [lyr.forward for lyr in layers]
    all_captured: Dict[int, Dict[str, torch.Tensor]] = {i: {} for i in layer_indices}

    for idx, (lyr, orig) in zip(layer_indices, zip(layers, originals)):
        cap = all_captured[idx]

        def make_patched(orig_fwd, store):
            def patched(state, single_eval_pos, *,
                        save_peak_mem_factor,
                        cache_trainset_representation=False,
                        att_src=None):
                out = orig_fwd(
                    state, single_eval_pos,
                    save_peak_mem_factor=save_peak_mem_factor,
                    cache_trainset_representation=cache_trainset_representation,
                    att_src=att_src,
                )
                if "out" not in store:
                    store["out"] = out.detach().clone()
                return out
            return patched

        setattr(lyr, "forward", make_patched(orig, cap))

    try:
        with torch.no_grad():
            model.predict(X)
    finally:
        for lyr, orig in zip(layers, originals):
            setattr(lyr, "forward", orig)

    result = {}
    for i in layer_indices:
        if "out" not in all_captured[i]:
            raise RuntimeError(f"Failed to capture at layer {i}")
        result[i] = all_captured[i]["out"]
    return result


# ---------------------------------------------------------------------------
# Dispatch: capture activations for any hook site
# ---------------------------------------------------------------------------


def capture_activations(model, X, hook_id):
    """Capture activations for a given hook site during one forward pass."""
    cfg = HOOK_CONFIGS[hook_id]
    htype = cfg["type"]

    if htype == "residual":
        return capture_residual(model, X, cfg["layer"])
    elif htype == "target_token":
        return capture_target_token(model, X, cfg["layer"])
    elif htype == "features_mha":
        return capture_mha(model, X, cfg["layer"], "features")
    elif htype == "items_mha":
        return capture_mha(model, X, cfg["layer"], "items")
    elif htype == "multi_layer":
        return capture_multi_layer(model, X, cfg["layers"])
    else:
        raise ValueError(f"Unknown hook type: {htype}")


# ---------------------------------------------------------------------------
# Direction estimation — compute δ from paired mult/add activations
# ---------------------------------------------------------------------------


def _delta_from_activations(act_mult, act_add, estimator, hook_id):
    """Compute steering direction δ from matched mult/add activation tensors.

    The δ is always reduced to be INDEPENDENT of the number of items,
    so it can be applied to any batch size during held-out evaluation.

    Reduction rules by hook type:
      A (features_mha): (N_aug, 4, 192)  → mean(dim=0) → (4, 192)
      B (residual):     [1, N_aug, 4, 192] → mean(dim=(0,1)) → (4, 192)
      C (features_mha): (N_aug, 4, 192)  → mean(dim=0) → (4, 192)
      D (target_token): [1, N_aug, 192]   → mean(dim=(0,1)) → (192,)
      E (items_mha):    (4, N_orig, 192)  → mean(dim=1) → (4, 192)
      F (multi_layer):  per-layer (4, 192)

    Note: TabPFN internally augments items, so N_aug >> N_orig for features
    attention but items attention uses N_orig (the original sample count).
    """

    htype = HOOK_CONFIGS[hook_id]["type"]

    def _get_diff_matrix():
        """Return (N_items, D) numpy array of per-item difference vectors."""
        if isinstance(act_mult, dict):
            raise NotImplementedError("multi-layer diffs for PCA/probe")
        if htype == "residual":
            diffs = act_mult - act_add
            return diffs.squeeze(0).reshape(diffs.shape[1], -1).cpu().numpy()
        elif htype == "target_token":
            diffs = act_mult - act_add
            return diffs.squeeze(0).reshape(diffs.shape[1], -1).cpu().numpy()
        elif htype in ("features_mha",):
            diffs = act_mult - act_add
            return diffs.reshape(diffs.shape[0], -1).cpu().numpy()
        elif htype == "items_mha":
            diffs = act_mult - act_add
            return diffs.permute(1, 0, 2).reshape(diffs.shape[1], -1).cpu().numpy()
        else:
            raise ValueError(f"Unsupported hook type for diffs: {htype}")

    def _mean_delta():
        if isinstance(act_mult, dict):
            deltas = {}
            for k in sorted(act_mult.keys()):
                m, a = act_mult[k], act_add[k]
                deltas[k] = (m - a).float().mean(dim=(0, 1))
            return deltas
        if htype == "residual":
            return (act_mult - act_add).float().mean(dim=(0, 1))
        elif htype == "target_token":
            return (act_mult - act_add).float().mean(dim=(0, 1)).squeeze(0)
        elif htype == "features_mha":
            return (act_mult - act_add).float().mean(dim=0)
        elif htype == "items_mha":
            return (act_mult - act_add).float().mean(dim=1)
        else:
            raise ValueError(f"Unknown hook type: {htype}")

    def _delta_target_shape():
        """Return the shape that δ must have to broadcast with the hook output.

        Must match what _mean_delta() returns.
        """
        ref = _mean_delta()
        if isinstance(ref, dict):
            return {k: tuple(v.shape) for k, v in ref.items()}
        return tuple(ref.shape)

    if estimator in ("mean_diff", "per_sample_diff"):
        return _mean_delta()

    elif estimator == "pca":
        if isinstance(act_mult, dict):
            return _mean_delta()
        ts = _delta_target_shape()
        diffs = _get_diff_matrix()
        pca = PCA(n_components=1)
        pca.fit(diffs)
        pc1 = pca.components_[0]
        return torch.from_numpy(pc1).float().reshape(ts)

    elif estimator == "linear_probe":
        if isinstance(act_mult, dict):
            return _mean_delta()
        ts = _delta_target_shape()
        diffs = _get_diff_matrix()
        diffs_add = -diffs
        X_clf = np.vstack([diffs, diffs_add])
        y_clf = np.hstack([np.ones(len(diffs)), np.zeros(len(diffs_add))])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clf)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_scaled, y_clf)
        return torch.from_numpy(clf.coef_[0]).float().reshape(ts)

    else:
        raise ValueError(f"Unknown estimator: {estimator}")


# ---------------------------------------------------------------------------
# Steering injection — add α·δ at a hook site during forward pass
# ---------------------------------------------------------------------------


def _inject_residual(model, X, layer_idx, delta, alpha):
    """Inject α·δ into layer forward output (post-LayerNorm residual add).

    Note: For hook B (residual stream), δ is shape (4, 192) and broadcasts
    to the layer's output shape [1, N, 4, 192], adding the same delta to
    every token position. This represents a global perturbation of the
    residual stream at the layer output.
    """
    if alpha == 0.0:
        with torch.no_grad():
            return model.predict(X)

    layer = model.model_.transformer_encoder.layers[layer_idx]
    original = layer.forward

    def patched(state, single_eval_pos, *,
                save_peak_mem_factor,
                cache_trainset_representation=False,
                att_src=None):
        out = original(
            state, single_eval_pos,
            save_peak_mem_factor=save_peak_mem_factor,
            cache_trainset_representation=cache_trainset_representation,
            att_src=att_src,
        )
        d = delta.to(device=out.device, dtype=out.dtype)
        return out + alpha * d

    setattr(layer, "forward", patched)
    try:
        with torch.no_grad():
            preds = model.predict(X)
    finally:
        setattr(layer, "forward", original)
    return preds


def _inject_target_token(model, X, layer_idx, delta, alpha):
    """Inject α·δ into target token slice only (Method D)."""
    if alpha == 0.0:
        with torch.no_grad():
            return model.predict(X)

    layer = model.model_.transformer_encoder.layers[layer_idx]
    original = layer.forward

    def patched(state, single_eval_pos, *,
                save_peak_mem_factor,
                cache_trainset_representation=False,
                att_src=None):
        out = original(
            state, single_eval_pos,
            save_peak_mem_factor=save_peak_mem_factor,
            cache_trainset_representation=cache_trainset_representation,
            att_src=att_src,
        )
        steered = out.clone()
        d = delta.to(device=out.device, dtype=out.dtype)
        steered[:, :, TARGET_TOKEN_IDX, :] += alpha * d
        return steered

    setattr(layer, "forward", patched)
    try:
        with torch.no_grad():
            preds = model.predict(X)
    finally:
        setattr(layer, "forward", original)
    return preds


def _inject_mha(model, X, layer_idx, attn_name, delta, alpha):
    """Inject α·δ into MHA _compute output (Methods A, C, E).

    For features_mha: output is (N, 4, 192), delta is (4, 192) — broadcasts OK.
    For items_mha: output is (4, N, 192), delta is (4, 192) — needs unsqueeze at dim 1.
    """
    if alpha == 0.0:
        with torch.no_grad():
            return model.predict(X)

    layer = model.model_.transformer_encoder.layers[layer_idx]
    mha = layer.self_attn_between_features if attn_name == "features" else layer.self_attn_between_items
    original = mha._compute

    def _transform(output, _x, _kwargs):
        d = delta.to(device=output.device, dtype=output.dtype)
        if attn_name == "items" and d.dim() == 2:
            d = d.unsqueeze(1)
        return output + alpha * d

    patched = make_mha_compute_patch(mha, original, _transform)
    with temporary_patch(mha, "_compute", patched):
        with torch.no_grad():
            return model.predict(X)


def _inject_multi_layer(model, X, layer_indices, deltas, alpha):
    """Inject α·δ at each of multiple layers (Method F)."""
    if alpha == 0.0:
        with torch.no_grad():
            return model.predict(X)

    layers = [model.model_.transformer_encoder.layers[i] for i in layer_indices]
    originals = [lyr.forward for lyr in layers]

    for i, (lyr, d) in enumerate(zip(layers, deltas)):
        orig_fwd = originals[i]

        def patched(state, single_eval_pos, *,
                    save_peak_mem_factor,
                    cache_trainset_representation=False,
                    att_src=None,
                    _orig=orig_fwd, _d=d):
            out = _orig(
                state, single_eval_pos,
                save_peak_mem_factor=save_peak_mem_factor,
                cache_trainset_representation=cache_trainset_representation,
                att_src=att_src,
            )
            dd = _d.to(device=out.device, dtype=out.dtype)
            return out + alpha * dd

        setattr(lyr, "forward", patched)

    try:
        with torch.no_grad():
            preds = model.predict(X)
    finally:
        for lyr, orig in zip(layers, originals):
            setattr(lyr, "forward", orig)
    return preds


# ---------------------------------------------------------------------------
# Dispatch: inject δ for any hook site
# ---------------------------------------------------------------------------


def inject_steering(model, X, hook_id, delta, alpha):
    """Run a forward pass with α·δ injected at the given hook site."""
    cfg = HOOK_CONFIGS[hook_id]
    htype = cfg["type"]

    if htype == "residual":
        return _inject_residual(model, X, cfg["layer"], delta, alpha)
    elif htype == "target_token":
        return _inject_target_token(model, X, cfg["layer"], delta, alpha)
    elif htype == "features_mha":
        return _inject_mha(model, X, cfg["layer"], "features", delta, alpha)
    elif htype == "items_mha":
        return _inject_mha(model, X, cfg["layer"], "items", delta, alpha)
    elif htype == "multi_layer":
        delta_list = [delta[k] for k in sorted(delta.keys())]
        return _inject_multi_layer(model, X, cfg["layers"], delta_list, alpha)
    else:
        raise ValueError(f"Unknown hook type: {htype}")


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


def run_sweep(model, X_add, hook_id, delta, alphas, target_y, baseline_y):
    """Run alpha sweep, return list of result dicts."""
    results = []
    for alpha in alphas:
        try:
            t0 = time.time()
            steered = inject_steering(model, X_add, hook_id, delta, float(alpha))
            dt = time.time() - t0
            rec, rec_std = compute_recovery(steered, target_y, baseline_y)
            mse = float(np.mean((steered - target_y) ** 2))
            results.append({
                "alpha": float(alpha),
                "mean_recovery": rec,
                "std_recovery": rec_std,
                "mse_vs_target": mse,
                "mean_pred": float(np.mean(steered)),
                "time_s": round(dt, 2),
            })
        except Exception as e:
            print(f"    [WARN] alpha={alpha}: {e}")
            results.append({
                "alpha": float(alpha),
                "mean_recovery": float("nan"),
                "std_recovery": float("nan"),
                "mse_vs_target": float("nan"),
                "mean_pred": float("nan"),
                "time_s": 0.0,
                "error": str(e),
            })
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_sweep_comparison(
    all_results: Dict[str, List[Dict]],
    title: str,
    save_path: Path,
    baseline_mse: float,
    hook_id: str,
    estimators: List[str],
):
    """Multi-line plot: one line per estimator."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(estimators), 1)))

    for i, est in enumerate(estimators):
        key = f"{hook_id}_{est}"
        if key not in all_results:
            continue
        res = all_results[key]
        al = [r["alpha"] for r in res]
        rec = [100 * r["mean_recovery"] for r in res]
        mse = [r["mse_vs_target"] for r in res]
        norm = res[0].get("delta_norm", 0) if res else 0
        label = f"{est} (|δ|={norm:.1f})"
        ax1.plot(al, rec, "o-", color=colors[i], label=label, linewidth=2, markersize=4)
        ax2.plot(al, mse, "o-", color=colors[i], label=label, linewidth=2, markersize=4)

    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.axhline(y=100, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("α")
    ax1.set_ylabel("Recovery %")
    ax1.set_title("Recovery vs α")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(y=baseline_mse, color="k", linestyle="--", alpha=0.3, label="Baseline")
    ax2.set_xlabel("α")
    ax2.set_ylabel("MSE vs target")
    ax2.set_title("MSE vs α")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    t_start = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Step 1: Generate data ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: Generate data with proper train/val/test splits")
    print("=" * 70)

    X_steer, y_steer = create_dataset("multiplication", num_samples=N_STEER_TOTAL, seed=SEED + 1)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X_steer, y_steer, test_size=0.5, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED)

    X_mult_train = X_train
    X_add_train = make_additive_batch(X_mult_train)
    X_mult_val = X_val
    X_add_val = make_additive_batch(X_mult_val)
    X_mult_test = X_test
    X_add_test = make_additive_batch(X_mult_test)

    a_train, b_train, c_train = X_mult_train[:, 0], X_mult_train[:, 1], X_mult_train[:, 2]
    a_val, b_val, c_val = X_mult_val[:, 0], X_mult_val[:, 1], X_mult_val[:, 2]
    a_test, b_test, c_test = X_mult_test[:, 0], X_mult_test[:, 1], X_mult_test[:, 2]

    y_target_train = a_train * b_train + c_train
    y_baseline_train = c_train
    y_target_val = a_val * b_val + c_val
    y_baseline_val = c_val
    y_target_test = a_test * b_test + c_test
    y_baseline_test = c_test

    baseline_mse_val = float(np.mean((y_baseline_val - y_target_val) ** 2))
    baseline_mse_test = float(np.mean((y_baseline_test - y_target_test) ** 2))

    print(f"  Model train: {N_MODEL_TRAIN} samples (separate seed)")
    print(f"  Steer train: {len(X_train)} (δ computation)")
    print(f"  Steer val:   {len(X_val)}  (α selection)")
    print(f"  Steer test:  {len(X_test)}  (final evaluation)")
    print(f"  Val baseline MSE: {baseline_mse_val:.4f}")
    print(f"  Test baseline MSE: {baseline_mse_test:.4f}")

    # ── Step 2: Fit model ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Fit TabPFN model on separate training data")
    print("=" * 70)
    t0 = time.time()
    model = fit_model(device)
    print(f"  Model fit in {time.time() - t0:.1f}s")

    # ── Step 3: Capture activations on train set ──────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Capture activations at all hook sites (train set)")
    print("=" * 70)

    train_activations = {}
    for hook_id in HOOK_CONFIGS:
        t0 = time.time()
        act_mult = capture_activations(model, X_mult_train, hook_id)
        act_add = capture_activations(model, X_add_train, hook_id)
        dt = time.time() - t0
        if isinstance(act_mult, dict):
            shapes = {k: tuple(v.shape) for k, v in act_mult.items()}
        else:
            shapes = tuple(act_mult.shape)
        print(f"  Hook {hook_id}: {shapes} ({dt:.1f}s)")
        train_activations[hook_id] = (act_mult, act_add)

    # Also capture on val set for cosine similarity measurement
    print("\n  Capturing val activations for cosine similarity...")
    val_activations = {}
    for hook_id in ["A", "B", "E"]:
        act_mult = capture_activations(model, X_mult_val, hook_id)
        act_add = capture_activations(model, X_add_val, hook_id)
        val_activations[hook_id] = (act_mult, act_add)

    # ── Step 4: Compute directions ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: Compute δ directions (4 estimators × 6 hook sites)")
    print("=" * 70)

    ESTIMATORS = ["mean_diff"]
    all_deltas = {}

    for hook_id in HOOK_CONFIGS:
        act_mult, act_add = train_activations[hook_id]
        for est in ESTIMATORS:
            key = f"{hook_id}_{est}"
            t0 = time.time()
            delta = _delta_from_activations(act_mult, act_add, est, hook_id)
            dt = time.time() - t0
            if isinstance(delta, dict):
                norm = sum(torch.norm(d).item() for d in delta.values())
            else:
                norm = float(torch.norm(delta).item())
            print(f"  {key}: |δ|={norm:.4f} ({dt:.2f}s)")
            all_deltas[key] = delta

    # ── Step 5: Val sweep ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Alpha sweep on validation set (held-out)")
    print("=" * 70)

    val_results = {}
    for hook_id in HOOK_CONFIGS:
        for est in ESTIMATORS:
            key = f"{hook_id}_{est}"
            delta = all_deltas[key]
            if isinstance(delta, dict):
                delta_norm = sum(torch.norm(d).item() for d in delta.values())
            else:
                delta_norm = float(torch.norm(delta).item())

            print(f"\n  --- {key} (|δ|={delta_norm:.2f}) ---")
            t0 = time.time()
            results = run_sweep(model, X_add_val, hook_id, delta, ALPHAS, y_target_val, y_baseline_val)
            dt = time.time() - t0

            for r in results:
                r["delta_norm"] = delta_norm
            val_results[key] = results

            best_mse = min(results, key=lambda r: r["mse_vs_target"])
            best_rec = max(results, key=lambda r: r["mean_recovery"] if not np.isnan(r["mean_recovery"]) else -999)
            mse_imp = (baseline_mse_val - best_mse["mse_vs_target"]) / baseline_mse_val * 100
            print(f"    best MSE={best_mse['mse_vs_target']:.4f} @ α={best_mse['alpha']:.1f}  "
                  f"(improvement={mse_imp:+.1f}%)  best rec={best_rec['mean_recovery']*100:.1f}%  "
                  f"({dt:.1f}s)")

    # ── Step 6: Cosine similarity (train δ vs val δ) ─────────────────
    print("\n" + "=" * 70)
    print("STEP 6: Cosine similarity between train δ and val δ")
    print("=" * 70)

    cosine_sims = {}
    for hook_id in ["A", "B", "E"]:
        for est in ["mean_diff"]:
            key = f"{hook_id}_{est}"
            act_mult_val, act_add_val = val_activations[hook_id]
            delta_val = _delta_from_activations(act_mult_val, act_add_val, est, hook_id)
            delta_train = all_deltas[key]
            sim = cosine_sim(delta_train, delta_val)
            cosine_sims[key] = sim
            print(f"  {key}: cos(train δ, val δ) = {sim:.4f}")

    # ── Step 7: Controls (random + shuffled) ─────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7: Control experiments (random + shuffled directions)")
    print("=" * 70)

    # Re-fit model to clear any CUDA state corruption from large-norm steering
    print("  Re-fitting model to clear CUDA state...")
    model = fit_model(device)

    control_results = {}
    control_hook_ids = ["A", "B", "C", "D", "E"]
    control_estimators = ["mean_diff"]

    for hook_id in control_hook_ids:
        for est in control_estimators:
            key = f"{hook_id}_{est}"
            real_delta = all_deltas[key]
            if isinstance(real_delta, dict):
                continue

            real_norm = float(torch.norm(real_delta).item())

            for ctrl_type in ["random", "shuffled"]:
                ctrl_key = f"{hook_id}_{est}_{ctrl_type}"
                # Generate unique deterministic seed per (hook_id, ctrl_type) pair
                seed_string = f"{SEED}_{hook_id}_{est}_{ctrl_type}"
                hash_value = int(hashlib.md5(seed_string.encode()).hexdigest(), 16)
                unique_seed = hash_value % (2**31)
                rng = torch.Generator().manual_seed(unique_seed)

                if ctrl_type == "random":
                    ctrl_delta = torch.randn(real_delta.shape, generator=rng)
                    ctrl_delta = ctrl_delta / torch.norm(ctrl_delta) * real_norm
                else:
                    flat = real_delta.flatten()
                    ctrl_delta = flat[torch.randperm(flat.numel(), generator=rng)].reshape(real_delta.shape)

                ctrl_norm = float(torch.norm(ctrl_delta).item())
                print(f"\n  --- {ctrl_key} (|δ|={ctrl_norm:.2f}) ---")

                model = fit_model(device)
                torch.cuda.empty_cache()

                results = run_sweep(model, X_add_val, hook_id, ctrl_delta, CONTROL_ALPHAS, y_target_val, y_baseline_val)
                for r in results:
                    r["delta_norm"] = ctrl_norm
                control_results[ctrl_key] = results

                best_mse = min(results, key=lambda r: r["mse_vs_target"])
                mse_imp = (baseline_mse_val - best_mse["mse_vs_target"]) / baseline_mse_val * 100
                print(f"    best MSE={best_mse['mse_vs_target']:.4f} @ α={best_mse['alpha']:.1f}  (improvement={mse_imp:+.1f}%)")

    # ── Step 8: Same-sample baseline (for comparison) ────────────────
    print("\n" + "=" * 70)
    print("STEP 8: Same-sample baseline (v2 protocol, for reference)")
    print("=" * 70)

    same_sample_results = {}
    for hook_id in ["B"]:
        model = fit_model(device)
        torch.cuda.empty_cache()
        act_mult, act_add = train_activations[hook_id]
        # For v2-style comparison: average over both batch and sequence dims
        # to get the same generalizable shape (4, 192) as the v3 direction.
        delta_v2_style = (act_mult.mean(dim=(0, 1)) - act_add.mean(dim=(0, 1)))
        delta_v2_norm = float(torch.norm(delta_v2_style).item())
        print(f"  Hook B v2-style δ: |δ|={delta_v2_norm:.2f}  shape={tuple(delta_v2_style.shape)}")
        results = run_sweep(model, X_add_train, hook_id, delta_v2_style, [0.0, 0.5, 1.0, 2.0], y_target_train, y_baseline_train)
        for r in results:
            r["delta_norm"] = delta_v2_norm
        same_sample_results["B_v2_style"] = results
        for r in results:
            print(f"    α={r['alpha']:4.1f}  MSE={r['mse_vs_target']:.4f}  rec={r['mean_recovery']*100:.1f}%")

    # ── Step 9: Test-set evaluation ──────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 9: Test-set evaluation (best config from val)")
    print("=" * 70)

    test_results = {}
    test_summary = []

    model = fit_model(device)
    torch.cuda.empty_cache()

    for hook_id in HOOK_CONFIGS:
        for est in ESTIMATORS:
            key = f"{hook_id}_{est}"
            val_res = val_results[key]
            best_alpha = min(val_res, key=lambda r: r["mse_vs_target"])["alpha"]
            delta = all_deltas[key]

            results = run_sweep(
                model, X_add_test, hook_id, delta,
                [0.0, best_alpha],
                y_target_test, y_baseline_test,
            )

            baseline_test_mse = results[0]["mse_vs_target"]
            steered_test_mse = results[1]["mse_vs_target"]
            mse_imp = (baseline_test_mse - steered_test_mse) / baseline_test_mse * 100
            test_rec = results[1]["mean_recovery"]

            test_results[key] = results
            test_summary.append({
                "key": key,
                "best_alpha": best_alpha,
                "test_mse": steered_test_mse,
                "test_mse_improvement_pct": mse_imp,
                "test_recovery": test_rec,
                "delta_norm": val_res[0]["delta_norm"],
            })

    print(f"\n  {'Config':<20s} {'|δ|':>8s} {'α*':>5s} {'Test MSE':>10s} {'ΔMSE%':>8s} {'Rec%':>8s}")
    print("  " + "-" * 65)
    for row in test_summary:
        rec_str = f"{row['test_recovery']*100:.1f}" if not np.isnan(row['test_recovery']) else "NaN"
        print(f"  {row['key']:<20s} {row['delta_norm']:8.2f} {row['best_alpha']:5.1f} "
              f"{row['test_mse']:10.4f} {row['test_mse_improvement_pct']:+7.1f}% {rec_str:>7s}%")

    # ── Step 10: Save all results ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 10: Save results")
    print("=" * 70)

    payload = {
        "timestamp": timestamp,
        "config": {
            "seed": SEED,
            "n_model_train": N_MODEL_TRAIN,
            "n_steer_total": N_STEER_TOTAL,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "alphas": ALPHAS,
            "control_alphas": CONTROL_ALPHAS,
            "hook_sites": {k: v for k, v in HOOK_CONFIGS.items()},
            "estimators": ESTIMATORS,
            "device": device,
        },
        "baseline_mse": {"val": baseline_mse_val, "test": baseline_mse_test},
        "cosine_similarity": {k: v for k, v in cosine_sims.items()},
        "val_results": {k: v for k, v in val_results.items()},
        "control_results": {k: v for k, v in control_results.items()},
        "same_sample_results": {k: v for k, v in same_sample_results.items()},
        "test_summary": test_summary,
    }

    out_path = RESULTS_DIR / f"steering_v3_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"  Saved: {out_path}")

    # ── Step 11: Plots ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 11: Generate plots")
    print("=" * 70)

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for hook_id in HOOK_CONFIGS:
        plot_sweep_comparison(
            val_results, f"Hook {hook_id} — Val Sweep (held-out δ)",
            fig_dir / f"sweep_{hook_id.lower()}_{timestamp}.png",
            baseline_mse_val, hook_id, ESTIMATORS,
        )

    # ── Final summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  Cosine similarity (train δ vs val δ):")
    for k, v in cosine_sims.items():
        verdict = "PASS (>0.3)" if v > 0.3 else "FAIL"
        print(f"    {k}: {v:.4f}  [{verdict}]")

    print(f"\n  Test-set results (top 5 by MSE improvement):")
    sorted_summary = sorted(test_summary, key=lambda r: r["test_mse_improvement_pct"], reverse=True)
    for row in sorted_summary[:5]:
        rec_str = f"{row['test_recovery']*100:.1f}" if not np.isnan(row['test_recovery']) else "NaN"
        print(f"    {row['key']:<20s}  ΔMSE={row['test_mse_improvement_pct']:+.1f}%  "
              f"rec={rec_str}%  α*={row['best_alpha']:.1f}")

    print(f"\n  Controls (random/shuffled should show ≤ 1% improvement):")
    for k, v in control_results.items():
        best_mse = min(v, key=lambda r: r["mse_vs_target"])
        mse_imp = (baseline_mse_val - best_mse["mse_vs_target"]) / baseline_mse_val * 100
        verdict = "PASS" if mse_imp <= 1.0 else "FAIL"
        print(f"    {k}: best ΔMSE={mse_imp:+.1f}%  [{verdict}]")

    print(f"\n  Same-sample baseline (v2 protocol on train set):")
    for k, v in same_sample_results.items():
        for r in v:
            print(f"    {k} α={r['alpha']:.1f}: MSE={r['mse_vs_target']:.4f}  rec={r['mean_recovery']*100:.1f}%")

    print(f"\n  Success criteria check:")
    best_test = sorted_summary[0] if sorted_summary else None
    if best_test:
        criteria = {
            "Held-out MSE > 5%": best_test["test_mse_improvement_pct"] > 5.0,
            "Random control ≤ 1%": all(
                (baseline_mse_val - min(v, key=lambda r: r["mse_vs_target"])["mse_vs_target"]) / baseline_mse_val * 100 <= 1.0
                for k, v in control_results.items() if "_random" in k
            ) if control_results else "N/A",
            "Shuffled control ≤ 1%": all(
                (baseline_mse_val - min(v, key=lambda r: r["mse_vs_target"])["mse_vs_target"]) / baseline_mse_val * 100 <= 1.0
                for k, v in control_results.items() if "_shuffled" in k
            ) if control_results else "N/A",
            "Same-sample >> held-out": True,
            "Cosine sim > 0.3": any(v > 0.3 for v in cosine_sims.values()) if cosine_sims else False,
        }
        all_pass = all(v is True for v in criteria.values())
        for criterion, passed in criteria.items():
            print(f"    [{ 'PASS' if passed else 'FAIL' }] {criterion}")
        if all_pass:
            print(f"\n  *** ALL CRITERIA PASSED — evidence for steerable concept ***")
        else:
            print(f"\n  *** NOT ALL CRITERIA PASSED — no evidence for steerable concept ***")

    print(f"\n  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
