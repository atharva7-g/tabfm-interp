from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention

from src.datasets.synthetic import create_dataset

SEED = 42
N_EVAL = 32
ALPHA_MIN = 0.0
ALPHA_MAX = 10.0
ALPHA_STEP = 0.5
DPI = 200

RESULTS_DIR = Path("results_contrastive_steering_v2")


def alpha_grid() -> np.ndarray:
    return np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_STEP / 2, ALPHA_STEP)


def fit_model(device: str) -> Tuple[TabPFNRegressor, np.ndarray, np.ndarray]:
    X, y = create_dataset("multiplication", num_samples=1000, seed=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.8,
        random_state=SEED,
    )
    model = TabPFNRegressor(device=device, n_estimators=1)
    model.fit(X_train, y_train)
    return model, X_test[:N_EVAL], y_test[:N_EVAL]


def make_additive_batch(X_mult: np.ndarray) -> np.ndarray:
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


def run_alpha_sweep(
    model: TabPFNRegressor,
    X_add: np.ndarray,
    delta: torch.Tensor,
    inject_fn: Callable[[TabPFNRegressor, np.ndarray, torch.Tensor, float], np.ndarray],
    alphas: Iterable[float],
    target_y: np.ndarray,
    baseline_y: np.ndarray,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for alpha in alphas:
        alpha_f = float(alpha)
        steered = inject_fn(model, X_add, delta, alpha_f)
        mean_rec, std_rec = compute_recovery(steered, target_y, baseline_y)
        mse = float(np.mean((steered - target_y) ** 2))
        results.append(
            {
                "alpha": alpha_f,
                "mean_recovery": mean_rec,
                "std_recovery": std_rec,
                "mse_vs_target": mse,
                "mean_pred": float(np.mean(steered)),
            }
        )
    return results


def plot_results(
    results: List[Dict[str, float]],
    title: str,
    save_path: Path,
    baseline_mse: float,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    alphas = [r["alpha"] for r in results]
    recovery = [100.0 * r["mean_recovery"] for r in results]
    mse = [r["mse_vs_target"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(alphas, recovery, "o-", color="tab:red", linewidth=2, markersize=4)
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.axhline(y=100, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Steering coefficient alpha")
    ax1.set_ylabel("Recovery %")
    ax1.set_title("Recovery vs steering strength")
    ax1.grid(True, alpha=0.3)

    ax2.plot(alphas, mse, "o-", color="tab:blue", linewidth=2, markersize=4)
    ax2.axhline(y=baseline_mse, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Steering coefficient alpha")
    ax2.set_ylabel("MSE vs target")
    ax2.set_title("MSE vs steering strength")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_results(
    method_name: str,
    config: Dict[str, Any],
    results: List[Dict[str, float]],
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"steering_{method_name}_{timestamp}.json"
    payload = {
        "config": config,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def print_results_summary(method_name: str, results: List[Dict[str, float]]) -> None:
    best_recovery = max(
        results,
        key=lambda r: r["mean_recovery"] if not np.isnan(r["mean_recovery"]) else -1e9,
    )
    best_mse = min(results, key=lambda r: r["mse_vs_target"])
    print(
        f"[{method_name}] best recovery: {best_recovery['mean_recovery'] * 100:.1f}% "
        f"at alpha={best_recovery['alpha']:.1f}"
    )
    print(
        f"[{method_name}] best MSE: {best_mse['mse_vs_target']:.4f} "
        f"at alpha={best_mse['alpha']:.1f}"
    )


@contextmanager
def temporary_patch(obj: Any, attribute: str, replacement: Any) -> Iterator[Any]:
    original = getattr(obj, attribute)
    setattr(obj, attribute, replacement)
    try:
        yield original
    finally:
        setattr(obj, attribute, original)


def make_mha_compute_patch(
    mha: MultiHeadAttention,
    original_compute: Callable[..., torch.Tensor],
    transform_fn: Callable[[torch.Tensor, torch.Tensor, Dict[str, Any]], torch.Tensor],
) -> Callable[..., torch.Tensor]:
    def patched_compute(x, x_kv, k_cache, v_cache, kv_cache, **kwargs):
        if kwargs.get("save_peak_mem_factor", None) is not None:
            return original_compute(x, x_kv, k_cache, v_cache, kv_cache, **kwargs)

        q, k, v, kv, qkv = mha.compute_qkv(
            x,
            x_kv,
            k_cache,
            v_cache,
            kv_cache,
            cache_kv=kwargs.get("cache_kv", False),
            use_cached_kv=kwargs.get("use_cached_kv", False),
            reuse_first_head_kv=kwargs.get("reuse_first_head_kv", False),
        )
        head_outputs = MultiHeadAttention.compute_attention_heads(
            q,
            k,
            v,
            kv,
            qkv,
            mha.dropout_p,
            mha.softmax_scale,
        )
        raw_output = torch.einsum("... h d, h d s -> ... s", head_outputs, mha._w_out)
        raw_output = transform_fn(raw_output, x, kwargs)

        if kwargs.get("add_input", False):
            return x + raw_output
        return raw_output

    return patched_compute


def capture_mha_output(
    model: TabPFNRegressor,
    X: np.ndarray,
    mha: MultiHeadAttention,
) -> torch.Tensor:
    captured: Dict[str, torch.Tensor] = {}
    original_compute = mha._compute

    def _transform(output: torch.Tensor, _x: torch.Tensor, _kwargs: Dict[str, Any]) -> torch.Tensor:
        if "output" not in captured:
            captured["output"] = output.detach().clone()
        return output

    patched_compute = make_mha_compute_patch(mha, original_compute, _transform)
    with temporary_patch(mha, "_compute", patched_compute):
        with torch.no_grad():
            model.predict(X)

    if "output" not in captured:
        raise RuntimeError("Failed to capture MHA output.")
    return captured["output"]
