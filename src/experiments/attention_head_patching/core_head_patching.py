from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention

from src.experiments.hooks.core_patching import create_corrupted_input


def _make_caching_compute(mha, cache: Dict[str, torch.Tensor], layer_name: str) -> Callable:
    def patched_compute(x, x_kv, k_cache, v_cache, kv_cache, **kwargs):
        add_input = kwargs.get("add_input", False)
        save_peak_mem_factor = kwargs.get("save_peak_mem_factor", None)

        if save_peak_mem_factor is not None:
            split_size = (x.size(0) + save_peak_mem_factor - 1) // save_peak_mem_factor
            n_splits = (x.size(0) + split_size - 1) // split_size
            other_args = []
            for arg in (x_kv, k_cache, v_cache, kv_cache):
                if isinstance(arg, torch.Tensor):
                    other_args.append(torch.split(arg, split_size))
                else:
                    other_args.append([arg] * n_splits)
            result = x.clone()
            for i, x_chunk in enumerate(torch.split(x, split_size)):
                out = _inner(
                    x_chunk,
                    other_args[0][i],
                    other_args[1][i],
                    other_args[2][i],
                    other_args[3][i],
                    kwargs,
                )
                if add_input:
                    result[: x_chunk.size(0)] += out
                else:
                    result[: x_chunk.size(0)] = out
            return result

        out = _inner(x, x_kv, k_cache, v_cache, kv_cache, kwargs)
        if add_input:
            return x + out
        return out

    def _inner(x, x_kv, k_cache, v_cache, kv_cache, kwargs):
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
            q, k, v, kv, qkv, mha.dropout_p, mha.softmax_scale
        )
        cache[layer_name] = head_outputs.detach().clone()
        return torch.einsum("... h d, h d s -> ... s", head_outputs, mha._w_out)

    return patched_compute


def _make_patching_compute(
    mha,
    cache: Dict[str, torch.Tensor],
    layer_name: str,
    head_idx: int,
) -> Callable:
    def patched_compute(x, x_kv, k_cache, v_cache, kv_cache, **kwargs):
        add_input = kwargs.get("add_input", False)
        save_peak_mem_factor = kwargs.get("save_peak_mem_factor", None)

        if save_peak_mem_factor is not None:
            split_size = (x.size(0) + save_peak_mem_factor - 1) // save_peak_mem_factor
            n_splits = (x.size(0) + split_size - 1) // split_size
            other_args = []
            for arg in (x_kv, k_cache, v_cache, kv_cache):
                if isinstance(arg, torch.Tensor):
                    other_args.append(torch.split(arg, split_size))
                else:
                    other_args.append([arg] * n_splits)
            result = x.clone()
            for i, x_chunk in enumerate(torch.split(x, split_size)):
                out = _inner(
                    x_chunk,
                    other_args[0][i],
                    other_args[1][i],
                    other_args[2][i],
                    other_args[3][i],
                    kwargs,
                )
                if add_input:
                    result[: x_chunk.size(0)] += out
                else:
                    result[: x_chunk.size(0)] = out
            return result

        out = _inner(x, x_kv, k_cache, v_cache, kv_cache, kwargs)
        if add_input:
            return x + out
        return out

    def _inner(x, x_kv, k_cache, v_cache, kv_cache, kwargs):
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
            q, k, v, kv, qkv, mha.dropout_p, mha.softmax_scale
        )
        cached = cache[layer_name]
        head_outputs[:, :, head_idx, :] = cached[:, :, head_idx, :]
        return torch.einsum("... h d, h d s -> ... s", head_outputs, mha._w_out)

    return patched_compute


def _get_mha_module(regressor: TabPFNRegressor, layer_idx: int):
    model = regressor.model_
    layer = model.transformer_encoder.layers[layer_idx]
    return layer.self_attn_between_features


def _get_num_heads(regressor: TabPFNRegressor) -> int:
    mha = _get_mha_module(regressor, 0)
    return mha._nhead


def _get_num_layers(regressor: TabPFNRegressor) -> int:
    model = regressor.model_
    return len(model.transformer_encoder.layers)


def cache_clean_heads(
    regressor: TabPFNRegressor,
    X_clean: np.ndarray,
    layer_indices: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    model = regressor.model_
    num_layers = _get_num_layers(regressor)
    if layer_indices is None:
        layer_indices = list(range(num_layers))

    clean_head_cache: Dict[str, torch.Tensor] = {}
    original_computes = {}

    for li in layer_indices:
        mha = _get_mha_module(regressor, li)
        layer_name = f"layer_{li}"
        original_computes[li] = mha._compute
        mha._compute = _make_caching_compute(mha, clean_head_cache, layer_name)

    with torch.no_grad():
        regressor.predict(X_clean)

    for li in layer_indices:
        mha = _get_mha_module(regressor, li)
        mha._compute = original_computes[li]

    return clean_head_cache


def compute_absolute_recovery(
    y_clean_arr: np.ndarray,
    y_corrupt_arr: np.ndarray,
    y_patched_arr: np.ndarray,
    min_gap_percentile: float = 25.0,
) -> Dict[str, float]:
    """
    Compute per-sample absolute recovery, filtering out samples with small gaps.
    
    This addresses sign-cancellation in low-gap regimes (e.g., gaussian_replace on pairwise_50).
    
    Returns:
    - 'per_sample_mean_gap': mean absolute gap across all samples
    - 'per_sample_mean_restoration': mean absolute restoration across all samples
    - 'filtered_recovery_fraction': recovery % on samples with above-threshold gaps
    - 'samples_improved': fraction of samples where patching moved closer to clean
    - 'gap_threshold': the percentile-based threshold used for filtering
    """
    # Per-sample quantities
    per_sample_gap = np.abs(y_clean_arr - y_corrupt_arr)
    per_sample_restoration = np.abs(y_patched_arr - y_corrupt_arr)
    per_sample_residual = np.abs(y_clean_arr - y_patched_arr)
    
    # Gap threshold: samples below this are excluded as too noisy
    gap_threshold = np.percentile(per_sample_gap, min_gap_percentile)
    meaningful_gap_mask = per_sample_gap >= gap_threshold
    n_meaningful = np.sum(meaningful_gap_mask)
    
    # Recovery fraction for samples with meaningful gaps
    # How much of the gap did patching close?
    if n_meaningful > 0:
        # improvement = original_distance - new_distance
        improvements = per_sample_gap[meaningful_gap_mask] - per_sample_residual[meaningful_gap_mask]
        filtered_recovery_fraction = np.mean(improvements / per_sample_gap[meaningful_gap_mask])
    else:
        filtered_recovery_fraction = 0.0
        n_meaningful = 1  # avoid division by zero
    
    # Samples improved: did patching reduce distance to clean?
    samples_improved = np.sum(per_sample_residual < per_sample_gap) / len(y_clean_arr)
    
    return {
        "per_sample_mean_gap": float(np.mean(per_sample_gap)),
        "per_sample_mean_restoration": float(np.mean(per_sample_restoration)),
        "filtered_recovery_fraction": float(filtered_recovery_fraction),
        "samples_improved_fraction": float(samples_improved),
        "gap_threshold_used": float(gap_threshold),
        "n_meaningful_samples": int(n_meaningful),
        "total_samples": int(len(y_clean_arr)),
    }


def run_single_head_patching(
    regressor: TabPFNRegressor,
    X_clean: np.ndarray,
    X_corrupt: np.ndarray,
    layer_idx: int,
    head_idx: int,
    clean_head_cache: Dict[str, torch.Tensor],
    n_train_samples: int,
    corrupt_idx: Union[int, List[int]],
    ratio_epsilon: float = 0.05,
    ratio_threshold: Optional[float] = None,
    y_scale: Optional[float] = None,
    metric_mode: str = "regime",
) -> Dict[str, float]:
    model = regressor.model_
    layer_name = f"layer_{layer_idx}"
    mha = _get_mha_module(regressor, layer_idx)

    original_compute = mha._compute
    mha._compute = _make_patching_compute(
        mha, clean_head_cache, layer_name, head_idx
    )

    with torch.no_grad():
        y_patched = regressor.predict(X_corrupt)

    mha._compute = original_compute

    with torch.no_grad():
        y_clean = regressor.predict(X_clean)
        y_corrupt = regressor.predict(X_corrupt)

    y_clean_arr = np.asarray(y_clean, dtype=np.float64).reshape(-1)
    y_corrupt_arr = np.asarray(y_corrupt, dtype=np.float64).reshape(-1)
    y_patched_arr = np.asarray(y_patched, dtype=np.float64).reshape(-1)

    delta_gap_arr = y_clean_arr - y_corrupt_arr
    delta_restoration_arr = y_patched_arr - y_corrupt_arr
    delta_residual_arr = y_clean_arr - y_patched_arr

    y_clean_val = float(np.mean(y_clean_arr))
    y_corrupt_val = float(np.mean(y_corrupt_arr))
    y_patched_val = float(np.mean(y_patched_arr))

    restoration = float(np.mean(delta_restoration_arr))
    restoration_abs_mean = float(np.mean(np.abs(delta_restoration_arr)))
    clean_corrupt_diff = float(np.mean(delta_gap_arr))
    clean_corrupt_gap_abs_mean = float(np.mean(np.abs(delta_gap_arr)))
    residual_abs_mean = float(np.mean(np.abs(delta_residual_arr)))

    if abs(clean_corrupt_diff) > 1e-10:
        recovery_ratio = restoration / clean_corrupt_diff
    else:
        recovery_ratio = 0.0

    ratio_threshold_val = (
        float(ratio_threshold) if ratio_threshold is not None else float(ratio_epsilon)
    )
    ratio_valid_signed = abs(clean_corrupt_diff) >= ratio_threshold_val
    ratio_valid_abs = clean_corrupt_gap_abs_mean >= ratio_threshold_val

    if ratio_valid_signed:
        recovery_fractional_signed = restoration / clean_corrupt_diff
    else:
        recovery_fractional_signed = None

    if ratio_valid_abs:
        recovery_fractional_abs = restoration_abs_mean / clean_corrupt_gap_abs_mean
    else:
        recovery_fractional_abs = None

    safe_denominator = max(abs(clean_corrupt_diff), ratio_epsilon)
    recovery_ratio_stable = restoration / safe_denominator
    recovery_score = 1.0 - (abs(y_patched_val - y_clean_val) / safe_denominator)
    recovery_score = float(np.clip(recovery_score, -1.0, 1.0))

    if y_scale is not None:
        y_scale_val = float(max(float(y_scale), 1e-12))
    else:
        y_scale_val = float(max(np.std(y_clean_arr), 1e-12))

    restoration_sigma = restoration_abs_mean / y_scale_val
    residual_sigma = residual_abs_mean / y_scale_val

    # Compute per-sample absolute recovery (addresses sign-cancellation)
    abs_recovery_metrics = compute_absolute_recovery(
        y_clean_arr, y_corrupt_arr, y_patched_arr, min_gap_percentile=25.0
    )

    low_gap_regime = not ratio_valid_signed
    if ratio_valid_signed and recovery_fractional_signed is not None:
        recovery_score_regime = recovery_fractional_signed
    else:
        recovery_score_regime = restoration_sigma

    if metric_mode == "legacy":
        recovery_primary = recovery_score
        recovery_primary_metric = "recovery_score"
    else:
        recovery_primary = recovery_score_regime
        recovery_primary_metric = (
            "restoration_sigma" if low_gap_regime else "recovery_fractional_signed"
        )

    return {
        "y_clean": y_clean_val,
        "y_corrupt": y_corrupt_val,
        "y_patched": y_patched_val,
        "restoration": restoration,
        "restoration_abs_mean": restoration_abs_mean,
        "clean_corrupt_gap_abs_mean": clean_corrupt_gap_abs_mean,
        "residual_abs_mean": residual_abs_mean,
        "recovery_ratio": recovery_ratio,
        "recovery_ratio_stable": recovery_ratio_stable,
        "recovery_score": recovery_score,
        "recovery_primary": recovery_primary,
        "recovery_primary_metric": recovery_primary_metric,
        "recovery_score_regime": recovery_score_regime,
        "recovery_fractional_signed": recovery_fractional_signed,
        "recovery_fractional_abs": recovery_fractional_abs,
        "ratio_threshold": ratio_threshold_val,
        "ratio_valid_signed": ratio_valid_signed,
        "ratio_valid_abs": ratio_valid_abs,
        "low_gap_regime": low_gap_regime,
        "restoration_sigma": restoration_sigma,
        "residual_sigma": residual_sigma,
        "y_scale": y_scale_val,
        "clean_corrupt_diff": clean_corrupt_diff,
        "safe_denominator": safe_denominator,
        "n_eval_samples": int(y_clean_arr.size),
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        # Per-sample absolute recovery metrics (addresses sign-cancellation)
        "per_sample_mean_gap": abs_recovery_metrics["per_sample_mean_gap"],
        "per_sample_mean_restoration": abs_recovery_metrics["per_sample_mean_restoration"],
        "filtered_recovery_fraction": abs_recovery_metrics["filtered_recovery_fraction"],
        "samples_improved_fraction": abs_recovery_metrics["samples_improved_fraction"],
        "gap_threshold_used": abs_recovery_metrics["gap_threshold_used"],
        "n_meaningful_samples": abs_recovery_metrics["n_meaningful_samples"],
    }


def sweep_heads_and_layers(
    regressor: TabPFNRegressor,
    X_clean: np.ndarray,
    X_corrupt: np.ndarray,
    corrupt_idx: Union[int, List[int]],
    n_train_samples: int,
    head_indices: Optional[List[int]] = None,
    layer_indices: Optional[List[int]] = None,
    ratio_epsilon: float = 0.05,
    ratio_threshold: Optional[float] = None,
    y_scale: Optional[float] = None,
    metric_mode: str = "regime",
) -> List[Dict[str, float]]:
    num_heads = _get_num_heads(regressor)
    num_layers = _get_num_layers(regressor)

    if head_indices is None:
        head_indices = list(range(num_heads))
    if layer_indices is None:
        layer_indices = list(range(num_layers))

    print(f"Caching clean attention head outputs for {num_layers} layers...")
    clean_head_cache = cache_clean_heads(regressor, X_clean, layer_indices)
    print(
        f"Cached {len(clean_head_cache)} layers, "
        f"head output shape: {list(clean_head_cache.values())[0].shape}"
    )

    results = []
    total = len(head_indices) * len(layer_indices)
    done = 0
    for head_idx in head_indices:
        for layer_idx in layer_indices:
            done += 1
            print(
                f"  [{done}/{total}] Head {head_idx}, Layer {layer_idx}...",
                end="",
                flush=True,
            )
            result = run_single_head_patching(
                regressor=regressor,
                X_clean=X_clean,
                X_corrupt=X_corrupt,
                layer_idx=layer_idx,
                head_idx=head_idx,
                clean_head_cache=clean_head_cache,
                n_train_samples=n_train_samples,
                corrupt_idx=corrupt_idx,
                ratio_epsilon=ratio_epsilon,
                ratio_threshold=ratio_threshold,
                y_scale=y_scale,
                metric_mode=metric_mode,
            )
            results.append(result)
            print(
                f" effect={result['restoration_abs_mean']:.4f} "
                f"({result['restoration_sigma']:.3f}σ)"
            )

    return results
