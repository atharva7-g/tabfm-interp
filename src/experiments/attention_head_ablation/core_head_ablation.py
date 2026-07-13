from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention


def _make_ablation_compute(mha, head_idx: int) -> Callable:
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
        head_outputs[:, :, head_idx, :] = 0.0
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


def run_single_head_ablation(
    regressor: TabPFNRegressor,
    X: np.ndarray,
    layer_idx: int,
    head_idx: int,
    ratio_epsilon: float = 0.05,
    y_scale: Optional[float] = None,
) -> Dict[str, float]:
    mha = _get_mha_module(regressor, layer_idx)
    original_compute = mha._compute

    mha._compute = _make_ablation_compute(mha, head_idx)

    with torch.no_grad():
        y_ablated = regressor.predict(X)

    mha._compute = original_compute

    with torch.no_grad():
        y_normal = regressor.predict(X)

    y_ablated_arr = np.asarray(y_ablated, dtype=np.float64).reshape(-1)
    y_normal_arr = np.asarray(y_normal, dtype=np.float64).reshape(-1)

    y_ablated_val = float(np.mean(y_ablated_arr))
    y_normal_val = float(np.mean(y_normal_arr))

    delta_arr = y_normal_arr - y_ablated_arr
    ablation_effect = float(np.mean(delta_arr))
    ablation_effect_abs_mean = float(np.mean(np.abs(delta_arr)))

    if abs(y_normal_val) > 1e-10:
        ablation_ratio = ablation_effect / abs(y_normal_val)
    else:
        ablation_ratio = 0.0

    if y_scale is not None:
        y_scale_val = float(max(y_scale, ratio_epsilon))
    else:
        y_scale_val = float(max(np.std(y_normal_arr), ratio_epsilon))

    stable_denominator = y_scale_val
    ablation_ratio_stable = ablation_effect / stable_denominator
    ablation_ratio_stable_abs = ablation_effect_abs_mean / stable_denominator

    return {
        "y_normal": y_normal_val,
        "y_ablated": y_ablated_val,
        "ablation_effect": ablation_effect,
        "ablation_effect_abs_mean": ablation_effect_abs_mean,
        "ablation_ratio": ablation_ratio,
        "ablation_ratio_stable": ablation_ratio_stable,
        "ablation_ratio_stable_abs": ablation_ratio_stable_abs,
        "ablation_effect_sigma": ablation_ratio_stable_abs,
        "ablation_effect_signed_sigma": ablation_ratio_stable,
        "y_scale": y_scale_val,
        "stable_denominator": stable_denominator,
        "ratio_epsilon": ratio_epsilon,
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "n_eval_samples": int(y_normal_arr.size),
    }


def sweep_heads_and_layers(
    regressor: TabPFNRegressor,
    X: np.ndarray,
    head_indices: Optional[List[int]] = None,
    layer_indices: Optional[List[int]] = None,
    ratio_epsilon: float = 0.05,
    y_scale: Optional[float] = None,
) -> List[Dict[str, float]]:
    num_heads = _get_num_heads(regressor)
    num_layers = _get_num_layers(regressor)

    if head_indices is None:
        head_indices = list(range(num_heads))
    if layer_indices is None:
        layer_indices = list(range(num_layers))

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
            result = run_single_head_ablation(
                regressor=regressor,
                X=X,
                layer_idx=layer_idx,
                head_idx=head_idx,
                ratio_epsilon=ratio_epsilon,
                y_scale=y_scale,
            )
            results.append(result)
            print(
                f" effect={result['ablation_effect_abs_mean']:.4f} "
                f"({result['ablation_effect_sigma']:.3f}σ)"
            )

    return results
