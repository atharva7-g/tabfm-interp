#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.utils import set_seed, get_project_root
from src.datasets import create_dataset, get_dataset_formula
from src.experiments.hooks.base import ExperimentConfig
from src.experiments.hooks.core_patching import create_corrupted_input
from src.experiments.attention_head_patching.core_head_patching import (
    _get_num_heads,
    _get_num_layers,
)
from src.experiments.attention_head_patching.head_patching_experiment import (
    HeadPatchingExperiment,
)
from src.tracking import AimExperimentTracker
from tabpfn import TabPFNRegressor
from sklearn.model_selection import train_test_split

VALID_CORRUPTION_MODES = [
    "gaussian_replace",
    "gaussian_add",
    "mean_shift",
    "scale",
    "sign_flip",
    "fixed",
    "zero",
    "permute",
]

VALID_METRIC_MODES = ["legacy", "regime"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run true MHA attention head patching experiments for TabPFN"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pairwise_50",
        help="Dataset type (default: pairwise_50)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=64,
        help="Number of test samples to evaluate (default: 64)",
    )
    parser.add_argument(
        "--corrupt-idx",
        type=str,
        default="0",
        help="Feature index to corrupt (int or comma-separated ints, default: 0)",
    )
    parser.add_argument(
        "--corruption-mode",
        type=str,
        default="gaussian_replace",
        choices=VALID_CORRUPTION_MODES,
        help="Corruption mode (default: gaussian_replace)",
    )
    parser.add_argument(
        "--corruption-strength",
        type=float,
        default=1.0,
        help="Corruption strength (default: 1.0)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=1.0,
        help="Noise std for corruption (default: 1.0)",
    )
    parser.add_argument(
        "--metric-mode",
        type=str,
        default="regime",
        choices=VALID_METRIC_MODES,
        help="Metric mode (default: regime)",
    )
    parser.add_argument(
        "--ratio-epsilon",
        type=float,
        default=0.05,
        help="Minimum denominator for stable metrics (default: 0.05)",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=None,
        help="Gap threshold for fractional recovery validity",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Max layers to sweep (default: all)",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default=None,
        help="Comma-separated head indices to patch (default: all 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of dataset samples (default: 1000)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.8,
        help="Test split fraction (default: 0.8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_attention_heads_mha",
        help="Output directory (default: results_attention_heads_mha)",
    )
    return parser.parse_args()


def parse_corrupt_idx(value: str):
    value = value.strip()
    if "," in value:
        return [int(v.strip()) for v in value.split(",") if v.strip() != ""]
    return int(value)


def parse_head_indices(value: str | None, num_heads: int) -> list[int]:
    if value is None:
        return list(range(num_heads))
    return [int(v.strip()) for v in value.split(",") if v.strip() != ""]


def compute_scale_and_threshold(
    y_eval: np.ndarray,
    y_reference: np.ndarray,
    ratio_threshold_arg: float | None,
) -> tuple[float, float, str, str]:
    min_scale = 1e-12
    eval_std = float(np.std(y_eval))
    if y_eval.size >= 2 and eval_std > min_scale:
        y_scale = eval_std
        y_scale_source = "eval_std"
    else:
        reference_std = float(np.std(y_reference))
        if reference_std > min_scale:
            y_scale = reference_std
            y_scale_source = "test_std_fallback"
        else:
            y_scale = float(max(np.mean(np.abs(y_reference)), min_scale))
            y_scale_source = "abs_mean_fallback"

    if ratio_threshold_arg is not None:
        ratio_threshold = float(ratio_threshold_arg)
        threshold_source = "user"
    else:
        ratio_threshold = 0.05 * y_scale
        threshold_source = "auto_0.05_y_scale"
    return y_scale, ratio_threshold, threshold_source, y_scale_source


def format_regime_metric(summary: dict[str, Any]) -> str:
    if summary.get("best_recovery_metric") == "recovery_score":
        return f"{summary['best_recovery'] * 100:.2f}%"
    if summary.get("best_recovery_metric") == "recovery_fractional_signed":
        value = summary.get("best_recovery_fractional_signed")
        if value is None:
            return "n/a"
        return f"{value * 100:.2f}%"
    return f"{summary['best_restoration_sigma']:.3f} std"


def main():
    args = parse_args()

    corrupt_idx = parse_corrupt_idx(args.corrupt_idx)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nDataset: {args.dataset}")
    formula = get_dataset_formula(args.dataset)
    print(f"Formula: {formula}")
    print(f"Creating dataset with {args.n_samples} samples...")

    set_seed(args.seed)
    X, y = create_dataset(args.dataset, num_samples=args.n_samples, seed=args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    n_eval = max(1, min(args.eval_samples, len(X_test)))
    X_clean = X_test[:n_eval]
    y_eval = np.asarray(y_test[:n_eval], dtype=np.float64).reshape(-1)
    y_test_all = np.asarray(y_test, dtype=np.float64).reshape(-1)

    y_scale, ratio_threshold, threshold_source, y_scale_source = (
        compute_scale_and_threshold(y_eval, y_test_all, args.ratio_threshold)
    )

    print(f"Test sample shape: {X_clean.shape}")
    print(f"y_scale: {y_scale:.6f} ({y_scale_source})")

    X_corrupt = create_corrupted_input(
        X_clean,
        corrupt_idx=corrupt_idx,
        noise_std=args.noise_std,
        seed=args.seed,
        corruption_mode=args.corruption_mode,
        corruption_strength=args.corruption_strength,
    )

    print("\nLoading TabPFN model...")
    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)
    print("Model fitted successfully")

    num_heads = _get_num_heads(regressor)
    num_layers = _get_num_layers(regressor)
    head_indices = parse_head_indices(args.heads, num_heads)
    layer_indices = (
        list(range(num_layers))
        if args.max_layers is None
        else list(range(min(args.max_layers, num_layers)))
    )

    print(f"\nModel: {num_heads} MHA heads, {num_layers} layers")
    print(f"Patching heads: {head_indices}")
    print(f"Sweeping layers: {len(layer_indices)} layers (0..{max(layer_indices)})")

    exp_config = ExperimentConfig(
        corrupt_idx=corrupt_idx,
        noise_std=args.noise_std,
        seed=args.seed,
        n_train_samples=len(X_train),
        corruption_mode=args.corruption_mode,
        corruption_strength=args.corruption_strength,
        patch_dim=None,
        max_layers=args.max_layers,
        ratio_epsilon=args.ratio_epsilon,
        ratio_threshold=ratio_threshold,
        y_scale=y_scale,
        metric_mode=args.metric_mode,
    )

    script_path = str(Path(__file__).relative_to(Path.cwd()))
    dataset_output_dir = Path(args.output_dir) / args.dataset
    experiment = HeadPatchingExperiment(
        regressor=regressor,
        config=exp_config,
        output_dir=str(dataset_output_dir),
        script_path=script_path,
    )

    print("\n" + "=" * 60)
    print("RUNNING MHA ATTENTION HEAD PATCHING EXPERIMENT")
    print("=" * 60)

    results = experiment.run_sweep(
        X_clean=X_clean,
        X_corrupt=X_corrupt,
        head_indices=head_indices,
        layer_indices=layer_indices,
    )

    all_summaries = experiment.build_per_head_summaries(results)

    for head_idx, summary in sorted(all_summaries.items()):
        experiment.save_head_results(head_idx, summary, script_path)

    if len(all_summaries) > 1:
        experiment.create_comparison_plot(all_summaries, script_path)
        experiment.create_heatmap(all_summaries, script_path)

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Evaluated samples: {n_eval}")
    print(f"Metric mode: {args.metric_mode}")
    print(f"y_scale: {y_scale:.6f} ({y_scale_source})")
    print(f"Corruption: idx={corrupt_idx}, mode={args.corruption_mode}, strength={args.corruption_strength}")
    print(f"Clean output: {all_summaries[head_indices[0]]['y_clean']:.6f}")
    print(f"Corrupted output: {all_summaries[head_indices[0]]['y_corrupt']:.6f}")
    gap = all_summaries[head_indices[0]]['clean_corrupt_gaps_signed'][0]
    print(f"Clean-corrupt gap (signed): {gap:.6f}")

    print(f"\n{'Head':<8} {'Best Metric':<14} {'Rest.σ':<10} {'Best Layer':<12}")
    print("-" * 50)
    for head_idx in sorted(all_summaries.keys()):
        s = all_summaries[head_idx]
        print(
            f"{head_idx:<8} "
            f"{format_regime_metric(s):<14} "
            f"{s['best_restoration_sigma']:<10.4f} "
            f"{s['best_layer']:<12}"
        )

    best_head = max(all_summaries.values(), key=lambda s: s["best_recovery"])
    print(
        f"\nBest head: {best_head['head_idx']} "
        f"({best_head['best_recovery_metric']}) at layer {best_head['best_layer']}"
    )

    print(f"\nResults saved to: {dataset_output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
