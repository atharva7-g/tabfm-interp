#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.utils import set_seed
from src.datasets import create_dataset, get_dataset_formula
from src.experiments.attention_head_ablation.core_head_ablation import (
    _get_num_heads,
    _get_num_layers,
)
from src.experiments.attention_head_ablation.head_ablation_experiment import (
    HeadAblationExperiment,
)
from tabpfn import TabPFNRegressor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run true MHA attention head ablation experiments for TabPFN"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="multiplication",
        help="Dataset type (default: multiplication)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=64,
        help="Number of test samples to evaluate (default: 64)",
    )
    parser.add_argument(
        "--ratio-epsilon",
        type=float,
        default=0.05,
        help="Minimum denominator for stable metrics (default: 0.05)",
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
        help="Comma-separated head indices (default: all)",
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
        default="results_ablation_heads_mha",
        help="Output directory (default: results_ablation_heads_mha)",
    )
    return parser.parse_args()


def parse_head_indices(value: str | None, num_heads: int) -> list[int]:
    if value is None:
        return list(range(num_heads))
    return [int(v.strip()) for v in value.split(",") if v.strip() != ""]


def main():
    args = parse_args()

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
    X_eval = X_test[:n_eval]
    y_eval = np.asarray(y_test[:n_eval], dtype=np.float64).reshape(-1)
    y_test_all = np.asarray(y_test, dtype=np.float64).reshape(-1)

    y_scale = float(np.std(y_eval)) if np.std(y_eval) > 1e-12 else float(np.std(y_test_all))

    print(f"Eval sample shape: {X_eval.shape}")
    print(f"y_scale: {y_scale:.6f}")

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
    print(f"Ablating heads: {head_indices}")
    print(f"Sweeping layers: {len(layer_indices)} layers (0..{max(layer_indices)})")

    script_path = str(Path(__file__).relative_to(Path.cwd()))
    dataset_output_dir = Path(args.output_dir) / args.dataset
    experiment = HeadAblationExperiment(
        regressor=regressor,
        output_dir=str(dataset_output_dir),
        script_path=script_path,
    )

    print("\n" + "=" * 60)
    print("RUNNING MHA ATTENTION HEAD ABLATION EXPERIMENT")
    print("=" * 60)

    results = experiment.run_sweep(
        X=X_eval,
        head_indices=head_indices,
        layer_indices=layer_indices,
        ratio_epsilon=args.ratio_epsilon,
        y_scale=y_scale,
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
    print(f"y_scale: {y_scale:.6f}")
    print(f"Normal output: {all_summaries[head_indices[0]]['y_normal']:.6f}")

    print(f"\n{'Head':<8} {'Best Effect σ':<14} {'Best Layer':<12} {'Peak |Effect|':<14}")
    print("-" * 52)
    for head_idx in sorted(all_summaries.keys()):
        s = all_summaries[head_idx]
        print(
            f"{head_idx:<8} "
            f"{s['best_effect_sigma']:<14.4f} "
            f"{s['best_layer']:<12} "
            f"{s['best_effect_abs']:<14.6f}"
        )

    best_head = max(all_summaries.values(), key=lambda s: s["best_effect_sigma"])
    print(
        f"\nMost important head: {best_head['head_idx']} "
        f"({best_head['best_effect_sigma']:.4f}σ) at layer {best_head['best_layer']}"
    )

    print(f"\nResults saved to: {dataset_output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
