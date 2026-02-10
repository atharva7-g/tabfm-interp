#!/usr/bin/env python3
"""
CLI script for running attention head patching experiments.

Example usage:
    python run_patching.py --heads 0 1 2 3 --corrupt-idx 1 --seed 42
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.utils import create_multiplication_dataset, set_seed
from src.experiments.hooks.base import ExperimentConfig
from src.experiments.hooks.core_patching import create_corrupted_input
from src.experiments.hooks.attention_patching import AttentionPatchingExperiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run attention head patching experiments"
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Head indices to patch (default: 0 1 2 3)",
    )
    parser.add_argument(
        "--corrupt-idx",
        type=int,
        default=1,
        help="Index of feature to corrupt (default: 1)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=1.0,
        help="Standard deviation of noise for corruption (default: 1.0)",
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
        help="Number of samples for dataset (default: 1000)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.5,
        help="Fraction of data for testing (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/attention_patching",
        help="Output directory for results (default: results/attention_patching)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Determine device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset
    print(f"\nCreating dataset with {args.n_samples} samples...")
    X, y = create_multiplication_dataset(num_samples=args.n_samples, seed=args.seed)

    # Split data
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    X_clean = X_test[0:1]
    print(f"Test sample shape: {X_clean.shape}")
    print(
        f"Test sample values: a={X_clean[0, 0]:.4f}, b={X_clean[0, 1]:.4f}, c={X_clean[0, 2]:.4f}"
    )

    # Create corrupted input
    X_corrupt = create_corrupted_input(
        X_clean, corrupt_idx=args.corrupt_idx, noise_std=args.noise_std, seed=args.seed
    )
    print(f"Corrupted input: b={X_corrupt[0, args.corrupt_idx]:.4f} (noise)")

    # Load TabPFN model
    print("\nLoading TabPFN model...")
    from tabpfn import TabPFNRegressor

    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)
    print("Model fitted successfully")

    # Create experiment config
    config = ExperimentConfig(
        corrupt_idx=args.corrupt_idx,
        noise_std=args.noise_std,
        seed=args.seed,
        n_train_samples=len(X_train),
        patch_dim=2,  # Attention heads
    )

    # Create experiment
    script_path = str(Path(__file__).relative_to(Path.cwd()))
    experiment = AttentionPatchingExperiment(
        regressor=regressor,
        config=config,
        output_dir=args.output_dir,
        script_path=script_path,
    )

    print("\n" + "=" * 60)
    print("RUNNING ATTENTION HEAD PATCHING EXPERIMENT")
    print("=" * 60)

    # Run patching for each head
    all_summaries = []
    for head_idx in args.heads:
        summary, raw_results = experiment.patch_single_head(
            head_idx=head_idx,
            X_clean=X_clean,
            X_corrupt=X_corrupt,
        )
        all_summaries.append(summary)

        # Save results for this head
        experiment.save_head_results(head_idx, summary, raw_results, script_path)

        print(
            f"  Best recovery: {summary['best_recovery'] * 100:.2f}% at layer {summary['best_layer']}"
        )

    # Create comparison plots
    if len(all_summaries) > 1:
        print("\n" + "=" * 60)
        experiment.create_comparison_plot(all_summaries, script_path)
        experiment.create_heatmap(all_summaries, script_path)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"\nClean output: {all_summaries[0]['y_clean']:.6f}")
    print(f"Corrupted output: {all_summaries[0]['y_corrupt']:.6f}")
    print(f"\nHead-by-head results:")
    print(f"{'Head':<8} {'Best Recovery':<15} {'Best Layer':<12}")
    print("-" * 35)
    for summary in all_summaries:
        print(
            f"{summary['head_idx']:<8} {summary['best_recovery'] * 100:<15.2f}% {summary['best_layer']:<12}"
        )

    best_head = max(all_summaries, key=lambda x: x["best_recovery"])
    print(
        f"\nBest head overall: {best_head['head_idx']} ({best_head['best_recovery'] * 100:.2f}% recovery)"
    )

    print(f"\nResults saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
