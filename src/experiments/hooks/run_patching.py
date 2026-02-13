#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.utils import set_seed
from src.datasets import create_dataset, get_dataset_formula
from src.experiments.hooks.base import ExperimentConfig
from src.experiments.hooks.core_patching import create_corrupted_input
from src.experiments.hooks.attention_patching import AttentionPatchingExperiment
from src.experiments.hooks.config import interactive_config, save_config


def main():
    config = interactive_config()

    if config is None:
        return

    save_config(config)

    set_seed(config["seed"])

    device = config["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nDataset: {config['dataset_type']}")
    formula = get_dataset_formula(config["dataset_type"])
    print(f"Formula: {formula}")
    print(f"\nCreating dataset with {config['n_samples']} samples...")
    X, y = create_dataset(
        config["dataset_type"], num_samples=config["n_samples"], seed=config["seed"]
    )

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["seed"]
    )

    X_clean = X_test[0:1]
    print(f"Test sample shape: {X_clean.shape}")
    print(
        f"Test sample values: a={X_clean[0, 0]:.4f}, b={X_clean[0, 1]:.4f}, c={X_clean[0, 2]:.4f}"
    )

    X_corrupt = create_corrupted_input(
        X_clean,
        corrupt_idx=config["corrupt_idx"],
        noise_std=config["noise_std"],
        seed=config["seed"],
    )
    print(f"Corrupted input: b={X_corrupt[0, config['corrupt_idx']]:.4f} (noise)")

    print("\nLoading TabPFN model...")
    from tabpfn import TabPFNRegressor

    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)
    print("Model fitted successfully")

    exp_config = ExperimentConfig(
        corrupt_idx=config["corrupt_idx"],
        noise_std=config["noise_std"],
        seed=config["seed"],
        n_train_samples=len(X_train),
        patch_dim=2,
    )

    script_path = str(Path(__file__).relative_to(Path.cwd()))
    dataset_output_dir = Path(config["output_dir"]) / config["dataset_type"]
    experiment = AttentionPatchingExperiment(
        regressor=regressor,
        config=exp_config,
        output_dir=str(dataset_output_dir),
        script_path=script_path,
    )

    print("\n" + "=" * 60)
    print("RUNNING ATTENTION HEAD PATCHING EXPERIMENT")
    print("=" * 60)

    all_summaries = []
    for head_idx in config["heads"]:
        summary, raw_results = experiment.patch_single_head(
            head_idx=head_idx,
            X_clean=X_clean,
            X_corrupt=X_corrupt,
        )
        all_summaries.append(summary)

        experiment.save_head_results(head_idx, summary, raw_results, script_path)

        print(
            f"  Best recovery: {summary['best_recovery'] * 100:.2f}% at layer {summary['best_layer']}"
        )

    if len(all_summaries) > 1:
        print("\n" + "=" * 60)
        experiment.create_comparison_plot(all_summaries, script_path)
        experiment.create_heatmap(all_summaries, script_path)

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

    print(f"\nResults saved to: {dataset_output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
