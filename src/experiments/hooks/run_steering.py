#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split

from src.utils.utils import set_seed
from src.datasets import create_dataset, get_dataset_formula
from src.experiments.hooks.attention_steering import (
    AttentionSteeringExperiment,
    SteeringConfig,
)
from src.tracking import AimExperimentTracker
from tabpfn import TabPFNRegressor


DATASET_TYPE = "multiplication"
HEADS = [0, 1, 2, 3]
SEED = 42
N_SAMPLES = 1000
TEST_SIZE = 0.5
OUTPUT_DIR = "results/"
STEER_DIM = 2
ALPHA = 1.0
DIRECTION_TYPE = "random"
MAX_LAYERS = None


def main():
    parser = argparse.ArgumentParser(
        description="Run attention head steering experiments for TabPFN"
    )
    parser.add_argument("--dataset", type=str, default=DATASET_TYPE)
    parser.add_argument("--heads", type=str, default="0,1,2,3")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--steer-dim", type=int, default=STEER_DIM)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--direction-type", type=str, default=DIRECTION_TYPE)
    parser.add_argument("--max-layers", type=int, default=MAX_LAYERS)
    args = parser.parse_args()

    heads = [int(h) for h in args.heads.split(",")]

    with AimExperimentTracker(
        experiment_name="steering",
        tags=[args.dataset, args.direction_type],
    ) as tracker:
        config = {
            "dataset_type": args.dataset,
            "heads": heads,
            "seed": args.seed,
            "n_samples": args.n_samples,
            "test_size": args.test_size,
            "output_dir": args.output_dir,
            "steer_dim": args.steer_dim,
            "alpha": args.alpha,
            "direction_type": args.direction_type,
        }
        tracker.log_params(config)

        set_seed(args.seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print(f"\nDataset: {args.dataset}")
        formula = get_dataset_formula(args.dataset)
        print(f"Formula: {formula}")
        print(f"\nCreating dataset with {args.n_samples} samples...")
        X, y = create_dataset(args.dataset, num_samples=args.n_samples, seed=args.seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )

        X_test_sample = X_test[0:1]
        print(f"Test sample shape: {X_test_sample.shape}")
        print(
            f"Test sample values: a={X_test_sample[0, 0]:.4f}, b={X_test_sample[0, 1]:.4f}, c={X_test_sample[0, 2]:.4f}"
        )
        print(
            f"Expected output: {X_test_sample[0, 0] * X_test_sample[0, 1] + X_test_sample[0, 2]:.4f}"
        )

        print("\nLoading TabPFN model...")

        regressor = TabPFNRegressor(device=device, n_estimators=1)
        regressor.fit(X_train, y_train)
        print("Model fitted successfully")

        exp_config = SteeringConfig(
            seed=args.seed,
            n_train_samples=len(X_train),
            steer_dim=args.steer_dim,
            alpha=args.alpha,
            direction_type=args.direction_type,
            max_layers=args.max_layers,
        )

        script_path = str(Path(__file__).relative_to(Path.cwd()))
        dataset_output_dir = Path(args.output_dir) / args.dataset / "steering"
        experiment = AttentionSteeringExperiment(
            regressor=regressor,
            config=exp_config,
            output_dir=str(dataset_output_dir),
            script_path=script_path,
        )

        print("\n" + "=" * 60)
        print("RUNNING STEERING EXPERIMENT")
        print("=" * 60)
        print(f"Steer dimension: {args.steer_dim}")
        print(f"Alpha: {args.alpha}")
        print(f"Direction type: {args.direction_type}")

        all_summaries = []

        if args.steer_dim is None:
            summary, raw_results = experiment.steer_full_layer(X=X_test_sample)
            all_summaries.append(summary)
            experiment.save_full_layer_results(summary, raw_results, script_path)

            print(
                f"  Best effect: {summary['best_effect']:.6f} at layer {summary['best_layer']}"
            )
        elif args.steer_dim == 1:
            for token_idx in range(3):
                summary, raw_results = experiment.steer_single_token(
                    token_idx=token_idx, X=X_test_sample
                )
                all_summaries.append(summary)
                experiment.save_token_results(
                    token_idx, summary, raw_results, script_path
                )

                print(
                    f"  Token {token_idx}: Best effect: {summary['best_effect']:.6f} at layer {summary['best_layer']}"
                )
        else:
            for head_idx in heads:
                summary, raw_results = experiment.steer_single_head(
                    head_idx=head_idx, X=X_test_sample
                )
                all_summaries.append(summary)
                experiment.save_head_results(
                    head_idx, summary, raw_results, script_path
                )

                print(
                    f"  Head {head_idx}: Best effect: {summary['best_effect']:.6f} at layer {summary['best_layer']}"
                )

        if len(all_summaries) > 1:
            print("\n" + "=" * 60)
            experiment.create_comparison_plot(all_summaries, script_path)

        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"\nNormal output: {all_summaries[0]['y_normal']:.6f}")
        print(f"Steered output: {all_summaries[0]['y_steered']:.6f}")

        if args.steer_dim is None:
            print("\nFull layer steering results:")
            print(f"{'Layer':<8} {'Best Effect':<15}")
            print("-" * 25)
            print(
                f"{all_summaries[0]['best_layer']:<8} {all_summaries[0]['best_effect']:<15.6f}"
            )
        elif args.steer_dim == 1:
            print("\nToken-by-token results:")
            print(f"{'Token':<8} {'Best Effect':<15} {'Best Layer':<12}")
            print("-" * 35)
            for summary in all_summaries:
                print(
                    f"{summary['token_idx']:<8} {summary['best_effect']:<15.6f} {summary['best_layer']:<12}"
                )

            best_token = max(all_summaries, key=lambda x: x["best_effect"])
            print(
                f"\nBest token overall: {best_token['token_idx']} ({best_token['best_effect']:.6f} effect)"
            )
        else:
            print("\nHead-by-head results:")
            print(f"{'Head':<8} {'Best Effect':<15} {'Best Layer':<12}")
            print("-" * 35)
            for summary in all_summaries:
                print(
                    f"{summary['head_idx']:<8} {summary['best_effect']:<15.6f} {summary['best_layer']:<12}"
                )

            best_head = max(all_summaries, key=lambda x: x["best_effect"])
            print(
                f"\nBest head overall: {best_head['head_idx']} ({best_head['best_effect']:.6f} effect)"
            )

        print(f"\nResults saved to: {dataset_output_dir}/")
        print("=" * 60)


if __name__ == "__main__":
    main()
