#!/usr/bin/env python3

import sys
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.utils import set_seed
from src.datasets import create_dataset, get_dataset_formula
from src.ablation.ablation_experiment import AblationExperiment, AblationConfig
from src.tracking import AimExperimentTracker
from tabpfn import TabPFNRegressor


DATASET_TYPE = "multiplication"
HEADS = [0, 1, 2, 3]
TOKENS = [0, 1, 2]
SEED = 42
N_SAMPLES = 1000
TEST_SIZE = 0.5
OUTPUT_DIR = "results/"
ABLATE_DIM = 2
ABLATION_TYPE = "zero"


def main():
    with AimExperimentTracker(
        experiment_name="ablation",
        tags=[
            DATASET_TYPE,
            ABLATION_TYPE,
        ],
    ) as tracker:
        config = {
            "dataset_type": DATASET_TYPE,
            "heads": HEADS,
            "tokens": TOKENS,
            "seed": SEED,
            "n_samples": N_SAMPLES,
            "test_size": TEST_SIZE,
            "output_dir": OUTPUT_DIR,
            "ablate_dim": ABLATE_DIM,
            "ablation_type": ABLATION_TYPE,
        }
        tracker.log_params(config)

        set_seed(SEED)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print(f"\nDataset: {DATASET_TYPE}")
        formula = get_dataset_formula(DATASET_TYPE)
        print(f"Formula: {formula}")
        print(f"\nCreating dataset with {N_SAMPLES} samples...")
        X, y = create_dataset(
            DATASET_TYPE,
            num_samples=N_SAMPLES,
            seed=SEED,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED
        )

        X_test_sample = X_test[0:1]
        print(f"Test sample shape: {X_test_sample.shape}")
        print(
            f"Test sample values: a={X_test_sample[0, 0]:.4f}, b={X_test_sample[0, 1]:.4f}, c={X_test_sample[0, 2]:.4f}"
        )

        print("\nLoading TabPFN model...")

        regressor = TabPFNRegressor(device=device, n_estimators=1)
        regressor.fit(X_train, y_train)
        print("Model fitted successfully")

        exp_config = AblationConfig(
            seed=SEED,
            n_train_samples=len(X_train),
            ablate_dim=ABLATE_DIM,
            ablation_type=ABLATION_TYPE,
        )

        script_path = str(Path(__file__).relative_to(Path.cwd()))
        dataset_output_dir = Path(OUTPUT_DIR) / DATASET_TYPE
        experiment = AblationExperiment(
            regressor=regressor,
            config=exp_config,
            output_dir=str(dataset_output_dir),
            script_path=script_path,
        )

        print("\n" + "=" * 60)
        print("RUNNING ABLATION EXPERIMENT")
        print("=" * 60)
        print(f"Ablation type: {ABLATION_TYPE}")
        print(f"Ablation dimension: {ABLATE_DIM}")

        all_summaries = []

        if ABLATE_DIM is None:
            summary, raw_results = experiment.ablate_full_layer(
                X=X_test_sample,
            )
            all_summaries.append(summary)
            experiment.save_full_layer_results(summary, raw_results, script_path)

            print(
                f"  Best effect: {summary['best_effect']:.6f} at layer {summary['best_layer']}"
            )
            tracker.log_ablation_layer(
                layer_idx=summary["best_layer"],
                effect=summary["best_effect"],
                ratio=abs(summary["best_effect"]) / (abs(summary["y_normal"]) + 1e-8),
            )
            tracker.log_summary(
                y_normal=summary["y_normal"],
                y_ablated=summary["y_ablated"],
                best_effect=summary["best_effect"],
                best_layer=summary["best_layer"],
            )
        elif ABLATE_DIM == 1:
            for token_idx in TOKENS:
                summary, raw_results = experiment.ablate_single_token(
                    token_idx=token_idx,
                    X=X_test_sample,
                )
                all_summaries.append(summary)

                experiment.save_token_results(
                    token_idx, summary, raw_results, script_path
                )

                print(
                    f"  Token {token_idx}: Best effect: {summary['best_effect']:.6f} at layer {summary['best_layer']}"
                )
                tracker.log_ablation_layer(
                    layer_idx=summary["best_layer"],
                    effect=summary["best_effect"],
                    ratio=abs(summary["best_effect"])
                    / (abs(summary["y_normal"]) + 1e-8),
                    token_idx=token_idx,
                )
        else:
            for head_idx in HEADS:
                summary, raw_results = experiment.ablate_single_head(
                    head_idx=head_idx,
                    X=X_test_sample,
                )
                all_summaries.append(summary)

                experiment.save_head_results(
                    head_idx, summary, raw_results, script_path
                )

                print(
                    f"  Head {head_idx}: Best effect: {summary['best_effect']:.6f} at layer {summary['best_layer']}"
                )
                tracker.log_ablation_layer(
                    layer_idx=summary["best_layer"],
                    effect=summary["best_effect"],
                    ratio=abs(summary["best_effect"])
                    / (abs(summary["y_normal"]) + 1e-8),
                    head_idx=head_idx,
                )

        if len(all_summaries) > 1:
            print("\n" + "=" * 60)
            experiment.create_comparison_plot(all_summaries, script_path)

        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"\nNormal output: {all_summaries[0]['y_normal']:.6f}")
        print(f"Ablated output: {all_summaries[0]['y_ablated']:.6f}")

        tracker.log_summary(
            y_normal=all_summaries[0]["y_normal"],
            y_ablated=all_summaries[0]["y_ablated"],
            best_effect=all_summaries[0]["best_effect"],
            best_layer=all_summaries[0]["best_layer"],
        )

        if ABLATE_DIM is None:
            print("\nFull layer ablation results:")
            print(f"{'Layer':<8} {'Best Effect':<15}")
            print("-" * 25)
            print(
                f"{all_summaries[0]['best_layer']:<8} {all_summaries[0]['best_effect']:<15.6f}"
            )
        elif ABLATE_DIM == 1:
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
            tracker.log_summary(
                best_token=best_token["token_idx"],
                best_token_effect=best_token["best_effect"],
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
            tracker.log_summary(
                best_head=best_head["head_idx"],
                best_head_effect=best_head["best_effect"],
            )

        print(f"\nResults saved to: {dataset_output_dir}/")
        print("=" * 60)


if __name__ == "__main__":
    main()
