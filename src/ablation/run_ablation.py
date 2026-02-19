#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.utils import set_seed, get_project_root
from src.datasets import create_dataset, get_dataset_formula
from src.ablation.ablation_experiment import AblationExperiment, AblationConfig
from src.ablation.config import (
    interactive_config,
    save_config,
    load_config,
)
from src.tracking import AimExperimentTracker
from tabpfn import TabPFNRegressor


def find_default_config():
    default_paths = [
        Path(f"{get_project_root()}/src/ablation/config.json"),
    ]
    for path in default_paths:
        if path.exists():
            return str(path)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation experiments for TabPFN")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config JSON file (default: look for config.json in standard locations)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Use interactive configuration mode instead of config file",
    )
    return parser.parse_args()


def get_config(args):
    if args.interactive:
        print("Using interactive configuration mode...")
        return interactive_config()

    config_path = args.config

    if not config_path:
        config_path = find_default_config()

    if config_path:
        print(f"Loading config from: {config_path}")
        try:
            return load_config(config_path)
        except FileNotFoundError:
            print(f"Error: Config file not found: {config_path}")
            return None
        except Exception as e:
            print(f"Error loading config: {e}")
            return None
    else:
        print("No config file found. Use --interactive for interactive mode.")
        print("\nOr specify a config file with --config <path>")
        return None


def main():
    args = parse_args()
    config = get_config(args)

    if config is None:
        return

    with AimExperimentTracker(
        experiment_name="ablation",
        tags=[
            config["dataset_type"],
            config["ablation_type"],
            f"corrupt_{config['corrupt_idx']}",
        ],
    ) as tracker:
        save_config(config)
        tracker.log_params(config)

        set_seed(config["seed"])

        device = config["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        print(f"\nDataset: {config['dataset_type']}")
        formula = get_dataset_formula(config["dataset_type"])
        print(f"Formula: {formula}")
        print(f"\nCreating dataset with {config['n_samples']} samples...")
        X, y = create_dataset(
            config["dataset_type"],
            num_samples=config["n_samples"],
            seed=config["seed"],
        )

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config["test_size"], random_state=config["seed"]
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

        ablate_dim = config.get("ablate_dim", 2)
        exp_config = AblationConfig(
            corrupt_idx=config["corrupt_idx"],
            noise_std=config["noise_std"],
            seed=config["seed"],
            n_train_samples=len(X_train),
            ablate_dim=ablate_dim,
            ablation_type=config.get("ablation_type", "zero"),
        )

        script_path = str(Path(__file__).relative_to(Path.cwd()))
        dataset_output_dir = Path(config["output_dir"]) / config["dataset_type"]
        experiment = AblationExperiment(
            regressor=regressor,
            config=exp_config,
            output_dir=str(dataset_output_dir),
            script_path=script_path,
        )

        print("\n" + "=" * 60)
        print("RUNNING ABLATION EXPERIMENT")
        print("=" * 60)
        print(f"Ablation type: {config.get('ablation_type', 'zero')}")
        print(f"Ablation dimension: {ablate_dim}")

        all_summaries = []

        if ablate_dim is None:
            summary, raw_results = experiment.ablate_full_layer(
                X=X_test_sample,
            )
            all_summaries.append(summary)
            experiment.save_full_layer_results(summary, raw_results, script_path)

            best_idx = summary["layer_indices"].index(summary["best_layer"])
            tracker.log_ablation_layer(
                layer_idx=summary["best_layer"],
                effect=summary["ablation_effects"][best_idx],
                ratio=summary["ablation_ratios"][best_idx],
            )

            print(
                f"  Best effect: {summary['best_effect']:.6f} at layer {summary['best_layer']}"
            )
        elif ablate_dim == 1:
            for token_idx in config["tokens"]:
                summary, raw_results = experiment.ablate_single_token(
                    token_idx=token_idx,
                    X=X_test_sample,
                )
                all_summaries.append(summary)

                experiment.save_token_results(
                    token_idx, summary, raw_results, script_path
                )

                best_idx = summary["layer_indices"].index(summary["best_layer"])
                tracker.log_ablation_layer(
                    layer_idx=summary["best_layer"],
                    effect=summary["ablation_effects"][best_idx],
                    ratio=summary["ablation_ratios"][best_idx],
                    token_idx=token_idx,
                )

                print(
                    f"  Best effect: {summary['best_effect']:.6f} at layer {summary['best_layer']}"
                )
        else:
            for head_idx in config["heads"]:
                summary, raw_results = experiment.ablate_single_head(
                    head_idx=head_idx,
                    X=X_test_sample,
                )
                all_summaries.append(summary)

                experiment.save_head_results(
                    head_idx, summary, raw_results, script_path
                )

                best_idx = summary["layer_indices"].index(summary["best_layer"])
                tracker.log_ablation_layer(
                    layer_idx=summary["best_layer"],
                    effect=summary["ablation_effects"][best_idx],
                    ratio=summary["ablation_ratios"][best_idx],
                    head_idx=head_idx,
                )

                print(
                    f"  Best effect: {summary['best_effect']:.6f} at layer {summary['best_layer']}"
                )

        if len(all_summaries) > 1:
            print("\n" + "=" * 60)
            experiment.create_comparison_plot(all_summaries, script_path)

        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"\nNormal output: {all_summaries[0]['y_normal']:.6f}")
        print(f"Ablated output: {all_summaries[0]['y_ablated']:.6f}")

        if ablate_dim is None:
            print("\nFull layer ablation results:")
            print(f"{'Layer':<8} {'Best Effect':<15}")
            print("-" * 25)
            print(
                f"{all_summaries[0]['best_layer']:<8} {all_summaries[0]['best_effect']:<15.6f}"
            )
            print(
                f"\nBest layer overall: {all_summaries[0]['best_layer']} ({all_summaries[0]['best_effect']:.6f} effect)"
            )
        elif ablate_dim == 1:
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

        if experiment.created_images:
            image_captions = {}
            dataset_name = config["dataset_type"]
            ablation_info = f"ablate_dim={ablate_dim}, ablation_type={config.get('ablation_type', 'zero')}"

            for img_path in experiment.created_images:
                filename = img_path.name
                if "full_layer_ablation" in filename:
                    caption = f"Full layer ablation - {dataset_name} - {ablation_info}"
                elif "head_" in filename and "ablation" in filename:
                    head_idx = filename.split("head_")[1].split("_")[0]
                    caption = (
                        f"Head {head_idx} ablation - {dataset_name} - {ablation_info}"
                    )
                elif "token_" in filename and "ablation" in filename:
                    token_idx = filename.split("token_")[1].split("_")[0]
                    caption = (
                        f"Token {token_idx} ablation - {dataset_name} - {ablation_info}"
                    )
                elif "comparison_ablation" in filename:
                    caption = f"Ablation comparison - {dataset_name} - {ablation_info}"
                else:
                    caption = f"{filename} - {dataset_name} - {ablation_info}"

                image_captions[img_path] = caption

            tracker.log_artifacts(image_captions)

        if ablate_dim is None:
            tracker.log_summary(
                y_normal=all_summaries[0]["y_normal"],
                y_ablated=all_summaries[0]["y_ablated"],
                best_effect=all_summaries[0]["best_effect"],
                best_layer=all_summaries[0]["best_layer"],
                ablation_type="full_layer",
            )
        elif ablate_dim == 1:
            best_overall = max(all_summaries, key=lambda x: x["best_effect"])
            tracker.log_summary(
                y_normal=all_summaries[0]["y_normal"],
                y_ablated=all_summaries[0]["y_ablated"],
                best_effect=best_overall["best_effect"],
                best_layer=best_overall["best_layer"],
                best_token=best_overall["token_idx"],
                tokens_tested=len(config["tokens"]),
                ablation_type="tokens",
            )
        else:
            best_overall = max(all_summaries, key=lambda x: x["best_effect"])
            tracker.log_summary(
                y_normal=all_summaries[0]["y_normal"],
                y_ablated=all_summaries[0]["y_ablated"],
                best_effect=best_overall["best_effect"],
                best_layer=best_overall["best_layer"],
                best_head=best_overall["head_idx"],
                heads_tested=len(config["heads"]),
                ablation_type="attention_heads",
            )

        print(f"\nResults saved to: {dataset_output_dir}/")
        print("=" * 60)


if __name__ == "__main__":
    main()
