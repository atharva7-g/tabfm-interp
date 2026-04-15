#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
from typing import Any
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.utils import set_seed, get_project_root
from src.datasets import create_dataset, get_dataset_formula
from src.experiments.hooks.base import ExperimentConfig
from src.experiments.hooks.core_patching import create_corrupted_input
from src.experiments.hooks.attention_patching import AttentionPatchingExperiment
from src.experiments.hooks.config import (
    interactive_config,
    save_config,
    load_config,
)
from src.tracking import AimExperimentTracker
from tabpfn import TabPFNRegressor

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


def find_default_config():
    default_paths = [
        Path(f"{get_project_root()}/src/experiments/hooks/config.json"),
    ]
    for path in default_paths:
        if path.exists():
            return str(path)
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run attention head patching experiments for TabPFN"
    )
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
        help="Minimum denominator for stable recovery metrics (default: 0.05)",
    )
    parser.add_argument(
        "--corrupt-idx",
        type=str,
        default=None,
        help="Override corrupt_idx from config; accepts int or comma-separated ints",
    )
    parser.add_argument(
        "--corruption-mode",
        type=str,
        default=None,
        choices=VALID_CORRUPTION_MODES,
        help="Corruption mode override",
    )
    parser.add_argument(
        "--corruption-strength",
        type=float,
        default=None,
        help="Corruption strength override (>=0)",
    )
    return parser.parse_args()


def parse_corrupt_idx(value: str):
    value = value.strip()
    if "," in value:
        return [int(v.strip()) for v in value.split(",") if v.strip() != ""]
    return int(value)


def format_corrupt_tag(corrupt_idx):
    if isinstance(corrupt_idx, list):
        if not corrupt_idx:
            return "none"
        if len(corrupt_idx) <= 4:
            return "-".join(map(str, corrupt_idx))
        return f"multi{len(corrupt_idx)}"
    return str(corrupt_idx)


def apply_corruption_defaults(config: dict[str, Any]) -> dict[str, Any]:
    if "corruption_mode" not in config or config["corruption_mode"] is None:
        config["corruption_mode"] = "gaussian_replace"
    if "corruption_strength" not in config or config["corruption_strength"] is None:
        config["corruption_strength"] = 1.0
    return config


def recommend_corrupt_idx(dataset_type: str, current_corrupt_idx):
    if current_corrupt_idx is not None:
        return current_corrupt_idx
    if dataset_type == "multiplication":
        return 2
    if dataset_type == "quadratic":
        return 2
    return 0


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

    if args.corrupt_idx is not None:
        config["corrupt_idx"] = parse_corrupt_idx(args.corrupt_idx)

    config = apply_corruption_defaults(config)

    if args.corruption_mode is not None:
        config["corruption_mode"] = args.corruption_mode

    if args.corruption_strength is not None:
        config["corruption_strength"] = args.corruption_strength

    config["eval_samples"] = args.eval_samples
    config["ratio_epsilon"] = args.ratio_epsilon
    config["corrupt_idx"] = recommend_corrupt_idx(
        config.get("dataset_type", "multiplication"),
        config.get("corrupt_idx"),
    )

    with AimExperimentTracker(
        experiment_name="attention-patching",
        tags=[
            config["dataset_type"],
            f"corrupt_{format_corrupt_tag(config['corrupt_idx'])}",
            f"mode_{config['corruption_mode']}",
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

        n_eval = max(1, min(args.eval_samples, len(X_test)))
        X_clean = X_test[:n_eval]
        print(f"Test sample shape: {X_clean.shape}")
        # print(
        #     f"Test sample values: a={X_clean[0, 0]:.4f}, b={X_clean[0, 1]:.4f}, c={X_clean[0, 2]:.4f}"
        # )

        X_corrupt = create_corrupted_input(
            X_clean,
            corrupt_idx=config["corrupt_idx"],
            noise_std=config["noise_std"],
            seed=config["seed"],
            corruption_mode=config["corruption_mode"],
            corruption_strength=config["corruption_strength"],
        )
        # print(f"Corrupted input: b={X_corrupt[0, config['corrupt_idx']]:.4f} (noise)")

        print("\nLoading TabPFN model...")

        regressor = TabPFNRegressor(device=device, n_estimators=1)
        regressor.fit(X_train, y_train)
        print("Model fitted successfully")

        patch_dim = config.get("patch_dim", 2)
        exp_config = ExperimentConfig(
            corrupt_idx=config["corrupt_idx"],
            noise_std=config["noise_std"],
            seed=config["seed"],
            n_train_samples=len(X_train),
            corruption_mode=config["corruption_mode"],
            corruption_strength=config["corruption_strength"],
            patch_dim=patch_dim,
            ratio_epsilon=args.ratio_epsilon,
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

        if patch_dim is None:
            summary, raw_results = experiment.patch_full_layer(
                X_clean=X_clean,
                X_corrupt=X_corrupt,
            )
            all_summaries.append(summary)
            experiment.save_full_layer_results(summary, raw_results, script_path)

            best_idx = summary["layer_indices"].index(summary["best_layer"])
            tracker.log_patching_layer(
                layer_idx=summary["best_layer"],
                restoration=summary["restorations"][best_idx],
                recovery_ratio=summary["recovery_ratios_stable"][best_idx],
            )

            print(
                f"  Best score: {summary['best_recovery'] * 100:.2f}% at layer {summary['best_layer']}"
            )
        elif patch_dim == 1:
            for token_idx in config["tokens"]:
                summary, raw_results = experiment.patch_single_token(
                    token_idx=token_idx,
                    X_clean=X_clean,
                    X_corrupt=X_corrupt,
                )
                all_summaries.append(summary)

                experiment.save_token_results(
                    token_idx, summary, raw_results, script_path
                )

                best_idx = summary["layer_indices"].index(summary["best_layer"])
                tracker.log_patching_layer(
                    layer_idx=summary["best_layer"],
                    restoration=summary["restorations"][best_idx],
                    recovery_ratio=summary["recovery_ratios_stable"][best_idx],
                    token_idx=token_idx,
                )

                print(
                    f"  Best score: {summary['best_recovery'] * 100:.2f}% at layer {summary['best_layer']}"
                )
        else:
            for head_idx in config["heads"]:
                summary, raw_results = experiment.patch_single_head(
                    head_idx=head_idx,
                    X_clean=X_clean,
                    X_corrupt=X_corrupt,
                )
                all_summaries.append(summary)

                experiment.save_head_results(
                    head_idx, summary, raw_results, script_path
                )

                best_idx = summary["layer_indices"].index(summary["best_layer"])
                tracker.log_patching_layer(
                    layer_idx=summary["best_layer"],
                    restoration=summary["restorations"][best_idx],
                    recovery_ratio=summary["recovery_ratios_stable"][best_idx],
                    head_idx=head_idx,
                )

                print(
                    f"  Best score: {summary['best_recovery'] * 100:.2f}% at layer {summary['best_layer']}"
                )

        if len(all_summaries) > 1:
            print("\n" + "=" * 60)
            experiment.create_comparison_plot(all_summaries, script_path)
            experiment.create_heatmap(all_summaries, script_path)

        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"\nEvaluated samples: {n_eval}")
        print(f"Stable ratio epsilon: {args.ratio_epsilon}")
        print(f"Corruption mode: {config['corruption_mode']}")
        print(f"Corruption strength: {config['corruption_strength']}")
        print(f"Corrupt idx: {config['corrupt_idx']}")
        print(f"\nClean output: {all_summaries[0]['y_clean']:.6f}")
        print(f"Corrupted output: {all_summaries[0]['y_corrupt']:.6f}")
        clean_corrupt_diff = all_summaries[0]["y_clean"] - all_summaries[0]["y_corrupt"]
        print(f"Clean-corrupt gap: {clean_corrupt_diff:.6f}")

        if patch_dim is None:
            print("\nFull layer patching results:")
            print(
                f"{'Layer':<8} {'Best Score':<12} {'Stable Ratio':<14} {'Raw |Ratio|':<12}"
            )
            print("-" * 52)
            print(
                f"{all_summaries[0]['best_layer']:<8} "
                f"{all_summaries[0]['best_recovery'] * 100:<11.2f}% "
                f"{all_summaries[0]['best_recovery_stable_abs'] * 100:<13.2f}% "
                f"{all_summaries[0]['best_recovery_raw_abs'] * 100:<11.2f}%"
            )
            print(
                f"\nBest layer overall: {all_summaries[0]['best_layer']} "
                f"({all_summaries[0]['best_recovery'] * 100:.2f}% score)"
            )
        elif patch_dim == 1:
            print("\nToken-by-token results:")
            print(
                f"{'Token':<8} {'Best Score':<12} {'Stable Ratio':<14} {'Raw |Ratio|':<12} {'Best Layer':<12}"
            )
            print("-" * 72)
            for summary in all_summaries:
                print(
                    f"{summary['token_idx']:<8} "
                    f"{summary['best_recovery'] * 100:<11.2f}% "
                    f"{summary['best_recovery_stable_abs'] * 100:<13.2f}% "
                    f"{summary['best_recovery_raw_abs'] * 100:<11.2f}% "
                    f"{summary['best_layer']:<12}"
                )

            best_token = max(all_summaries, key=lambda x: x["best_recovery"])
            print(
                f"\nBest token overall: {best_token['token_idx']} ({best_token['best_recovery'] * 100:.2f}% score)"
            )
        else:
            print("\nHead-by-head results:")
            print(
                f"{'Head':<8} {'Best Score':<12} {'Stable Ratio':<14} {'Raw |Ratio|':<12} {'Best Layer':<12}"
            )
            print("-" * 72)
            for summary in all_summaries:
                print(
                    f"{summary['head_idx']:<8} "
                    f"{summary['best_recovery'] * 100:<11.2f}% "
                    f"{summary['best_recovery_stable_abs'] * 100:<13.2f}% "
                    f"{summary['best_recovery_raw_abs'] * 100:<11.2f}% "
                    f"{summary['best_layer']:<12}"
                )

            best_head = max(all_summaries, key=lambda x: x["best_recovery"])
            print(
                f"\nBest head overall: {best_head['head_idx']} ({best_head['best_recovery'] * 100:.2f}% score)"
            )

        if experiment.created_images:
            # Build captions for each image based on filename and context
            image_captions = {}
            dataset_name = config["dataset_type"]
            corrupt_info = (
                f"corrupt_idx={config['corrupt_idx']}, noise_std={config['noise_std']}, "
                f"mode={config['corruption_mode']}, strength={config['corruption_strength']}"
            )

            for img_path in experiment.created_images:
                filename = img_path.name
                if "full_layer_restoration" in filename:
                    caption = (
                        f"Full layer restoration - {dataset_name} - {corrupt_info}"
                    )
                elif "head_" in filename and "restoration" in filename:
                    # Extract head_idx from filename like "head_0_restoration_123456.png"
                    head_idx = filename.split("head_")[1].split("_")[0]
                    caption = (
                        f"Head {head_idx} restoration - {dataset_name} - {corrupt_info}"
                    )
                elif "comparison_all_heads" in filename:
                    heads = config["heads"]
                    caption = f"Restoration comparison across heads {heads} - {dataset_name} - {corrupt_info}"
                elif "comparison_heatmap" in filename:
                    caption = f"Recovery heatmap (layers x heads) - {dataset_name} - {corrupt_info}"
                else:
                    caption = f"{filename} - {dataset_name} - {corrupt_info}"

                image_captions[img_path] = caption

            tracker.log_artifacts(image_captions)

        if patch_dim is None:
            tracker.log_summary(
                y_clean=all_summaries[0]["y_clean"],
                y_corrupt=all_summaries[0]["y_corrupt"],
                best_recovery=all_summaries[0]["best_recovery"],
                best_layer=all_summaries[0]["best_layer"],
                patch_type="full_layer",
                best_recovery_raw_abs=all_summaries[0]["best_recovery_raw_abs"],
                best_recovery_stable_abs=all_summaries[0]["best_recovery_stable_abs"],
                eval_samples=n_eval,
                ratio_epsilon=args.ratio_epsilon,
            )
        elif patch_dim == 1:
            best_overall = max(all_summaries, key=lambda x: x["best_recovery"])
            tracker.log_summary(
                y_clean=all_summaries[0]["y_clean"],
                y_corrupt=all_summaries[0]["y_corrupt"],
                best_recovery=best_overall["best_recovery"],
                best_layer=best_overall["best_layer"],
                best_token=best_overall["token_idx"],
                tokens_tested=len(config["tokens"]),
                patch_type="tokens",
                best_recovery_raw_abs=best_overall["best_recovery_raw_abs"],
                best_recovery_stable_abs=best_overall["best_recovery_stable_abs"],
                eval_samples=n_eval,
                ratio_epsilon=args.ratio_epsilon,
            )
        else:
            best_overall = max(all_summaries, key=lambda x: x["best_recovery"])
            tracker.log_summary(
                y_clean=all_summaries[0]["y_clean"],
                y_corrupt=all_summaries[0]["y_corrupt"],
                best_recovery=best_overall["best_recovery"],
                best_layer=best_overall["best_layer"],
                best_head=best_overall["head_idx"],
                heads_tested=len(config["heads"]),
                patch_type="attention_heads",
                best_recovery_raw_abs=best_overall["best_recovery_raw_abs"],
                best_recovery_stable_abs=best_overall["best_recovery_stable_abs"],
                eval_samples=n_eval,
                ratio_epsilon=args.ratio_epsilon,
            )

        print(f"\nResults saved to: {dataset_output_dir}/")
        print("=" * 60)


if __name__ == "__main__":
    main()
