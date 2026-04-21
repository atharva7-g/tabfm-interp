#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
import numpy as np
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
TEST_SIZE = 0.8
OUTPUT_DIR = "results/"
ABLATE_DIM = 2
ABLATION_TYPE = "zero"
SCALE_MODE = "y_scale"

VALID_SCALE_MODES = ["legacy", "y_scale"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run TabPFN ablation experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset type (default: multiplication)",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default=None,
        help="Override heads as comma-separated ints (e.g. '0,1,2,3')",
    )
    parser.add_argument(
        "--ablate-dim",
        type=int,
        default=None,
        help="Override ablate dim: 1=tokens, 2=heads, null=full layer",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="Override tokens as comma-separated ints (e.g. '0,1,2')",
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
        help="Minimum denominator for stable ablation ratios (default: 0.05)",
    )
    parser.add_argument(
        "--scale-mode",
        type=str,
        default=SCALE_MODE,
        choices=VALID_SCALE_MODES,
        help="Ablation normalization mode: legacy or y_scale",
    )
    return parser.parse_args()


def compute_ablation_scale(
    y_eval: np.ndarray,
    y_reference: np.ndarray,
    ratio_epsilon: float,
    scale_mode: str,
) -> tuple[float, str]:
    if scale_mode == "legacy":
        return 1.0, "legacy_disabled"

    min_scale = float(max(ratio_epsilon, 1e-12))
    eval_std = float(np.std(y_eval))
    if y_eval.size >= 2 and eval_std > min_scale:
        return eval_std, "eval_std"

    reference_std = float(np.std(y_reference))
    if reference_std > min_scale:
        return reference_std, "test_std_fallback"

    abs_mean = float(np.mean(np.abs(y_reference)))
    return float(max(abs_mean, min_scale)), "abs_mean_fallback"


def main():
    args = parse_args()

    dataset_type = args.dataset or DATASET_TYPE
    heads = [int(h) for h in args.heads.split(",")] if args.heads else HEADS
    tokens = [int(t) for t in args.tokens.split(",")] if args.tokens else TOKENS
    ablate_dim = args.ablate_dim if args.ablate_dim is not None else ABLATE_DIM

    with AimExperimentTracker(
        experiment_name="ablation",
        tags=[
            dataset_type,
            ABLATION_TYPE,
        ],
    ) as tracker:
        config = {
            "dataset_type": dataset_type,
            "heads": heads,
            "tokens": tokens,
            "seed": SEED,
            "n_samples": N_SAMPLES,
            "test_size": TEST_SIZE,
            "output_dir": OUTPUT_DIR,
            "ablate_dim": ablate_dim,
            "ablation_type": ABLATION_TYPE,
            "eval_samples": args.eval_samples,
            "ratio_epsilon": args.ratio_epsilon,
            "scale_mode": args.scale_mode,
        }

        set_seed(SEED)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print(f"\nDataset: {dataset_type}")
        formula = get_dataset_formula(dataset_type)
        print(f"Formula: {formula}")
        print(f"\nCreating dataset with {N_SAMPLES} samples...")
        X, y = create_dataset(
            dataset_type,
            num_samples=N_SAMPLES,
            seed=SEED,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED
        )

        n_eval = max(1, min(args.eval_samples, len(X_test)))
        X_eval = X_test[:n_eval]
        y_eval = np.asarray(y_test[:n_eval], dtype=np.float64).reshape(-1)
        y_test_all = np.asarray(y_test, dtype=np.float64).reshape(-1)
        y_scale, y_scale_source = compute_ablation_scale(
            y_eval=y_eval,
            y_reference=y_test_all,
            ratio_epsilon=args.ratio_epsilon,
            scale_mode=args.scale_mode,
        )
        config["y_scale"] = y_scale
        config["y_scale_source"] = y_scale_source
        tracker.log_params(config)
        print(f"Eval sample shape: {X_eval.shape}")
        print(
            f"First eval row (first 3 features): "
            f"x0={X_eval[0, 0]:.4f}, x1={X_eval[0, 1]:.4f}, x2={X_eval[0, 2]:.4f}"
        )

        print("\nLoading TabPFN model...")

        regressor = TabPFNRegressor(device=device, n_estimators=1)
        regressor.fit(X_train, y_train)
        print("Model fitted successfully")

        exp_config = AblationConfig(
            seed=SEED,
            n_train_samples=len(X_train),
            ablate_dim=ablate_dim,
            ablation_type=ABLATION_TYPE,
            ratio_epsilon=args.ratio_epsilon,
            y_scale=y_scale,
            scale_mode=args.scale_mode,
        )

        script_path = str(Path(__file__).relative_to(Path.cwd()))
        dataset_output_dir = Path(OUTPUT_DIR) / dataset_type
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
        print(f"Ablation dimension: {ablate_dim}")

        all_summaries = []

        if ablate_dim is None:
            summary, raw_results = experiment.ablate_full_layer(
                X=X_eval,
            )
            all_summaries.append(summary)
            experiment.save_full_layer_results(summary, raw_results, script_path)

            print(
                f"  Best effect: {summary['best_effect']:.6f} (sigma={summary['best_effect_sigma']:.4f}) at layer {summary['best_layer']}"
            )
            best_idx = summary["layer_indices"].index(summary["best_layer"])
            tracker.log_ablation_layer(
                layer_idx=summary["best_layer"],
                effect=summary["best_effect"],
                ratio=summary["ablation_ratios_stable"][best_idx],
            )
        elif ablate_dim == 1:
            for token_idx in tokens:
                summary, raw_results = experiment.ablate_single_token(
                    token_idx=token_idx,
                    X=X_eval,
                )
                all_summaries.append(summary)

                experiment.save_token_results(
                    token_idx, summary, raw_results, script_path
                )

                print(
                    f"  Token {token_idx}: Best effect: {summary['best_effect']:.6f} (sigma={summary['best_effect_sigma']:.4f}) at layer {summary['best_layer']}"
                )
                best_idx = summary["layer_indices"].index(summary["best_layer"])
                tracker.log_ablation_layer(
                    layer_idx=summary["best_layer"],
                    effect=summary["best_effect"],
                    ratio=summary["ablation_ratios_stable"][best_idx],
                    token_idx=token_idx,
                )
        else:
            for head_idx in heads:
                summary, raw_results = experiment.ablate_single_head(
                    head_idx=head_idx,
                    X=X_eval,
                )
                all_summaries.append(summary)

                experiment.save_head_results(
                    head_idx, summary, raw_results, script_path
                )

                print(
                    f"  Head {head_idx}: Best effect: {summary['best_effect']:.6f} (sigma={summary['best_effect_sigma']:.4f}) at layer {summary['best_layer']}"
                )
                best_idx = summary["layer_indices"].index(summary["best_layer"])
                tracker.log_ablation_layer(
                    layer_idx=summary["best_layer"],
                    effect=summary["best_effect"],
                    ratio=summary["ablation_ratios_stable"][best_idx],
                    head_idx=head_idx,
                )

        if len(all_summaries) > 1:
            print("\n" + "=" * 60)
            experiment.create_comparison_plot(all_summaries, script_path)

        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"\nEvaluated samples: {n_eval}")
        print(f"Scale mode: {args.scale_mode}")
        print(f"Stable ratio epsilon: {args.ratio_epsilon}")
        print(f"y scale: {y_scale:.6f} ({y_scale_source})")
        print(f"\nNormal output: {all_summaries[0]['y_normal']:.6f}")
        print(f"Ablated output: {all_summaries[0]['y_ablated']:.6f}")

        overall_best = (
            all_summaries[0]
            if ablate_dim is None
            else max(all_summaries, key=lambda x: x["best_effect"])
        )

        tracker.log_summary(
            y_normal=all_summaries[0]["y_normal"],
            y_ablated=all_summaries[0]["y_ablated"],
            best_effect=overall_best["best_effect"],
            best_layer=overall_best["best_layer"],
            best_ratio_raw_abs=overall_best["best_ratio_raw_abs"],
            best_ratio_stable_abs=overall_best["best_ratio_stable_abs"],
            best_effect_sigma=overall_best["best_effect_sigma"],
            y_scale=y_scale,
            ratio_epsilon=args.ratio_epsilon,
            scale_mode=args.scale_mode,
            y_scale_source=y_scale_source,
        )

        if ablate_dim is None:
            print("\nFull layer ablation results:")
            print(
                f"{'Layer':<8} {'Best Effect':<15} {'Effect Sigma':<13} {'Raw |Ratio|':<12}"
            )
            print("-" * 52)
            print(
                f"{all_summaries[0]['best_layer']:<8} "
                f"{all_summaries[0]['best_effect']:<15.6f} "
                f"{all_summaries[0]['best_effect_sigma']:<13.4f} "
                f"{all_summaries[0]['best_ratio_raw_abs'] * 100:<11.2f}%"
            )
        elif ablate_dim == 1:
            print("\nToken-by-token results:")
            print(
                f"{'Token':<8} {'Best Effect':<15} {'Effect Sigma':<13} {'Raw |Ratio|':<12} {'Best Layer':<12}"
            )
            print("-" * 72)
            for summary in all_summaries:
                print(
                    f"{summary['token_idx']:<8} "
                    f"{summary['best_effect']:<15.6f} "
                    f"{summary['best_effect_sigma']:<13.4f} "
                    f"{summary['best_ratio_raw_abs'] * 100:<11.2f}% "
                    f"{summary['best_layer']:<12}"
                )

            best_token = max(all_summaries, key=lambda x: x["best_effect"])
            print(
                f"\nBest token overall: {best_token['token_idx']} ({best_token['best_effect']:.6f} effect)"
            )
            tracker.log_summary(
                best_token=best_token["token_idx"],
                best_token_effect=best_token["best_effect"],
                best_token_ratio_raw_abs=best_token["best_ratio_raw_abs"],
                best_token_ratio_stable_abs=best_token["best_ratio_stable_abs"],
                best_token_effect_sigma=best_token["best_effect_sigma"],
                scale_mode=args.scale_mode,
                y_scale=y_scale,
                y_scale_source=y_scale_source,
            )
        else:
            print("\nHead-by-head results:")
            print(
                f"{'Head':<8} {'Best Effect':<15} {'Effect Sigma':<13} {'Raw |Ratio|':<12} {'Best Layer':<12}"
            )
            print("-" * 72)
            for summary in all_summaries:
                print(
                    f"{summary['head_idx']:<8} "
                    f"{summary['best_effect']:<15.6f} "
                    f"{summary['best_effect_sigma']:<13.4f} "
                    f"{summary['best_ratio_raw_abs'] * 100:<11.2f}% "
                    f"{summary['best_layer']:<12}"
                )

            best_head = max(all_summaries, key=lambda x: x["best_effect"])
            print(
                f"\nBest head overall: {best_head['head_idx']} ({best_head['best_effect']:.6f} effect)"
            )
            tracker.log_summary(
                best_head=best_head["head_idx"],
                best_head_effect=best_head["best_effect"],
                best_head_ratio_raw_abs=best_head["best_ratio_raw_abs"],
                best_head_ratio_stable_abs=best_head["best_ratio_stable_abs"],
                best_head_effect_sigma=best_head["best_effect_sigma"],
                scale_mode=args.scale_mode,
                y_scale=y_scale,
                y_scale_source=y_scale_source,
            )

        print(f"\nResults saved to: {dataset_output_dir}/")
        print("=" * 60)


if __name__ == "__main__":
    main()
