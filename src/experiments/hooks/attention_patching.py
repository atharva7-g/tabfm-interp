from typing import List, Dict, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor
import sys
from pathlib import Path

# Add parent to path to import base
sys.path.insert(0, str(Path(__file__).parent))
from base import BaseExperiment, ExperimentConfig

# Import existing patching functions
from src.experiments.hooks.core_patching import sweep_layers


class AttentionPatchingExperiment(BaseExperiment):
    """Experiment class for attention head patching."""

    def __init__(
        self,
        regressor: TabPFNRegressor,
        config: ExperimentConfig,
        output_dir: str = "results/attention_patching",
        script_path: str = "",
    ):
        super().__init__(output_dir)
        self.regressor = regressor
        self.config = config
        self.script_path = script_path
        self.created_images: list = []

    @staticmethod
    def _optional_to_float_list(values: List[float | None]) -> List[float]:
        return [float("nan") if v is None else float(v) for v in values]

    def _build_summary(self, results: List[Dict], key: str, value) -> Dict:
        raw_recovery_ratios = [float(r["recovery_ratio"]) for r in results]
        stable_recovery_ratios = [float(r["recovery_ratio_stable"]) for r in results]
        recovery_scores = [float(r["recovery_score"]) for r in results]
        recovery_scores_regime = [float(r["recovery_score_regime"]) for r in results]
        recovery_fractional_signed = [r["recovery_fractional_signed"] for r in results]
        recovery_fractional_abs = [r["recovery_fractional_abs"] for r in results]
        best_by_primary = max(results, key=lambda x: x["recovery_primary"])

        valid_fractional_abs = [
            float(x) for x in recovery_fractional_abs if x is not None
        ]
        best_fractional_abs_overall = (
            max(valid_fractional_abs) if len(valid_fractional_abs) > 0 else None
        )
        valid_fractional_signed = [
            float(x) for x in recovery_fractional_signed if x is not None
        ]
        best_fractional_signed_overall = (
            max(valid_fractional_signed) if len(valid_fractional_signed) > 0 else None
        )

        best_metric = str(best_by_primary["recovery_primary_metric"])

        summary = {
            key: value,
            "y_clean": results[0]["y_clean"],
            "y_corrupt": results[0]["y_corrupt"],
            "restorations": [float(r["restoration"]) for r in results],
            "restorations_abs_mean": [
                float(r["restoration_abs_mean"]) for r in results
            ],
            "clean_corrupt_gaps_signed": [
                float(r["clean_corrupt_diff"]) for r in results
            ],
            "clean_corrupt_gap_abs_means": [
                float(r["clean_corrupt_gap_abs_mean"]) for r in results
            ],
            "residual_abs_means": [float(r["residual_abs_mean"]) for r in results],
            "restoration_sigmas": [float(r["restoration_sigma"]) for r in results],
            "residual_sigmas": [float(r["residual_sigma"]) for r in results],
            "recovery_ratios": raw_recovery_ratios,
            "recovery_ratios_stable": stable_recovery_ratios,
            "recovery_scores": recovery_scores,
            "recovery_scores_regime": recovery_scores_regime,
            "recovery_fractional_signed": recovery_fractional_signed,
            "recovery_fractional_abs": recovery_fractional_abs,
            "recovery_fractional_signed_numeric": self._optional_to_float_list(
                recovery_fractional_signed
            ),
            "recovery_fractional_abs_numeric": self._optional_to_float_list(
                recovery_fractional_abs
            ),
            "ratio_valid_signed": [bool(r["ratio_valid_signed"]) for r in results],
            "ratio_valid_abs": [bool(r["ratio_valid_abs"]) for r in results],
            "low_gap_regimes": [bool(r["low_gap_regime"]) for r in results],
            "layer_indices": [r["layer_idx"] for r in results],
            "ratio_threshold": float(results[0]["ratio_threshold"]),
            "y_scale": float(results[0]["y_scale"]),
            "n_eval_samples": int(results[0]["n_eval_samples"]),
            "best_recovery": float(best_by_primary["recovery_primary"]),
            "best_recovery_metric": best_metric,
            "best_recovery_raw_abs": max([abs(x) for x in raw_recovery_ratios]),
            "best_recovery_stable_abs": max([abs(x) for x in stable_recovery_ratios]),
            "best_layer": int(best_by_primary["layer_idx"]),
            "best_restoration_abs_mean": float(best_by_primary["restoration_abs_mean"]),
            "best_clean_corrupt_gap_abs_mean": float(
                best_by_primary["clean_corrupt_gap_abs_mean"]
            ),
            "best_residual_abs_mean": float(best_by_primary["residual_abs_mean"]),
            "best_restoration_sigma": float(best_by_primary["restoration_sigma"]),
            "best_residual_sigma": float(best_by_primary["residual_sigma"]),
            "best_recovery_fractional_abs": best_by_primary["recovery_fractional_abs"],
            "best_recovery_fractional_signed": best_by_primary[
                "recovery_fractional_signed"
            ],
            "metric_mode": str(self.config.metric_mode),
            "best_fractional_abs_overall": best_fractional_abs_overall,
            "best_fractional_signed_overall": best_fractional_signed_overall,
            "fractional_abs_valid_layers": int(
                sum(1 for x in results if bool(x["ratio_valid_abs"]))
            ),
            "fractional_signed_valid_layers": int(
                sum(1 for x in results if bool(x["ratio_valid_signed"]))
            ),
        }
        return summary

    def patch_single_head(
        self, head_idx: int, X_clean: np.ndarray, X_corrupt: np.ndarray
    ) -> Tuple[Dict, List[Dict]]:
        """Patch a single attention head across all layers."""
        print(f"\nPatching head {head_idx}...")

        results = sweep_layers(
            regressor=self.regressor,
            X_clean=X_clean,
            X_corrupt=X_corrupt,
            corrupt_idx=self.config.corrupt_idx,
            n_train_samples=self.config.n_train_samples,
            patch_indices=head_idx,
            patch_dim=self.config.patch_dim,
            ratio_epsilon=self.config.ratio_epsilon,
            ratio_threshold=self.config.ratio_threshold,
            y_scale=self.config.y_scale,
            metric_mode=self.config.metric_mode,
            max_layers=self.config.max_layers,
        )

        summary = self._build_summary(results, "head_idx", head_idx)

        return summary, results

    def patch_multiple_heads(
        self, head_indices: List[int], X_clean: np.ndarray, X_corrupt: np.ndarray
    ) -> Tuple[Dict, List[Dict]]:
        """Patch multiple attention heads at once."""
        print(f"\nPatching heads {head_indices}...")

        results = sweep_layers(
            regressor=self.regressor,
            X_clean=X_clean,
            X_corrupt=X_corrupt,
            corrupt_idx=self.config.corrupt_idx,
            n_train_samples=self.config.n_train_samples,
            patch_indices=head_indices,
            patch_dim=self.config.patch_dim,
            ratio_epsilon=self.config.ratio_epsilon,
            ratio_threshold=self.config.ratio_threshold,
            y_scale=self.config.y_scale,
            metric_mode=self.config.metric_mode,
            max_layers=self.config.max_layers,
        )

        summary = self._build_summary(results, "head_indices", head_indices)

        return summary, results

    def patch_single_token(
        self, token_idx: int, X_clean: np.ndarray, X_corrupt: np.ndarray
    ) -> Tuple[Dict, List[Dict]]:
        """Patch a single token across all layers."""
        print(f"\nPatching token {token_idx}...")

        results = sweep_layers(
            regressor=self.regressor,
            X_clean=X_clean,
            X_corrupt=X_corrupt,
            corrupt_idx=self.config.corrupt_idx,
            n_train_samples=self.config.n_train_samples,
            patch_indices=token_idx,
            patch_dim=self.config.patch_dim,
            ratio_epsilon=self.config.ratio_epsilon,
            ratio_threshold=self.config.ratio_threshold,
            y_scale=self.config.y_scale,
            metric_mode=self.config.metric_mode,
            max_layers=self.config.max_layers,
        )

        summary = self._build_summary(results, "token_idx", token_idx)

        return summary, results

    def patch_multiple_tokens(
        self, token_indices: List[int], X_clean: np.ndarray, X_corrupt: np.ndarray
    ) -> Tuple[Dict, List[Dict]]:
        """Patch multiple tokens at once."""
        print(f"\nPatching tokens {token_indices}...")

        results = sweep_layers(
            regressor=self.regressor,
            X_clean=X_clean,
            X_corrupt=X_corrupt,
            corrupt_idx=self.config.corrupt_idx,
            n_train_samples=self.config.n_train_samples,
            patch_indices=token_indices,
            patch_dim=self.config.patch_dim,
            ratio_epsilon=self.config.ratio_epsilon,
            ratio_threshold=self.config.ratio_threshold,
            y_scale=self.config.y_scale,
            metric_mode=self.config.metric_mode,
            max_layers=self.config.max_layers,
        )

        summary = self._build_summary(results, "token_indices", token_indices)

        return summary, results

    def patch_full_layer(
        self, X_clean: np.ndarray, X_corrupt: np.ndarray
    ) -> Tuple[Dict, List[Dict]]:
        """Patch the full layer output (all tokens and heads)."""
        print("\nPatching full layer output (all tokens and heads)...")

        results = sweep_layers(
            regressor=self.regressor,
            X_clean=X_clean,
            X_corrupt=X_corrupt,
            corrupt_idx=self.config.corrupt_idx,
            n_train_samples=self.config.n_train_samples,
            patch_indices=0,
            patch_dim=None,
            ratio_epsilon=self.config.ratio_epsilon,
            ratio_threshold=self.config.ratio_threshold,
            y_scale=self.config.y_scale,
            metric_mode=self.config.metric_mode,
            max_layers=self.config.max_layers,
        )

        summary = self._build_summary(results, "patch_dim", None)

        return summary, results

    def save_head_results(
        self, head_idx: int, summary: Dict, raw_results: List[Dict], script_path: str
    ):
        """Save results for a single head."""
        subdir = f"head_{head_idx}"

        # Save summary JSON
        self.save_results(summary, subdir, "summary", script_path)

        # Save tensors separately
        tensors = {
            "y_clean": summary["y_clean"],
            "y_corrupt": summary["y_corrupt"],
            "restorations": torch.tensor(summary["restorations"]),
            "recovery_ratios": torch.tensor(summary["recovery_ratios"]),
            "recovery_ratios_stable": torch.tensor(summary["recovery_ratios_stable"]),
            "recovery_scores": torch.tensor(summary["recovery_scores"]),
            "layer_indices": torch.tensor(summary["layer_indices"]),
        }
        self.save_tensors(tensors, "tensors", f"head_{head_idx}")

        # Create and save plot
        self._plot_single_head(summary, head_idx)

    def save_token_results(
        self, token_idx: int, summary: Dict, raw_results: List[Dict], script_path: str
    ):
        """Save results for a single token."""
        subdir = f"token_{token_idx}"

        # Save summary JSON
        self.save_results(summary, subdir, "summary", script_path)

        # Save tensors separately
        tensors = {
            "y_clean": summary["y_clean"],
            "y_corrupt": summary["y_corrupt"],
            "restorations": torch.tensor(summary["restorations"]),
            "recovery_ratios": torch.tensor(summary["recovery_ratios"]),
            "recovery_ratios_stable": torch.tensor(summary["recovery_ratios_stable"]),
            "recovery_scores": torch.tensor(summary["recovery_scores"]),
            "layer_indices": torch.tensor(summary["layer_indices"]),
        }
        self.save_tensors(tensors, "tensors", f"token_{token_idx}")

        # Create and save plot
        self._plot_single_token(summary, token_idx)

    def save_full_layer_results(
        self, summary: Dict, raw_results: List[Dict], script_path: str
    ):
        """Save results for full layer patching."""
        subdir = "full_layer"

        # Save summary JSON
        self.save_results(summary, subdir, "summary", script_path)

        # Save tensors separately
        tensors = {
            "y_clean": summary["y_clean"],
            "y_corrupt": summary["y_corrupt"],
            "restorations": torch.tensor(summary["restorations"]),
            "recovery_ratios": torch.tensor(summary["recovery_ratios"]),
            "recovery_ratios_stable": torch.tensor(summary["recovery_ratios_stable"]),
            "recovery_scores": torch.tensor(summary["recovery_scores"]),
            "layer_indices": torch.tensor(summary["layer_indices"]),
        }
        self.save_tensors(tensors, "tensors", "full_layer")

        # Create and save plot
        self._plot_full_layer(summary)

    def _plot_full_layer(self, summary: Dict):
        """Create restoration plot for full layer patching."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        layer_indices = summary["layer_indices"]
        restorations = summary["restorations"]
        recovery_ratios = summary.get(
            "recovery_ratios_stable", summary["recovery_ratios"]
        )
        y_clean = summary["y_clean"]
        y_corrupt = summary["y_corrupt"]

        # Restoration plot
        ax1.plot(layer_indices, restorations, "o-", linewidth=2, markersize=8)
        ax1.axhline(
            y=y_clean - y_corrupt,
            color="r",
            linestyle="--",
            label=f"Target ({y_clean - y_corrupt:.4f})",
        )
        ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Restoration")
        ax1.set_title("Full Layer Patching: Restoration by Layer")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Recovery ratio plot
        ax2.plot(
            layer_indices,
            [r * 100 for r in recovery_ratios],
            "o-",
            linewidth=2,
            markersize=8,
            color="green",
        )
        ax2.axhline(y=100, color="r", linestyle="--", label="Full Recovery")
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Recovery %")
        ax2.set_title("Full Layer Patching: Stable Recovery Ratio")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / f"full_layer_restoration_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved plot: {save_path}")

    def _plot_single_head(self, summary: Dict, head_idx: int):
        """Create restoration plot for a single head."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        layer_indices = summary["layer_indices"]
        restorations = summary["restorations"]
        recovery_ratios = summary.get(
            "recovery_ratios_stable", summary["recovery_ratios"]
        )
        y_clean = summary["y_clean"]
        y_corrupt = summary["y_corrupt"]

        # Restoration plot
        ax1.plot(layer_indices, restorations, "o-", linewidth=2, markersize=8)
        ax1.axhline(
            y=y_clean - y_corrupt,
            color="r",
            linestyle="--",
            label=f"Target ({y_clean - y_corrupt:.4f})",
        )
        ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Restoration")
        ax1.set_title(f"Head {head_idx}: Restoration by Layer")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Recovery ratio plot
        ax2.plot(
            layer_indices,
            [r * 100 for r in recovery_ratios],
            "o-",
            linewidth=2,
            markersize=8,
            color="green",
        )
        ax2.axhline(y=100, color="r", linestyle="--", label="Full Recovery")
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Recovery %")
        ax2.set_title(f"Head {head_idx}: Stable Recovery Ratio")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = (
            self.output_dir / f"head_{head_idx}_restoration_{self.timestamp}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved plot: {save_path}")

    def _plot_single_token(self, summary: Dict, token_idx: int):
        """Create restoration plot for a single token."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        layer_indices = summary["layer_indices"]
        restorations = summary["restorations"]
        recovery_ratios = summary.get(
            "recovery_ratios_stable", summary["recovery_ratios"]
        )
        y_clean = summary["y_clean"]
        y_corrupt = summary["y_corrupt"]

        ax1.plot(layer_indices, restorations, "o-", linewidth=2, markersize=8)
        ax1.axhline(
            y=y_clean - y_corrupt,
            color="r",
            linestyle="--",
            label=f"Target ({y_clean - y_corrupt:.4f})",
        )
        ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Restoration")
        ax1.set_title(f"Token {token_idx}: Restoration by Layer")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            layer_indices,
            [r * 100 for r in recovery_ratios],
            "o-",
            linewidth=2,
            markersize=8,
            color="green",
        )
        ax2.axhline(y=100, color="r", linestyle="--", label="Full Recovery")
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Recovery %")
        ax2.set_title(f"Token {token_idx}: Stable Recovery Ratio")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = (
            self.output_dir / f"token_{token_idx}_restoration_{self.timestamp}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved plot: {save_path}")

    def create_comparison_plot(self, all_summaries: List[Dict], script_path: str):
        """Create comparison plot across heads or tokens."""
        print("\nCreating comparison plots...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if "head_idx" in all_summaries[0]:
            index_key = "head_idx"
            label_prefix = "Head"
            comparison_label = "heads"
        elif "token_idx" in all_summaries[0]:
            index_key = "token_idx"
            label_prefix = "Token"
            comparison_label = "tokens"
        else:
            index_key = None
            label_prefix = "Item"
            comparison_label = "items"

        colors = ["blue", "green", "red", "purple", "orange", "brown"]

        for i, summary in enumerate(all_summaries):
            if index_key is None:
                item_idx = i
            else:
                item_idx = summary[index_key]
            layer_indices = summary["layer_indices"]
            restorations = summary["restorations"]
            recovery_ratios = summary.get(
                "recovery_ratios_stable", summary["recovery_ratios"]
            )

            ax1.plot(
                layer_indices,
                restorations,
                "o-",
                linewidth=2,
                markersize=6,
                label=f"{label_prefix} {item_idx}",
                color=colors[i % len(colors)],
            )
            ax2.plot(
                layer_indices,
                [r * 100 for r in recovery_ratios],
                "o-",
                linewidth=2,
                markersize=6,
                label=f"{label_prefix} {item_idx}",
                color=colors[i % len(colors)],
            )

        # Restoration comparison
        ax1.axhline(
            y=all_summaries[0]["y_clean"] - all_summaries[0]["y_corrupt"],
            color="k",
            linestyle="--",
            alpha=0.5,
            label="Target",
        )
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Restoration")
        ax1.set_title(f"Restoration Comparison: All {comparison_label.title()}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Recovery comparison
        ax2.axhline(y=100, color="k", linestyle="--", alpha=0.5, label="Full Recovery")
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Recovery %")
        ax2.set_title(f"Recovery Ratio: All {comparison_label.title()}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = (
            self.output_dir / f"comparison_all_{comparison_label}_{self.timestamp}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved comparison plot: {save_path}")

        # Save combined metadata
        best_summary = max(all_summaries, key=lambda x: x["best_recovery"])
        combined_summary = {
            "all_items": all_summaries,
            "comparison_type": comparison_label,
            "best_recovery_overall": max([s["best_recovery"] for s in all_summaries]),
        }
        if comparison_label == "heads":
            combined_summary["all_heads"] = all_summaries
            combined_summary["best_head"] = best_summary["head_idx"]
        elif comparison_label == "tokens":
            combined_summary["all_tokens"] = all_summaries
            combined_summary["best_token"] = best_summary["token_idx"]
        elif index_key is not None:
            combined_summary["best_item"] = best_summary[index_key]
        self.save_results(combined_summary, "comparisons", "summary", script_path)

    def create_heatmap(self, all_summaries: List[Dict], script_path: str):
        """Create heatmap of recovery ratios (layers × items)."""
        print("\nCreating heatmap...")

        if "head_idx" in all_summaries[0]:
            index_key = "head_idx"
            label_prefix = "Head"
            x_axis_label = "Attention Head"
            title_suffix = "Heads"
            comparison_label = "heads"
        elif "token_idx" in all_summaries[0]:
            index_key = "token_idx"
            label_prefix = "Token"
            x_axis_label = "Token"
            title_suffix = "Tokens"
            comparison_label = "tokens"
        else:
            index_key = None
            label_prefix = "Item"
            x_axis_label = "Item"
            title_suffix = "Items"
            comparison_label = "items"

        num_layers = len(all_summaries[0]["layer_indices"])
        num_items = len(all_summaries)

        # Build heatmap data
        heatmap_data = np.zeros((num_layers, num_items))
        for i, summary in enumerate(all_summaries):
            for layer_idx, recovery in zip(
                summary["layer_indices"],
                summary.get("recovery_ratios_stable", summary["recovery_ratios"]),
            ):
                heatmap_data[layer_idx, i] = recovery * 100

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=-100, vmax=100)

        ax.set_xticks(range(num_items))
        if index_key is None:
            item_indices = list(range(num_items))
        else:
            item_indices = [s[index_key] for s in all_summaries]
        ax.set_xticklabels([f"{label_prefix} {h}" for h in item_indices])
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f"Layer {i}" for i in range(num_layers)])
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel("Layer Index")
        ax.set_title(f"Recovery Ratio Heatmap (%): {title_suffix}")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Recovery %")

        plt.tight_layout()

        save_path = (
            self.output_dir
            / f"comparison_heatmap_{comparison_label}_{self.timestamp}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved heatmap: {save_path}")

        # Save heatmap data as tensor
        self.save_tensors({"heatmap": torch.tensor(heatmap_data)}, "tensors", "heatmap")
