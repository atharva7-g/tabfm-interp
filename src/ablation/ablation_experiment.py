import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.experiments.hooks.base import BaseExperiment

from src.ablation.core_ablation import sweep_layers_for_ablation


@dataclass
class AblationConfig:
    seed: int
    n_train_samples: int
    ablate_dim: int = 2
    ablation_type: str = "zero"
    max_layers: Optional[int] = None
    ratio_epsilon: float = 0.05
    y_scale: Optional[float] = None
    scale_mode: str = "y_scale"


class AblationExperiment(BaseExperiment):
    def __init__(
        self,
        regressor: TabPFNRegressor,
        config: AblationConfig,
        output_dir: str = "results/ablation",
        script_path: str = "",
    ):
        super().__init__(output_dir)
        self.regressor = regressor
        self.config = config
        self.script_path = script_path
        self.created_images: list = []

    def ablate_single_head(
        self, head_idx: int, X: np.ndarray
    ) -> Tuple[Dict, List[Dict]]:
        """Ablate a single attention head across all layers."""
        print(f"\nAblating head {head_idx}...")

        results = sweep_layers_for_ablation(
            regressor=self.regressor,
            X=X,
            ablate_indices=head_idx,
            ablate_dim=self.config.ablate_dim,
            ablation_type=self.config.ablation_type,
            max_layers=self.config.max_layers,
            ratio_epsilon=self.config.ratio_epsilon,
            y_scale=self.config.y_scale,
            scale_mode=self.config.scale_mode,
        )

        raw_ratios = [r["ablation_ratio"] for r in results]
        stable_ratios = [r["ablation_ratio_stable"] for r in results]
        stable_abs_ratios = [r["ablation_ratio_stable_abs"] for r in results]
        best_by_stable_abs = max(results, key=lambda x: x["ablation_ratio_stable_abs"])

        summary = {
            "head_idx": head_idx,
            "y_normal": results[0]["y_normal"],
            "y_ablated": results[0]["y_ablated"],
            "ablation_effects": [r["ablation_effect"] for r in results],
            "ablation_effects_abs_mean": [
                r["ablation_effect_abs_mean"] for r in results
            ],
            "ablation_effect_sigmas": [r["ablation_effect_sigma"] for r in results],
            "ablation_ratios": raw_ratios,
            "ablation_ratios_stable": stable_ratios,
            "ablation_ratios_stable_abs": stable_abs_ratios,
            "layer_indices": [r["layer_idx"] for r in results],
            "best_effect": best_by_stable_abs["ablation_effect_abs_mean"],
            "best_effect_raw_abs": max([abs(r["ablation_effect"]) for r in results]),
            "best_ratio_raw_abs": max([abs(x) for x in raw_ratios]),
            "best_ratio_stable_abs": max(stable_abs_ratios),
            "best_effect_sigma": best_by_stable_abs["ablation_effect_sigma"],
            "y_scale": results[0]["y_scale"],
            "best_layer": best_by_stable_abs["layer_idx"],
        }

        return summary, results

    def ablate_multiple_heads(
        self, head_indices: List[int], X: np.ndarray
    ) -> Tuple[Dict, List[Dict]]:
        """Ablate multiple attention heads at once."""
        print(f"\nAblating heads {head_indices}...")

        results = sweep_layers_for_ablation(
            regressor=self.regressor,
            X=X,
            ablate_indices=head_indices,
            ablate_dim=self.config.ablate_dim,
            ablation_type=self.config.ablation_type,
            max_layers=self.config.max_layers,
            ratio_epsilon=self.config.ratio_epsilon,
            y_scale=self.config.y_scale,
            scale_mode=self.config.scale_mode,
        )

        raw_ratios = [r["ablation_ratio"] for r in results]
        stable_ratios = [r["ablation_ratio_stable"] for r in results]
        stable_abs_ratios = [r["ablation_ratio_stable_abs"] for r in results]
        best_by_stable_abs = max(results, key=lambda x: x["ablation_ratio_stable_abs"])

        summary = {
            "head_indices": head_indices,
            "y_normal": results[0]["y_normal"],
            "y_ablated": results[0]["y_ablated"],
            "ablation_effects": [r["ablation_effect"] for r in results],
            "ablation_effects_abs_mean": [
                r["ablation_effect_abs_mean"] for r in results
            ],
            "ablation_effect_sigmas": [r["ablation_effect_sigma"] for r in results],
            "ablation_ratios": raw_ratios,
            "ablation_ratios_stable": stable_ratios,
            "ablation_ratios_stable_abs": stable_abs_ratios,
            "layer_indices": [r["layer_idx"] for r in results],
            "best_effect": best_by_stable_abs["ablation_effect_abs_mean"],
            "best_effect_raw_abs": max([abs(r["ablation_effect"]) for r in results]),
            "best_ratio_raw_abs": max([abs(x) for x in raw_ratios]),
            "best_ratio_stable_abs": max(stable_abs_ratios),
            "best_effect_sigma": best_by_stable_abs["ablation_effect_sigma"],
            "y_scale": results[0]["y_scale"],
            "best_layer": best_by_stable_abs["layer_idx"],
        }

        return summary, results

    def ablate_single_token(
        self, token_idx: int, X: np.ndarray
    ) -> Tuple[Dict, List[Dict]]:
        """Ablate a single token across all layers."""
        print(f"\nAblating token {token_idx}...")

        results = sweep_layers_for_ablation(
            regressor=self.regressor,
            X=X,
            ablate_indices=token_idx,
            ablate_dim=self.config.ablate_dim,
            ablation_type=self.config.ablation_type,
            max_layers=self.config.max_layers,
            ratio_epsilon=self.config.ratio_epsilon,
            y_scale=self.config.y_scale,
            scale_mode=self.config.scale_mode,
        )

        raw_ratios = [r["ablation_ratio"] for r in results]
        stable_ratios = [r["ablation_ratio_stable"] for r in results]
        stable_abs_ratios = [r["ablation_ratio_stable_abs"] for r in results]
        best_by_stable_abs = max(results, key=lambda x: x["ablation_ratio_stable_abs"])

        summary = {
            "token_idx": token_idx,
            "y_normal": results[0]["y_normal"],
            "y_ablated": results[0]["y_ablated"],
            "ablation_effects": [r["ablation_effect"] for r in results],
            "ablation_effects_abs_mean": [
                r["ablation_effect_abs_mean"] for r in results
            ],
            "ablation_effect_sigmas": [r["ablation_effect_sigma"] for r in results],
            "ablation_ratios": raw_ratios,
            "ablation_ratios_stable": stable_ratios,
            "ablation_ratios_stable_abs": stable_abs_ratios,
            "layer_indices": [r["layer_idx"] for r in results],
            "best_effect": best_by_stable_abs["ablation_effect_abs_mean"],
            "best_effect_raw_abs": max([abs(r["ablation_effect"]) for r in results]),
            "best_ratio_raw_abs": max([abs(x) for x in raw_ratios]),
            "best_ratio_stable_abs": max(stable_abs_ratios),
            "best_effect_sigma": best_by_stable_abs["ablation_effect_sigma"],
            "y_scale": results[0]["y_scale"],
            "best_layer": best_by_stable_abs["layer_idx"],
        }

        return summary, results

    def ablate_multiple_tokens(
        self, token_indices: List[int], X: np.ndarray
    ) -> Tuple[Dict, List[Dict]]:
        """Ablate multiple tokens at once."""
        print(f"\nAblating tokens {token_indices}...")

        results = sweep_layers_for_ablation(
            regressor=self.regressor,
            X=X,
            ablate_indices=token_indices,
            ablate_dim=self.config.ablate_dim,
            ablation_type=self.config.ablation_type,
            max_layers=self.config.max_layers,
            ratio_epsilon=self.config.ratio_epsilon,
            y_scale=self.config.y_scale,
            scale_mode=self.config.scale_mode,
        )

        raw_ratios = [r["ablation_ratio"] for r in results]
        stable_ratios = [r["ablation_ratio_stable"] for r in results]
        stable_abs_ratios = [r["ablation_ratio_stable_abs"] for r in results]
        best_by_stable_abs = max(results, key=lambda x: x["ablation_ratio_stable_abs"])

        summary = {
            "token_indices": token_indices,
            "y_normal": results[0]["y_normal"],
            "y_ablated": results[0]["y_ablated"],
            "ablation_effects": [r["ablation_effect"] for r in results],
            "ablation_effects_abs_mean": [
                r["ablation_effect_abs_mean"] for r in results
            ],
            "ablation_effect_sigmas": [r["ablation_effect_sigma"] for r in results],
            "ablation_ratios": raw_ratios,
            "ablation_ratios_stable": stable_ratios,
            "ablation_ratios_stable_abs": stable_abs_ratios,
            "layer_indices": [r["layer_idx"] for r in results],
            "best_effect": best_by_stable_abs["ablation_effect_abs_mean"],
            "best_effect_raw_abs": max([abs(r["ablation_effect"]) for r in results]),
            "best_ratio_raw_abs": max([abs(x) for x in raw_ratios]),
            "best_ratio_stable_abs": max(stable_abs_ratios),
            "best_effect_sigma": best_by_stable_abs["ablation_effect_sigma"],
            "y_scale": results[0]["y_scale"],
            "best_layer": best_by_stable_abs["layer_idx"],
        }

        return summary, results

    def ablate_full_layer(self, X: np.ndarray) -> Tuple[Dict, List[Dict]]:
        """Ablate the full layer output (all tokens and heads)."""
        print("\nAblating full layer output (all tokens and heads)...")

        results = sweep_layers_for_ablation(
            regressor=self.regressor,
            X=X,
            ablate_indices=0,
            ablate_dim=None,
            ablation_type=self.config.ablation_type,
            max_layers=self.config.max_layers,
            ratio_epsilon=self.config.ratio_epsilon,
            y_scale=self.config.y_scale,
            scale_mode=self.config.scale_mode,
        )

        raw_ratios = [r["ablation_ratio"] for r in results]
        stable_ratios = [r["ablation_ratio_stable"] for r in results]
        stable_abs_ratios = [r["ablation_ratio_stable_abs"] for r in results]
        best_by_stable_abs = max(results, key=lambda x: x["ablation_ratio_stable_abs"])

        summary = {
            "ablate_dim": None,
            "y_normal": results[0]["y_normal"],
            "y_ablated": results[0]["y_ablated"],
            "ablation_effects": [r["ablation_effect"] for r in results],
            "ablation_effects_abs_mean": [
                r["ablation_effect_abs_mean"] for r in results
            ],
            "ablation_effect_sigmas": [r["ablation_effect_sigma"] for r in results],
            "ablation_ratios": raw_ratios,
            "ablation_ratios_stable": stable_ratios,
            "ablation_ratios_stable_abs": stable_abs_ratios,
            "layer_indices": [r["layer_idx"] for r in results],
            "best_effect": best_by_stable_abs["ablation_effect_abs_mean"],
            "best_effect_raw_abs": max([abs(r["ablation_effect"]) for r in results]),
            "best_ratio_raw_abs": max([abs(x) for x in raw_ratios]),
            "best_ratio_stable_abs": max(stable_abs_ratios),
            "best_effect_sigma": best_by_stable_abs["ablation_effect_sigma"],
            "y_scale": results[0]["y_scale"],
            "best_layer": best_by_stable_abs["layer_idx"],
        }

        return summary, results

    def save_head_results(
        self, head_idx: int, summary: Dict, raw_results: List[Dict], script_path: str
    ):
        """Save results for a single head."""
        subdir = f"head_{head_idx}"

        self.save_results(summary, subdir, "summary", script_path)

        tensors = {
            "y_normal": summary["y_normal"],
            "y_ablated": summary["y_ablated"],
            "ablation_effects": torch.tensor(summary["ablation_effects"]),
            "ablation_ratios": torch.tensor(summary["ablation_ratios"]),
            "ablation_ratios_stable": torch.tensor(summary["ablation_ratios_stable"]),
            "ablation_ratios_stable_abs": torch.tensor(
                summary["ablation_ratios_stable_abs"]
            ),
            "layer_indices": torch.tensor(summary["layer_indices"]),
        }
        self.save_tensors(tensors, "tensors", f"head_{head_idx}")

        self._plot_single_head(summary, head_idx)

    def save_token_results(
        self, token_idx: int, summary: Dict, raw_results: List[Dict], script_path: str
    ):
        """Save results for a single token."""
        subdir = f"token_{token_idx}"

        self.save_results(summary, subdir, "summary", script_path)

        tensors = {
            "y_normal": summary["y_normal"],
            "y_ablated": summary["y_ablated"],
            "ablation_effects": torch.tensor(summary["ablation_effects"]),
            "ablation_ratios": torch.tensor(summary["ablation_ratios"]),
            "ablation_ratios_stable": torch.tensor(summary["ablation_ratios_stable"]),
            "ablation_ratios_stable_abs": torch.tensor(
                summary["ablation_ratios_stable_abs"]
            ),
            "layer_indices": torch.tensor(summary["layer_indices"]),
        }
        self.save_tensors(tensors, "tensors", f"token_{token_idx}")

        self._plot_single_token(summary, token_idx)

    def save_full_layer_results(
        self, summary: Dict, raw_results: List[Dict], script_path: str
    ):
        """Save results for full layer ablation."""
        subdir = "full_layer"

        self.save_results(summary, subdir, "summary", script_path)

        tensors = {
            "y_normal": summary["y_normal"],
            "y_ablated": summary["y_ablated"],
            "ablation_effects": torch.tensor(summary["ablation_effects"]),
            "ablation_ratios": torch.tensor(summary["ablation_ratios"]),
            "ablation_ratios_stable": torch.tensor(summary["ablation_ratios_stable"]),
            "ablation_ratios_stable_abs": torch.tensor(
                summary["ablation_ratios_stable_abs"]
            ),
            "layer_indices": torch.tensor(summary["layer_indices"]),
        }
        self.save_tensors(tensors, "tensors", "full_layer")

        self._plot_full_layer(summary)

    def _plot_full_layer(self, summary: Dict):
        """Create ablation plot for full layer."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        layer_indices = summary["layer_indices"]
        ablation_effects = summary["ablation_effects"]
        ablation_ratios = summary.get(
            "ablation_ratios_stable", summary["ablation_ratios"]
        )

        ax1.plot(
            layer_indices,
            ablation_effects,
            "o-",
            linewidth=2,
            markersize=8,
            label="Ablation effect",
        )
        ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3, label="Baseline")
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Ablation Effect")
        ax1.set_title("Full Layer Ablation: Effect by Layer")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            layer_indices,
            [r * 100 for r in ablation_ratios],
            "o-",
            linewidth=2,
            markersize=8,
            color="orange",
            label="Stable ratio",
        )
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3, label="Baseline")
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Ablation Ratio %")
        ax2.set_title("Full Layer Ablation: Stable Ratio")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / f"full_layer_ablation_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved plot: {save_path}")

    def _plot_single_head(self, summary: Dict, head_idx: int):
        """Create ablation plot for a single head."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        layer_indices = summary["layer_indices"]
        ablation_effects = summary["ablation_effects"]
        ablation_ratios = summary.get(
            "ablation_ratios_stable", summary["ablation_ratios"]
        )

        ax1.plot(
            layer_indices,
            ablation_effects,
            "o-",
            linewidth=2,
            markersize=8,
            label="Ablation effect",
        )
        ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3, label="Baseline")
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Ablation Effect")
        ax1.set_title(f"Head {head_idx}: Ablation Effect by Layer")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            layer_indices,
            [r * 100 for r in ablation_ratios],
            "o-",
            linewidth=2,
            markersize=8,
            color="orange",
            label="Stable ratio",
        )
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3, label="Baseline")
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Ablation Ratio %")
        ax2.set_title(f"Head {head_idx}: Stable Ablation Ratio")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / f"head_{head_idx}_ablation_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved plot: {save_path}")

    def _plot_single_token(self, summary: Dict, token_idx: int):
        """Create ablation plot for a single token."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        layer_indices = summary["layer_indices"]
        ablation_effects = summary["ablation_effects"]
        ablation_ratios = summary.get(
            "ablation_ratios_stable", summary["ablation_ratios"]
        )

        ax1.plot(
            layer_indices,
            ablation_effects,
            "o-",
            linewidth=2,
            markersize=8,
            label="Ablation effect",
        )
        ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3, label="Baseline")
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Ablation Effect")
        ax1.set_title(f"Token {token_idx}: Ablation Effect by Layer")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            layer_indices,
            [r * 100 for r in ablation_ratios],
            "o-",
            linewidth=2,
            markersize=8,
            color="orange",
            label="Stable ratio",
        )
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3, label="Baseline")
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Ablation Ratio %")
        ax2.set_title(f"Token {token_idx}: Stable Ablation Ratio")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / f"token_{token_idx}_ablation_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved plot: {save_path}")

    def create_comparison_plot(self, all_summaries: List[Dict], script_path: str):
        """Create comparison plot across all heads/tokens."""
        print("\nCreating comparison plots...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        colors = ["blue", "green", "red", "purple", "orange", "brown"]

        for i, summary in enumerate(all_summaries):
            if "head_idx" in summary:
                idx = summary["head_idx"]
                label = f"Head {idx}"
            elif "token_idx" in summary:
                idx = summary["token_idx"]
                label = f"Token {idx}"
            else:
                idx = i
                label = f"Item {idx}"

            layer_indices = summary["layer_indices"]
            ablation_effects = summary["ablation_effects"]
            ablation_ratios = summary.get(
                "ablation_ratios_stable", summary["ablation_ratios"]
            )

            ax1.plot(
                layer_indices,
                ablation_effects,
                "o-",
                linewidth=2,
                markersize=6,
                label=label,
                color=colors[i % len(colors)],
            )
            ax2.plot(
                layer_indices,
                [r * 100 for r in ablation_ratios],
                "o-",
                linewidth=2,
                markersize=6,
                label=label,
                color=colors[i % len(colors)],
            )

        ax1.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Ablation Effect")
        ax1.set_title("Ablation Effect Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Ablation Ratio %")
        ax2.set_title("Stable Ablation Ratio Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / f"comparison_ablation_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved comparison plot: {save_path}")

        combined_summary = {
            "all_items": all_summaries,
            "best_item": max(all_summaries, key=lambda x: x["best_effect"]),
            "best_effect_overall": max([s["best_effect"] for s in all_summaries]),
        }
        self.save_results(combined_summary, "comparisons", "summary", script_path)
