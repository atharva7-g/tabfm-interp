from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor

from src.experiments.hooks.base import BaseExperiment
from src.experiments.attention_head_ablation.core_head_ablation import (
    sweep_heads_and_layers,
    _get_num_heads,
    _get_num_layers,
)


class HeadAblationExperiment(BaseExperiment):
    def __init__(
        self,
        regressor: TabPFNRegressor,
        output_dir: str = "results_ablation_heads_mha",
        script_path: str = "",
    ):
        super().__init__(output_dir)
        self.regressor = regressor
        self.script_path = script_path
        self.created_images: list = []

    def run_sweep(
        self,
        X: np.ndarray,
        head_indices: Optional[List[int]] = None,
        layer_indices: Optional[List[int]] = None,
        ratio_epsilon: float = 0.05,
        y_scale: Optional[float] = None,
    ) -> List[Dict]:
        if head_indices is None:
            head_indices = list(range(_get_num_heads(self.regressor)))
        if layer_indices is None:
            layer_indices = list(range(_get_num_layers(self.regressor)))

        return sweep_heads_and_layers(
            regressor=self.regressor,
            X=X,
            head_indices=head_indices,
            layer_indices=layer_indices,
            ratio_epsilon=ratio_epsilon,
            y_scale=y_scale,
        )

    def build_per_head_summaries(self, results: List[Dict]) -> Dict[int, Dict]:
        by_head: Dict[int, List[Dict]] = {}
        for r in results:
            by_head.setdefault(r["head_idx"], []).append(r)

        summaries = {}
        for head_idx, head_results in by_head.items():
            summaries[head_idx] = self._build_summary(head_results)
        return summaries

    def _build_summary(self, results: List[Dict]) -> Dict:
        head_idx = results[0]["head_idx"]
        raw_ratios = [float(r["ablation_ratio"]) for r in results]
        stable_ratios = [float(r["ablation_ratio_stable"]) for r in results]
        stable_abs_ratios = [float(r["ablation_ratio_stable_abs"]) for r in results]
        best_by_sigma = max(results, key=lambda x: x["ablation_effect_sigma"])

        summary = {
            "head_idx": head_idx,
            "y_normal": results[0]["y_normal"],
            "y_ablated": results[0]["y_ablated"],
            "ablation_effects": [float(r["ablation_effect"]) for r in results],
            "ablation_effects_abs_mean": [
                float(r["ablation_effect_abs_mean"]) for r in results
            ],
            "ablation_effect_sigmas": [
                float(r["ablation_effect_sigma"]) for r in results
            ],
            "ablation_ratios": raw_ratios,
            "ablation_ratios_stable": stable_ratios,
            "ablation_ratios_stable_abs": stable_abs_ratios,
            "layer_indices": [r["layer_idx"] for r in results],
            "best_effect_abs": float(best_by_sigma["ablation_effect_abs_mean"]),
            "best_effect_sigma": float(best_by_sigma["ablation_effect_sigma"]),
            "best_layer": int(best_by_sigma["layer_idx"]),
            "best_ratio_raw_abs": max(abs(x) for x in raw_ratios),
            "best_ratio_stable_abs": max(stable_abs_ratios),
            "y_scale": float(results[0]["y_scale"]),
        }
        return summary

    def save_head_results(self, head_idx: int, summary: Dict, script_path: str):
        subdir = f"head_{head_idx}"
        self.save_results(summary, subdir, "summary", script_path)

        tensors = {
            "ablation_effects": torch.tensor(summary["ablation_effects"]),
            "ablation_effects_abs_mean": torch.tensor(
                summary["ablation_effects_abs_mean"]
            ),
            "ablation_ratios_stable": torch.tensor(
                summary["ablation_ratios_stable"]
            ),
            "ablation_ratios_stable_abs": torch.tensor(
                summary["ablation_ratios_stable_abs"]
            ),
            "layer_indices": torch.tensor(summary["layer_indices"]),
        }
        self.save_tensors(tensors, "tensors", f"head_{head_idx}")

    def create_comparison_plot(self, all_summaries: Dict[int, Dict], script_path: str):
        print("\nCreating MHA head ablation comparison plots...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        colors = ["blue", "green", "red", "purple", "orange"]
        for i, (head_idx, summary) in enumerate(sorted(all_summaries.items())):
            layer_indices = summary["layer_indices"]
            effects = summary["ablation_effects_abs_mean"]
            sigmas = summary["ablation_effect_sigmas"]
            color = colors[i % len(colors)]

            ax1.plot(
                layer_indices,
                effects,
                "o-",
                linewidth=2,
                markersize=6,
                label=f"Head {head_idx}",
                color=color,
            )
            ax2.plot(
                layer_indices,
                sigmas,
                "o-",
                linewidth=2,
                markersize=6,
                label=f"Head {head_idx}",
                color=color,
            )

        ax1.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Ablation Effect (abs mean)")
        ax1.set_title("MHA Head Ablation: Effect by Layer")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Effect (σ)")
        ax2.set_title("MHA Head Ablation: Effect Sigma by Layer")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = (
            self.output_dir / f"mha_heads_ablation_comparison_{self.timestamp}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved: {save_path}")

        combined = {
            "all_heads": {str(k): v for k, v in all_summaries.items()},
            "best_head_overall": max(
                all_summaries.values(), key=lambda s: s["best_effect_sigma"]
            )["head_idx"],
        }
        self.save_results(combined, "comparisons", "summary", script_path)

    def create_heatmap(self, all_summaries: Dict[int, Dict], script_path: str):
        print("\nCreating MHA head ablation heatmap...")
        num_layers = len(list(all_summaries.values())[0]["layer_indices"])
        num_heads = len(all_summaries)
        head_indices = sorted(all_summaries.keys())

        heatmap_data = np.zeros((num_layers, num_heads))
        for j, head_idx in enumerate(head_indices):
            summary = all_summaries[head_idx]
            for i, sigma in enumerate(summary["ablation_effect_sigmas"]):
                heatmap_data[i, j] = sigma

        fig, ax = plt.subplots(figsize=(6, 8))
        im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([f"Head {h}" for h in head_indices])
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f"L{i}" for i in range(num_layers)])
        ax.set_xlabel("Attention Head (MHA)")
        ax.set_ylabel("Layer")
        ax.set_title("MHA Head Ablation: Effect (σ)")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Effect (σ)")
        plt.tight_layout()

        save_path = self.output_dir / f"mha_heads_ablation_heatmap_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved: {save_path}")
        self.save_tensors(
            {"heatmap": torch.tensor(heatmap_data)}, "tensors", "heatmap"
        )
