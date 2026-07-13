from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor

from src.experiments.hooks.base import BaseExperiment, ExperimentConfig
from src.experiments.attention_head_patching.core_head_patching import (
    sweep_heads_and_layers,
    _get_num_heads,
    _get_num_layers,
)


class HeadPatchingExperiment(BaseExperiment):
    def __init__(
        self,
        regressor: TabPFNRegressor,
        config: ExperimentConfig,
        output_dir: str = "results_attention_heads_mha",
        script_path: str = "",
    ):
        super().__init__(output_dir)
        self.regressor = regressor
        self.config = config
        self.script_path = script_path
        self.created_images: list = []

    def run_sweep(
        self,
        X_clean: np.ndarray,
        X_corrupt: np.ndarray,
        head_indices: Optional[List[int]] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> List[Dict]:
        if head_indices is None:
            head_indices = list(range(_get_num_heads(self.regressor)))
        if layer_indices is None:
            layer_indices = list(range(_get_num_layers(self.regressor)))

        results = sweep_heads_and_layers(
            regressor=self.regressor,
            X_clean=X_clean,
            X_corrupt=X_corrupt,
            corrupt_idx=self.config.corrupt_idx,
            n_train_samples=self.config.n_train_samples,
            head_indices=head_indices,
            layer_indices=layer_indices,
            ratio_epsilon=self.config.ratio_epsilon,
            ratio_threshold=self.config.ratio_threshold,
            y_scale=self.config.y_scale,
            metric_mode=self.config.metric_mode,
        )
        return results

    @staticmethod
    def _optional_to_float_list(values: List[float | None]) -> List[float]:
        return [float("nan") if v is None else float(v) for v in values]

    def build_per_head_summaries(
        self, results: List[Dict]
    ) -> Dict[int, Dict]:
        by_head: Dict[int, List[Dict]] = {}
        for r in results:
            hi = r["head_idx"]
            by_head.setdefault(hi, []).append(r)

        summaries = {}
        for head_idx, head_results in by_head.items():
            summaries[head_idx] = self._build_summary(head_results, "head_idx", head_idx)
        return summaries

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
            # Per-sample absolute recovery metrics (addresses sign-cancellation)
            "filtered_recovery_fractions": [
                float(r.get("filtered_recovery_fraction", 0.0)) for r in results
            ],
            "samples_improved_fractions": [
                float(r.get("samples_improved_fraction", 0.0)) for r in results
            ],
            "per_sample_mean_gaps": [
                float(r.get("per_sample_mean_gap", 0.0)) for r in results
            ],
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

    def save_head_results(
        self, head_idx: int, summary: Dict, script_path: str
    ):
        subdir = f"head_{head_idx}"
        self.save_results(summary, subdir, "summary", script_path)

        tensors = {
            "restorations": torch.tensor(summary["restorations"]),
            "recovery_ratios": torch.tensor(summary["recovery_ratios"]),
            "recovery_ratios_stable": torch.tensor(summary["recovery_ratios_stable"]),
            "recovery_scores": torch.tensor(summary["recovery_scores"]),
            "layer_indices": torch.tensor(summary["layer_indices"]),
        }
        self.save_tensors(tensors, "tensors", f"head_{head_idx}")

    def create_comparison_plot(self, all_summaries: Dict[int, Dict], script_path: str):
        print("\nCreating MHA head comparison plots...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        colors = ["blue", "green", "red", "purple", "orange"]
        for i, (head_idx, summary) in enumerate(sorted(all_summaries.items())):
            layer_indices = summary["layer_indices"]
            restorations = summary["restorations"]
            recovery_ratios = summary.get(
                "recovery_ratios_stable", summary["recovery_ratios"]
            )
            color = colors[i % len(colors)]

            ax1.plot(
                layer_indices,
                restorations,
                "o-",
                linewidth=2,
                markersize=6,
                label=f"Head {head_idx}",
                color=color,
            )
            ax2.plot(
                layer_indices,
                [r * 100 for r in recovery_ratios],
                "o-",
                linewidth=2,
                markersize=6,
                label=f"Head {head_idx}",
                color=color,
            )

        target = all_summaries[list(all_summaries.keys())[0]]["y_clean"] - \
                 all_summaries[list(all_summaries.keys())[0]]["y_corrupt"]
        ax1.axhline(y=target, color="k", linestyle="--", alpha=0.5, label="Target")
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Restoration")
        ax1.set_title("MHA Head Patching: Restoration by Layer")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.axhline(y=100, color="k", linestyle="--", alpha=0.5, label="Full Recovery")
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Recovery %")
        ax2.set_title("MHA Head Patching: Recovery Ratio by Layer")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / f"mha_heads_comparison_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved: {save_path}")

        combined = {
            "all_heads": {str(k): v for k, v in all_summaries.items()},
            "best_head_overall": max(
                all_summaries.values(), key=lambda s: s["best_recovery"]
            )["head_idx"],
        }
        self.save_results(combined, "comparisons", "summary", script_path)

    def create_heatmap(self, all_summaries: Dict[int, Dict], script_path: str):
        print("\nCreating MHA head heatmap...")
        num_layers = len(list(all_summaries.values())[0]["layer_indices"])
        num_heads = len(all_summaries)
        head_indices = sorted(all_summaries.keys())

        heatmap_data = np.zeros((num_layers, num_heads))
        for j, head_idx in enumerate(head_indices):
            summary = all_summaries[head_idx]
            for i, recovery in enumerate(
                summary.get("recovery_ratios_stable", summary["recovery_ratios"])
            ):
                heatmap_data[i, j] = recovery * 100

        fig, ax = plt.subplots(figsize=(6, 8))
        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=-100, vmax=100)
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([f"Head {h}" for h in head_indices])
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f"L{i}" for i in range(num_layers)])
        ax.set_xlabel("Attention Head (MHA)")
        ax.set_ylabel("Layer")
        ax.set_title("MHA Head Patching: Recovery % (Layer x Head)")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Recovery %")
        plt.tight_layout()

        save_path = self.output_dir / f"mha_heads_heatmap_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved: {save_path}")
        self.save_tensors(
            {"heatmap": torch.tensor(heatmap_data)}, "tensors", "heatmap"
        )
