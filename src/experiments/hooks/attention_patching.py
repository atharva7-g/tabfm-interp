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
            max_layers=self.config.max_layers,
        )

        # Extract key metrics
        summary = {
            "head_idx": head_idx,
            "y_clean": results[0]["y_clean"],
            "y_corrupt": results[0]["y_corrupt"],
            "restorations": [r["restoration"] for r in results],
            "recovery_ratios": [r["recovery_ratio"] for r in results],
            "layer_indices": [r["layer_idx"] for r in results],
            "best_recovery": max([abs(r["recovery_ratio"]) for r in results]),
            "best_layer": max(results, key=lambda x: abs(x["recovery_ratio"]))[
                "layer_idx"
            ],
        }

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
            max_layers=self.config.max_layers,
        )

        summary = {
            "head_indices": head_indices,
            "y_clean": results[0]["y_clean"],
            "y_corrupt": results[0]["y_corrupt"],
            "restorations": [r["restoration"] for r in results],
            "recovery_ratios": [r["recovery_ratio"] for r in results],
            "layer_indices": [r["layer_idx"] for r in results],
            "best_recovery": max([abs(r["recovery_ratio"]) for r in results]),
            "best_layer": max(results, key=lambda x: abs(x["recovery_ratio"]))[
                "layer_idx"
            ],
        }

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
            max_layers=self.config.max_layers,
        )

        summary = {
            "patch_dim": None,
            "y_clean": results[0]["y_clean"],
            "y_corrupt": results[0]["y_corrupt"],
            "restorations": [r["restoration"] for r in results],
            "recovery_ratios": [r["recovery_ratio"] for r in results],
            "layer_indices": [r["layer_idx"] for r in results],
            "best_recovery": max([abs(r["recovery_ratio"]) for r in results]),
            "best_layer": max(results, key=lambda x: abs(x["recovery_ratio"]))[
                "layer_idx"
            ],
        }

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
            "layer_indices": torch.tensor(summary["layer_indices"]),
        }
        self.save_tensors(tensors, "tensors", f"head_{head_idx}")

        # Create and save plot
        self._plot_single_head(summary, head_idx)

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
        recovery_ratios = summary["recovery_ratios"]
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
        ax2.set_title("Full Layer Patching: Recovery Ratio")
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
        recovery_ratios = summary["recovery_ratios"]
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
        ax2.set_title(f"Head {head_idx}: Recovery Ratio")
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

    def create_comparison_plot(self, all_summaries: List[Dict], script_path: str):
        """Create comparison plot across all heads."""
        print("\nCreating comparison plots...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        colors = ["blue", "green", "red", "purple"]

        for i, summary in enumerate(all_summaries):
            head_idx = summary["head_idx"]
            layer_indices = summary["layer_indices"]
            restorations = summary["restorations"]
            recovery_ratios = summary["recovery_ratios"]

            ax1.plot(
                layer_indices,
                restorations,
                "o-",
                linewidth=2,
                markersize=6,
                label=f"Head {head_idx}",
                color=colors[i % len(colors)],
            )
            ax2.plot(
                layer_indices,
                [r * 100 for r in recovery_ratios],
                "o-",
                linewidth=2,
                markersize=6,
                label=f"Head {head_idx}",
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
        ax1.set_title("Restoration Comparison: All Heads")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Recovery comparison
        ax2.axhline(y=100, color="k", linestyle="--", alpha=0.5, label="Full Recovery")
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Recovery %")
        ax2.set_title("Recovery Ratio: All Heads")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / f"comparison_all_heads_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved comparison plot: {save_path}")

        # Save combined metadata
        combined_summary = {
            "all_heads": all_summaries,
            "best_head": max(all_summaries, key=lambda x: x["best_recovery"])[
                "head_idx"
            ],
            "best_recovery_overall": max([s["best_recovery"] for s in all_summaries]),
        }
        self.save_results(combined_summary, "comparisons", "summary", script_path)

    def create_heatmap(self, all_summaries: List[Dict], script_path: str):
        """Create heatmap of recovery ratios (layers × heads)."""
        print("\nCreating heatmap...")

        num_layers = len(all_summaries[0]["layer_indices"])
        num_heads = len(all_summaries)

        # Build heatmap data
        heatmap_data = np.zeros((num_layers, num_heads))
        for i, summary in enumerate(all_summaries):
            head_idx = summary["head_idx"]
            for layer_idx, recovery in zip(
                summary["layer_indices"], summary["recovery_ratios"]
            ):
                heatmap_data[layer_idx, i] = (
                    recovery * 100
                )  # Use column index i, not head_idx

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=-100, vmax=100)

        ax.set_xticks(range(num_heads))
        head_indices = [s["head_idx"] for s in all_summaries]
        ax.set_xticklabels([f"Head {h}" for h in head_indices])
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f"Layer {i}" for i in range(num_layers)])
        ax.set_xlabel("Attention Head")
        ax.set_ylabel("Layer Index")
        ax.set_title("Recovery Ratio Heatmap (%)")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Recovery %")

        plt.tight_layout()

        save_path = self.output_dir / f"comparison_heatmap_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved heatmap: {save_path}")

        # Save heatmap data as tensor
        self.save_tensors({"heatmap": torch.tensor(heatmap_data)}, "tensors", "heatmap")
