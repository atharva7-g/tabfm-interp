from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from base import BaseExperiment, ExperimentConfig

from src.experiments.hooks.core_patching import (
    sweep_steering_layers,
    create_random_direction,
    create_direction_from_difference,
)


class SteeringConfig:
    def __init__(
        self,
        seed: int = 42,
        n_train_samples: int = 500,
        steer_dim: int = 2,
        alpha: float = 1.0,
        direction_type: str = "random",
        max_layers: Optional[int] = None,
    ):
        self.seed = seed
        self.n_train_samples = n_train_samples
        self.steer_dim = steer_dim
        self.alpha = alpha
        self.direction_type = direction_type
        self.max_layers = max_layers


class AttentionSteeringExperiment(BaseExperiment):
    """Experiment class for attention head steering."""

    def __init__(
        self,
        regressor: TabPFNRegressor,
        config: SteeringConfig,
        output_dir: str = "results/attention_steering",
        script_path: str = "",
    ):
        super().__init__(output_dir)
        self.regressor = regressor
        self.config = config
        self.script_path = script_path
        self.created_images: list = []

    def _get_activation_shape(self, X: np.ndarray) -> Tuple[int, ...]:
        model = self.regressor.model_
        layer = model.transformer_encoder.layers[0]
        attention_module = layer.self_attn_between_features

        activation_shape = None

        def shape_hook(module, inputs, output):
            nonlocal activation_shape
            if isinstance(output, (tuple, list)):
                output_tensor = output[0]
            else:
                output_tensor = output
            activation_shape = output_tensor.shape

        handle = attention_module.register_forward_hook(shape_hook)
        with torch.no_grad():
            self.regressor.predict(X)
        handle.remove()

        return activation_shape

    def _get_device(self) -> torch.device:
        return next(self.regressor.model_.parameters()).device

    def _create_direction(self, shape: Tuple[int, ...]) -> torch.Tensor:
        device = self._get_device()
        if self.config.direction_type == "random":
            return create_random_direction(
                shape, seed=self.config.seed, normalize=True, device=device
            )
        else:
            raise ValueError(f"Unknown direction_type: {self.config.direction_type}")

    def steer_single_head(
        self, head_idx: int, X: np.ndarray
    ) -> Tuple[Dict, List[Dict]]:
        """Steer a single attention head across all layers."""
        print(f"\nSteering head {head_idx} with alpha={self.config.alpha}...")

        activation_shape = self._get_activation_shape(X)
        direction = self._create_direction(activation_shape)

        results = sweep_steering_layers(
            regressor=self.regressor,
            X=X,
            direction=direction,
            steer_indices=head_idx,
            steer_dim=self.config.steer_dim,
            alpha=self.config.alpha,
            max_layers=self.config.max_layers,
        )

        summary = {
            "head_idx": head_idx,
            "y_normal": results[0]["y_normal"],
            "y_steered": results[0]["y_steered"],
            "steering_effects": [r["steering_effect"] for r in results],
            "layer_indices": [r["layer_idx"] for r in results],
            "best_effect": max([abs(r["steering_effect"]) for r in results]),
            "best_layer": max(results, key=lambda x: abs(x["steering_effect"]))[
                "layer_idx"
            ],
            "alpha": self.config.alpha,
        }

        return summary, results

    def steer_multiple_heads(
        self, head_indices: List[int], X: np.ndarray
    ) -> Tuple[Dict, List[Dict]]:
        """Steer multiple attention heads at once."""
        print(f"\nSteering heads {head_indices} with alpha={self.config.alpha}...")

        activation_shape = self._get_activation_shape(X)
        direction = self._create_direction(activation_shape)

        results = sweep_steering_layers(
            regressor=self.regressor,
            X=X,
            direction=direction,
            steer_indices=head_indices,
            steer_dim=self.config.steer_dim,
            alpha=self.config.alpha,
            max_layers=self.config.max_layers,
        )

        summary = {
            "head_indices": head_indices,
            "y_normal": results[0]["y_normal"],
            "y_steered": results[0]["y_steered"],
            "steering_effects": [r["steering_effect"] for r in results],
            "layer_indices": [r["layer_idx"] for r in results],
            "best_effect": max([abs(r["steering_effect"]) for r in results]),
            "best_layer": max(results, key=lambda x: abs(x["steering_effect"]))[
                "layer_idx"
            ],
            "alpha": self.config.alpha,
        }

        return summary, results

    def steer_single_token(
        self, token_idx: int, X: np.ndarray
    ) -> Tuple[Dict, List[Dict]]:
        """Steer a single token across all layers."""
        print(f"\nSteering token {token_idx} with alpha={self.config.alpha}...")

        activation_shape = self._get_activation_shape(X)
        direction = self._create_direction(activation_shape)

        results = sweep_steering_layers(
            regressor=self.regressor,
            X=X,
            direction=direction,
            steer_indices=token_idx,
            steer_dim=self.config.steer_dim,
            alpha=self.config.alpha,
            max_layers=self.config.max_layers,
        )

        summary = {
            "token_idx": token_idx,
            "y_normal": results[0]["y_normal"],
            "y_steered": results[0]["y_steered"],
            "steering_effects": [r["steering_effect"] for r in results],
            "layer_indices": [r["layer_idx"] for r in results],
            "best_effect": max([abs(r["steering_effect"]) for r in results]),
            "best_layer": max(results, key=lambda x: abs(x["steering_effect"]))[
                "layer_idx"
            ],
            "alpha": self.config.alpha,
        }

        return summary, results

    def steer_full_layer(self, X: np.ndarray) -> Tuple[Dict, List[Dict]]:
        """Steer the full layer output (all tokens and heads)."""
        print(f"\nSteering full layer with alpha={self.config.alpha}...")

        activation_shape = self._get_activation_shape(X)
        direction = self._create_direction(activation_shape)

        results = sweep_steering_layers(
            regressor=self.regressor,
            X=X,
            direction=direction,
            steer_indices=0,
            steer_dim=None,
            alpha=self.config.alpha,
            max_layers=self.config.max_layers,
        )

        summary = {
            "steer_dim": None,
            "y_normal": results[0]["y_normal"],
            "y_steered": results[0]["y_steered"],
            "steering_effects": [r["steering_effect"] for r in results],
            "layer_indices": [r["layer_idx"] for r in results],
            "best_effect": max([abs(r["steering_effect"]) for r in results]),
            "best_layer": max(results, key=lambda x: abs(x["steering_effect"]))[
                "layer_idx"
            ],
            "alpha": self.config.alpha,
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
            "y_steered": summary["y_steered"],
            "steering_effects": torch.tensor(summary["steering_effects"]),
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
            "y_steered": summary["y_steered"],
            "steering_effects": torch.tensor(summary["steering_effects"]),
            "layer_indices": torch.tensor(summary["layer_indices"]),
        }
        self.save_tensors(tensors, "tensors", f"token_{token_idx}")

        self._plot_single_token(summary, token_idx)

    def save_full_layer_results(
        self, summary: Dict, raw_results: List[Dict], script_path: str
    ):
        """Save results for full layer steering."""
        subdir = "full_layer"
        self.save_results(summary, subdir, "summary", script_path)

        tensors = {
            "y_normal": summary["y_normal"],
            "y_steered": summary["y_steered"],
            "steering_effects": torch.tensor(summary["steering_effects"]),
            "layer_indices": torch.tensor(summary["layer_indices"]),
        }
        self.save_tensors(tensors, "tensors", "full_layer")

        self._plot_full_layer(summary)

    def _plot_full_layer(self, summary: Dict):
        """Create steering plot for full layer."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        layer_indices = summary["layer_indices"]
        steering_effects = summary["steering_effects"]
        y_normal = summary["y_normal"]

        ax1.plot(layer_indices, steering_effects, "o-", linewidth=2, markersize=8)
        ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Steering Effect")
        ax1.set_title("Full Layer Steering: Effect by Layer")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            layer_indices,
            [e / (abs(y_normal) + 1e-8) * 100 for e in steering_effects],
            "o-",
            linewidth=2,
            markersize=8,
            color="purple",
        )
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Effect %")
        ax2.set_title("Full Layer Steering: Effect Ratio %")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / f"full_layer_steering_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved plot: {save_path}")

    def _plot_single_head(self, summary: Dict, head_idx: int):
        """Create steering plot for a single head."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        layer_indices = summary["layer_indices"]
        steering_effects = summary["steering_effects"]
        y_normal = summary["y_normal"]

        ax1.plot(layer_indices, steering_effects, "o-", linewidth=2, markersize=8)
        ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Steering Effect")
        ax1.set_title(f"Head {head_idx}: Steering Effect by Layer")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            layer_indices,
            [e / (abs(y_normal) + 1e-8) * 100 for e in steering_effects],
            "o-",
            linewidth=2,
            markersize=8,
            color="purple",
        )
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Effect %")
        ax2.set_title(f"Head {head_idx}: Effect Ratio %")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / f"head_{head_idx}_steering_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.created_images.append(save_path)
        print(f"  Saved plot: {save_path}")

    def _plot_single_token(self, summary: Dict, token_idx: int):
        """Create steering plot for a single token."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        layer_indices = summary["layer_indices"]
        steering_effects = summary["steering_effects"]
        y_normal = summary["y_normal"]

        ax1.plot(layer_indices, steering_effects, "o-", linewidth=2, markersize=8)
        ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Steering Effect")
        ax1.set_title(f"Token {token_idx}: Steering Effect by Layer")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            layer_indices,
            [e / (abs(y_normal) + 1e-8) * 100 for e in steering_effects],
            "o-",
            linewidth=2,
            markersize=8,
            color="purple",
        )
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Effect %")
        ax2.set_title(f"Token {token_idx}: Effect Ratio %")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / f"token_{token_idx}_steering_{self.timestamp}.png"
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
            steering_effects = summary["steering_effects"]
            y_normal = summary["y_normal"]

            ax1.plot(
                layer_indices,
                steering_effects,
                "o-",
                linewidth=2,
                markersize=6,
                label=label,
                color=colors[i % len(colors)],
            )
            ax2.plot(
                layer_indices,
                [e / (abs(y_normal) + 1e-8) * 100 for e in steering_effects],
                "o-",
                linewidth=2,
                markersize=6,
                label=label,
                color=colors[i % len(colors)],
            )

        ax1.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Steering Effect")
        ax1.set_title("Steering Effect Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Effect %")
        ax2.set_title("Steering Effect Ratio Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / f"comparison_steering_{self.timestamp}.png"
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
