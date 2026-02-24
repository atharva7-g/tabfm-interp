#!/usr/bin/env python3

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

from utils.utils import set_seed
from src.datasets import create_dataset
from src.experiments.hooks.core_patching import (
	create_cache_hook,
	create_corrupted_input,
	create_steer_hook,
)


def get_layer_activation(
		regressor: TabPFNRegressor,
		X: np.ndarray,
		layer_idx: int,
) -> torch.Tensor:
	model = regressor.model_
	layer = model.transformer_encoder.layers[layer_idx]
	attention_module = layer.self_attn_between_features

	activations = {}
	cache_handle = attention_module.register_forward_hook(
		create_cache_hook(activations, "cache")
	)

	with torch.no_grad():
		regressor.predict(X)

	cache_handle.remove()

	return activations["cache"]


def extract_direction_from_corruption(
		regressor: TabPFNRegressor,
		X_clean: np.ndarray,
		X_corrupt: np.ndarray,
		layer_idx: int,
		normalize: bool = True,
) -> torch.Tensor:
	activation_clean = get_layer_activation(regressor, X_clean, layer_idx)
	activation_corrupt = get_layer_activation(regressor, X_corrupt, layer_idx)

	direction = activation_clean - activation_corrupt

	if normalize:
		direction = direction / (direction.norm() + 1e-8)

	return direction


def run_steering_test(
		regressor: TabPFNRegressor,
		X: np.ndarray,
		layer_idx: int,
		direction: torch.Tensor,
		alpha_values: List[float],
		steer_indices: int = 0,
		steer_dim: int = 2,
) -> List[Dict[str, float]]:
	model = regressor.model_
	layer = model.transformer_encoder.layers[layer_idx]
	attention_module = layer.self_attn_between_features

	with torch.no_grad():
		y_normal = regressor.predict(X)
	y_normal_val = float(y_normal[0]) if len(y_normal.shape) > 0 else float(y_normal)

	results = []
	for alpha in alpha_values:
		steer_hook = create_steer_hook(
			direction=direction,
			steer_indices=steer_indices,
			steer_dim=steer_dim,
			alpha=alpha,
		)
		handle = attention_module.register_forward_hook(steer_hook)

		with torch.no_grad():
			y_steered = regressor.predict(X)

		handle.remove()

		y_steered_val = (
			float(y_steered[0]) if len(y_steered.shape) > 0 else float(y_steered)
		)
		effect = y_steered_val - y_normal_val

		results.append(
			{
				"alpha": alpha,
				"y_normal": y_normal_val,
				"y_steered": y_steered_val,
				"effect": effect,
			}
		)

	return results


def run_scaling_test(
		regressor: TabPFNRegressor,
		X_clean: np.ndarray,
		X_corrupt: np.ndarray,
		layer_idx: int,
		alpha_values: List[float],
) -> List[Dict[str, float]]:
	direction = extract_direction_from_corruption(
		regressor, X_clean, X_corrupt, layer_idx, normalize=True
	)

	results = run_steering_test(regressor, X_clean, layer_idx, direction, alpha_values)

	return results


def run_subtraction_test(
		regressor: TabPFNRegressor,
		X_clean: np.ndarray,
		X_corrupt: np.ndarray,
		layer_idx: int,
) -> Dict[str, float]:
	direction = extract_direction_from_corruption(
		regressor, X_clean, X_corrupt, layer_idx, normalize=True
	)

	model = regressor.model_
	layer = model.transformer_encoder.layers[layer_idx]
	attention_module = layer.self_attn_between_features

	with torch.no_grad():
		y_clean = regressor.predict(X_clean)
	y_clean_val = float(y_clean[0]) if len(y_clean.shape) > 0 else float(y_clean)

	steer_hook = create_steer_hook(
		direction=direction,
		steer_indices=0,
		steer_dim=2,
		alpha=-1.0,
	)
	handle = attention_module.register_forward_hook(steer_hook)

	with torch.no_grad():
		y_subtracted = regressor.predict(X_clean)

	handle.remove()

	y_subtracted_val = (
		float(y_subtracted[0]) if len(y_subtracted.shape) > 0 else float(y_subtracted)
	)

	return {
		"y_clean": y_clean_val,
		"y_with_subtracted_direction": y_subtracted_val,
		"effect": y_subtracted_val - y_clean_val,
	}


def run_replacement_test(
		regressor: TabPFNRegressor,
		X_sample_a: np.ndarray,
		X_sample_b: np.ndarray,
		layer_idx: int,
) -> Dict[str, float]:
	X_corrupt_a = create_corrupted_input(
		X_sample_a, corrupt_idx=0, noise_std=1.0, seed=42
	)

	direction = extract_direction_from_corruption(
		regressor, X_sample_a, X_corrupt_a, layer_idx, normalize=True
	)

	model = regressor.model_
	layer = model.transformer_encoder.layers[layer_idx]
	attention_module = layer.self_attn_between_features

	with torch.no_grad():
		y_b_normal = regressor.predict(X_sample_b)
	y_b_normal_val = (
		float(y_b_normal[0]) if len(y_b_normal.shape) > 0 else float(y_b_normal)
	)

	steer_hook = create_steer_hook(
		direction=direction,
		steer_indices=0,
		steer_dim=2,
		alpha=1.0,
	)
	handle = attention_module.register_forward_hook(steer_hook)

	with torch.no_grad():
		y_b_with_a_direction = regressor.predict(X_sample_b)

	handle.remove()

	y_b_with_a_direction_val = (
		float(y_b_with_a_direction[0])
		if len(y_b_with_a_direction.shape) > 0
		else float(y_b_with_a_direction)
	)

	return {
		"y_b_normal": y_b_normal_val,
		"y_b_with_a_direction": y_b_with_a_direction_val,
		"effect": y_b_with_a_direction_val - y_b_normal_val,
	}


def test_direction_geometry_at_layer(
		regressor: TabPFNRegressor,
		X_clean: np.ndarray,
		X_corrupt: np.ndarray,
		layer_idx: int,
		alpha_values=None,
) -> Dict:
	if alpha_values is None:
		alpha_values = [0.25, 0.5, 1.0, 2.0, 4.0]
	print(f"\n{'=' * 60}")
	print(f"Testing layer {layer_idx}")
	print(f"{'=' * 60}")

	scaling_results = run_scaling_test(
		regressor, X_clean, X_corrupt, layer_idx, alpha_values
	)

	print("\nScaling test:")
	print(f"{'Alpha':<10} {'Effect':<15} {'Effect/Alpha':<15}")
	print("-" * 40)
	for r in scaling_results:
		ratio = r["effect"] / r["alpha"] if r["alpha"] != 0 else 0
		print(f"{r['alpha']:<10.2f} {r['effect']:<15.6f} {ratio:<15.6f}")

	subtraction_result = run_subtraction_test(regressor, X_clean, X_corrupt, layer_idx)

	print("\nSubtraction test (alpha=-1):")
	print(f"  y_clean: {subtraction_result['y_clean']:.6f}")
	print(
		f"  y_with_subtracted: {subtraction_result['y_with_subtracted_direction']:.6f}"
	)
	print(f"  effect: {subtraction_result['effect']:.6f}")

	return {
		"layer_idx": layer_idx,
		"scaling_results": scaling_results,
		"subtraction_result": subtraction_result,
	}


def sweep_all_layers(
		regressor: TabPFNRegressor,
		X_clean: np.ndarray,
		X_corrupt: np.ndarray,
		max_layers: int = None,
		alpha_values=None,
) -> List[Dict]:
	if alpha_values is None:
		alpha_values = [0.25, 0.5, 1.0, 2.0, 4.0]
	model = regressor.model_
	total_layers = len(model.transformer_encoder.layers)
	num_layers = max_layers if max_layers is not None else total_layers
	num_layers = min(num_layers, total_layers)

	all_results = []

	for layer_idx in range(num_layers):
		print(f"\nProcessing layer {layer_idx}/{num_layers - 1}...")
		try:
			result = test_direction_geometry_at_layer(
				regressor, X_clean, X_corrupt, layer_idx, alpha_values
			)
			all_results.append(result)
		except Exception as e:
			print(f"  Error at layer {layer_idx}: {e}")
			import traceback

			traceback.print_exc()
			continue

	return all_results


def analyze_linearity(results: List[Dict]) -> Dict:
	print("\n" + "=" * 60)
	print("LINEARITY ANALYSIS")
	print("=" * 60)

	for result in results:
		layer_idx = result["layer_idx"]
		scaling = result["scaling_results"]

		effects = [r["effect"] for r in scaling]
		alphas = [r["alpha"] for r in scaling]

		mean_ratio = np.mean([e / a for e, a in zip(effects, alphas) if a != 0])
		std_ratio = np.std([e / a for e, a in zip(effects, alphas) if a != 0])

		print(f"\nLayer {layer_idx}:")
		print(f"  Mean effect/alpha: {mean_ratio:.6f}")
		print(f"  Std effect/alpha: {std_ratio:.6f}")
		print(
			f"  Linear? {('YES' if std_ratio / (abs(mean_ratio) + 1e-8) < 0.5 else 'NO')}"
		)

	return {}


def main():
	set_seed(42)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")

	print("Creating multiplication dataset...")
	X, y = create_dataset("multiplication", num_samples=100, seed=42)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	X_clean = X_test[0:1]
	X_corrupt = create_corrupted_input(X_clean, corrupt_idx=0, noise_std=1.0, seed=42)

	print(f"Clean input: {X_clean}")
	print(f"Corrupt input: {X_corrupt}")

	print("\nLoading TabPFN model...")
	regressor = TabPFNRegressor(device=device, n_estimators=1)
	regressor.fit(X_train, y_train)

	with torch.no_grad():
		y_clean_pred = regressor.predict(X_clean)
		y_corrupt_pred = regressor.predict(X_corrupt)

	print(f"Clean prediction: {y_clean_pred[0]:.6f}")
	print(f"Corrupt prediction: {y_corrupt_pred[0]:.6f}")
	print(f"Target (a*b): {X_clean[0, 0] * X_clean[0, 1]:.6f}")

	print("\nRunning direction geometry test across all layers...")
	all_results = sweep_all_layers(
		regressor, X_clean, X_corrupt, alpha_values=[0.25, 0.5, 1.0, 2.0, 4.0]
	)

	analyze_linearity(all_results)

	print("\n" + "=" * 60)
	print("EXPERIMENT COMPLETE")
	print("=" * 60)

	return all_results


if __name__ == "__main__":
	main()
