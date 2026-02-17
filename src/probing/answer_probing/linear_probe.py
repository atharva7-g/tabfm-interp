import numpy as np
import torch
from tabpfn import TabPFNRegressor
from typing import Dict
from src.utils.utils import set_seed


def train_model(X: np.ndarray, y: np.ndarray, seed: int = 42) -> TabPFNRegressor:
    """Train a TabPFN model on the given dataset"""
    set_seed(seed)  # Reset seed every time a model is trained
    regressor = TabPFNRegressor(device="cuda", n_estimators=1)
    regressor.fit(X, y)
    return regressor


def extract_activations(
    regressor, model, X_data: np.ndarray, device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Extract activations from all transformer layers"""
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output[0].detach()[-len(X_data) :]
            print(f"Layer {name} activations shape: {activations[name].shape}")

        return hook

    # Register hooks for all transformer layers
    hook_handles = []
    for i, layer in enumerate(model.transformer_encoder.layers):
        handle = layer.register_forward_hook(get_activation(f"layer_{i}"))
        hook_handles.append(handle)

    # Forward pass to extract activations
    with torch.no_grad():
        _ = regressor.predict(X_data)

    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    return activations
