import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch.nn as nn
from typing import List, Tuple, Dict
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# def create_dummy_dataset(weights: List[float], num_samples: int = 1000, bias: bool = False) -> Tuple[np.ndarray, np.ndarray]:
#     """Create a dummy dataset with given weights"""
#     X = np.random.randn(num_samples, len(weights))
#     y = X @ weights
#     if bias:
#         y += np.random.randn(num_samples)
#     return X, y

def create_dummy_dataset(num_samples: int = 1000, bias: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a dummy dataset where each input X has 4 values: a, b, c, d, 
    and the target y = a * b + c
    """
    X = np.random.randn(num_samples, 3)
    a = X[:, 0]
    b = X[:, 1]
    c = X[:, 2]
    y = a * b + c
    if bias:
        y += np.random.randn(num_samples)
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> Tuple[TabPFNRegressor, float, float]:
    """Train a TabPFN model on the given dataset"""
    regressor = TabPFNRegressor(device='cuda', n_estimators=1)
    regressor.fit(X, y)
    return regressor


def extract_activations(regressor, model, X_data: np.ndarray, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Extract activations from all transformer layers"""
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output[0].detach()[-len(X_data):]
            print(f"Layer {name} activations shape: {activations[name].shape}")
        return hook

    # Register hooks for all transformer layers
    hook_handles = []
    for i, layer in enumerate(model.transformer_encoder.layers):
        handle = layer.register_forward_hook(get_activation(f'layer_{i}'))
        hook_handles.append(handle)

    # Forward pass to extract activations
    with torch.no_grad():
        _ = regressor.predict(X_data)

    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    return activations


def main():
    """Main function"""
    set_seed(42)
    # X_train, y_train = create_dummy_dataset(
    #     weights=[10, 13], num_samples=1000, bias=False)
    # X_test, y_test = create_dummy_dataset(
    #     weights=[1, 2], num_samples=10000, bias=False)

    X_train, y_train = create_dummy_dataset(num_samples=1000, bias=False)
    X_test, y_test = create_dummy_dataset(num_samples=10000, bias=False)

    regressor = train_model(X_train, y_train)
    activations = extract_activations(
        regressor, regressor.model_, X_test, device='cuda')

    layer_ids = []
    train_r2_scores = []
    test_r2_scores = []
    train_mse_scores = []
    test_mse_scores = []

    for layer, activation in activations.items():
        # taking the activation at the last token (representing the y prediction) only for the probe
        # Convert to float64 explicitly to avoid overflow during conversion
        linear_train = activation[:8000, -1, :].detach().cpu().double().numpy()
        linear_test = activation[8000:, -1, :].detach().cpu().double().numpy()
        # probe target is a + c from X_test (keep TabPFN training target unchanged)
        probe_target = X_test[:, 0]
        linear_y_train = probe_target[:8000]
        linear_y_test = probe_target[8000:]

        # Check for NaN, Inf, or extremely large values
        if np.isnan(linear_train).any() or np.isinf(linear_train).any():
            print(f"Warning: {layer} contains NaN or Inf values in activations")
            continue
        
        # Always normalize to prevent overflow and improve numerical stability
        train_mean = linear_train.mean(axis=0, keepdims=True)
        train_std = linear_train.std(axis=0, keepdims=True) + 1e-8
        linear_train = (linear_train - train_mean) / train_std
        linear_test = (linear_test - train_mean) / train_std

        model = LinearRegression()
        try:
            model.fit(linear_train, linear_y_train)
            
            # Check predictions for NaN
            train_pred = model.predict(linear_train)
            if np.isnan(train_pred).any() or np.isinf(train_pred).any():
                print(f"Warning: {layer} predictions contain NaN/Inf, skipping")
                continue
        except Exception as e:
            print(f"Warning: {layer} failed to fit: {e}")
            continue

        print(f"Layer_{layer}_Stats")
        print("--------------------------------")
        train_mse = mean_squared_error(linear_y_train, train_pred)
        train_r2 = model.score(linear_train, linear_y_train)
        print(f"Layer {layer} train mse: {train_mse:.4f}")
        print(f"Layer {layer} train r2 score: {train_r2:.4f}")
        
        test_pred = model.predict(linear_test)
        if np.isnan(test_pred).any() or np.isinf(test_pred).any():
            print(f"Warning: {layer} test predictions contain NaN/Inf, skipping test metrics")
            print("--------------------------------")
            continue
        
        test_mse = mean_squared_error(linear_y_test, test_pred)
        test_r2 = r2_score(linear_y_test, test_pred)
        print(f"Layer {layer} test mse: {test_mse:.4f}")
        print(f"Layer {layer} test r2 score: {test_r2:.4f}")
        print("--------------------------------")

        # Track metrics for plotting
        try:
            layer_idx = int(layer.split('_')[-1])
        except ValueError:
            layer_idx = layer
        layer_ids.append(layer_idx)
        train_r2_scores.append(train_r2)
        test_r2_scores.append(test_r2)
        train_mse_scores.append(train_mse)
        test_mse_scores.append(test_mse)

    if layer_ids:
        # Ensure plotting order follows layer index if numeric
        sorted_indices = sorted(range(len(layer_ids)), key=lambda i: layer_ids[i])
        sorted_layers = [layer_ids[i] for i in sorted_indices]
        sorted_train_r2 = [train_r2_scores[i] for i in sorted_indices]
        sorted_test_r2 = [test_r2_scores[i] for i in sorted_indices]
        sorted_train_mse = [train_mse_scores[i] for i in sorted_indices]
        sorted_test_mse = [test_mse_scores[i] for i in sorted_indices]

        # Font size settings
        axis_label_fontsize = 20
        tick_label_fontsize = 16
        legend_fontsize = 16

        # Create and save combined R² + MSE graph with dual y-axes
        print("\nGenerating combined R² + MSE graph...")
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        r2_train_line = ax1.plot(sorted_layers, sorted_train_r2, 'o-', color='tab:blue', label='Train R²', linewidth=2, markersize=6)
        r2_test_line = ax1.plot(sorted_layers, sorted_test_r2, 's-', color='tab:cyan', label='Test R²', linewidth=2, markersize=6)

        mse_train_line = ax2.plot(sorted_layers, sorted_train_mse, 'o--', color='tab:red', label='Train MSE', linewidth=2, markersize=6)
        mse_test_line = ax2.plot(sorted_layers, sorted_test_mse, 's--', color='tab:orange', label='Test MSE', linewidth=2, markersize=6)

        ax1.set_xlabel('Layer', fontsize=axis_label_fontsize)
        ax1.set_ylabel('R² Score', fontsize=axis_label_fontsize, color='tab:blue')
        ax2.set_ylabel('MSE', fontsize=axis_label_fontsize, color='tab:red')

        ax1.tick_params(axis='both', labelsize=tick_label_fontsize, colors='black')
        ax2.tick_params(axis='both', labelsize=tick_label_fontsize, colors='black')

        lines = r2_train_line + r2_test_line + mse_train_line + mse_test_line
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, fontsize=legend_fontsize, loc='best')

        ax1.grid(True, alpha=0.3)
        fig.tight_layout()

        output_path = 'answer_probe_results_r2_mse.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined R²+MSE plot to {output_path}")
        plt.close()
    else:
        print("No valid layer metrics were collected; skipping R² plot.")


if __name__ == "__main__":
    main()

