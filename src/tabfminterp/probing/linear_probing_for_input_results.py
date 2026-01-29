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


def train_model(X: np.ndarray, y: np.ndarray) -> TabPFNRegressor:
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
    probe_targets = {
        'a': X_test[:, 0],
        'b': X_test[:, 1],
        'c': X_test[:, 2],
        'a_plus_c': X_test[:, 0] + X_test[:, 2],
        'a_plus_b': X_test[:, 0] + X_test[:, 1],
    }

    regressor = train_model(X_train, y_train)
    activations = extract_activations(
        regressor, regressor.model_, X_test, device='cuda')

    probe_metrics = {
        name: {'layers': [], 'train_r2': [], 'test_r2': [], 'train_mse': [], 'test_mse': []}
        for name in probe_targets
    }

    for layer, activation in activations.items():
        # taking the activation at the last token (representing the y prediction) only for the probe
        # Convert to float64 explicitly to avoid overflow during conversion
        linear_train = activation[:8000, -1, :].detach().cpu().double().numpy()
        linear_test = activation[8000:, -1, :].detach().cpu().double().numpy()
        # Check for NaN, Inf, or extremely large values
        if np.isnan(linear_train).any() or np.isinf(linear_train).any():
            print(f"Warning: {layer} contains NaN or Inf values in activations")
            continue
        
        # Always normalize to prevent overflow and improve numerical stability
        train_mean = linear_train.mean(axis=0, keepdims=True)
        train_std = linear_train.std(axis=0, keepdims=True) + 1e-8
        linear_train = (linear_train - train_mean) / train_std
        linear_test = (linear_test - train_mean) / train_std

        try:
            layer_idx = int(layer.split('_')[-1])
        except ValueError:
            layer_idx = layer

        for target_name, target_values in probe_targets.items():
            linear_y_train = target_values[:8000]
            linear_y_test = target_values[8000:]
            model = LinearRegression()
            try:
                model.fit(linear_train, linear_y_train)
                train_pred = model.predict(linear_train)
                if np.isnan(train_pred).any() or np.isinf(train_pred).any():
                    print(f"Warning: {layer} - {target_name} predictions contain NaN/Inf, skipping")
                    continue
            except Exception as e:
                print(f"Warning: {layer} - {target_name} failed to fit: {e}")
                continue

            print(f"{layer} - Target: {target_name}")
            print("--------------------------------")
            train_mse = mean_squared_error(linear_y_train, train_pred)
            train_r2 = model.score(linear_train, linear_y_train)
            print(f"Train mse: {train_mse:.4f}")
            print(f"Train r2 score: {train_r2:.4f}")
            
            test_pred = model.predict(linear_test)
            if np.isnan(test_pred).any() or np.isinf(test_pred).any():
                print(f"Warning: {layer} - {target_name} test predictions contain NaN/Inf, skipping test metrics")
                print("--------------------------------")
                continue
            
            test_mse = mean_squared_error(linear_y_test, test_pred)
            test_r2 = r2_score(linear_y_test, test_pred)
            print(f"Test mse: {test_mse:.4f}")
            print(f"Test r2 score: {test_r2:.4f}")
            print("--------------------------------")

            probe_metrics[target_name]['layers'].append(layer_idx)
            probe_metrics[target_name]['train_r2'].append(train_r2)
            probe_metrics[target_name]['test_r2'].append(test_r2)
            probe_metrics[target_name]['train_mse'].append(train_mse)
            probe_metrics[target_name]['test_mse'].append(test_mse)

    # Original R²-only graph (commented out as requested, kept for reference)
    # plotted = False
    # plt.figure(figsize=(10, 6))
    # for target_name, metrics in probe_metrics.items():
    #     if not metrics['layers']:
    #         continue
    #     plotted = True
    #     sorted_indices = sorted(range(len(metrics['layers'])), key=lambda i: metrics['layers'][i])
    #     sorted_layers = [metrics['layers'][i] for i in sorted_indices]
    #     sorted_test_r2 = [metrics['test_r2'][i] for i in sorted_indices]
    #     plt.plot(sorted_layers, sorted_test_r2, marker='o', label=target_name)
    #
    # if plotted:
    #     plt.xlabel('Layer')
    #     plt.ylabel('Test R² Score')
    #     plt.title('Probe Test R² by Transformer Layer')
    #     plt.legend()
    #     plt.grid(True, linestyle='--', alpha=0.5)
    #     plt.tight_layout()
    #     output_path = 'input_probe_results_r2.png'
    #     plt.savefig(output_path)
    #     plt.close()
    #     print(f"Saved combined R² plot to {output_path}")
    # else:
    #     plt.close()
    #     print("No valid probe metrics were collected; skipping R² plot.")

    # R² graph with formatting
    axis_label_fontsize = 20
    tick_label_fontsize = 16
    legend_fontsize = 16

    # Mapping from internal names to display names with math notation
    display_names = {
        'a': 'a',
        'b': 'b',
        'c': 'c',
        'a_plus_c': 'a+c',
        'a_plus_b': 'a+b',
    }

    plotted = False
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette and markers for different targets
    colors_r2 = ['tab:blue', 'tab:cyan', 'tab:green', 'tab:purple', 'tab:pink']
    markers = ['o', 's', '^', 'D', 'v']

    color_idx = 0
    all_lines = []
    all_labels = []

    for target_name, metrics in probe_metrics.items():
        if not metrics['layers']:
            continue
        plotted = True
        sorted_indices = sorted(range(len(metrics['layers'])), key=lambda i: metrics['layers'][i])
        sorted_layers = [metrics['layers'][i] for i in sorted_indices]
        sorted_test_r2 = [metrics['test_r2'][i] for i in sorted_indices]

        display_name = display_names.get(target_name, target_name)

        # Plot R²
        r2_line = ax.plot(
            sorted_layers,
            sorted_test_r2,
            marker=markers[color_idx % len(markers)],
            color=colors_r2[color_idx % len(colors_r2)],
            label=display_name,
            linewidth=2,
            markersize=6,
        )
        all_lines.extend(r2_line)
        all_labels.append(display_name)

        color_idx += 1

    if plotted:
        ax.set_xlabel('Layer', fontsize=axis_label_fontsize)
        ax.set_ylabel('R² Score', fontsize=axis_label_fontsize)

        ax.tick_params(axis='both', labelsize=tick_label_fontsize, colors='black')

        # Move legend below x-axis
        ax.legend(all_lines, all_labels, fontsize=legend_fontsize, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=True)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        output_path = 'input_probe_results_r2_mse.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved R² plot to {output_path}")
    else:
        plt.close()
        print("No valid probe metrics were collected; skipping R² plot.")


if __name__ == "__main__":
    main()


