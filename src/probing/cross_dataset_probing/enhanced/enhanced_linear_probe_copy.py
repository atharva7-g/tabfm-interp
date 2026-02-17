import torch
import numpy as np
import random
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch.nn as nn
from typing import List, Tuple, Dict


def set_seed(seed):
    """Set random seeds for reproducibility across all sources of randomness"""
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)  # All GPU devices
    # Set deterministic behavior for PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dummy_dataset(
    weights: List[float], num_samples: int = 1000, bias: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a dummy dataset with given weights"""
    X = np.random.randn(num_samples, len(weights))
    y = X @ weights
    if bias:
        y += np.random.randn(num_samples)
    return X, y


def generate_multiple_datasets(
    num_datasets: int = 10, samples_per_dataset: int = 1000
) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[float]]]:
    """Generate multiple datasets with different weight combinations"""
    datasets_X = []
    datasets_y = []
    weight_combinations = []

    # Set random seed for reproducible random generation
    np.random.seed(42)

    for i in range(num_datasets):
        # Random alpha and beta values between [1, 10] with equal probability
        alpha = np.random.randint(1, 10)  # [1, 10] inclusive
        beta = np.random.randint(1, 10)  # [1, 10] inclusive
        weights = [alpha, beta]
        X, y = create_dummy_dataset(weights, samples_per_dataset)
        datasets_X.append(X)
        datasets_y.append(y)
        weight_combinations.append(weights)

    return datasets_X, datasets_y, weight_combinations


class LinearProbe(nn.Module):
    """1-layer neural network probe for activation analysis"""

    def __init__(self, input_dim: int, output_dim: int = 2, hidden_dim: int = 256):
        super(LinearProbe, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def extract_activations(
    regressor, model, X_data: np.ndarray, device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Extract activations from all transformer layers"""
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output[0].detach()[-len(X_data) :]
            # print(f"Layer {name} activations shape: {activations[name].shape}")

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


def train_linear_probe(
    activations: torch.Tensor,
    targets: torch.Tensor,
    device: str = "cuda",
    epochs: int = 100,
    lr: float = 0.001,
) -> Tuple[LinearProbe, float, float]:
    """Train a linear probe on given activations"""
    # Flatten activations
    activation_size = activations.shape[1] * activations.shape[2]
    activation_flat = activations.view(-1, activation_size)

    # Split data
    activation_train, activation_test, target_train, target_test = train_test_split(
        activation_flat, targets, test_size=0.5, random_state=42
    )

    # Move to device
    activation_train = activation_train.to(device).float()
    activation_test = activation_test.to(device).float()
    target_train = target_train.to(device).float()
    target_test = target_test.to(device).float()

    # Initialize probe
    probe = LinearProbe(input_dim=activation_size, output_dim=2, hidden_dim=256).to(
        device
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = probe(activation_train)
        loss = criterion(outputs, target_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    with torch.no_grad():
        test_outputs = probe(activation_test)
        test_loss = criterion(test_outputs, target_test)
        # Calculate R2 for each coefficient separately and average
        r2_alpha = r2_score(
            target_test[:, 0].cpu().numpy(), test_outputs[:, 0].cpu().numpy()
        )
        r2_beta = r2_score(
            target_test[:, 1].cpu().numpy(), test_outputs[:, 1].cpu().numpy()
        )
        r2 = (r2_alpha + r2_beta) / 2

    return probe, test_loss.item(), r2


def test_single_layer(
    activations_data, train_indices, test_indices, layer_name, device="cuda"
):
    """Test a single layer and return performance metrics"""
    print(f"\nTesting layer: {layer_name}")

    # Combine activations from training datasets
    layer_activations_list = []
    layer_targets_list = []

    for dataset_idx in train_indices:
        layer_activations = activations_data[dataset_idx]["activations"][layer_name]

        # Get the original weights and data
        alpha_value = activations_data[dataset_idx]["weights"][0]
        beta_value = activations_data[dataset_idx]["weights"][1]
        X_test = activations_data[dataset_idx]["X_test"]
        y_test = activations_data[dataset_idx]["y_test"]
        # Use training distribution for normalization
        X_train = activations_data[dataset_idx]["X_train"]
        y_train = activations_data[dataset_idx]["y_train"]

        # Calculate standard deviations
        std_x1 = np.std(X_train[:, 0])  # std_dev of x1 (from train)
        std_x2 = np.std(X_train[:, 1])  # std_dev of x2 (from train)
        std_z = np.std(y_train)  # std_dev of z (from train)

        # Calculate new targets: alpha*std_dev(x1)/std_dev(z) and beta*std_dev(x2)/std_dev(z)
        target_alpha = alpha_value * std_x1 / std_z
        target_beta = beta_value * std_x2 / std_z

        # Create target array with same length as activations, with 2D targets
        layer_targets = np.full(
            (layer_activations.shape[0], 2), [target_alpha, target_beta]
        )

        layer_activations_list.append(layer_activations)
        layer_targets_list.append(layer_targets)

    # Concatenate activations and targets
    combined_activations = torch.cat(layer_activations_list, dim=0)
    combined_targets = np.concatenate(layer_targets_list, axis=0)

    # Calculate activation size
    activation_size = combined_activations.shape[1] * combined_activations.shape[2]

    # Train linear probe
    targets = torch.tensor(combined_targets, dtype=torch.float32)
    probe, train_loss, train_r2 = train_linear_probe(
        combined_activations, targets, device, epochs=100, lr=0.001
    )

    # Test on remaining datasets
    test_losses = []
    test_r2s = []

    for test_idx in test_indices:
        # Get pre-extracted activations
        layer_activations = activations_data[test_idx]["activations"][layer_name]

        # Get the original weights and data
        alpha_value = activations_data[test_idx]["weights"][0]
        beta_value = activations_data[test_idx]["weights"][1]
        X_test = activations_data[test_idx]["X_test"]
        y_test = activations_data[test_idx]["y_test"]
        # Use training distribution for normalization
        X_train = activations_data[test_idx]["X_train"]
        y_train = activations_data[test_idx]["y_train"]

        # Calculate standard deviations
        std_x1 = np.std(X_train[:, 0])  # std_dev of x1 (from train)
        std_x2 = np.std(X_train[:, 1])  # std_dev of x2 (from train)
        std_z = np.std(y_train)  # std_dev of z (from train)

        # Calculate new targets: alpha*std_dev(x1)/std_dev(z) and beta*std_dev(x2)/std_dev(z)
        target_alpha = alpha_value * std_x1 / std_z
        target_beta = beta_value * std_x2 / std_z

        # Create target array with same length as activations, with 2D targets
        target_values = np.full(
            (layer_activations.shape[0], 2), [target_alpha, target_beta]
        )

        # Flatten activations
        activation_flat = layer_activations.view(-1, activation_size).to(device).float()

        # Make predictions
        with torch.no_grad():
            predictions = probe(activation_flat)

        # Calculate metrics
        test_loss = nn.MSELoss()(
            predictions, torch.tensor(target_values, dtype=torch.float32).to(device)
        )
        # Calculate R2 for each coefficient separately
        r2_alpha = r2_score(target_values[:, 0], predictions[:, 0].cpu().numpy())
        r2_beta = r2_score(target_values[:, 1], predictions[:, 1].cpu().numpy())
        test_r2 = (r2_alpha + r2_beta) / 2

        test_losses.append(test_loss.item())
        test_r2s.append(test_r2)

    # Calculate average performance across test datasets
    avg_test_loss = np.mean(test_losses)
    avg_test_r2 = np.mean(test_r2s)

    print(f"  Training Loss: {train_loss:.4f}, Training R2: {train_r2:.4f}")
    print(
        f"  Average Test Loss: {avg_test_loss:.4f}, Average Test R2: {avg_test_r2:.4f}"
    )

    return {
        "layer": layer_name,
        "train_loss": train_loss,
        "train_r2": train_r2,
        "avg_test_loss": avg_test_loss,
        "avg_test_r2": avg_test_r2,
        "test_losses": test_losses,
        "test_r2s": test_r2s,
    }


def main():
    # Set random seed
    set_seed(42)

    # Configuration
    num_datasets = 25
    samples_per_dataset = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print("Testing ALL layers of TabPFN to find the best performing layer")

    # Generate multiple datasets
    print("Generating datasets...")
    datasets_X, datasets_y, weight_combinations = generate_multiple_datasets(
        num_datasets, samples_per_dataset
    )

    # Randomly sample 8 datasets for training, 2 for testing
    np.random.seed(42)  # For reproducible random sampling
    all_indices = list(range(num_datasets))
    train_indices = np.random.choice(all_indices, size=8, replace=False)
    test_indices = [idx for idx in all_indices if idx not in train_indices]

    print(
        f"Randomly selected training datasets: {[weight_combinations[i] for i in train_indices]}"
    )
    print(f"Testing datasets: {[weight_combinations[i] for i in test_indices]}")

    # Extract activations from all datasets at once
    print("\nExtracting activations from all datasets...")
    all_activations = {}  # Will store activations from all datasets

    for i in range(num_datasets):
        print(f"Processing dataset {i + 1} with weights {weight_combinations[i]}...")

        X_data = datasets_X[i]
        y_data = datasets_y[i]
        # Split dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.5, random_state=42
        )

        set_seed(42)

        # Train TabPFN on training split
        regressor = TabPFNRegressor(device=device, n_estimators=1)
        regressor.fit(X_train, y_train)

        # set_seed(42)

        # Evaluate TabPFN performance on test split
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"  MSE: {mse:.4f}, R²: {r2:.4f}")

        # Extract activations from test split
        activations = extract_activations(regressor, regressor.model_, X_test, device)

        # Store activations for this dataset
        all_activations[i] = {
            "activations": activations,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "weights": weight_combinations[i],
        }

    # Get available layers from the first dataset
    available_layers = list(all_activations[0]["activations"].keys())
    print(f"\nAvailable layers: {available_layers}")
    print(f"Total layers to test: {len(available_layers)}")

    # Test all layers
    print("\n" + "=" * 60)
    print("TESTING ALL LAYERS")
    print("=" * 60)

    layer_results = []

    for layer_name in available_layers:
        try:
            result = test_single_layer(
                all_activations, train_indices, test_indices, layer_name, device
            )
            layer_results.append(result)
        except Exception as e:
            print(f"Error testing layer {layer_name}: {e}")
            continue

    # Find best performing layer
    if layer_results:
        best_layer_by_r2 = max(layer_results, key=lambda x: x["avg_test_r2"])
        best_layer_by_loss = min(layer_results, key=lambda x: x["avg_test_loss"])

        print("\n" + "=" * 60)
        print("LAYER PERFORMANCE SUMMARY")
        print("=" * 60)

        # Sort by R2 score
        layer_results_sorted = sorted(
            layer_results, key=lambda x: x["avg_test_r2"], reverse=True
        )

        print("\nAll layers ranked by Average Test R² Score:")
        print("-" * 60)
        print(f"{'Layer':<12} {'Train R²':<10} {'Test R²':<10} {'Test Loss':<12}")
        print("-" * 60)

        for result in layer_results_sorted:
            print(
                f"{result['layer']:<12} {result['train_r2']:<10.4f} {result['avg_test_r2']:<10.4f} {result['avg_test_loss']:<12.4f}"
            )

        print("\n" + "=" * 60)
        print("BEST PERFORMING LAYER")
        print("=" * 60)
        print(f"Best layer by R² Score: {best_layer_by_r2['layer']}")
        print(f"  Training R²: {best_layer_by_r2['train_r2']:.4f}")
        print(f"  Average Test R²: {best_layer_by_r2['avg_test_r2']:.4f}")
        print(f"  Average Test Loss: {best_layer_by_r2['avg_test_loss']:.4f}")
        print(
            f"  Individual Test R² scores: {[f'{r:.4f}' for r in best_layer_by_r2['test_r2s']]}"
        )

        print(f"\nBest layer by Loss: {best_layer_by_loss['layer']}")
        print(f"  Training Loss: {best_layer_by_loss['train_loss']:.4f}")
        print(f"  Average Test Loss: {best_layer_by_loss['avg_test_loss']:.4f}")
        print(f"  Average Test R²: {best_layer_by_loss['avg_test_r2']:.4f}")

        # Detailed analysis of best layer
        print("\n" + "=" * 60)
        print(f"DETAILED ANALYSIS OF BEST LAYER: {best_layer_by_r2['layer']}")
        print("=" * 60)

        for i, test_idx in enumerate(test_indices):
            print(f"\nTest Dataset {i + 1} (weights {weight_combinations[test_idx]}):")
            print(f"  R² Score: {best_layer_by_r2['test_r2s'][i]:.4f}")
            print(f"  Loss: {best_layer_by_r2['test_losses'][i]:.4f}")

    else:
        print("No layers were successfully tested!")


if __name__ == "__main__":
    main()
