import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch.nn as nn
from typing import List, Tuple, Dict


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_dummy_dataset(
    weights: List[float], num_samples: int = 1000, bias: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a dummy dataset with given weights"""
    X = np.random.randn(num_samples, len(weights))
    y = X @ weights
    if bias:
        y += np.random.randn(num_samples)
    return X, y


def generate_combined_dataset(
    num_relationships: int = 10, samples_per_relationship: int = 1000
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[int, int]]:
    """Generate one combined dataset where feature 3 controls the linear relationship"""
    all_X = []
    all_y = []
    all_relationship_ids = []

    random_key = {
        i: np.random.randint(0, 1000) for i in range(1, num_relationships + 1)
    }
    print(f"Random key: {random_key}")

    for i in range(1, num_relationships + 1):
        # Generate features 1 and 2 (random)
        X_features = np.random.randn(samples_per_relationship, 2)
        # Feature 3 is the relationship ID
        relationship_id = np.full((samples_per_relationship, 1), random_key[i])
        # Combine all features
        X_combined = np.hstack([X_features, relationship_id])

        # Calculate y using the relationship-specific weights
        weights = [i, np.random.rand()]  # [1,1], [2,2], ..., [10,10]
        y = X_features @ weights

        all_X.append(X_combined)
        all_y.append(y)
        all_relationship_ids.extend([random_key[i]] * samples_per_relationship)

    # Combine all data
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)

    return X_combined, y_combined, all_relationship_ids, random_key


# class LinearProbe(nn.Module):
#     """1-layer neural network probe for activation analysis"""
#     def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 256):
#         super(LinearProbe, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.network(x)


class LinearProbe(nn.Module):
    # linear probe with no hidden layer and no activation
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


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

    # Move to device
    activation_flat = activation_flat.to(device).float()
    targets = targets.to(device).float()

    # Initialize probe
    probe = LinearProbe(input_dim=activation_size, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = probe(activation_flat).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Evaluate on training data (since we don't have separate test data)
    with torch.no_grad():
        train_outputs = probe(activation_flat).squeeze()
        train_loss = criterion(train_outputs, targets)
        r2 = r2_score(targets.cpu().numpy(), train_outputs.cpu().numpy())

    return probe, train_loss.item(), r2


def main():
    # Set random seed
    set_seed(42)

    # Configuration
    num_relationships = 10
    samples_per_relationship = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_LAYER = "layer_10"  # Train probe only on this layer

    print(f"Using device: {device}")
    print(f"Training probe only on layer: {TARGET_LAYER}")

    # Generate combined dataset
    print("Generating combined dataset...")
    X_combined, y_combined, relationship_ids, random_key = generate_combined_dataset(
        num_relationships, samples_per_relationship
    )
    print(f"Combined dataset shape: X={X_combined.shape}, y={y_combined.shape}")
    print(f"Relationship IDs: {set(relationship_ids)}")

    # Split combined dataset into train and test
    X_train, X_test, y_train, y_test, train_relationship_ids, test_relationship_ids = (
        train_test_split(
            X_combined, y_combined, relationship_ids, test_size=0.5, random_state=42
        )
    )
    print(f"Train split for TabPFN: X={X_train.shape}, y={y_train.shape}")
    print(f"Test split for TabPFN: X={X_test.shape}, y={y_test.shape}")

    # Train TabPFN on training split
    print("\nTraining TabPFN on training split...")
    regressor = TabPFNRegressor(device=device, n_estimators=1)
    # print(X_train)
    regressor.fit(X_train, y_train)

    # Evaluate TabPFN performance on test split
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"TabPFN Performance on Test Split - MSE: {mse:.4f}, R²: {r2:.4f}")

    #########################################################
    ## Probe Training
    #########################################################

    # Extract activations from test split
    print("\nExtracting activations from test split...")
    activations = extract_activations(regressor, regressor.model_, X_test, device)

    # Train linear probe using random train/test split
    print(
        f"\nTraining linear probe for {TARGET_LAYER} using random train/test split..."
    )

    # Create mapping from random UIDs to original relationship IDs
    uid_to_original = {random_key[i]: i for i in range(1, num_relationships + 1)}
    print(f"UID to original mapping: {uid_to_original}")

    # Get all activations and convert UIDs to original relationship IDs
    all_activations_at_layer = activations[TARGET_LAYER]
    all_uid_targets = np.array(test_relationship_ids)
    all_targets = np.array([uid_to_original[uid] for uid in all_uid_targets])

    print(f"All activations shape: {all_activations_at_layer.shape}")
    print(f"All targets shape: {all_targets.shape}")
    print(f"All target values (linear weights): {set(all_targets)}")

    # Split activations using train_test_split
    activation_train, activation_test, target_train, target_test = train_test_split(
        all_activations_at_layer, all_targets, test_size=0.2, random_state=42
    )

    print(f"Probe train activations: {activation_train.shape}")
    print(f"Probe test activations: {activation_test.shape}")
    print(f"Probe train targets: {target_train.shape}")
    print(f"Probe test targets: {target_test.shape}")

    # Calculate activation size
    activation_size = activation_train.shape[1] * activation_train.shape[2]
    print(f"Activation size for probe: {activation_size}")

    # Train linear probe
    train_targets_tensor = torch.tensor(target_train, dtype=torch.float32)
    probe, train_loss, train_r2 = train_linear_probe(
        activation_train, train_targets_tensor, device, epochs=100, lr=0.001
    )

    print(f"  Training Loss: {train_loss:.4f}, R2 Score: {train_r2:.4f}")

    # Test the probe
    activation_flat = activation_test.view(-1, activation_size).to(device).float()
    test_targets_tensor = torch.tensor(target_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = probe(activation_flat).squeeze()

    test_loss = nn.MSELoss()(predictions, test_targets_tensor)
    test_r2 = r2_score(target_test, predictions.cpu().numpy())

    print(f"  Test Loss: {test_loss.item():.4f}, R2 Score: {test_r2:.4f}")

    # Summary
    print("\n" + "=" * 50)
    print("PROBE PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Training Loss: {train_loss:.4f}, Training R2: {train_r2:.4f}")
    print(f"Test Loss: {test_loss.item():.4f}, Test R2: {test_r2:.4f}")

    # # Summary of results
    # print("\n" + "="*50)
    # print("SUMMARY")
    # print("="*50)
    # print(f"Layer: {TARGET_LAYER}")
    # print(f"Training Loss: {train_loss:.4f}")
    # print(f"Training R2 Score: {train_r2:.4f}")


if __name__ == "__main__":
    main()
