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

def create_dummy_dataset(weights: List[float], num_samples: int = 1000, bias: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Create a dummy dataset with given weights"""
    X = np.random.randn(num_samples, len(weights))
    y = X @ weights
    if bias:
        y += np.random.randn(num_samples)
    return X, y

def generate_multiple_datasets(num_datasets: int = 10, samples_per_dataset: int = 1000) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[float]]]:
    """Generate multiple datasets with different weight combinations"""
    datasets_X = []
    datasets_y = []
    weight_combinations = []
    
    for i in range(1, num_datasets + 1):
        weights = [i, i]  # [1,1], [2,2], ..., [10,10]
        X, y = create_dummy_dataset(weights, samples_per_dataset)
        datasets_X.append(X)
        datasets_y.append(y)
        weight_combinations.append(weights)
    
    return datasets_X, datasets_y, weight_combinations

class LinearProbe(nn.Module):
    """1-layer neural network probe for activation analysis"""
    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 256):
        super(LinearProbe, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def extract_activations(regressor, model, X_data: np.ndarray, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Extract activations from all transformer layers"""
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output[0].detach()[-len(X_data):]
            # print(f"Layer {name} activations shape: {activations[name].shape}")
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

def train_linear_probe(activations: torch.Tensor, targets: np.ndarray, 
                      device: str = 'cuda', epochs: int = 100, lr: float = 0.001) -> Tuple[LinearProbe, float, float]:
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
    probe = LinearProbe(input_dim=activation_size, output_dim=1, hidden_dim=256).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = probe(activation_train).squeeze()
        loss = criterion(outputs, target_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    with torch.no_grad():
        test_outputs = probe(activation_test).squeeze()
        test_loss = criterion(test_outputs, target_test)
        r2 = r2_score(target_test.cpu().numpy(), test_outputs.cpu().numpy())
    
    return probe, test_loss.item(), r2

def main():
    # Set random seed
    set_seed(42)
    
    # Configuration
    num_datasets = 10
    samples_per_dataset = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TARGET_LAYER = 'layer_1'  # Train probe only on this layer
    
    print(f"Using device: {device}")
    print(f"Training probe only on layer: {TARGET_LAYER}")
    
    # Generate multiple datasets
    print("Generating datasets...")
    datasets_X, datasets_y, weight_combinations = generate_multiple_datasets(num_datasets, samples_per_dataset)
    
    # Randomly sample 8 datasets for training, 2 for testing
    np.random.seed(42)  # For reproducible random sampling
    all_indices = list(range(10))
    train_indices = np.random.choice(all_indices, size=8, replace=False)
    test_indices = [idx for idx in all_indices if idx not in train_indices]
    
    print(f"Randomly selected training datasets: {[weight_combinations[i] for i in train_indices]}")
    print(f"Testing datasets: {[weight_combinations[i] for i in test_indices]}")
    
    # Extract activations from all datasets at once
    print("\nExtracting activations from all datasets...")
    all_activations = {}  # Will store activations from all datasets
    
    for i in range(num_datasets):
        print(f"Processing dataset {i+1} with weights {weight_combinations[i]}...")
        
        X_data = datasets_X[i]
        y_data = datasets_y[i]
        # Split dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=42)
        
        # Train TabPFN on training split
        regressor = TabPFNRegressor(device=device, n_estimators=1)
        regressor.fit(X_train, y_train)
        
        # Evaluate TabPFN performance on test split
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"  MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Extract activations from test split
        activations = extract_activations(regressor, regressor.model_, X_test, device)
        
        # Store activations for this dataset
        all_activations[i] = {
            'activations': activations,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'weights': weight_combinations[i]
        }
    
    # Train linear probe on training datasets
    print(f"\nTraining linear probe for {TARGET_LAYER} on training datasets...")
    
    # Combine activations from training datasets
    layer_activations_list = []
    layer_targets_list = []
    
    for dataset_idx in train_indices:
        layer_activations = all_activations[dataset_idx]['activations'][TARGET_LAYER]
        # print(f"  Dataset {dataset_idx+1} activations shape: {layer_activations.shape}")
        # Target is the actual input values (X_test) that correspond to these activations
        # We want to probe whether we can recover the input from the activations
        layer_targets = all_activations[dataset_idx]['X_test'][:, 0]  # Use the actual test inputs
        
        layer_activations_list.append(layer_activations)
        layer_targets_list.append(layer_targets)
    
    # Concatenate activations and targets
    combined_activations = torch.cat(layer_activations_list, dim=0)
    combined_targets = np.concatenate(layer_targets_list)
    
    print(f"Combined activations shape: {combined_activations.shape}")
    print(f"Combined targets shape: {combined_targets.shape}")
    # print(f"Target values: {[all_activations[idx]['X_train'][:, 0] for idx in train_indices]}")
    
    # Calculate activation size once (constant across all datasets)
    activation_size = combined_activations.shape[1] * combined_activations.shape[2]
    print(f"Activation size: {activation_size}")
    
    # Train linear probe
    targets = torch.tensor(combined_targets, dtype=torch.float32)
    probe, train_loss, train_r2 = train_linear_probe(
        combined_activations, targets, device, epochs=100, lr=0.001
    )
    
    print(f"  Training Loss: {train_loss:.4f}, R2 Score: {train_r2:.4f}")
    
    # Test on remaining datasets
    print("\n" + "="*50)
    print("TESTING ON REMAINING DATASETS")
    print("="*50)
    
    for test_idx in test_indices:
        print(f"\nTesting on dataset with weights {weight_combinations[test_idx]}:")
        
        # Get pre-extracted activations
        layer_activations = all_activations[test_idx]['activations'][TARGET_LAYER]
        # Target is the actual input values (X_test) that correspond to these activations
        target_values = all_activations[test_idx]['X_test'][:, 0]
        
        # Flatten activations (using pre-calculated activation_size)
        activation_flat = layer_activations.view(-1, activation_size).to(device).float()
        
        # Make predictions
        with torch.no_grad():
            predictions = probe(activation_flat).squeeze()
        
        # Calculate metrics
        test_loss = nn.MSELoss()(predictions, torch.tensor(target_values, dtype=torch.float32).to(device))
        test_r2 = r2_score(target_values, predictions.cpu().numpy())
        
        print(f"  {TARGET_LAYER}: Loss={test_loss.item():.4f}, R2={test_r2:.4f}")
        print(f"  True weight[0]: {target_values[0]}, Predicted: {predictions[0].item():.4f}")
    
    # Summary of results
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Layer: {TARGET_LAYER}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training R2 Score: {train_r2:.4f}")

if __name__ == "__main__":
    main()
