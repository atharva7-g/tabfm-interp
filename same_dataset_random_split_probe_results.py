import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch.nn as nn
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


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

def generate_combined_dataset(num_relationships: int = 10, samples_per_relationship: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[int, int]]:
    """Generate one combined dataset where feature 3 controls the linear relationship"""
    all_X = []
    all_y = []
    all_relationship_ids = []

    random_key = {i: np.random.randint(0, 1000) for i in range(1, num_relationships + 1)}
    print(f"Random key: {random_key}")

    for i in range(1, num_relationships + 1):
        # Generate features 1 and 2 (random)
        X_features = np.random.randn(samples_per_relationship, 2)
        # Feature 3 is the relationship ID
        relationship_id = np.full((samples_per_relationship, 1), random_key[i])
        # Combine all features
        X_combined = np.hstack([X_features, relationship_id])
        
        # Calculate y using the relationship-specific weights
        weights = [i, i]  # [1,1], [2,2], ..., [10,10]
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

class VariableComplexityProbe(nn.Module):
    """Linear probe with variable number of hidden layers"""
    def __init__(self, input_dim: int, output_dim: int = 1, num_hidden_layers: int = 0, hidden_dim: int = 256):
        super(VariableComplexityProbe, self).__init__()
        layers = []
        
        if num_hidden_layers == 0:
            # No hidden layers - just linear mapping
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # First layer: input -> hidden
            layers.append(nn.Linear(input_dim, hidden_dim))
            # Hidden layers
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            # Output layer: hidden -> output
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
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

def train_linear_probe(activations: torch.Tensor, targets: torch.Tensor, 
                      device: str = 'cuda', epochs: int = 100, lr: float = 0.001) -> Tuple[LinearProbe, float, float]:
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

def train_variable_complexity_probe(activation_train: torch.Tensor, activation_test: torch.Tensor,
                                     target_train: torch.Tensor, target_test: torch.Tensor,
                                     num_hidden_layers: int, device: str = 'cuda', 
                                     epochs: int = 100, lr: float = 0.001, hidden_dim: int = 256) -> Tuple[float, float, float, float]:
    """Train a probe with variable complexity and return train/test R2 and MSE"""
    # Flatten activations
    activation_size = activation_train.shape[1] * activation_train.shape[2]
    activation_train_flat = activation_train.view(-1, activation_size)
    activation_test_flat = activation_test.view(-1, activation_size)
    
    # Move to device
    activation_train_flat = activation_train_flat.to(device).float()
    activation_test_flat = activation_test_flat.to(device).float()
    target_train = target_train.to(device).float()
    target_test = target_test.to(device).float()
    
    # Initialize probe
    probe = VariableComplexityProbe(
        input_dim=activation_size, 
        output_dim=1, 
        num_hidden_layers=num_hidden_layers,
        hidden_dim=hidden_dim
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = probe(activation_train_flat).squeeze()
        loss = criterion(outputs, target_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate on training and test data
    with torch.no_grad():
        train_outputs = probe(activation_train_flat).squeeze()
        train_r2 = r2_score(target_train.cpu().numpy(), train_outputs.cpu().numpy())
        train_mse = mean_squared_error(target_train.cpu().numpy(), train_outputs.cpu().numpy())
        
        test_outputs = probe(activation_test_flat).squeeze()
        test_r2 = r2_score(target_test.cpu().numpy(), test_outputs.cpu().numpy())
        test_mse = mean_squared_error(target_test.cpu().numpy(), test_outputs.cpu().numpy())
    
    return train_r2, test_r2, train_mse, test_mse

def main():
    # Set random seed
    set_seed(42)
    
    # Configuration
    num_relationships = 10
    samples_per_relationship = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Training probes on all layers\n")
    
    # Generate combined dataset
    print("Generating combined dataset...")
    X_combined, y_combined, relationship_ids, random_key = generate_combined_dataset(num_relationships, samples_per_relationship)
    print(f"Combined dataset shape: X={X_combined.shape}, y={y_combined.shape}")
    print(f"Relationship IDs: {set(relationship_ids)}\n")
    
    # Split combined dataset into train and test
    X_train, X_test, y_train, y_test, train_relationship_ids, test_relationship_ids = train_test_split(
        X_combined, y_combined, relationship_ids, test_size=0.5, random_state=42
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
    
    # Create mapping from random UIDs to original relationship IDs
    uid_to_original = {random_key[i]: i for i in range(1, num_relationships + 1)}
    print(f"UID to original mapping: {uid_to_original}")
    
    # Get all targets
    all_uid_targets = np.array(test_relationship_ids)
    all_targets = np.array([uid_to_original[uid] for uid in all_uid_targets])
    
    # Get sorted layer names
    layer_names = sorted(activations.keys(), key=lambda x: int(x.split('_')[1]))
    
    # Store R2 scores for each layer
    layer_numbers = []
    train_r2_scores = []
    test_r2_scores = []
    train_mse_scores = []
    test_mse_scores = []
    
    print(f"\nTraining linear probes on all layers...")
    print(f"Number of layers: {len(layer_names)}\n")
    
    # Train probe on each layer
    for layer_name in layer_names:
        layer_num = int(layer_name.split('_')[1])
        layer_numbers.append(layer_num)
        
        print(f"Processing {layer_name}...")
        
        # Get activations for this layer
        all_activations_at_layer = activations[layer_name]
        # The following code removes the 4th token (index 3) from the sequence dimension of activations,
        # i.e., it keeps all tokens except the one at position 3.
        # This is correct if your intention is to exclude only the 4th token from each sample.
        all_activations_at_layer = torch.cat(
            (all_activations_at_layer[:, :3, :], all_activations_at_layer[:, 4:, :]), dim=1
        )  # use every token except the 4th token (index 3)
        print(f"Activations at layer {layer_name}: {all_activations_at_layer.shape}")
        
        # Split activations using train_test_split
        activation_train, activation_test, probe_target_train, probe_target_test = train_test_split(
            all_activations_at_layer, all_targets, test_size=0.2, random_state=42
        )
        
        # Calculate activation size
        activation_size = activation_train.shape[1] * activation_train.shape[2]
        
        # Train linear probe
        train_targets_tensor = torch.tensor(probe_target_train, dtype=torch.float32)
        probe, train_loss, train_r2 = train_linear_probe(
            activation_train, train_targets_tensor, device, epochs=100, lr=0.001
        )
        
        train_r2_scores.append(train_r2)
        train_mse_scores.append(train_loss)
        
        # Test the probe
        activation_flat = activation_test.view(-1, activation_size).to(device).float()
        test_targets_tensor = torch.tensor(probe_target_test, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            predictions = probe(activation_flat).squeeze()
        
        test_r2 = r2_score(probe_target_test, predictions.cpu().numpy())
        test_r2_scores.append(test_r2)
        test_mse = mean_squared_error(probe_target_test, predictions.cpu().numpy())
        test_mse_scores.append(test_mse)
        
        print(f"  Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}, Train MSE: {train_loss:.4f}, Test MSE: {test_mse:.4f}")
    
    # Create and save graph
    print("\nGenerating graph...")
    axis_label_fontsize = 20
    tick_label_fontsize = 16
    legend_fontsize = 16

    plt.figure(figsize=(10, 6))
    plt.plot(layer_numbers, train_r2_scores, 'o-', label='Train R²', linewidth=2, markersize=6)
    plt.plot(layer_numbers, test_r2_scores, 's-', label='Test R²', linewidth=2, markersize=6)
    plt.xlabel('Layer', fontsize=axis_label_fontsize)
    plt.ylabel('R² Score', fontsize=axis_label_fontsize)
    plt.xticks(fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the graph
    output_filename = 'same_dataset_random_split_probe_results_across_layers.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {output_filename}")
    plt.close()
    
    # Create and save MSE graph
    print("\nGenerating MSE graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(layer_numbers, train_mse_scores, 'o-', label='Train MSE', linewidth=2, markersize=6)
    plt.plot(layer_numbers, test_mse_scores, 's-', label='Test MSE', linewidth=2, markersize=6)
    plt.xlabel('Layer', fontsize=axis_label_fontsize)
    plt.ylabel('MSE', fontsize=axis_label_fontsize)
    plt.xticks(fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the graph
    mse_output_filename = 'same_dataset_random_split_probe_results_across_layers_mse.png'
    plt.savefig(mse_output_filename, dpi=300, bbox_inches='tight')
    print(f"MSE graph saved as: {mse_output_filename}")
    plt.close()

    # Create and save combined R2 + MSE graph with dual y-axes
    print("\nGenerating combined R² + MSE graph...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    r2_train_line = ax1.plot(layer_numbers, train_r2_scores, 'o-', color='tab:blue', label='Train R²', linewidth=2, markersize=6)
    r2_test_line = ax1.plot(layer_numbers, test_r2_scores, 's-', color='tab:cyan', label='Test R²', linewidth=2, markersize=6)

    mse_train_line = ax2.plot(layer_numbers, train_mse_scores, 'o--', color='tab:red', label='Train MSE', linewidth=2, markersize=6)
    mse_test_line = ax2.plot(layer_numbers, test_mse_scores, 's--', color='tab:orange', label='Test MSE', linewidth=2, markersize=6)

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

    combined_output_filename = 'same_dataset_random_split_probe_results_across_layers_r2_mse.png'
    plt.savefig(combined_output_filename, dpi=300, bbox_inches='tight')
    print(f"Combined R²+MSE graph saved as: {combined_output_filename}")
    plt.close()
    
    # Summary
    print("\n" + "="*50)
    print("PROBE PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Best Train R2: {max(train_r2_scores):.4f} at layer {layer_numbers[np.argmax(train_r2_scores)]}")
    print(f"Best Test R2: {max(test_r2_scores):.4f} at layer {layer_numbers[np.argmax(test_r2_scores)]}")
    
    #########################################################
    ## Probe Complexity Analysis for Layer 8
    #########################################################
    
    print("\n" + "="*50)
    print("PROBE COMPLEXITY ANALYSIS FOR LAYER 8")
    print("="*50)
    
    # Extract activations for layer 8
    target_layer = 'layer_8'
    if target_layer not in activations:
        print(f"Warning: {target_layer} not found. Available layers: {list(activations.keys())}")
        # Try to find the closest layer
        available_layers = sorted(activations.keys(), key=lambda x: int(x.split('_')[1]))
        if available_layers:
            target_layer = available_layers[min(8, len(available_layers)-1)]
            print(f"Using {target_layer} instead")
    
    print(f"\nAnalyzing probe complexity for {target_layer}...")
    
    # Get activations for layer 8
    all_activations_at_layer = activations[target_layer]
    all_activations_at_layer = torch.cat(
        (all_activations_at_layer[:, :2, :], all_activations_at_layer[:, 4:, :]), dim=1
    )  # use every token except the 4th token (index 3)
    
    # Split activations using train_test_split
    activation_train, activation_test, probe_target_train, probe_target_test = train_test_split(
        all_activations_at_layer, all_targets, test_size=0.2, random_state=42
    )
    
    # Convert to tensors
    activation_train_tensor = torch.tensor(activation_train, dtype=torch.float32)
    activation_test_tensor = torch.tensor(activation_test, dtype=torch.float32)
    target_train_tensor = torch.tensor(probe_target_train, dtype=torch.float32)
    target_test_tensor = torch.tensor(probe_target_test, dtype=torch.float32)
    
    # Test different probe complexities (0 to 5 hidden layers)
    max_hidden_layers = 5
    complexities = list(range(max_hidden_layers + 1))  # 0, 1, 2, 3, 4, 5
    complexity_train_r2 = []
    complexity_test_r2 = []
    complexity_train_mse = []
    complexity_test_mse = []
    
    print(f"\nTraining probes with 0 to {max_hidden_layers} hidden layers...")
    
    for num_hidden in complexities:
        print(f"  Training probe with {num_hidden} hidden layer(s)...", end=" ")
        train_r2, test_r2, train_mse, test_mse = train_variable_complexity_probe(
            activation_train_tensor, activation_test_tensor,
            target_train_tensor, target_test_tensor,
            num_hidden_layers=num_hidden,
            device=device,
            epochs=100,
            lr=0.001,
            hidden_dim=256
        )
        complexity_train_r2.append(train_r2)
        complexity_test_r2.append(test_r2)
        complexity_train_mse.append(train_mse)
        complexity_test_mse.append(test_mse)
        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    
    # Create and save complexity graph
    print("\nGenerating complexity graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(complexities, complexity_train_r2, 'o-', label='Train R²', linewidth=2, markersize=8)
    plt.plot(complexities, complexity_test_r2, 's-', label='Test R²', linewidth=2, markersize=8)
    plt.xlabel('Number of Hidden Layers', fontsize=axis_label_fontsize)
    plt.ylabel('R² Score', fontsize=axis_label_fontsize)
    plt.xticks(complexities, fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the graph
    complexity_output_filename = 'same_dataset_random_split_probe_results_probe_complexity_layer8.png'
    plt.savefig(complexity_output_filename, dpi=300, bbox_inches='tight')
    print(f"Complexity graph saved as: {complexity_output_filename}")
    plt.close()
    
    # Create and save complexity MSE graph
    print("\nGenerating complexity MSE graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(complexities, complexity_train_mse, 'o-', label='Train MSE', linewidth=2, markersize=8)
    plt.plot(complexities, complexity_test_mse, 's-', label='Test MSE', linewidth=2, markersize=8)
    plt.xlabel('Number of Hidden Layers', fontsize=axis_label_fontsize)
    plt.ylabel('MSE', fontsize=axis_label_fontsize)
    plt.xticks(complexities, fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the graph
    complexity_mse_output_filename = 'same_dataset_random_split_probe_results_probe_complexity_layer8_mse.png'
    plt.savefig(complexity_mse_output_filename, dpi=300, bbox_inches='tight')
    print(f"Complexity MSE graph saved as: {complexity_mse_output_filename}")
    plt.close()

    # Create and save combined R2 + MSE graph for complexity sweep
    print("\nGenerating combined complexity R² + MSE graph...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    r2_train_line = ax1.plot(complexities, complexity_train_r2, 'o-', color='tab:blue', label='Train R²', linewidth=2, markersize=8)
    r2_test_line = ax1.plot(complexities, complexity_test_r2, 's-', color='tab:cyan', label='Test R²', linewidth=2, markersize=8)

    mse_train_line = ax2.plot(complexities, complexity_train_mse, 'o--', color='tab:red', label='Train MSE', linewidth=2, markersize=8)
    mse_test_line = ax2.plot(complexities, complexity_test_mse, 's--', color='tab:orange', label='Test MSE', linewidth=2, markersize=8)

    ax1.set_xlabel('Number of Hidden Layers', fontsize=axis_label_fontsize)
    ax1.set_ylabel('R² Score', fontsize=axis_label_fontsize, color='tab:blue')
    ax2.set_ylabel('MSE', fontsize=axis_label_fontsize, color='tab:red')

    ax1.tick_params(axis='both', labelsize=tick_label_fontsize, colors='black')
    ax2.tick_params(axis='both', labelsize=tick_label_fontsize, colors='black')

    lines = r2_train_line + r2_test_line + mse_train_line + mse_test_line
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=legend_fontsize, loc='best')

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    complexity_combined_output_filename = 'same_dataset_random_split_probe_results_probe_complexity_layer8_r2_mse.png'
    plt.savefig(complexity_combined_output_filename, dpi=300, bbox_inches='tight')
    print(f"Combined complexity R²+MSE graph saved as: {complexity_combined_output_filename}")
    plt.close()
    
    # Summary for complexity analysis
    print("\n" + "="*50)
    print("COMPLEXITY ANALYSIS SUMMARY")
    print("="*50)
    print(f"Best Train R2: {max(complexity_train_r2):.4f} with {complexities[np.argmax(complexity_train_r2)]} hidden layer(s)")
    print(f"Best Test R2: {max(complexity_test_r2):.4f} with {complexities[np.argmax(complexity_test_r2)]} hidden layer(s)")
    
    # # Summary of results
    # print("\n" + "="*50)
    # print("SUMMARY")
    # print("="*50)
    # print(f"Layer: {TARGET_LAYER}")
    # print(f"Training Loss: {train_loss:.4f}")
    # print(f"Training R2 Score: {train_r2:.4f}")

if __name__ == "__main__":
    main()
